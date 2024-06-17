#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Visual-Inertial Flow Estimator (Production + Tunable Parameters)
=========================================================================
Рассчитывает линейную скорость дрона (X=вперёд, Y=вправо) и позицию.
Гибридная логика:
  🔹 VISION MODE: Оптический поток + IMU-компенсация + KF update
  🔹 COAST MODE: KF predict с физическим затуханием + рост ковариации
Всегда публикует данные (без "стоп-кадров"). Все ключевые коэффициенты вынесены в rosparam.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Imu, Image, Range
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from collections import deque

class HybridKalmanFilter:
    """1D KF с параметризируемым затуханием скорости при коастинге"""
    def __init__(self, R=0.7, Q=0.12, damping=0.0):
        self.R = R          # Шум измерения (доверие к Vision)
        self.Q = Q          # Шум процесса (допустимое ускорение)
        self.damping = damping  # Затухание скорости при потче зрения (1/с)
        self.x = 0.0        # Оценка скорости
        self.P = 1.0        # Дисперсия оценки

    def predict(self, dt=1.0):
        """Шаг прогноза (используется при потче текстуры)"""
        self.P += self.Q * dt
        self.P = min(self.P, 5.0)
        # Физическое затухание: скорость плавно стремится к 0
        self.x *= max(0.0, 1.0 - self.damping * dt)

    def update(self, z, dt=1.0):
        """Шаг коррекции (используется при наличии фич)"""
        self.P += self.Q * dt
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1.0 - K)
        return self.x

class HybridFlowNode:
    def __init__(self):
        rospy.init_node('hybrid_flow_node', anonymous=True)
        rospy.loginfo("🟢 Hybrid Flow Node initializing...")

        # === 1. КАЛИБРОВКА ===
        self.fx = rospy.get_param('~fx', 560.7238)
        self.fy = rospy.get_param('~fy', 554.3852)
        R_list = rospy.get_param('~R_cam_imu', [
            -0.02674005, -0.99928846, -0.02659981,
            -0.98749351,  0.03054164, -0.15467312,
             0.15537547,  0.02213117, -0.98760755
        ])
        self.R_cam_imu = np.array(R_list).reshape(3, 3)

        # === 2. ПАРАМЕТРЫ ТРЕКИНГА ===
        self.max_features = rospy.get_param('~max_features', 120)
        self.min_features = rospy.get_param('~min_features', 14)
        self.flow_win = rospy.get_param('~flow_window_size', 21)
        self.max_flow_err = rospy.get_param('~max_flow_error', 25.0)
        self.quality_level = rospy.get_param('~quality_level', 0.06)
        self.min_dist = rospy.get_param('~min_distance', 8)
        self.block_size = rospy.get_param('~block_size', 9)
        self.imu_buffer_len = 60
        self.max_sync_offset = rospy.get_param('~max_sync_offset', 0.06)
        self.vel_deadband = rospy.get_param('~vel_deadband', 0.015)

        # === 3. 🔧 ТЮНИНГ ПОВЕДЕНИЯ (все вынесены в rosparam) ===
        self.kf_R = rospy.get_param('~kf_R', 0.7)
        self.kf_Q = rospy.get_param('~kf_Q', 0.12)
        self.coast_damping = rospy.get_param('~coast_damping', 0.5)      # 1/с
        self.cov_growth_rate = rospy.get_param('~cov_growth_rate', 0.015)
        self.max_coast_frames = rospy.get_param('~max_coast_frames', 50)

        # Топики
        self.image_topic = rospy.get_param('~image_topic', '/data')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_raw')
        self.lidar_topic = '/tfmini_ros_node/range'
        self.output_vel_topic = '/drone/ground_velocity'
        self.output_odom_topic = '/drone/odom'

        self.bridge = CvBridge()
        self.imu_buffer = deque(maxlen=self.imu_buffer_len)
        self.prev_gray, self.prev_pts, self.last_img_time = None, None, None
        self.current_height = 1.0
        self.lidar_received = False

        # 🔑 HYBRID STATE
        self.kf_x = HybridKalmanFilter(R=self.kf_R, Q=self.kf_Q, damping=self.coast_damping)
        self.kf_y = HybridKalmanFilter(R=self.kf_R, Q=self.kf_Q, damping=self.coast_damping)
        self.vision_active = True
        self.coast_counter = 0
        self.cov_xy = 0.05  # Базовая ковариация

        # Позиция
        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None

        # Publishers & Subscribers
        self.vel_pub = rospy.Publisher(self.output_vel_topic, TwistStamped, queue_size=10)
        self.odom_pub = rospy.Publisher(self.output_odom_topic, Odometry, queue_size=10)
        rospy.Subscriber('/drone/reset_pos', Empty, self.reset_position_callback)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(self.lidar_topic, Range, self.lidar_callback)

        rospy.loginfo(f"📡 Listening: {self.image_topic}, {self.imu_topic}, {self.lidar_topic}")
        rospy.loginfo(f"⚙️ KF: R={self.kf_R}, Q={self.kf_Q}, Damp={self.coast_damping}, CovRate={self.cov_growth_rate}")
        rospy.loginfo("✅ Hybrid Node ready. Vision/IMU Coast mode active.")

    def reset_position_callback(self, msg):
        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None
        self.coast_counter = 0
        self.vision_active = True
        self.cov_xy = 0.05
        rospy.loginfo("📍 Position reset. Vision mode restored.")

    def imu_callback(self, msg):
        if rospy.is_shutdown(): return
        self.imu_buffer.append({
            't': msg.header.stamp.to_sec(),
            'w': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        })

    def lidar_callback(self, msg):
        if rospy.is_shutdown(): return
        if 0.1 <= msg.range <= 12.0:
            self.current_height = msg.range
            self.lidar_received = True

    def image_callback(self, msg):
        if rospy.is_shutdown(): return
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Bridge error: {e}")
            return

        img_time = msg.header.stamp.to_sec()
        
        # Инициализация
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_pts = self._detect_features(gray)
            self.last_img_time = img_time
            return

        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = self._detect_features(gray)
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        pts_in = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)

        try:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts_in, None, 
                winSize=(self.flow_win, self.flow_win), maxLevel=2, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 12, 0.03)
            )
        except cv2.error:
            self.prev_pts = self._detect_features(gray)
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        # Проверка валидности трекинга
        if curr_pts is not None and status is not None:
            status_flat = status.flatten()
            err_flat = err.flatten() if err is not None else np.full_like(status_flat, np.inf)
            valid = (status_flat == 1) & np.isfinite(err_flat) & (err_flat < self.max_flow_err)
            
            good_new = curr_pts[valid].reshape(-1, 2)
            good_old = pts_in[valid].reshape(-1, 2)
            n_valid = len(good_new)

            # === ГЛАВНАЯ ЛОГИКА: HYBRID MODE ===
            vision_ok = n_valid >= self.min_features
            
            dt = img_time - self.last_img_time
            if dt < 0.015: dt = 0.033
            if dt > 0.05: dt = 0.05

            if vision_ok:
                # 🔹 VISION MODE: Обновляем KF измерением
                self.coast_counter = 0
                if not self.vision_active:
                    rospy.loginfo("✅ Vision recovered. Switching to Visual Update mode.")
                self.vision_active = True
                self.cov_xy = 0.05

                median_flow = np.median(good_new - good_old, axis=0)
                flow_x_px = median_flow[0] / dt
                flow_y_px = median_flow[1] / dt

                imu_w = self._get_closest_imu(img_time)
                if imu_w is not None:
                    imu_vec = np.array(imu_w['w'])
                    imu_in_cam = self.R_cam_imu @ imu_vec
                    flow_x_comp = flow_x_px + imu_in_cam[1] * self.fx
                    flow_y_comp = flow_y_px - imu_in_cam[0] * self.fy
                else:
                    flow_x_comp = flow_x_px
                    flow_y_comp = flow_y_px

                vx_cam = flow_x_comp * self.current_height / self.fx
                vy_cam = flow_y_comp * self.current_height / self.fy
                vx_drone = -vy_cam
                vy_drone = vx_cam

                vx_filt = self.kf_x.update(vx_drone, dt)
                vy_filt = self.kf_y.update(vy_drone, dt)

            else:
                # 🔹 COAST MODE: Только прогноз KF (инерциальный дрейф с затуханием)
                self.coast_counter += 1
                if self.coast_counter == 1:
                    rospy.logwarn("⚠️ Texture lost! Switching to IMU Coast mode (damping active).")
                self.vision_active = False
                
                # Ковариация растёт со временем без коррекции
                self.cov_xy = min(0.05 + self.coast_counter * self.cov_growth_rate, 2.0)

                self.kf_x.predict(dt)
                self.kf_y.predict(dt)
                vx_filt = self.kf_x.x  # ✅ Исправлено: атрибут .x
                vy_filt = self.kf_y.x  # ✅ Исправлено: атрибут .x

            # Защита от дрейфа в статике (ZUPT)
            if abs(vx_filt) < self.vel_deadband: vx_filt = 0.0
            if abs(vy_filt) < self.vel_deadband: vy_filt = 0.0

            # Если коастим слишком долго, мягко гасим скорость
            if self.coast_counter > self.max_coast_frames:
                vx_filt *= 0.95
                vy_filt *= 0.95

            # === ПУБЛИКАЦИЯ И ИНТЕГРАЦИЯ (ВСЕГДА, ЕСЛИ ЛИДАР ГОТОВ) ===
            if self.lidar_received and not rospy.is_shutdown():
                msg_vel = TwistStamped()
                msg_vel.header.stamp = msg.header.stamp
                msg_vel.twist.linear.x, msg_vel.twist.linear.y = vx_filt, vy_filt
                self.vel_pub.publish(msg_vel)

                if self.last_integ_time is None:
                    self.last_integ_time = img_time
                else:
                    dt_pos = img_time - self.last_img_time
                    if 0.01 < dt_pos < 0.15:
                        self.pos_x += vx_filt * dt_pos
                        self.pos_y += vy_filt * dt_pos
                        self.last_integ_time = img_time

                        odom = Odometry()
                        odom.header.stamp = msg.header.stamp
                        odom.header.frame_id = "odom"
                        odom.child_frame_id = "base_link"
                        odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = self.pos_x, self.pos_y, self.current_height
                        odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx_filt, vy_filt
                        
                        # 🔑 Динамическая ковариация для EKF полётника
                        c = self.cov_xy
                        odom.pose.covariance[0], odom.pose.covariance[7] = c, c
                        odom.twist.covariance[0], odom.twist.covariance[7] = c*0.5, c*0.5
                        self.odom_pub.publish(odom)

                        mode_str = "VISION" if self.vision_active else f"COAST ({self.coast_counter})"
                        rospy.loginfo_throttle(1.5, f"📍 {mode_str} | X={self.pos_x:+.2f} Y={self.pos_y:+.2f} vx={vx_filt:.3f} cov={c:.2f}")

        # Обновление состояния
        self.prev_gray = gray.copy()
        if curr_pts is not None and status is not None:
            valid_pts = curr_pts[status == 1]
            self.prev_pts = valid_pts.reshape(-1, 1, 2) if len(valid_pts) > 0 else None
        self.last_img_time = img_time

    def _detect_features(self, gray):
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=9)
        return pts if pts is not None and len(pts) > 0 else None

    def _get_closest_imu(self, t_img):
        if not self.imu_buffer: return None
        closest = min(self.imu_buffer, key=lambda x: abs(x['t'] - t_img))
        return closest if abs(closest['t'] - t_img) <= self.max_sync_offset else None

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        HybridFlowNode().run()
    except rospy.ROSInterruptException:
        pass
