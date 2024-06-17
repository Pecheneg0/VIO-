#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Gyro & Velocity Estimator for GNSS-Denied Navigation
============================================================
Рассчитывает:
1. Угловые скорости (ωx, ωy) из оптического потока, скомпенсированные IMU
2. Линейные скорости (Vx, Vy) из оптического потока + высота с лидара
Все вычисления выполняются в системе координат камеры.
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Imu, Image, Range
from geometry_msgs.msg import Vector3, TwistStamped
from cv_bridge import CvBridge
from collections import deque

class VisualGyroEstimator:
    def __init__(self):
        """
        Инициализация ноды.
        ВАЖНО: Publishers создаются ПЕРЕД подписчиками, чтобы избежать гонки,
        когда первый кадр с камеры приходит до инициализации издателей.
        """
        rospy.init_node('visual_gyro_node', anonymous=True)
        rospy.loginfo(" Node initializing...")

        # === 1. КАЛИБРОВКА КАМЕРЫ И ЭКСТРИНСИКИ ===
        self.fx = rospy.get_param('~fx', 560.7238)
        self.fy = rospy.get_param('~fy', 554.3852)
        R_list = rospy.get_param('~R_cam_imu', [
            -0.02674005, -0.99928846, -0.02659981,
            -0.98749351,  0.03054164, -0.15467312,
             0.15537547,  0.02213117, -0.98760755
        ])
        self.R_cam_imu = np.array(R_list).reshape(3, 3)

        # === 2. ПАРАМЕТРЫ ТРЕКИНГА И СИНХРОНИЗАЦИИ ===
        self.max_features = rospy.get_param('~max_features', 120)
        self.min_features = rospy.get_param('~min_features', 15)
        self.flow_win = rospy.get_param('~flow_window_size', 21)
        self.max_flow_err = rospy.get_param('~max_flow_error', 15.0)
        self.quality_level = rospy.get_param('~quality_level', 0.1)
        self.min_dist = rospy.get_param('~min_distance', 10)
        self.block_size = rospy.get_param('~block_size', 9)
        self.imu_buffer_len = rospy.get_param('~imu_buffer_length', 60)
        self.max_sync_offset = rospy.get_param('~max_sync_offset_sec', 0.05)

        # === 3. ТОПИКИ (ABSOLUTE NAMES) ===
        self.image_topic = rospy.get_param('~image_topic', '/data')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_raw')
        self.output_angular_topic = rospy.get_param('~output_angular_topic', '/drone/clean_flow_vel')
        self.output_linear_topic = rospy.get_param('~output_linear_topic', '/drone/ground_velocity')
        self.lidar_topic = '/tfmini_ros_node/range'

        self.bridge = CvBridge()
        self.imu_buffer = deque(maxlen=self.imu_buffer_len)
        
        # Состояние оптического потока
        self.prev_gray = None
        self.prev_pts = None
        self.last_img_time = None

        # Состояние лидара
        self.current_height = 1.0
        self.lidar_received = False

        # === 4. ПУБЛИШЕРЫ СОЗДАЁМ ПЕРВЫМИ (критично!) ===
        self.ang_vel_pub = rospy.Publisher(self.output_angular_topic, Vector3, queue_size=10)
        self.lin_vel_pub = rospy.Publisher(self.output_linear_topic, TwistStamped, queue_size=10)
        rospy.loginfo(f" Publishers created: {self.output_angular_topic}, {self.output_linear_topic}")

        # === 5. ПОДПИСЧИКИ СОЗДАЁМ ПОСЛЕДНИМИ ===
        self.cam_sub = rospy.Subscriber(self.image_topic, Image, self.image_callback)
        self.imu_sub = rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        self.lidar_sub = rospy.Subscriber(self.lidar_topic, Range, self.lidar_callback)
        rospy.loginfo(f" Subscribed to: {self.image_topic}, {self.imu_topic}, {self.lidar_topic}")

        rospy.loginfo(" Visual Gyro Estimator ready.")

    def imu_callback(self, msg):
        """Буферизация IMU для последующей синхронизации с кадрами камеры"""
        if rospy.is_shutdown(): return
        t = msg.header.stamp.to_sec()
        self.imu_buffer.append({
            't': t,
            'w': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        })

    def lidar_callback(self, msg):
        """Приём высоты с лидара. Простая валидация диапазона 0.1-12.0 м"""
        if rospy.is_shutdown(): return
        if 0.1 <= msg.range <= 12.0:
            self.current_height = msg.range
            self.lidar_received = True
            rospy.loginfo_throttle(1.0, f" Lidar OK | H={self.current_height:.2f}m")
        else:
            rospy.logwarn_throttle(2.0, f" Lidar out of range: {msg.range:.2f}m")

    def image_callback(self, msg):
        """
        ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ.
        Защита: проверяем, что издатели существуют, перед публикацией.
        """
        if rospy.is_shutdown(): return
        
        # Защита от публикации до инициализации (на всякий случай)
        if not hasattr(self, 'ang_vel_pub') or not hasattr(self, 'lin_vel_pub'):
            return
        
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Bridge error: {e}")
            return

        img_time = msg.header.stamp.to_sec()
        
        # Инициализация при первом кадре
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self._get_feature_params())
            self.last_img_time = img_time
            return

        # Пересоздание точек, если они потерялись
        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self._get_feature_params())
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        pts_in = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)

        try:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts_in, None, **self._get_lk_params()
            )
        except cv2.error as e:
            rospy.logwarn_throttle(1.0, f"OpenCV LK error: {e}")
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self._get_feature_params())
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        if curr_pts is not None and status is not None:
            status_flat = status.flatten()
            err_flat = err.flatten()
            valid = (status_flat == 1) & (err_flat < self.max_flow_err)
            
            good_new = curr_pts[valid].reshape(-1, 2)
            good_old = pts_in[valid].reshape(-1, 2)

            if len(good_new) >= self.min_features:
                median_flow = np.median(good_new - good_old, axis=0)
                
                dt = img_time - self.last_img_time
                if dt < 0.01: dt = 0.033
                
                # Угловые скорости
                omega_cam_x = (median_flow[0] / dt) / self.fx
                omega_cam_y = (median_flow[1] / dt) / self.fy
                
                # Компенсация IMU
                imu_w = self._get_closest_imu(img_time)
                if imu_w is not None:
                    imu_vec = np.array(imu_w['w'])
                    imu_in_cam = self.R_cam_imu @ imu_vec
                    clean_omega_x = omega_cam_x - imu_in_cam[0]
                    clean_omega_y = imu_in_cam[1] - omega_cam_y
                else:
                    clean_omega_x, clean_omega_y = omega_cam_x, -omega_cam_y

                # Публикация угловых скоростей
                if not rospy.is_shutdown():
                    ang_msg = Vector3()
                    ang_msg.x, ang_msg.y, ang_msg.z = clean_omega_x, clean_omega_y, 0.0
                    self.ang_vel_pub.publish(ang_msg)

                # Линейные скорости (только если лидар активен)
                if self.lidar_received and self.current_height > 0.05 and not rospy.is_shutdown():
                    vx = (median_flow[0] / dt) * self.current_height / self.fx
                    vy = (median_flow[1] / dt) * self.current_height / self.fy
                    
                    lin_msg = TwistStamped()
                    lin_msg.header.stamp = msg.header.stamp
                    lin_msg.twist.linear.x = vx
                    lin_msg.twist.linear.y = vy
                    self.lin_vel_pub.publish(lin_msg)

        # Обновление состояния
        self.prev_gray = gray.copy()
        if curr_pts is not None and status is not None:
            valid_pts = curr_pts[status == 1]
            self.prev_pts = valid_pts.reshape(-1, 1, 2) if len(valid_pts) > 0 else cv2.goodFeaturesToTrack(gray, **self._get_feature_params())
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, **self._get_feature_params())
        self.last_img_time = img_time

    def _get_closest_imu(self, t_img):
        """Поиск ближайшего по времени замера IMU"""
        if not self.imu_buffer: return None
        closest = min(self.imu_buffer, key=lambda x: abs(x['t'] - t_img))
        return closest if abs(closest['t'] - t_img) <= self.max_sync_offset else None

    def _get_feature_params(self):
        return dict(
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_dist,
            blockSize=self.block_size
        )

    def _get_lk_params(self):
        return dict(
            winSize=(self.flow_win, self.flow_win),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02)
        )

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        VisualGyroEstimator().run()
    except rospy.ROSInterruptException:
        pass
