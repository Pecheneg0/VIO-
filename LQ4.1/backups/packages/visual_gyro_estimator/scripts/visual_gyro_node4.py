#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Flow Velocity Estimator (Production + IMU Compensation + Odometry)
==========================================================================
Рассчитывает линейную скорость дрона над землёй (X=вперёд, Y=вправо)
Интегрирует позицию, публикует TwistStamped и nav_msgs/Odometry
Содержит:
  - Оптический поток (Lucas-Kanade)
  - Компенсацию вращения IMU (физически точная модель)
  - Kalman Filter для сглаживания скоростей
  - ZUPT-защиту от дрейфа в статике
  - Отладочные логи для диагностики публикации
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

class SimpleKalmanFilter:
    """
    1D Kalman Filter для сглаживания скорости.
    Модель: постоянная скорость между кадрами + процессный шум Q.
    Адаптивно взвешивает предсказание и новое измерение R.
    """
    def __init__(self, R=0.5, Q=0.1):
        self.R = R  # Шум измерения (чем больше, тем сильнее сглаживание)
        self.Q = Q  # Шум процесса (допустимое ускорение)
        self.x = 0.0  # Текущая оценка скорости
        self.P = 1.0  # Неуверенность оценки

    def update(self, z, dt=1.0):
        self.P += self.Q * dt          # Прогноз неуверенности
        K = self.P / (self.P + self.R) # Коэффициент Калмана (0..1)
        self.x += K * (z - self.x)     # Коррекция оценки
        self.P *= (1.0 - K)            # Обновление неуверенности
        return self.x

class FlowVelocityNode:
    def __init__(self):
        rospy.init_node('flow_velocity_node', anonymous=True)
        rospy.loginfo("🟢 Flow Velocity Node initializing...")

        # === 1. КАЛИБРОВКА И ПАРАМЕТРЫ ===
        self.fx = rospy.get_param('~fx', 560.7238)
        self.fy = rospy.get_param('~fy', 554.3852)
        
        # Матрица поворота IMU -> Camera (из kalibr)
        R_list = rospy.get_param('~R_cam_imu', [
            -0.02674005, -0.99928846, -0.02659981,
            -0.98749351,  0.03054164, -0.15467312,
             0.15537547,  0.02213117, -0.98760755
        ])
        self.R_cam_imu = np.array(R_list).reshape(3, 3)

        # Параметры трекинга
        self.max_features = rospy.get_param('~max_features', 120)
        self.min_features = rospy.get_param('~min_features', 15)
        self.flow_win = rospy.get_param('~flow_window_size', 21)
        self.max_flow_err = rospy.get_param('~max_flow_error', 15.0)
        self.quality_level = rospy.get_param('~quality_level', 0.1)
        self.min_dist = rospy.get_param('~min_distance', 10)
        self.block_size = rospy.get_param('~block_size', 9)
        self.imu_buffer_len = 60
        self.max_sync_offset = 0.05
        self.vel_deadband = rospy.get_param('~vel_deadband', 0.02)  # м/с, порог обнуления

        # Топики
        self.image_topic = rospy.get_param('~image_topic', '/data')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_raw')
        self.lidar_topic = '/tfmini_ros_node/range'
        self.output_vel_topic = '/drone/ground_velocity'
        self.output_odom_topic = '/drone/odom'

        # Состояние
        self.bridge = CvBridge()
        self.imu_buffer = deque(maxlen=self.imu_buffer_len)
        self.prev_gray, self.prev_pts, self.last_img_time = None, None, None
        self.current_height = 1.0
        self.lidar_received = False

        # Фильтры
        self.kf_x = SimpleKalmanFilter(R=0.6, Q=0.15)
        self.kf_y = SimpleKalmanFilter(R=0.6, Q=0.15)

        # Позиция (интеграл скоростей)
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
        rospy.loginfo(f"📤 Publishing: {self.output_vel_topic}, {self.output_odom_topic}")
        rospy.loginfo("✅ Node ready.")

    def reset_position_callback(self, msg):
        """Сброс позиции в (0,0) через топик"""
        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None
        rospy.loginfo("📍 Position reset to (0,0)")

    def imu_callback(self, msg):
        """Буферизация IMU с таймстемпами для последующей синхронизации"""
        if rospy.is_shutdown(): return
        self.imu_buffer.append({
            't': msg.header.stamp.to_sec(),
            'w': [msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z]
        })

    def lidar_callback(self, msg):
        """Приём высоты. Устанавливаем флаг готовности для одометрии"""
        if rospy.is_shutdown(): return
        if 0.1 <= msg.range <= 12.0:
            self.current_height = msg.range
            self.lidar_received = True
            rospy.loginfo_throttle(5.0, f"📏 Lidar active | H={self.current_height:.2f}m")

    def image_callback(self, msg):
        """
        ГЛАВНЫЙ ЦИКЛ ОБРАБОТКИ.
        Выполняется при каждом новом кадре камеры.
        """
        if rospy.is_shutdown(): return

        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Bridge error: {e}")
            return

        img_time = msg.header.stamp.to_sec()
        
        # Инициализация первого кадра
        if self.prev_gray is None:
            self.prev_gray = gray.copy()
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
            self.last_img_time = img_time
            return

        # Пересоздание точек, если трекинг потерян
        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        pts_in = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)

        # Расчёт оптического потока
        try:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(
                self.prev_gray, gray, pts_in, None, 
                winSize=(self.flow_win, self.flow_win), maxLevel=2, 
                criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02)
            )
        except cv2.error:
            rospy.logwarn_throttle(1.0, "⚠️ OpenCV LK failed. Re-detecting.")
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        # === ФИЛЬТРАЦИЯ ТРЕКОВ + ЗАЩИТА ОТ NaN ===
        if curr_pts is not None and status is not None:
            status_flat = status.flatten()
            # err может быть None или содержать NaN/Inf. np.isfinite() отсекает их.
            err_flat = err.flatten() if err is not None else np.full_like(status_flat, np.inf)
            is_finite = np.isfinite(err_flat)
            
            valid = (status_flat == 1) & is_finite & (err_flat < self.max_flow_err)
            
            good_new = curr_pts[valid].reshape(-1, 2)
            good_old = pts_in[valid].reshape(-1, 2)

            if len(good_new) >= self.min_features:
                # Медиана устойчива к выбросам (блики, тени)
                median_flow = np.median(good_new - good_old, axis=0)
                dt = img_time - self.last_img_time
                if dt < 0.01: dt = 0.033  # Защита под ~30Hz

                # 1. Пиксельный поток (пиксели/сек)
                flow_x_px = median_flow[0] / dt
                flow_y_px = median_flow[1] / dt

                # 2. 🔥 КОМПЕНСАЦИЯ ВРАЩЕНИЯ IMU (Физически точная модель)
                # Вращение камеры создаёт ложный поток: ω * focal_length
                imu_w = self._get_closest_imu(img_time)
                if imu_w is not None:
                    imu_vec = np.array(imu_w['w'])
                    imu_in_cam = self.R_cam_imu @ imu_vec
                    # Вычитаем вращательную компоненту ДО перевода в м/с
                    flow_x_comp = flow_x_px + imu_in_cam[1] * self.fx
                    flow_y_comp = flow_y_px - imu_in_cam[0] * self.fy
                else:
                    flow_x_comp = flow_x_px
                    flow_y_comp = flow_y_px

                # 3. Перевод в линейную скорость (м/с) в системе камеры
                vx_cam = flow_x_comp * self.current_height / self.fx
                vy_cam = flow_y_comp * self.current_height / self.fy

                # 4. ПОВОРОТ в систему дрона (X=вперёд, Y=вправо)
                vx_drone = -vy_cam
                vy_drone = vx_cam

                # 5. Kalman Filter + Deadband (ZUPT-логика)
                vx_filt = self.kf_x.update(vx_drone, dt)
                vy_filt = self.kf_y.update(vy_drone, dt)
                
                if abs(vx_filt) < self.vel_deadband: vx_filt = 0.0
                if abs(vy_filt) < self.vel_deadband: vy_filt = 0.0

                # 6. Публикация скорости
                if not rospy.is_shutdown() and self.lidar_received:
                    msg_vel = TwistStamped()
                    msg_vel.header.stamp = msg.header.stamp
                    msg_vel.twist.linear.x = vx_filt
                    msg_vel.twist.linear.y = vy_filt
                    self.vel_pub.publish(msg_vel)

                    # 📍 ИНТЕГРАЦИЯ ПОЗИЦИИ + ОДОМЕТРИЯ
                    if self.last_integ_time is None:
                        self.last_integ_time = img_time
                    else:
                        dt_pos = img_time - self.last_img_time
                        
                        # ОТЛАДОЧНЫЙ ЛОГ: показывает, почему одометрия не публикуется
                        rospy.loginfo_throttle(2.0, 
                            f"📍 Check: lidar={self.lidar_received}, dt={dt_pos:.3f}s, "
                            f"vx={vx_filt:.3f}, vy={vy_filt:.3f}")
                        
                        if 0.005 < dt_pos < 0.15:
                            self.pos_x += vx_filt * dt_pos
                            self.pos_y += vy_filt * dt_pos
                            self.last_integ_time = img_time

                            # Формируем стандартное сообщение Odometry
                            odom = Odometry()
                            odom.header.stamp = msg.header.stamp
                            odom.header.frame_id = "odom"
                            odom.child_frame_id = "base_link"
                            odom.pose.pose.position.x = self.pos_x
                            odom.pose.pose.position.y = self.pos_y
                            odom.pose.pose.position.z = self.current_height
                            odom.twist.twist.linear.x = vx_filt
                            odom.twist.twist.linear.y = vy_filt
                            
                            # Ковариации: доверие к данным для EKF полётника
                            odom.pose.covariance[0] = 0.05
                            odom.pose.covariance[7] = 0.05
                            odom.twist.covariance[0] = 0.02
                            odom.twist.covariance[7] = 0.02
                            
                            self.odom_pub.publish(odom)
                            rospy.loginfo_throttle(2.0, f"✅ Odometry OK | X={self.pos_x:+.2f} Y={self.pos_y:+.2f}")
                        else:
                            rospy.logwarn_throttle(2.0, f"⚠️ dt_pos out of range: {dt_pos:.3f}s")
                else:
                    rospy.logwarn_throttle(2.0, f"⚠️ Lidar not ready (received={self.lidar_received})")

        # Обновление состояния для следующего кадра
        self.prev_gray = gray.copy()
        if curr_pts is not None and status is not None:
            valid_pts = curr_pts[status == 1]
            self.prev_pts = valid_pts.reshape(-1, 1, 2) if len(valid_pts) > 0 else cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
        self.last_img_time = img_time

    def _get_closest_imu(self, t_img):
        """Находит замер IMU, ближайший по времени к кадру камеры"""
        if not self.imu_buffer: return None
        closest = min(self.imu_buffer, key=lambda x: abs(x['t'] - t_img))
        return closest if abs(closest['t'] - t_img) <= self.max_sync_offset else None

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        FlowVelocityNode().run()
    except rospy.ROSInterruptException:
        pass
