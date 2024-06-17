#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visual Flow Velocity Estimator (Production)
============================================
Рассчитывает линейную скорость дрона над землёй (X=вперёд, Y=влево)
на основе оптического потока + высота лидара + IMU-компенсация вращения.
Публикует: /drone/ground_velocity (TwistStamped, м/с)
"""

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Imu, Image, Range
from geometry_msgs.msg import TwistStamped, Vector3
from cv_bridge import CvBridge
from collections import deque

class SimpleKalmanFilter:
    def __init__(self, R=0.5, Q=0.1):
        self.R, self.Q = R, Q
        self.x, self.P = 0.0, 1.0
    def update(self, z, dt=1.0):
        self.P += self.Q * dt
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x)
        self.P *= (1.0 - K)
        return self.x

class FlowVelocityNode:
    def __init__(self):
        rospy.init_node('flow_velocity_node', anonymous=True)
        rospy.loginfo(" Flow Velocity Node starting...")

        # Калибровка
        self.fx = rospy.get_param('~fx', 560.7238)
        self.fy = rospy.get_param('~fy', 554.3852)
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
        self.vel_deadband = rospy.get_param('~vel_deadband', 0.02)  # м/с

        # Топики
        self.image_topic = rospy.get_param('~image_topic', '/data')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_raw')
        self.lidar_topic = '/tfmini_ros_node/range'
        self.output_topic = '/drone/ground_velocity'

        self.bridge = CvBridge()
        self.imu_buffer = deque(maxlen=self.imu_buffer_len)
        self.prev_gray, self.prev_pts, self.last_img_time = None, None, None
        self.current_height = 1.0
        self.lidar_received = False

        # KF для сглаживания линейных скоростей
        self.kf_x = SimpleKalmanFilter(R=0.6, Q=0.15)
        self.kf_y = SimpleKalmanFilter(R=0.6, Q=0.15)

        # Publishers & Subscribers
        self.vel_pub = rospy.Publisher(self.output_topic, TwistStamped, queue_size=10)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(self.lidar_topic, Range, self.lidar_callback)

        rospy.loginfo(f" Listening: {self.image_topic}, {self.imu_topic}, {self.lidar_topic}")
        rospy.loginfo(f" Publishing: {self.output_topic} (X=forward, Y=left)")
        rospy.loginfo(" Ready.")

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
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
            self.last_img_time = img_time
            return

        if self.prev_pts is None or len(self.prev_pts) == 0:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        pts_in = self.prev_pts.reshape(-1, 1, 2).astype(np.float32)

        try:
            curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, pts_in, None, winSize=(self.flow_win, self.flow_win), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 15, 0.02))
        except cv2.error:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
            self.prev_gray = gray.copy()
            self.last_img_time = img_time
            return

        if curr_pts is not None and status is not None:
            valid = (status.flatten() == 1) & (err.flatten() < self.max_flow_err)
            good_new = curr_pts[valid].reshape(-1, 2)
            good_old = pts_in[valid].reshape(-1, 2)

            if len(good_new) >= self.min_features:
                median_flow = np.median(good_new - good_old, axis=0)
                dt = img_time - self.last_img_time
                if dt < 0.01: dt = 0.033

                # 1. Сырые скорости в системе КАМЕРЫ (пиксели/с -> м/с)
                # Камера: +X вправо, +Y вниз. Дрон смотрит вниз.
                vx_cam = (median_flow[0] / dt) * self.current_height / self.fx
                vy_cam = (median_flow[1] / dt) * self.current_height / self.fy

                # 2. ПОВОРОТ в систему ДРОНА (REP-103: X=вперёд, Y=влево)
                # Физика: если дрон летит вперёд (+X), земля в кадре движется ВВЕРХ (-Y_cam)
                # Если дрон летит влево (+Y), земля движется ВПРАВО (+X_cam)
                vx_drone = -vy_cam  # Вперёд
                vy_drone = vx_cam   # Влево

                # 3. Компенсация вращения IMU (убирает "ветер" от кренов)
                imu_w = self._get_closest_imu(img_time)
                if imu_w is not None:
                    imu_vec = np.array(imu_w['w'])
                    imu_in_cam = self.R_cam_imu @ imu_vec
                    # Угловая скорость вокруг Z камеры создает линейный поток на краях
                    # Простейшая компенсация: вычитаем вращательную компоненту
                    rot_comp_x = imu_in_cam[1] * self.current_height * 0.5  # упрощённо
                    rot_comp_y = -imu_in_cam[0] * self.current_height * 0.5
                    vx_drone -= rot_comp_x
                    vy_drone -= rot_comp_y

                # 4. Kalman Filter + Deadband (защита от дрейфа)
                vx_filt = self.kf_x.update(vx_drone, dt)
                vy_filt = self.kf_y.update(vy_drone, dt)
                
                if abs(vx_filt) < self.vel_deadband: vx_filt = 0.0
                if abs(vy_filt) < self.vel_deadband: vy_filt = 0.0

                # 5. Публикация
                if not rospy.is_shutdown() and self.lidar_received:
                    msg_out = TwistStamped()
                    msg_out.header.stamp = msg.header.stamp
                    msg_out.twist.linear.x = vx_filt
                    msg_out.twist.linear.y = vy_filt
                    self.vel_pub.publish(msg_out)

        self.prev_gray = gray.copy()
        if curr_pts is not None and status is not None:
            self.prev_pts = curr_pts[status == 1].reshape(-1, 1, 2)
        else:
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=self.block_size)
        self.last_img_time = img_time

    def _get_closest_imu(self, t_img):
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
