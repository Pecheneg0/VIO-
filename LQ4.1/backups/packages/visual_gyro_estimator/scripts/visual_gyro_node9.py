#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Hybrid Visual-Inertial Flow Estimator (Global N-E Frame + Dynamic Yaw)
=======================================================================
Рассчитывает глобальную позицию (North, East) и курс (Yaw) в реальном времени.
Логика:
   GROUND: Дрон на земле. Позиция не интегрируется.
   WAIT_LIFT: Высота > 2.5м. Ждём 3 сек стабилизации.
   ARMED: Позиция сброшена в (0,0). Скорости поворачиваются на динамический Yaw.
   Публикует Odometry в глобальной системе координат (N-E).
"""

import rospy
import cv2
import math
import numpy as np
import tf
from tf import transformations
from sensor_msgs.msg import Imu, Image, Range
from geometry_msgs.msg import TwistStamped
from nav_msgs.msg import Odometry
from std_msgs.msg import Empty
from cv_bridge import CvBridge
from collections import deque

class HybridKalmanFilter:
    def __init__(self, R=0.7, Q=0.12, damping=0.0):
        self.R, self.Q, self.damping = R, Q, damping
        self.x, self.P = 0.0, 1.0
    def predict(self, dt=1.0):
        self.P += self.Q * dt; self.P = min(self.P, 5.0)
        self.x *= max(0.0, 1.0 - self.damping * dt)
    def update(self, z, dt=1.0):
        self.P += self.Q * dt
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x); self.P *= (1.0 - K)
        return self.x

class HybridFlowNode:
    def __init__(self):
        rospy.init_node('hybrid_flow_node', anonymous=True)
        rospy.loginfo(" Global Flow Node initializing...")

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
        
        raw_db = rospy.get_param('~vel_deadband', 0.015)
        self.vel_deadband = max(0.001, min(float(raw_db), 0.030))
        #rospy.loginfo(f"️ vel_deadband clamped: {raw_db} → {self.vel_deadband:.3f} m/s")
        self.kf_R = rospy.get_param('~kf_R', 0.7)
        self.kf_Q = rospy.get_param('~kf_Q', 0.12)
        self.coast_damping = rospy.get_param('~coast_damping', 0.5)
        self.cov_growth_rate = rospy.get_param('~cov_growth_rate', 0.015)
        self.max_coast_frames = rospy.get_param('~max_coast_frames', 50)
        self.scale_factor = rospy.get_param('~scale_factor', 1.0)

        # === 3.  ГЛОБАЛЬНАЯ НАВИГАЦИЯ & СОСТОЯНИЯ ===
        self.initial_heading_deg = rospy.get_param('~initial_heading_deg', 0.0)
        self.yaw = math.radians(self.initial_heading_deg)
        self.last_yaw_time = None
        
        self.state = 'GROUND'  # GROUND -> WAIT_LIFT -> ARMED
        self.lift_start_time = None
        self.lift_height_threshold = rospy.get_param('~lift_height', 2.5)
        self.lift_confirm_delay = rospy.get_param('~lift_delay', 3.0)

        # Топики
        self.image_topic = rospy.get_param('~image_topic', '/data')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_raw')
        self.lidar_topic = '/tfmini_ros_node/range_20hz'
        self.output_vel_topic = '/drone/ground_velocity'
        self.output_odom_topic = '/drone/odom'

        self.bridge = CvBridge()
        self.imu_buffer = deque(maxlen=self.imu_buffer_len)
        self.prev_gray, self.prev_pts, self.last_img_time = None, None, None
        self.current_height = 1.0
        self.lidar_received = False

        # HYBRID STATE
        self.kf_x = HybridKalmanFilter(R=self.kf_R, Q=self.kf_Q, damping=self.coast_damping)
        self.kf_y = HybridKalmanFilter(R=self.kf_R, Q=self.kf_Q, damping=self.coast_damping)
        self.vision_active = True
        self.coast_counter = 0
        self.cov_xy = 0.05

        # Позиция (Глобальная N-E)
        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None

        # Publishers & Subscribers
        self.vel_pub = rospy.Publisher(self.output_vel_topic, TwistStamped, queue_size=10)
        self.odom_pub = rospy.Publisher(self.output_odom_topic, Odometry, queue_size=10)
        self.tf_br = tf.TransformBroadcaster()
        
        rospy.Subscriber('/drone/reset_pos', Empty, self.reset_position_callback)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(self.lidar_topic, Range, self.lidar_callback)

        rospy.loginfo(f" Listening: {self.image_topic}, {self.imu_topic}, {self.lidar_topic}")
        rospy.loginfo(f" Initial Heading: {self.initial_heading_deg}° | State: {self.state}")
        rospy.loginfo(" Global Node ready. Waiting for lift-off...")

    def reset_position_callback(self, msg):
        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None
        self.coast_counter = 0
        self.vision_active = True
        self.cov_xy = 0.05
        self.state = 'ARMED' # Принудительный сброс в боевой режим
        rospy.loginfo("📍 Position manually reset. ARMED.")

    def imu_callback(self, msg):
        if rospy.is_shutdown(): return
        t = msg.header.stamp.to_sec()
        w_raw = msg.angular_velocity
        self.imu_buffer.append({'t': t, 'w': [w_raw.x, w_raw.y, w_raw.z]})

        #  ДИНАМИЧЕСКИЙ РАСЧЁТ YAW
        if self.state != 'GROUND':
            wz = w_raw.z
            # Deadband для защиты от дрейфа в статике
            if abs(wz) < 0.005: wz = 0.0
            
            if self.last_yaw_time is not None:
                dt_yaw = t - self.last_yaw_time
                if 0.005 < dt_yaw < 0.1:
                    self.yaw -= wz * dt_yaw
                    # Нормализация в [-π, π]
                    self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
            self.last_yaw_time = t

    def lidar_callback(self, msg):
        if rospy.is_shutdown(): return
        h = msg.range
        if 0.1 <= h <= 12.0:
            self.current_height = h
            self.lidar_received = True

            #  МАШИНА СОСТОЯНИЙ
            if self.state == 'GROUND' and h > self.lift_height_threshold:
                self.state = 'WAIT_LIFT'
                self.lift_start_time = rospy.Time.now().to_sec()
                rospy.loginfo(f" Lift detected (H={h:.2f}m). Stabilizing for {self.lift_confirm_delay}s...")
            elif self.state == 'WAIT_LIFT':
                if h < self.lift_height_threshold: # Если сел обратно
                    self.state = 'GROUND'
                    self.lift_start_time = None
                    rospy.loginfo("Landed. Back to GROUND state.")
                elif rospy.Time.now().to_sec() - self.lift_start_time > self.lift_confirm_delay:
                    self.state = 'ARMED'
                    self.pos_x, self.pos_y = 0.0, 0.0
                    self.last_integ_time = None
                    rospy.loginfo(f" ARMED! Position reset. Yaw={math.degrees(self.yaw):.1f}°")

    def image_callback(self, msg):
        if rospy.is_shutdown(): return
        try:
            gray = self.bridge.imgmsg_to_cv2(msg, desired_encoding="mono8")
        except Exception as e:
            rospy.logwarn_throttle(1.0, f"Bridge error: {e}")
            return

        img_time = msg.header.stamp.to_sec()
        
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

        if curr_pts is not None and status is not None:
            status_flat = status.flatten()
            err_flat = err.flatten() if err is not None else np.full_like(status_flat, np.inf)
            valid = (status_flat == 1) & np.isfinite(err_flat) & (err_flat < self.max_flow_err)
            
            good_new = curr_pts[valid].reshape(-1, 2)
            good_old = pts_in[valid].reshape(-1, 2)
            n_valid = len(good_new)
            vision_ok = n_valid >= self.min_features
            
            dt = img_time - self.last_img_time
            if dt < 0.015: dt = 0.033
            if dt > 0.05: dt = 0.05

            if vision_ok:
                self.coast_counter = 0
                if not self.vision_active:
                    rospy.loginfo(" Vision recovered.")
                self.vision_active = True
                self.cov_xy = 0.05

                median_flow = np.median(good_new - good_old, axis=0)
                flow_x_px, flow_y_px = median_flow[0]/dt, median_flow[1]/dt

                imu_w = self._get_closest_imu(img_time)
                if imu_w is not None:
                    imu_in_cam = self.R_cam_imu @ np.array(imu_w['w'])
                    flow_x_comp = flow_x_px + imu_in_cam[1] * self.fx
                    flow_y_comp = flow_y_px - imu_in_cam[0] * self.fy
                else:
                    flow_x_comp, flow_y_comp = flow_x_px, flow_y_px

                vx_cam = flow_x_comp * self.current_height / self.fx
                vy_cam = flow_y_comp * self.current_height / self.fy
                vx_drone = -vy_cam * self.scale_factor
                vy_drone = vx_cam * self.scale_factor

                vx_filt = self.kf_x.update(vx_drone, dt)
                vy_filt = self.kf_y.update(vy_drone, dt)
            else:
                self.coast_counter += 1
                if self.coast_counter == 1: rospy.logwarn(" Texture lost! Coast mode.")
                self.vision_active = False
                self.cov_xy = min(0.05 + self.coast_counter * self.cov_growth_rate, 2.0)
                self.kf_x.predict(dt); self.kf_y.predict(dt)
                vx_filt, vy_filt = self.kf_x.x, self.kf_y.x

            if abs(vx_filt) < self.vel_deadband: vx_filt = 0.0
            if abs(vy_filt) < self.vel_deadband: vy_filt = 0.0
            if self.coast_counter > self.max_coast_frames:
                vx_filt *= 0.95; vy_filt *= 0.95

            # === ПУБЛИКАЦИЯ & ИНТЕГРАЦИЯ ===
            if self.lidar_received and not rospy.is_shutdown():
                msg_vel = TwistStamped()
                msg_vel.header.stamp = msg.header.stamp
                msg_vel.twist.linear.x, msg_vel.twist.linear.y = vx_filt, vy_filt
                self.vel_pub.publish(msg_vel)

                if self.state == 'ARMED':
                    #  ПОВОРОТ В ГЛОБАЛЬНУЮ СИСТЕМУ (N-E)
                    cy, sy = math.cos(self.yaw), math.sin(self.yaw)
                    v_north = vx_filt * cy - vy_filt * sy
                    v_east  = vx_filt * sy + vy_filt * cy

                    if self.last_integ_time is None:
                        self.last_integ_time = img_time
                    else:
                        dt_pos = img_time - self.last_img_time
                        if 0.01 < dt_pos < 0.15:
                            self.pos_x += v_north * dt_pos
                            self.pos_y += v_east * dt_pos
                            self.last_integ_time = img_time

                            # Odometry
                            odom = Odometry()
                            odom.header.stamp = msg.header.stamp
                            odom.header.frame_id = "odom"
                            odom.child_frame_id = "base_link"
                            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = self.pos_x, self.pos_y, self.current_height
                            
                            # Кватернион из yaw
                            q = transformations.quaternion_from_euler(0.0, 0.0, self.yaw)
                            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = q[0], q[1]
                            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q[2], q[3]
                            
                            # Twist остаётся в теле (стандарт ROS)
                            odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx_filt, vy_filt
                            c = self.cov_xy
                            odom.pose.covariance[0], odom.pose.covariance[7] = c, c
                            odom.twist.covariance[0], odom.twist.covariance[7] = c*0.5, c*0.5
                            self.odom_pub.publish(odom)

                            # TF
                            self.tf_br.sendTransform((self.pos_x, self.pos_y, self.current_height), q, msg.header.stamp, "base_link", "odom")
                            rospy.loginfo_throttle(1.5, f" ARMED | Yaw={math.degrees(self.yaw):.1f}° | N={self.pos_x:+.2f} E={self.pos_y:+.2f}")
                else:
                    rospy.loginfo_throttle(3.0, f" State: {self.state} | Waiting for flight...")

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
