#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    def update(self, z, dt=1.0):
        self.P += self.Q * dt
        K = self.P / (self.P + self.R)
        self.x += K * (z - self.x); self.P *= (1.0 - K)
        return self.x

class HybridFlowNode:
    def __init__(self):
        rospy.init_node('hybrid_flow_node', anonymous=True)
        rospy.loginfo("Оптический поток, инициализация.")

        # === 1. КАЛИБРОВКА ===
        self.fx = rospy.get_param('~fx', 922.531465)
        self.fy = rospy.get_param('~fy', 925.020034)
        R_list = rospy.get_param('~R_cam_imu', [
            0.024071420357756907, -0.9994387711955938, -0.023296123132475963,
            -0.9997047740849597, -0.023987684351386294, -0.0038672562883473715,
            0.0033062658244114654, 0.02338233586495525, -0.9997211286032712,
        ])
        self.R_cam_imu = np.array(R_list).reshape(3, 3)

        # === 2. ПАРАМЕТРЫ ТРЕКИНГА ===
        self.max_features = rospy.get_param('~max_features', 120)
        self.min_features = rospy.get_param('~min_features', 8)
        self.flow_win = rospy.get_param('~flow_window_size', 40)
        self.max_flow_err = rospy.get_param('~max_flow_error', 40.0)
        self.quality_level = rospy.get_param('~quality_level', 0.06)
        self.min_dist = rospy.get_param('~min_distance', 8)
        self.block_size = rospy.get_param('~block_size', 9)
        self.imu_buffer_len = 60
        self.max_sync_offset = rospy.get_param('~max_sync_offset', 0.06)
        
        raw_db = rospy.get_param('~vel_deadband', 0.015)
        self.vel_deadband = max(0.001, min(float(raw_db), 0.030))
        self.kf_R = rospy.get_param('~kf_R', 0.7)
        self.kf_Q = rospy.get_param('~kf_Q', 0.12)
        self.coast_damping = rospy.get_param('~coast_damping', 0.10)
        self.cov_growth_rate = rospy.get_param('~cov_growth_rate', 0.015)
        self.max_coast_frames = rospy.get_param('~max_coast_frames', 50)
        self.scale_factor = rospy.get_param('~scale_factor', 1.0)

        # === 3.  ПАРАМЕТРЫ IMU (ГРАВИТАЦИЯ + ДОВЕРИЕ) ===
        self.imu_acc_weight = rospy.get_param('~imu_acc_weight', 0.7)  # Доверие к IMU (0.0-1.0)
        self.gravity = rospy.get_param('~gravity', 9.81)               # Ускорение свободного падения
        self.comp_alpha = rospy.get_param('~comp_alpha', 0.98)         # Вес гироскопа в компл. фильтре
        self.acc_deadband = rospy.get_param('~acc_deadband', 0.1)      # Порог обнуления ускорения (м/с²)
        
        rospy.loginfo(f"IMU: weight={self.imu_acc_weight}, g={self.gravity}, α={self.comp_alpha}")

        # === 4. ГЛОБАЛЬНАЯ НАВИГАЦИЯ & СОСТОЯНИЯ ===
        self.initial_heading_deg = rospy.get_param('~initial_heading_deg', 0.0)
        self.yaw = math.radians(self.initial_heading_deg)
        self.roll = 0.0    #  Крен
        self.pitch = 0.0   #  Тангаж
        self.last_yaw_time = None
        
        self.state = 'GROUND'
        self.lift_start_time = None
        self.lift_height_threshold = rospy.get_param('~lift_height', 2.5)
        self.lift_confirm_delay = rospy.get_param('~lift_delay', 3.0)

        # Топики
        self.image_topic = rospy.get_param('~image_topic', '/data')
        self.imu_topic = rospy.get_param('~imu_topic', '/imu/data_raw')
        self.lidar_topic = '/tfmini_ros_node/range_20hz'
        self.output_vel_topic = '/drone/ground_velocity'
        self.output_odom_topic = '/drone/odom'

        self.imu_buffer = deque(maxlen=self.imu_buffer_len)
        self.imu_acc_buffer = deque(maxlen=self.imu_buffer_len)

        self.bridge = CvBridge()
        self.prev_gray, self.prev_pts, self.last_img_time = None, None, None
        self.current_height = 1.0
        self.lidar_received = False

        self.kf_x = HybridKalmanFilter(R=self.kf_R, Q=self.kf_Q, damping=self.coast_damping)
        self.kf_y = HybridKalmanFilter(R=self.kf_R, Q=self.kf_Q, damping=self.coast_damping)
        self.vision_active = True
        self.coast_counter = 0
        self.cov_xy = 0.05

        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None

        self.vel_pub = rospy.Publisher(self.output_vel_topic, TwistStamped, queue_size=10)
        self.odom_pub = rospy.Publisher(self.output_odom_topic, Odometry, queue_size=10)
        self.tf_br = tf.TransformBroadcaster()
        
        rospy.Subscriber('/drone/reset_pos', Empty, self.reset_position_callback)
        rospy.Subscriber(self.image_topic, Image, self.image_callback)
        rospy.Subscriber(self.imu_topic, Imu, self.imu_callback)
        rospy.Subscriber(self.lidar_topic, Range, self.lidar_callback)

        rospy.loginfo(f" Подписка на топики {self.image_topic}, {self.imu_topic}, {self.lidar_topic}")
        rospy.loginfo(f" Курс {self.initial_heading_deg}° | State: {self.state}")
        rospy.loginfo(" Система оптической навигации готова, ожидание взлёта")

    def reset_position_callback(self, msg):
        self.pos_x, self.pos_y = 0.0, 0.0
        self.last_integ_time = None
        self.coast_counter = 0
        self.vision_active = True
        self.cov_xy = 0.05
        self.state = 'ARMED'
        rospy.loginfo("Позиция обнулена. ARMED.")

    def imu_callback(self, msg):
        if rospy.is_shutdown(): return
        t = msg.header.stamp.to_sec()
        w_raw = msg.angular_velocity
        a_raw = msg.linear_acceleration
        
        self.imu_buffer.append({'t': t, 'w': [w_raw.x, w_raw.y, w_raw.z]})
        self.imu_acc_buffer.append({'t': t, 'a': [a_raw.x, a_raw.y, a_raw.z]})

        #  ОЦЕНКА ОРИЕНТАЦИИ (КОМПЛЕМЕНТАРНЫЙ ФИЛЬТР)
        # Используем акселерометр для коррекции дрейфа roll/pitch
        ax, ay, az = a_raw.x, a_raw.y, a_raw.z
        
        # Roll и Pitch из акселерометра (работает в статике и при медленных движениях)
        roll_acc = math.atan2(ay, az)
        pitch_acc = math.atan2(-ax, math.sqrt(ay**2 + az**2))
        
        if self.last_yaw_time is not None:
            dt_yaw = t - self.last_yaw_time
            if 0.005 < dt_yaw < 0.1:
                # Roll и Pitch: комплементарный фильтр
                alpha = self.comp_alpha
                self.roll  = alpha * (self.roll  + w_raw.x * dt_yaw) + (1 - alpha) * roll_acc
                self.pitch = alpha * (self.pitch + w_raw.y * dt_yaw) + (1 - alpha) * pitch_acc
                
                # Yaw: только гироскоп (с инверсией, как было)
                wz = w_raw.z
                if abs(wz) < 0.005: wz = 0.0
                self.yaw -= wz * dt_yaw
                self.yaw = math.atan2(math.sin(self.yaw), math.cos(self.yaw))
        else:
            # Инициализация roll/pitch из акселерометра
            self.roll = roll_acc
            self.pitch = pitch_acc
        
        self.last_yaw_time = t

    def lidar_callback(self, msg):
        if rospy.is_shutdown(): return
        h = msg.range
        if 0.1 <= h <= 12.0:
            self.current_height = h
            self.lidar_received = True

            if self.state == 'GROUND' and h > self.lift_height_threshold:
                self.state = 'WAIT_LIFT'
                self.lift_start_time = rospy.Time.now().to_sec()
                rospy.loginfo(f"Подъём обнаружен, (H={h:.2f}m). Стабилизация в течение {self.lift_confirm_delay}с...")
            elif self.state == 'WAIT_LIFT':
                if h < self.lift_height_threshold:
                    self.state = 'GROUND'
                    self.lift_start_time = None
                    rospy.loginfo("Снижение. Возврат состояние в GROUND")
                elif rospy.Time.now().to_sec() - self.lift_start_time > self.lift_confirm_delay:
                    self.state = 'ARMED'
                    self.pos_x, self.pos_y = 0.0, 0.0
                    self.last_integ_time = None
                    rospy.loginfo(f" ARMED! Позиция обнулена, курс={math.degrees(self.yaw):.1f}°")

    def _get_closest_imu(self, t_img):
        if not self.imu_buffer: return None
        closest = min(self.imu_buffer, key=lambda x: abs(x['t'] - t_img))
        return closest if abs(closest['t'] - t_img) <= self.max_sync_offset else None

    def _get_closest_imu_acc(self, t_img):
        if not self.imu_acc_buffer: return None
        closest = min(self.imu_acc_buffer, key=lambda x: abs(x['t'] - t_img))
        return closest if abs(closest['t'] - t_img) <= self.max_sync_offset else None

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
            if dt < 0.05: dt = 0.067
            if dt > 0.15: dt = 0.067

            if vision_ok:
                self.coast_counter = 0
                if not self.vision_active:
                    rospy.loginfo("Оптический поток восстановлен")
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
                #  COAST MODE С КОМПЕНСАЦИЕЙ ГРАВИТАЦИИ
                self.coast_counter += 1
                if self.coast_counter == 1: rospy.logwarn("Потеряны текстуры, переход на инерциальные данные.")
                self.vision_active = False
                self.cov_xy = min(0.05 + self.coast_counter * self.cov_growth_rate, 2.0)
                
                imu_acc = self._get_closest_imu_acc(img_time)
                vx_drone = self.kf_x.x
                vy_drone = self.kf_y.x

                if imu_acc is not None:
                    ax, ay, az = imu_acc['a']
                    
                    #  ШАГ 1: ВЫЧИТАЕМ ГРАВИТАЦИЮ (через roll/pitch)
                    cr, sr = math.cos(self.roll), math.sin(self.roll)
                    cp, sp = math.cos(self.pitch), math.sin(self.pitch)
                    
                    # Гравитация в системе IMU (при наклоне проецируется на X/Y)
                    gx = -self.gravity * sp
                    gy =  self.gravity * sr * cp
                    # gz =  self.gravity * cr * cp  # не нужно для горизонтальных скоростей
                    
                    # Линейное ускорение = сырое - гравитация
                    ax_lin = ax - gx
                    ay_lin = ay - gy
                    
                    #  ШАГ 2: ПРАВИЛЬНОЕ ПРЕОБРАЗОВАНИЕ IMU -> DRONE
                    # Ваш IMU: X=назад, Y=вправо, Z=вверх
                    # Дрон:     X=вперёд, Y=вправо, Z=вверх
                    # Значит: X_drone = -X_imu, Y_drone = +Y_imu
                    acc_x_drone = -ax_lin
                    acc_y_drone =  ay_lin
                    
                    #  ШАГ 3: ПРИМЕНЯЕМ ВЕС ДОВЕРИЯ
                    acc_x_drone *= self.imu_acc_weight
                    acc_y_drone *= self.imu_acc_weight
                    
                    #  ШАГ 4: ZUPT ДЛЯ УСКОРЕНИЯ (защита от дрейфа в статике)
                    if abs(acc_x_drone) < self.acc_deadband: acc_x_drone = 0.0
                    if abs(acc_y_drone) < self.acc_deadband: acc_y_drone = 0.0
                    
                    # Интегрируем ускорение в скорость
                    vx_drone += acc_x_drone * dt
                    vy_drone += acc_y_drone * dt
                    
                    # ZUPT для скорости
                    if abs(vx_drone) < self.vel_deadband: vx_drone = 0.0
                    if abs(vy_drone) < self.vel_deadband: vy_drone = 0.0
                    
                    vx_filt = vx_drone
                    vy_filt = vy_drone
                    self.kf_x.x = vx_filt
                    self.kf_y.x = vy_filt
                    
                    # Отладка (раз в 2 сек)
                    rospy.loginfo_throttle(2.0, 
                        f" IMU компенсация | roll={math.degrees(self.roll):+.1f}° pitch={math.degrees(self.pitch):+.1f}° | "
                        f"ax_lin={ax_lin:+.2f} ay_lin={ay_lin:+.2f} | "
                        f"acc_drone=[{acc_x_drone:+.2f}, {acc_y_drone:+.2f}]")
                else:
                    vx_filt = self.kf_x.x
                    vy_filt = self.kf_y.x

            if abs(vx_filt) < self.vel_deadband: vx_filt = 0.0
            if abs(vy_filt) < self.vel_deadband: vy_filt = 0.0
            if self.coast_counter > self.max_coast_frames:
                vx_filt *= 0.95; vy_filt *= 0.95

            if self.lidar_received and not rospy.is_shutdown():
                msg_vel = TwistStamped()
                msg_vel.header.stamp = msg.header.stamp
                msg_vel.twist.linear.x, msg_vel.twist.linear.y = vx_filt, vy_filt
                self.vel_pub.publish(msg_vel)

                if self.state == 'ARMED':
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

                            odom = Odometry()
                            odom.header.stamp = msg.header.stamp
                            odom.header.frame_id = "odom"
                            odom.child_frame_id = "base_link"
                            odom.pose.pose.position.x, odom.pose.pose.position.y, odom.pose.pose.position.z = self.pos_x, self.pos_y, self.current_height
                            
                            q = transformations.quaternion_from_euler(self.roll, self.pitch, self.yaw)
                            odom.pose.pose.orientation.x, odom.pose.pose.orientation.y = q[0], q[1]
                            odom.pose.pose.orientation.z, odom.pose.pose.orientation.w = q[2], q[3]
                            
                            odom.twist.twist.linear.x, odom.twist.twist.linear.y = vx_filt, vy_filt
                            c = self.cov_xy
                            odom.pose.covariance[0], odom.pose.covariance[7] = c, c
                            odom.twist.covariance[0], odom.twist.covariance[7] = c*0.5, c*0.5
                            self.odom_pub.publish(odom)

                            self.tf_br.sendTransform((self.pos_x, self.pos_y, self.current_height), q, msg.header.stamp, "base_link", "odom")
                            rospy.loginfo_throttle(1.5, f" ARMED | Yaw={math.degrees(self.yaw):.1f}° | N={self.pos_x:+.2f} E={self.pos_y:+.2f} | Coast={self.coast_counter}")
                else:
                    rospy.loginfo_throttle(3.0, f"Ожидание: {self.state} | Waiting for flight...")

        self.prev_gray = gray.copy()
        if curr_pts is not None and status is not None:
            valid_pts = curr_pts[status == 1]
            self.prev_pts = valid_pts.reshape(-1, 1, 2) if len(valid_pts) > 0 else None
        self.last_img_time = img_time

    def _detect_features(self, gray):
        pts = cv2.goodFeaturesToTrack(gray, maxCorners=self.max_features, qualityLevel=self.quality_level, minDistance=self.min_dist, blockSize=9)
        return pts if pts is not None and len(pts) > 0 else None

    def run(self):
        rospy.spin()

if __name__ == '__main__':
    try:
        HybridFlowNode().run()
    except rospy.ROSInterruptException:
        pass