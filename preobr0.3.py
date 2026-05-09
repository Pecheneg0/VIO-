#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus

# ================= КОНФИГУРАЦИЯ =================
VIO_TOPIC = "/run_subscribe_msckf/odomimu"  # или /vio/odom
PUB_TOPIC = "/vio/global_position"
FRAME_ID = "map"
ANOMALOUS_HEIGHT = 14.5  # Геоидная поправка для МО
# =================================================

class VIOtoWGS84Node:
    def __init__(self):
        rospy.init_node('vio_to_wgs84_node', anonymous=False)
        
        # Ввод координат старта (можно вынести в launch-файл через ~params)
        self.lat0 = float(input("\n   Широта старта (градусы): "))
        self.lon0 = float(input("   Долгота старта (градусы): "))
        self.h_msl0 = float(input("   Высота MSL старта (метры): "))
        
        # Переход к эллипсоидальной высоте
        self.h_ell0 = self.h_msl0 + ANOMALOUS_HEIGHT
        
        # Расчёт стартовых ECEF координат (ГОСТ 32453-2017)
        self.X0, self.Y0, self.Z0 = self._geo_to_ecef(self.lat0, self.lon0, self.h_ell0)
        
        # Матрица поворота ENU -> ECEF для точки старта
        self.R_enu2ecef = self._get_enu_to_ecef_matrix(self.lat0, self.lon0)
        
        rospy.loginfo(f"Старт зафиксирован: Lat={self.lat0:.6f}, Lon={self.lon0:.6f}, Alt={self.h_msl0:.1f}m")
        rospy.loginfo(f"Эллипсоидальная высота: {self.h_ell0:.3f}m")
        rospy.loginfo(f"Пространственные прямоугольные координаты точки старта: X={self.X0:.3f}, Y={self.Y0:.3f}, Z={self.Z0:.3f}")
        
        # Подписка на OpenVINS
        self.vio_sub = rospy.Subscriber(VIO_TOPIC, Odometry, self.vio_callback, queue_size=10)
        
        # Публикатор GPS
        self.gps_pub = rospy.Publisher(PUB_TOPIC, NavSatFix, queue_size=10)
        
        rospy.loginfo(f"Узел запущен. Ожидание данных из {VIO_TOPIC}")

    def _geo_to_ecef(self, lat_deg, lon_deg, h):
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2*f - f**2
        B = math.radians(lat_deg)
        L = math.radians(lon_deg)
        N = a / math.sqrt(1 - e2 * math.sin(B)**2)
        X = (N + h) * math.cos(B) * math.cos(L)
        Y = (N + h) * math.cos(B) * math.sin(L)
        Z = ((1 - e2) * N + h) * math.sin(B)
        return X, Y, Z

    def _get_enu_to_ecef_matrix(self, lat_deg, lon_deg):
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sin_lat, cos_lat = math.sin(lat), math.cos(lat)
        sin_lon, cos_lon = math.sin(lon), math.cos(lon)
        return np.array([
            [-sin_lon,           -sin_lat * cos_lon,   cos_lat * cos_lon],
            [cos_lon,            -sin_lat * sin_lon,   cos_lat * sin_lon],
            [0.0,                 cos_lat,              sin_lat]
        ])

    def _ecef_to_geo_bowring(self, X, Y, Z):
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2*f - f**2
        b = a * math.sqrt(1 - e2)
        ep2 = e2 / (1 - e2)
        Q = math.sqrt(X**2 + Y**2)
        if Q == 0:
            return (90.0 if Z > 0 else -90.0), 0.0, (abs(Z) - b)
        r = math.sqrt(Z**2 + Q**2 * (1 - e2))
        num = r**3 + b * ep2 * Z**2
        den = r**3 - b * e2 * (1 - e2) * Q**2
        B = math.atan((Z / Q) * (num / den))
        L = math.atan2(Y, X)
        N = a / math.sqrt(1 - e2 * math.sin(B)**2)
        H = Q / math.cos(B) - N
        return math.degrees(B), math.degrees(L), H

    def vio_callback(self, msg):
        # OpenVINS отдаёт локальное смещение в ENU (x=East, y=North, z=Up)
        dx = msg.pose.pose.position.x
        dy = msg.pose.pose.position.y
        dz = msg.pose.pose.position.z
        
        # Преобразование локального вектора в приращение ECEF
        d_enu = np.array([dx, dy, dz])
        d_ecef = self.R_enu2ecef @ d_enu
        
        # Новые абсолютные координаты ECEF
        X_new = self.X0 + d_ecef[0]
        Y_new = self.Y0 + d_ecef[1]
        Z_new = self.Z0 + d_ecef[2]
        
        # Обратный переход в WGS84
        lat, lon, h_ell = self._ecef_to_geo_bowring(X_new, Y_new, Z_new)
        h_msl = h_ell - ANOMALOUS_HEIGHT
        
        # Формирование сообщения NavSatFix
        gps_msg = NavSatFix()
        gps_msg.header = msg.header
        gps_msg.header.frame_id = FRAME_ID
        gps_msg.status.status = NavSatStatus.STATUS_FIX
        gps_msg.status.service = NavSatStatus.SERVICE_GPS
        gps_msg.latitude = lat
        gps_msg.longitude = lon
        gps_msg.altitude = h_msl
        
        # Ковариация: OpenVINS не отдаёт GPS-ковариацию, ставим умеренную
        #gps_msg.position_covariance = [0.1, 0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 0.1]
        #gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        
        self.gps_pub.publish(gps_msg)

if __name__ == '__main__':
    try:
        node = VIOtoWGS84Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    except KeyboardInterrupt:
        rospy.loginfo("Узел остановлен пользователем.")
