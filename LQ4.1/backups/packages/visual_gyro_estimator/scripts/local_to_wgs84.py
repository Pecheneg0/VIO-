#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local (N-E-U) to Global (WGS84) Converter
==========================================
Подписывается на /drone/odom (x=North, y=East, z=Up)
Публикует /vio/global_position (NavSatFix)
Математика: ENU -> ECEF -> WGS84 (Bowring) с учётом геоида
"""
import rospy
import math
import numpy as np
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix, NavSatStatus

class LocalToWGS84Node:
    def __init__(self):
        rospy.init_node('local_to_wgs84_node', anonymous=False)
        
        # === ПАРАМЕТРЫ ===
        self.lat0 = rospy.get_param('~start_lat', 55.692065)
        self.lon0 = rospy.get_param('~start_lon', 37.516265)
        self.h_msl0 = rospy.get_param('~start_alt', 150.0)
        self.odom_topic = rospy.get_param('~odom_topic', '/drone/odom')
        self.pub_topic = rospy.get_param('~pub_topic', '/vio/global_position')
        self.geoid_height = rospy.get_param('~geoid_height', 14.5)  # N(geoid) для вашего региона
        self.frame_id = rospy.get_param('~frame_id', 'map')
        
        rospy.loginfo(f"📍 Start: Lat={self.lat0:.6f}, Lon={self.lon0:.6f}, Alt(MSL)={self.h_msl0}m")
        rospy.loginfo(f"🌍 Geoid height (N): {self.geoid_height}m")
        
        # Начальная эллипсоидальная высота
        h_ell0 = self.h_msl0 + self.geoid_height
        
        # ECEF координаты точки старта
        self.X0, self.Y0, self.Z0 = self._geo_to_ecef(self.lat0, self.lon0, h_ell0)
        rospy.loginfo(f"🌐 ECEF Start: X={self.X0:.3f}, Y={self.Y0:.3f}, Z={self.Z0:.3f}")
        
        # Матрица поворота ENU -> ECEF
        self.R_enu2ecef = self._get_enu_to_ecef_matrix(self.lat0, self.lon0)
        
        # Подписка и публикация
        self.odom_sub = rospy.Subscriber(self.odom_topic, Odometry, self.odom_callback, queue_size=10)
        self.gps_pub = rospy.Publisher(self.pub_topic, NavSatFix, queue_size=10)
        
        rospy.loginfo(f"✅ Node ready. Subscribed: {self.odom_topic} → Publishing: {self.pub_topic}")

    def _geo_to_ecef(self, lat_deg, lon_deg, h):
        """WGS-84: Геодезические -> ECEF"""
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
        """Матрица поворота ENU -> ECEF"""
        lat = math.radians(lat_deg)
        lon = math.radians(lon_deg)
        sl, cl = math.sin(lat), math.cos(lat)
        so, co = math.sin(lon), math.cos(lon)
        return np.array([
            [-so,    -sl*co,  cl*co],
            [ co,    -sl*so,  cl*so],
            [ 0.0,     cl,     sl]
        ])

    def _ecef_to_geo_bowring(self, X, Y, Z):
        """ECEF -> WGS-84 (Метод Боуринга, точность < 10^-9)"""
        a = 6378137.0
        f = 1.0 / 298.257223563
        e2 = 2*f - f**2
        b = a * math.sqrt(1 - e2)
        p = math.sqrt(X**2 + Y**2)
        if p < 1e-9:
            return (90.0 if Z > 0 else -90.0), 0.0, abs(Z) - b
        
        theta = math.atan2(Z * a, p * b)
        sin_t, cos_t = math.sin(theta), math.cos(theta)
        lat = math.atan2(Z + (e2 / (1 - e2)) * b * sin_t**3, 
                         p - e2 * a * cos_t**3)
        lon = math.atan2(Y, X)
        N = a / math.sqrt(1 - e2 * math.sin(lat)**2)
        h = p / math.cos(lat) - N
        return math.degrees(lat), math.degrees(lon), h

    def odom_callback(self, msg):
        try:
            # 📐 ВАША СИСТЕМА: x=North, y=East, z=Up
            dx_north = msg.pose.pose.position.x
            dy_east  = msg.pose.pose.position.y
            dz_up    = msg.pose.pose.position.z
            
            # 🔑 КЛЮЧЕВОЙ ШАГ: Приводим к ENU (East, North, Up) для стандартной матрицы
            d_enu = np.array([dy_east, dx_north, dz_up])
            
            # Приращение в ECEF
            d_ecef = self.R_enu2ecef @ d_enu
            
            # Новые абсолютные координаты
            X_new = self.X0 + d_ecef[0]
            Y_new = self.Y0 + d_ecef[1]
            Z_new = self.Z0 + d_ecef[2]
            
            # Обратное преобразование в WGS-84
            lat, lon, h_ell = self._ecef_to_geo_bowring(X_new, Y_new, Z_new)
            
            # Высота над уровнем моря: MSL = Ellipsoid - Geoid
            h_msl = h_ell - self.geoid_height
            
            # Формирование NavSatFix
            gps_msg = NavSatFix()
            gps_msg.header = msg.header
            gps_msg.header.frame_id = self.frame_id
            gps_msg.status.status = NavSatStatus.STATUS_FIX
            gps_msg.status.service = NavSatStatus.SERVICE_GPS
            gps_msg.latitude = lat
            gps_msg.longitude = lon
            gps_msg.altitude = h_msl
            
            # Ковариация: типичная для optical-flow + lidar dead-reckoning
            gps_msg.position_covariance = [0.15, 0.0, 0.0, 0.0, 0.15, 0.0, 0.0, 0.0, 0.2]
            gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            
            self.gps_pub.publish(gps_msg)
            
        except Exception as e:
            rospy.logerr(f"❌ Conversion error: {e}")

if __name__ == '__main__':
    try:
        LocalToWGS84Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
