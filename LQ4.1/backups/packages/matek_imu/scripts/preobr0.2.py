#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Нода для преобразования локальных координат OpenVINS (метрические смещения)
в глобальные координаты WGS-84 (GPS) для подмены сигнала.
Использует математику ГОСТ 32453-2017 без pyproj.
"""

import rospy
import numpy as np
import math
import sys
from nav_msgs.msg import Odometry
from sensor_msgs.msg import NavSatFix

# ==========================================
# МАТЕМАТИЧЕСКИЙ БЛОК (ГОСТ 32453-2017)
# Взято из вашего файла preobr3.2.py
# ==========================================

# Параметры эллипсоида WGS-84
a_wgs84 = 6378137.0
f_wgs84 = 1/298.257223563
e2_wgs84 = 2 * f_wgs84 - f_wgs84**2

# Геоидная поправка для Московской области (аномальная высота)
# H_ellipsoid = H_MSL + 14.5
GEOID_UNDULATION_MOSCOW = 14.5

def geo_to_pr(B, L, H, a, e2):
    """Преобразование геодезических координат в прямоугольные (ECEF)"""
    N = a / np.sqrt(1 - e2 * np.sin(B)**2)
    X = (N + H) * np.cos(B) * np.cos(L)
    Y = (N + H) * np.cos(B) * np.sin(L)
    Z = ((1 - e2) * N + H) * np.sin(B)
    return X, Y, Z

def pr_to_geo_bowring(X, Y, Z, a, e2):
    """Преобразование прямоугольных (ECEF) в геодезические (Метод Боуринга)"""
    b = a * np.sqrt(1 - e2)
    ep2 = e2 / (1 - e2) 
    Q = np.sqrt(X**2 + Y**2)

    if Q == 0:
        B = np.pi/2 if Z > 0 else -np.pi/2
        L = 0.0
        H = abs(Z) - b
        return B, L, H

    r = np.sqrt(Z**2 + (X**2 + Y**2) * (1 - e2))
    numerator = r**3 + b * ep2 * Z**2
    denominator = r**3 - b * e2 * (1 - e2) * Q**2
    
    B = np.arctan((Z / Q) * (numerator / denominator))
    L = np.arctan2(Y, X)
    
    N = a / np.sqrt(1 - e2 * np.sin(B)**2)
    H = Q / np.cos(B) - N
    
    return B, L, H

def get_enu_to_ecef_matrix(lat, lon):
    """
    Возвращает матрицу поворота 3x3 из локальной системы ENU (Восток-Север-Вверх)
    в систему ECEF (X, Y, Z относительно центра Земли).
    """
    cos_lat = np.cos(lat)
    sin_lat = np.sin(lat)
    cos_lon = np.cos(lon)
    sin_lon = np.sin(lon)

    # Матрица поворота R (ENU -> ECEF)
    # Столбцы матрицы - это направления осей E (East), N (North), U (Up) в системе ECEF
    R = np.array([
        [-sin_lon,           -sin_lat * cos_lon,   cos_lat * cos_lon],
        [cos_lon,            -sin_lat * sin_lon,   cos_lat * sin_lon],
        [0.0,                 cos_lat,              sin_lat]
    ])
    return R

# ==========================================
# ROS NODE
# ==========================================

class VIOTOGPSNode:
    def __init__(self):
        rospy.init_node('vio_to_gps_node', anonymous=True)
        
        # 1. Запрашиваем координаты точки старта
        print("\n" + "="*60)
        print(" ВВОД КООРДИНАТ ТОЧКИ СТАРТА (MSL - над уровнем моря) ")
        print("="*60)
        try:
            lat_msl = float(input("   Широта (градусы): "))
            lon_msl = float(input("   Долгота (градусы): "))
            alt_msl = float(input("   Высота MSL (метры): "))
        except ValueError:
            rospy.logerr("Ошибка ввода координат. Завершение.")
            sys.exit(1)

        # 2. Инициализация переменных
        self.start_lat = np.radians(lat_msl)
        self.start_lon = np.radians(lon_msl)
        
        # Переводим высоту старта в эллипсоидальную для математики
        self.start_h_ellipsoid = alt_msl + GEOID_UNDULATION_MOSCOW
        
        # Считаем ECEF координаты старта ОДИН раз при инициализации
        self.start_X, self.start_Y, self.start_Z = geo_to_pr(
            self.start_lat, self.start_lon, self.start_h_ellipsoid, 
            a_wgs84, e2_wgs84
        )
        
        # Матрица поворота для точки старта (зависит от широты/долготы старта)
        self.R_matrix = get_enu_to_ecef_matrix(self.start_lat, self.start_lon)

        rospy.loginfo(f"Старт зафиксирован: Lat={lat_msl:.6f}, Lon={lon_msl:.6f}, Alt={alt_msl:.2f}m")
        rospy.loginfo(f"Эллипсоидальная высота старта: {self.start_h_ellipsoid:.3f}m")
        rospy.loginfo(f"Координаты ECEF старта: X={self.start_X:.3f}, Y={self.start_Y:.3f}, Z={self.start_Z:.3f}")

        # 3. Подписки и Публикации
        # Topic OpenVINS (обычно /vio/odom или /openvins/odom)
        self.sub_odom = rospy.Subscriber('/vio/odom', Odometry, self.callback_odom, queue_size=10)
        
        # Topic для MAVROS (подмена GPS)
        self.pub_gps = rospy.Publisher('/mavros/global_position/global', NavSatFix, queue_size=10)
        
        rospy.spin()

    def callback_odom(self, msg):
        try:
            # Получаем локальные смещения от OpenVINS (x, y, z)
            # Предполагается, что OpenVINS выдает данные в системе ENU (Восток, Север, Вверх)
            dx = msg.pose.pose.position.x
            dy = msg.pose.pose.position.y
            dz = msg.pose.pose.position.z
            
            # Вектор локального смещения
            local_offset = np.array([dx, dy, dz])
            
            # Поворачиваем смещение в систему ECEF
            # dX_ecef = R * dX_local
            offset_ecef = np.dot(self.R_matrix, local_offset)
            
            # Прибавляем смещение к стартовым координатам ECEF
            current_X = self.start_X + offset_ecef[0]
            current_Y = self.start_Y + offset_ecef[1]
            current_Z = self.start_Z + offset_ecef[2]
            
            # Переводим текущие ECEF обратно в Геодезические
            lat_rad, lon_rad, h_ellipsoid = pr_to_geo_bowring(
                current_X, current_Y, current_Z, 
                a_wgs84, e2_wgs84
            )
            
            # Переводим в градусы
            lat_deg = np.degrees(lat_rad)
            lon_deg = np.degrees(lon_rad)
            
            # Переводим высоту обратно в MSL (над уровнем моря) для MAVROS
            h_msl = h_ellipsoid - GEOID_UNDULATION_MOSCOW
            
            # Публикуем
            gps_msg = NavSatFix()
            gps_msg.header = msg.header # Копируем таймстамп
            gps_msg.status.status = NavSatFix.STATUS_FIX
            gps_msg.status.service = NavSatFix.SERVICE_GPS
            
            gps_msg.latitude = lat_deg
            gps_msg.longitude = lon_deg
            gps_msg.altitude = h_msl
            
            # Ковариация (для OpenVINS можно ставить небольшую, т.к. VIO точное локально)
            # Но для MAVROS лучше не завышать доверие, если нет RTK
            gps_msg.position_covariance = [0.1, 0, 0, 0, 0.1, 0, 0, 0, 0.1] 
            gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
            
            self.pub_gps.publish(gps_msg)
            
            # Логирование раз в 2 секунды, чтобы не засорять консоль
            if int(msg.header.stamp.to_sec()) % 2 == 0:
                 rospy.loginfo(f"GPS Out: Lat={lat_deg:.6f}, Lon={lon_deg:.6f}, Alt={h_msl:.2f} | VIO Off: {dx:.2f}, {dy:.2f}, {dz:.2f}")
            
        except Exception as e:
            rospy.logerr(f"Ошибка в callback: {e}")

if __name__ == '__main__':
    try:
        VIOTOGPSNode()
    except rospy.ROSInterruptException:
        pass
