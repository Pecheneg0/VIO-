#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import pyproj
import numpy as np
import sys
import re
from geometry_msgs.msg import PoseStamped
from sensor_msgs.msg import NavSatFix

# ==============================================================================
# БЛОК ФУНКЦИЙ ПРЕОБРАЗОВАНИЯ (из preobr3.2.py, логика не изменена)
# ==============================================================================
def parse_dms_input(value_str):
    value_str = value_str.strip().replace(',', '.')
    try:
        return float(value_str)
    except ValueError:
        pass
    dms_pattern = r'(\d+)\s*°?\s*(\d+)\s*\'?\s*([\d.]+)\s*"?\s*([NSWE])?'
    match = re.match(dms_pattern, value_str, re.IGNORECASE)
    if match:
        d = float(match.group(1))
        m = float(match.group(2))
        s = float(match.group(3))
        direction = match.group(4)
        decimal = d + m/60 + s/3600
        if direction and direction.upper() in ['S', 'W']:
            decimal = -decimal
        return decimal
    space_pattern = r'(\d+)\s+(\d+)\s+([\d.]+)'
    match = re.match(space_pattern, value_str)
    if match:
        d = float(match.group(1))
        m = float(match.group(2))
        s = float(match.group(3))
        return d + m/60 + s/3600
    raise ValueError(f"Неверный формат координаты: {value_str}")

def geo_to_pr(B, L, H, a, e2):
    N = a / np.sqrt(1 - e2 * np.sin(B)**2)
    X = (N + H) * np.cos(B) * np.cos(L)
    Y = (N + H) * np.cos(B) * np.sin(L)
    Z = ((1 - e2) * N + H) * np.sin(B)
    return X, Y, Z

def pr_to_geo_bowring(X, Y, Z, a, e2):
    b = a * np.sqrt(1 - e2)
    ep2 = e2 / (1 - e2)
    Q = np.sqrt(X**2 + Y**2)
    if Q == 0:
        B = np.pi/2 if Z > 0 else -np.pi/2
        L = 0.0
        H = abs(Z) - b
        return B, L, H
    r = np.sqrt(Z**2 + Q**2 * (1 - e2))
    numerator = r**3 + b * ep2 * Z**2
    denominator = r**3 - b * e2 * (1 - e2) * Q**2
    B = np.arctan((Z / Q) * (numerator / denominator))
    L = np.arctan2(Y, X)
    N = a / np.sqrt(1 - e2 * np.sin(B)**2)
    H = Q / np.cos(B) - N
    return B, L, H

def transform_sk42_to_pz90(X_sk42, Y_sk42, Z_sk42):
    dx, dy, dz = 23.557, -140.844, -79.778
    m = -0.228 * 1e-6
    X_pz = (1 + m) * (X_sk42 - 3.850439 * 10**(-6) * Y_sk42 + 1.679685 * 10**(-6) * Z_sk42) + dx
    Y_pz = (1 + m) * (3.850439 * 10**(-6) * X_sk42 + Y_sk42 - 1.115071 * 10**(-8) * Z_sk42) + dy
    Z_pz = (1 + m) * (-1.679685 * 10**(-6) * X_sk42 + 1.115071 * 10**(-8) * Y_sk42 + Z_sk42) + dz
    return X_pz, Y_pz, Z_pz

def transform_pz90_to_wgs84(X_pz, Y_pz, Z_pz):
    dx, dy, dz = -0.013, +0.106, +0.022
    m = -0.008 * 1e-6
    X_wgs = (1 + m) * (1 * X_pz + 2.041066 * 10**(-8) * Y_pz  + 1.716240 * 10**(-8) * Z_pz) - dx
    Y_wgs = (1 + m) * (-2.041066 * 10**(-8) * X_pz + 1 * Y_pz + 1.115071 * 10**(-8) * Z_pz) - dy
    Z_wgs = (1 + m) * (-1.716240 * 10**(-8) * X_pz + 1.115071 * 10**(-8) * Y_pz + 1 * Z_pz) - dz
    return X_wgs, Y_wgs, Z_wgs

# Параметры эллипсоидов (ГОСТ 32453-2017)
a_wgs84, f_wgs84 = 6378137.0, 1/298.257223563
e2_wgs84 = 2*f_wgs84 - f_wgs84**2
a_pz90, f_pz90 = 6378136.0, 1/298.25784
e2_pz90 = 2*f_pz90 - f_pz90**2
a_sk42, f_sk42 = 6378245.0, 1/298.3
e2_sk42 = 2*f_sk42 - f_sk42**2

def get_start_coordinates_cli():
    """Интерактивный ввод координат старта с выбором СК и пересчетом в WGS84"""
    print("\n" + "="*60)
    print("  ВВОД КООРДИНАТ ТОЧКИ СТАРТА (WGS84 / ПЗ-90.11 / СК-42)")
    print("="*60)
    print("Выберите систему координат для ввода:")
    print("  1 - WGS-84")
    print("  2 - ПЗ-90.11")
    print("  3 - СК-42")
    
    while True:
        try:
            choice = input("\nВаш выбор (1/2/3): ").strip()
            if choice in ['1', '2', '3']:
                break
            print("Пожалуйста, введите 1, 2 или 3.")
        except KeyboardInterrupt:
            sys.exit(0)

    print("\nПоддерживаемые форматы ввода:")
    print("  • Десятичные градусы: 55.754066")
    print("  • Г°М'С: 55 45 14.638 или 55°45'14.638\"")
    print("  • С направлением: 55°45'14.638\"N")

    try:
        lat_deg = parse_dms_input(input("   Широта: "))
        lon_deg = parse_dms_input(input("   Долгота: "))
        h_m = float(input("   Высота H (м, эллипсоидальная): ").replace(',', '.'))
    except (ValueError, KeyboardInterrupt) as e:
        print(f"\nОшибка ввода: {e}")
        sys.exit(1)

    B = np.radians(lat_deg)
    L = np.radians(lon_deg)
    H = h_m

    if choice == '1':
        print("\n[INFO] Введены координаты WGS-84. Пересчет не требуется.")
        return lat_deg, lon_deg, H
    elif choice == '2':
        a_in, e2_in = a_pz90, e2_pz90
        print("\n[INFO] Введены координаты ПЗ-90.11. Выполняется пересчет в WGS-84...")
    else: # choice == '3'
        a_in, e2_in = a_sk42, e2_sk42
        print("\n[INFO] Введены координаты СК-42. Выполняется пересчет в WGS-84 (через ПЗ-90.11)...")

    # 1. Геодезические -> Прямоугольные (X,Y,Z) в исходной СК
    X, Y, Z = geo_to_pr(B, L, H, a_in, e2_in)

    # 2. Пересчет датумов
    if choice == '3': # СК-42 -> ПЗ-90.11 -> WGS-84
        X, Y, Z = transform_sk42_to_pz90(X, Y, Z)
    X, Y, Z = transform_pz90_to_wgs84(X, Y, Z)

    # 3. Прямоугольные WGS-84 -> Геодезические WGS-84
    B_wgs, L_wgs, H_wgs = pr_to_geo_bowring(X, Y, Z, a_wgs84, e2_wgs84)

    lat_wgs = np.degrees(B_wgs)
    lon_wgs = np.degrees(L_wgs)

    print(f"[OK] Пересчет завершен. WGS-84: {lat_wgs:.9f}°, {lon_wgs:.9f}°, H={H_wgs:.3f} м")
    return lat_wgs, lon_wgs, H_wgs
# ==============================================================================


class VioToWgs84Node:
    def __init__(self):
        # 1. Инициализация ROS узла
        rospy.init_node('vio_to_wgs84_node', anonymous=True)
        
        # 2. Интерактивный ввод и пересчет координат в WGS84
        self.lat_0, self.lon_0, self.alt_0 = get_start_coordinates_cli()
        self.geoid_offset = rospy.get_param('~geoid_offset', 0.0)
        
        rospy.loginfo(f"Инициализация узла. Старт (WGS84): {self.lat_0}, {self.lon_0}, {self.alt_0}")

        # --- Настройка pyproj ---
        self.wgs84 = pyproj.CRS("EPSG:4326")
        self.ecef = pyproj.CRS("EPSG:4978")

        self.transformer_to_ecef = pyproj.Transformer.from_crs(self.wgs84, self.ecef, always_xy=True)
        self.transformer_to_wgs84 = pyproj.Transformer.from_crs(self.ecef, self.wgs84, always_xy=True)

        # --- Преобразование стартовой точки в ECEF (выполняется один раз) ---
        self.X0, self.Y0, self.Z0 = self.transformer_to_ecef.transform(self.lon_0, self.lat_0, self.alt_0)
        
        # Матрица поворота ENU -> ECEF
        self.R_enu2ecef = self._get_enu_to_ecef_matrix(self.lat_0, self.lon_0)

        rospy.loginfo(f"Стартовая точка в ECEF: X={self.X0:.3f}, Y={self.Y0:.3f}, Z={self.Z0:.3f}")

        # --- Подписка и Публикация ---
        self.vio_sub = rospy.Subscriber("/vio/local_pose", PoseStamped, self.vio_callback)
        self.wgs84_pub = rospy.Publisher("/vio/global_position", NavSatFix, queue_size=10)

        rospy.loginfo("Ожидание данных VIO...")

    def _get_enu_to_ecef_matrix(self, lat, lon):
        lat_rad = np.radians(lat)
        lon_rad = np.radians(lon)
        sin_lat = np.sin(lat_rad)
        cos_lat = np.cos(lat_rad)
        sin_lon = np.sin(lon_rad)
        cos_lon = np.cos(lon_rad)
        R = np.array([
            [-sin_lon,           -sin_lat * cos_lon,   cos_lat * cos_lon],
            [cos_lon,            -sin_lat * sin_lon,   cos_lat * sin_lon],
            [0.0,                 cos_lat,              sin_lat]
        ])
        return R

    def vio_callback(self, msg):
        # 1. Извлекаем локальные смещения (ENU: x=East, y=North, z=Up)
        local_east = msg.pose.position.x
        local_north = msg.pose.position.y
        local_up = msg.pose.position.z

        # 2. Вектор локального смещения
        vec_local = np.array([local_east, local_north, local_up])

        # 3. Поворот смещения в систему ECEF
        vec_ecef_offset = np.dot(self.R_enu2ecef, vec_local)

        # 4. Сложение со стартовой точкой ECEF
        X_new = self.X0 + vec_ecef_offset[0]
        Y_new = self.Y0 + vec_ecef_offset[1]
        Z_new = self.Z0 + vec_ecef_offset[2]

        # 5. Обратное преобразование ECEF -> WGS84 (Lon, Lat, Alt)
        lon, lat, alt_ellipsoidal = self.transformer_to_wgs84.transform(X_new, Y_new, Z_new)

        # 6. Формирование сообщения
        fix_msg = NavSatFix()
        fix_msg.header = msg.header
        fix_msg.header.frame_id = "wgs84"
        
        fix_msg.latitude = lat
        fix_msg.longitude = lon
        
        # Высота: эллипсоидальная WGS84
        fix_msg.altitude = alt_ellipsoidal - self.geoid_offset

        fix_msg.status.status = NavSatFix.STATUS_FIX
        fix_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_UNKNOWN

        self.wgs84_pub.publish(fix_msg)

if __name__ == '__main__':
    try:
        VioToWgs84Node()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
