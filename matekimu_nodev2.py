#!/usr/bin/env python3
"""
ACO Binary → ROS1 /imu Bridge
Парсит бинарный поток UART (сигнатура ACO) и публикует IMU данные для OpenVINS.
"""

import rospy
import struct
import math
import time
from pathlib import Path
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from tf.transformations import quaternion_from_euler

# ===== КОНФИГУРАЦИЯ =====
INPUT_BIN = Path("uart_raw.bin")
FRAME_ID = "imu_link"          # Фрейм IMU (должен совпадать с URDF/OpenVINS)
PUBLISH_RATE = 100             # Гц (частота IMU, подберите под реальный датчик)

# Структура бинарного пакета (совпадает с decoder.py)
SIG = b"ACO"
_PREFIX_LEN = len(SIG) + 1
_SEQ_LEN = 1
_CRC_LEN = 2
# <H: packetNumber, 2d: lat/lon, 12f: телеметрия, i: flags, h: state, 6f: acc/mag
STRUCT = struct.Struct("<H2d12fih6f")

# Ковариации (OpenVINS чувствителен к ним. Подстройте под спецификацию вашего датчика)
COV_ANG_VEL = 1e-4  # rad^2/s^2
COV_LIN_ACC = 1e-3  # m^2/s^4
COV_ORIENT = 1e-2   # rad^2 (если доверяете ориентации из Euler)

def euler_deg_to_quat(pitch_deg, roll_deg, yaw_deg):
    """Конвертация углов Эйлера (градусы) → Кватернион (ROS convention: X=roll, Y=pitch, Z=yaw)"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    # tf.transformations использует порядок (roll, pitch, yaw)
    q = quaternion_from_euler(roll, pitch, yaw, axes='sxyz')
    return q

def main():
    rospy.init_node('aco_imu_bridge', anonymous=True)
    imu_pub = rospy.Publisher('/imu', Imu, queue_size=10)
    rate = rospy.Rate(PUBLISH_RATE)

    if not INPUT_BIN.exists():
        rospy.logerr(f"Файл {INPUT_BIN} не найден. Положите его в директорию запуска ноды.")
        return

    data = INPUT_BIN.read_bytes()
    end = len(data)
    j = 0
    n_packets = 0

    rospy.loginfo(f"Запуск ACO IMU Bridge. Чтение {INPUT_BIN}...")

    while j < end  and not rospy.is_shutdown():
        # Поиск сигнатуры
        sig_idx = data.find(SIG, j)
        if sig_idx < 0 or sig_idx + _PREFIX_LEN > end:
            break

        L = data[sig_idx + len(SIG)]
        if L < 3:
            j = sig_idx + 1
            continue

        frame_len = _PREFIX_LEN + L
        if sig_idx + frame_len > end:
            break

        payload_start = sig_idx + _PREFIX_LEN
        telemetry_start = payload_start + _SEQ_LEN
        telemetry_len = L - _SEQ_LEN - _CRC_LEN
        body = data[telemetry_start : telemetry_start + telemetry_len]

        if len(body) != STRUCT.size:
            j = sig_idx + 1
            continue

        # Распаковка телеметрии
        u = STRUCT.unpack(body)
        
        # Индексы согласно <H2d12fih6f>
        pitch_deg = u[5]
        roll_deg  = u[6]
        yaw_deg   = u[7]
        rate_x    = u[8]
        rate_y    = u[9]
        rate_z    = u[10]
        acc_x     = u[17]
        acc_y     = u[18]
        acc_z     = u[19]

        # Формирование сообщения Imu
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = FRAME_ID
        imu_msg.header.seq = n_packets

        # Ориентация (Euler → Quaternion)
        q = euler_deg_to_quat(pitch_deg, roll_deg, yaw_deg)
        imu_msg.orientation = Quaternion(*q)
        imu_msg.orientation_covariance = [COV_ORIENT, 0, 0, 0, COV_ORIENT, 0, 0, 0, COV_ORIENT]

        # Угловая скорость
        imu_msg.angular_velocity = Vector3(rate_x, rate_y, rate_z)
        imu_msg.angular_velocity_covariance = [COV_ANG_VEL, 0, 0, 0, COV_ANG_VEL, 0, 0, 0, COV_ANG_VEL]

        # Линейное ускорение
        imu_msg.linear_acceleration = Vector3(acc_x, acc_y, acc_z)
        imu_msg.linear_acceleration_covariance = [COV_LIN_ACC, 0, 0, 0, COV_LIN_ACC, 0, 0, 0, COV_LIN_ACC]

        # Публикация
        imu_pub.publish(imu_msg)
        n_packets += 1

        # Эмуляция реального времени (уберите rate.sleep(), если нужна мгновенная выгрузка)
        rate.sleep()
        j = sig_idx + frame_len

    rospy.loginfo(f"Опубликовано {n_packets} пакетов IMU. Завершение.")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
