#!/usr/bin/env python3
"""
ACO Binary → ROS1 /imu Bridge
Читает UART поток в реальном времени, парсит протокол ACO и публикует IMU данные для OpenVINS.
"""

import rospy
import struct
import math
import time
import serial
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3, Quaternion
from tf.transformations import quaternion_from_euler

# ===== КОНФИГУРАЦИЯ =====
SERIAL_PORT = "/dev/ttyS0"   # Измените под вашу плату (ttyAMA0, ttyUSB0 и т.д.)
BAUDRATE = 921600
SERIAL_TIMEOUT = 0.005
FRAME_ID = "imu_link"        # Должен совпадать с URDF и конфигом OpenVINS
MAX_BUFFER_SIZE = 524288     # 512 KB защита от переполнения при потере синхронизации

# Протокол ACO (совпадает с decoder.py)
SIG = b"ACO"
_PREFIX_LEN = len(SIG) + 1   # 4 байта: ACO(3) + L(1)
_SEQ_LEN = 1
_CRC_LEN = 2
# <H: packetNumber, 2d: lat/lon, 12f: телеметрия, i: flags, h: state, 6f: acc/mag
STRUCT = struct.Struct("<H2d12fih6f")

# Ковариации (настройте под датчик, OpenVINS чувствителен к этим значениям)
COV_ANG_VEL = 1e-4   # rad^2/s^2
COV_LIN_ACC = 1e-3   # m^2/s^4
COV_ORIENT  = 1e-2   # rad^2

def euler_deg_to_quat(pitch_deg, roll_deg, yaw_deg):
    """Конвертация углов Эйлера (градусы) → Кватернион (ROS convention: X=roll, Y=pitch, Z=yaw)"""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)
    # tf.transformations использует порядок (roll, pitch, yaw)
    return quaternion_from_euler(roll, pitch, yaw, axes='sxyz')

def main():
    rospy.init_node('aco_imu_bridge', anonymous=True)
    imu_pub = rospy.Publisher('/imu', Imu, queue_size=10)

    # Открытие UART
    try:
        ser = serial.Serial(SERIAL_PORT, BAUDRATE, timeout=SERIAL_TIMEOUT)
        rospy.loginfo(f"UART открыт: {SERIAL_PORT} @ {BAUDRATE} baud")
    except serial.SerialException as e:
        rospy.logerr(f"Не удалось открыть порт {SERIAL_PORT}: {e}")
        return

    buffer = bytearray()
    n_packets = 0
    last_log_time = time.time()

    rospy.loginfo("Ожидание кадров ACO...")

    while not rospy.is_shutdown():
        # Читаем все доступные байты
        chunk = ser.read(ser.in_waiting or 1)
        if chunk:
            buffer.extend(chunk)

        # Защита от переполнения буфера при потере синхронизации
        if len(buffer) > MAX_BUFFER_SIZE:
            rospy.logwarn("Buffer overflow! Очищаю буфер. Проверьте целостность линии.")
            buffer.clear()
            continue

        # Поиск сигнатуры
        sig_idx = buffer.find(SIG)
        if sig_idx < 0:
            continue  # Сигнатура не найдена, ждём данные

        # Отрезаем мусор перед сигнатурой
        if sig_idx > 0:
            del buffer[:sig_idx]
            sig_idx = 0

        # Проверка наличия байта длины (L)
        if len(buffer) < _PREFIX_LEN:
            continue

        L = buffer[len(SIG)]
        if L < 3:
            # Невалидная длина, пропускаем сигнатуру
            del buffer[:len(SIG)]
            continue

        frame_len = _PREFIX_LEN + L
        if len(buffer) < frame_len:
            continue  # Фрейм не полный, ждём остаток

        # Извлекаем и удаляем фрейм из буфера
        frame = buffer[:frame_len]
        del buffer[:frame_len]

        # Парсинг полезной нагрузки
        payload_start = _PREFIX_LEN
        telemetry_start = payload_start + _SEQ_LEN
        telemetry_len = L - _SEQ_LEN - _CRC_LEN
        body = frame[telemetry_start : telemetry_start + telemetry_len]

        if len(body) != STRUCT.size:
            rospy.logwarn(f"Несоответствие размера телеметрии: {len(body)} vs {STRUCT.size}")
            continue

        try:
            u = STRUCT.unpack(body)
        except struct.error as e:
            rospy.logwarn(f"Ошибка распаковки struct: {e}")
            continue

        # Извлечение полей (индексы согласно <H2d12fih6f)
        pitch_deg = u[5]
        roll_deg  = u[6]
        yaw_deg   = u[7]
        rate_x    = u[8]
        rate_y    = u[9]
        rate_z    = u[10]
        acc_x     = u[17]
        acc_y     = u[18]
        acc_z     = u[19]

        # Формирование сообщения ROS
        imu_msg = Imu()
        imu_msg.header.stamp = rospy.Time.now()
        imu_msg.header.frame_id = FRAME_ID
        imu_msg.header.seq = n_packets

        # Ориентация
        q = euler_deg_to_quat(pitch_deg, roll_deg, yaw_deg)
        imu_msg.orientation = Quaternion(*q)
        imu_msg.orientation_covariance = [COV_ORIENT, 0, 0, 0, COV_ORIENT, 0, 0, 0, COV_ORIENT]

        # Угловая скорость
        imu_msg.angular_velocity = Vector3(rate_x, rate_y, rate_z)
        imu_msg.angular_velocity_covariance = [COV_ANG_VEL, 0, 0, 0, COV_ANG_VEL, 0, 0, 0, COV_ANG_VEL]

        # Линейное ускорение
        imu_msg.linear_acceleration = Vector3(acc_x, acc_y, acc_z)
        imu_msg.linear_acceleration_covariance = [COV_LIN_ACC, 0, 0, 0, COV_LIN_ACC, 0, 0, 0, COV_LIN_ACC]

        # Публикация (без rate.sleep(), чтобы OpenVINS получал данные с реальной частотой датчика)
        imu_pub.publish(imu_msg)
        n_packets += 1

        # Троттлинг логов
        if time.time() - last_log_time > 5.0:
            rospy.loginfo(f"Опубликовано пакетов: {n_packets} | Буфер: {len(buffer)} байт")
            last_log_time = time.time()

    ser.close()
    rospy.loginfo(f"Мост остановлен. Всего опубликовано: {n_packets}")

if __name__ == "__main__":
    try:
        main()
    except rospy.ROSInterruptException:
        pass
