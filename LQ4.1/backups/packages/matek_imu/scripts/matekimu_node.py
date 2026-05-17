#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import rospy
import struct
import serial
from sensor_msgs.msg import NavSatFix, Imu
from std_msgs.msg import Header, Float32, UInt8
from geometry_msgs.msg import Vector3Stamped

# === Конфигурация протокола (из вашего кода) ===
SIG = b"ACO"
_PREFIX_LEN = len(SIG) + 1  # ACO(3) + L(1)
_SEQ_LEN = 1
_CRC_LEN = 2
STRUCT = struct.Struct("<H2d12fih6f")  # Ваш формат телеметрии

# === ROS-параметры (можно переопределить в launch-файле) ===
DEFAULT_PORT = '/dev/ttyAMA0'
DEFAULT_BAUD = 57600
DEFAULT_FRAME_ID = 'imu_link'
PUBLISH_RATE = 50  # Гц


class ACOUartNode:
    def __init__(self):
        rospy.init_node('aco_uart_driver', anonymous=False)
        
        # Параметры из ROS param server
        self.port = rospy.get_param('~port', DEFAULT_PORT)
        self.baud = rospy.get_param('~baud', DEFAULT_BAUD)
        self.frame_id = rospy.get_param('~frame_id', DEFAULT_FRAME_ID)
        
        # Publishers
        self.pub_gps = rospy.Publisher('gps/fix', NavSatFix, queue_size=10)
        self.pub_imu = rospy.Publisher('imu/data', Imu, queue_size=10)
        self.pub_alt = rospy.Publisher('altitude', Float32, queue_size=10)
        self.pub_seq = rospy.Publisher('seq', UInt8, queue_size=10)
        
        # Состояние
        self.serial_conn = None
        self.running = True
        self.last_seq = -1
        
        rospy.loginfo(f"ACO UART driver started: {self.port}@{self.baud}")
        rospy.on_shutdown(self.shutdown)
        
        self.connect_serial()
        self.run()
    
    def connect_serial(self):
        """Установка соединения с UART"""
        try:
            self.serial_conn = serial.Serial(
                port=self.port,
                baudrate=self.baud,
                timeout=0.1,  # Non-blocking read
                bytesize=serial.EIGHTBITS,
                parity=serial.PARITY_NONE,
                stopbits=serial.STOPBITS_ONE
            )
            rospy.loginfo(f"Serial connected: {self.port}")
        except serial.SerialException as e:
            rospy.logerr(f"Failed to open serial: {e}")
            rospy.signal_shutdown("Serial error")
    
    def find_frame(self, buffer):
        """Поиск сигнатуры ACO в буфере"""
        idx = buffer.find(SIG)
        if idx < 0:
            return None, buffer
        return idx, buffer
    
    def parse_frame(self, data, start_idx):
        """Парсинг одного кадра, возврат (success, frame_len, telemetry_dict)"""
        end = len(data)
        
        # Проверка границ
        if start_idx + _PREFIX_LEN > end:
            return False, 0, None
        
        # Чтение L
        L = data[start_idx + len(SIG)]
        frame_len = _PREFIX_LEN + L
        
        if start_idx + frame_len > end:
            return False, 0, None  # Кадр обрезан, ждём больше данных
        
        # Извлечение payload
        payload_start = start_idx + _PREFIX_LEN
        seq = data[payload_start]
        
        telemetry_len = L - _SEQ_LEN - _CRC_LEN
        if telemetry_len != STRUCT.size:
            rospy.logwarn(f"Unexpected telemetry size: {telemetry_len} vs {STRUCT.size}")
            return False, frame_len, None
        
        # unpack телеметрии
        telemetry_start = payload_start + _SEQ_LEN
        body = data[telemetry_start : telemetry_start + telemetry_len]
        
        try:
            u = STRUCT.unpack(body)
        except struct.error as e:
            rospy.logerr(f"Struct unpack error: {e}")
            return False, frame_len, None
        
        # Маппинг полей (адаптируйте под ваши имена)
        telemetry = {
            'seq': seq,
            'packetNumber': u[0],
            'lat': u[1],
            'lon': u[2],
            'gps_height': u[3],
            'hAcc': u[4],
            'IMU_PITCH': u[5],
            'IMU_ROLL': u[6],
            'IMU_YAW': u[7],
            'rateX': u[8],
            'rateY': u[9],
            'rateZ': u[10],
            'ALTITUDE': u[11],
            'presAltOffsetAtGround': u[12],
            'ms5611_altitude': u[13],
            'lidar_range': u[14],
            'flags': u[15],
            'state': u[16],
            'IMU_ACCX': u[17],
            'IMU_ACCY': u[18],
            'IMU_ACCZ': u[19],
            'IMU_MAGX': u[20],
            'IMU_MAGY': u[21],
            'IMU_MAGZ': u[22],
        }
        
        return True, frame_len, telemetry
    
    def publish_telemetry(self, telemetry):
        """Публикация данных в ROS топики"""
        now = rospy.Time.now()
        header = Header(stamp=now, frame_id=self.frame_id)
        
        # GPS fix
        gps_msg = NavSatFix()
        gps_msg.header = header
        gps_msg.latitude = telemetry['lat']
        gps_msg.longitude = telemetry['lon']
        gps_msg.altitude = telemetry['gps_height']
        gps_msg.position_covariance = [telemetry['hAcc']**2] * 9  # diagonal
        gps_msg.position_covariance_type = NavSatFix.COVARIANCE_TYPE_DIAGONAL_KNOWN
        gps_msg.status.status = NavSatFix.STATUS_FIX
        self.pub_gps.publish(gps_msg)
        
        # IMU data
        imu_msg = Imu()
        imu_msg.header = header
        # Euler -> Quaternion (упрощённо, добавьте proper conversion if needed)
        # Для продакшена используйте tf.transformations.quaternion_from_euler
        imu_msg.orientation.w = 1.0  # placeholder
        imu_msg.angular_velocity.x = telemetry['rateX']
        imu_msg.angular_velocity.y = telemetry['rateY']
        imu_msg.angular_velocity.z = telemetry['rateZ']
        imu_msg.linear_acceleration.x = telemetry['IMU_ACCX']
        imu_msg.linear_acceleration.y = telemetry['IMU_ACCY']
        imu_msg.linear_acceleration.z = telemetry['IMU_ACCZ']
        self.pub_imu.publish(imu_msg)
        
        # Altitude
        alt_msg = Float32()
        alt_msg.data = telemetry['ALTITUDE']
        self.pub_alt.publish(alt_msg)
        
        # Sequence number
        seq_msg = UInt8()
        seq_msg.data = telemetry['seq']
        self.pub_seq.publish(seq_msg)
    
    def run(self):
        """Основной цикл чтения UART"""
        buffer = b""
        rate = rospy.Rate(PUBLISH_RATE)
        
        while not rospy.is_shutdown() and self.running:
            try:
                if self.serial_conn and self.serial_conn.in_waiting:
                    chunk = self.serial_conn.read(self.serial_conn.in_waiting)
                    buffer += chunk
                    
                    # Поиск и парсинг кадров
                    while True:
                        idx = buffer.find(SIG)
                        if idx < 0:
                            break
                        
                        success, frame_len, telemetry = self.parse_frame(buffer, idx)
                        
                        if success and telemetry:
                            # Проверка на пропуски последовательности
                            if self.last_seq >= 0 and telemetry['seq'] != (self.last_seq + 1) % 256:
                                rospy.logwarn(f"Sequence gap: {self.last_seq} -> {telemetry['seq']}")
                            
                            self.publish_telemetry(telemetry)
                            self.last_seq = telemetry['seq']
                            buffer = buffer[idx + frame_len:]  # сдвиг буфера
                        else:
                            # Невалидный или неполный кадр — сдвигаемся на 1 байт
                            buffer = buffer[idx + 1:] if idx >= 0 else buffer
                
                rate.sleep()
                
            except serial.SerialException as e:
                rospy.logerr(f"Serial error: {e}")
                rospy.sleep(1.0)
                self.connect_serial()
            except Exception as e:
                rospy.logerr(f"Unexpected error: {e}")
                rospy.sleep(0.1)
    
    def shutdown(self):
        """Корректное завершение"""
        self.running = False
        if self.serial_conn and self.serial_conn.is_open:
            self.serial_conn.close()
        rospy.loginfo("ACO UART driver shutdown")


if __name__ == '__main__':
    try:
        ACOUartNode()
    except rospy.ROSInterruptException:
        pass
