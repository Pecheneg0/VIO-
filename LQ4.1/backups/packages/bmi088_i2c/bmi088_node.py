#!/usr/bin/env python3
import rospy
import smbus2
import time
from sensor_msgs.msg import Imu
from geometry_msgs.msg import Vector3

# Адреса I2C
ACC_ADDR = 0x19
GYR_ADDR = 0x69

# Регистры BMI088
ACC_CHIP_ID = 0x00
ACC_PWR_CONF = 0x7C
ACC_PWR_CTRL = 0x7D
ACC_DATA = 0x12

GYR_CHIP_ID = 0x00
GYR_DATA = 0x02

class BMI088Driver:
    def __init__(self):
        rospy.init_node('bmi088_i2c_node', anonymous=True)
        self.pub = rospy.Publisher('imu/data_raw', Imu, queue_size=10)
        self.bus = smbus2.SMBus(1)
        self.frame_id = rospy.get_param('~frame_id', 'imu_link')
        
        self.init_sensor()
        
    def init_sensor(self):
        # 1. Проверка связи
        if self.bus.read_byte_data(ACC_ADDR, ACC_CHIP_ID) != 0x1E:
            rospy.logerr("Акселерометр BMI088 не обнаружен!")
        if self.bus.read_byte_data(GYR_ADDR, GYR_CHIP_ID) != 0x0F:
            rospy.logerr("Гироскоп BMI088 не обнаружен!")
            
        # 2. Активация акселерометра (по умолчанию он в режиме глубокого сна)
        self.bus.write_byte_data(ACC_ADDR, ACC_PWR_CONF, 0x00) # Включение питания
        time.sleep(0.05)
        self.bus.write_byte_data(ACC_ADDR, ACC_PWR_CTRL, 0x04) # Включение акселерометра
        time.sleep(0.05)
        rospy.loginfo("Датчик BMI088 успешно инициализирован по I2C")

    def read_word(self, addr, reg):
        # Чтение двух байт (младший, затем старший)
        low = self.bus.read_byte_data(addr, reg)
        high = self.bus.read_byte_data(addr, reg + 1)
        value = (high << 8) | low
        if value & 0x8000: # Обработка знака (двухкомпонентный код)
            value -= 0x10000
        return value

    def run(self):
        rate = rospy.Rate(100) # Частота публикации 100 Гц
        while not rospy.is_shutdown():
            try:
                imu_msg = Imu()
                imu_msg.header.stamp = rospy.Time.now()
                imu_msg.header.frame_id = self.frame_id
                
                # Чтение акселерометра (Диапазон по умолчанию +-6g, чувствительность: 0.183 mg/LSB)
                # Переводим в м/с^2 (1 mg = 0.00980665 м/с^2)
                scale_acc = 0.000183 * 9.80665
                ax = self.read_word(ACC_ADDR, ACC_DATA) * scale_acc
                ay = self.read_word(ACC_ADDR, ACC_DATA + 2) * scale_acc
                az = self.read_word(ACC_ADDR, ACC_DATA + 4) * scale_acc
                
                # Чтение гироскопа (Диапазон по умолчанию +-2000 dps, чувствительность: 16.384 LSB/dps)
                # Переводим в Радианы в секунду
                scale_gyr = (1.0 / 16.384) * (3.14159 / 180.0)
                gx = self.read_word(GYR_ADDR, GYR_DATA) * scale_gyr
                gy = self.read_word(GYR_ADDR, GYR_DATA + 2) * scale_gyr
                gz = self.read_word(GYR_ADDR, GYR_DATA + 4) * scale_gyr
                
                # Заполнение ROS сообщения
                imu_msg.linear_acceleration = Vector3(ax, ay, az)
                imu_msg.angular_velocity = Vector3(gx, gy, gz)
                
                # Ковариация не заполнена (-1 означает, что она неизвестна)
                imu_msg.orientation_covariance[0] = -1
                imu_msg.linear_acceleration_covariance[0] = -1
                imu_msg.angular_velocity_covariance[0] = -1
                
                self.pub.publish(imu_msg)
                
            except Exception as e:
                rospy.logwarn(f"Ошибка чтения данных по I2C: {e}")
                
            rate.sleep()

if __name__ == '__main__':
    try:
        driver = BMI088Driver()
        driver.run()
    except rospy.ROSInterruptException:
        pass
