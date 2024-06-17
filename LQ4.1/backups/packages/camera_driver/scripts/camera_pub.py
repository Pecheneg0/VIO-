#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import time

class CameraPublisher:
    def __init__(self):
        rospy.init_node('camera_publisher', anonymous=True)
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher('/data', Image, queue_size=10)
        
        # Настройка камеры
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            rospy.logerr("Не удалось открыть камеру!")
            exit(1)
            
        # Параметры камеры
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 20)
        
        self.rate = rospy.Rate(20)  # 15 Гц
        rospy.loginfo("Камера успешно инициализирована")
        
    def run(self):
        try:
            while not rospy.is_shutdown():
                ret, frame = self.cap.read()
                if not ret:
                    rospy.logwarn("Не удалось получить кадр с камеры")
                    continue
                    
                try:
                    # Публикация изображения
                    # 1. Конвертируем кадр из BGR в Mono8 (Grayscale)
                    #mono_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    img_msg = self.bridge.cv2_to_imgmsg(frame, encoding="bgr8")
                    img_msg.header.stamp = rospy.Time.now()
                    img_msg.header.frame_id = "camera_link"
                    self.image_pub.publish(img_msg)
                    
                except CvBridgeError as e:
                    rospy.logerr(f"Ошибка cv_bridge: {e}")
                    
                self.rate.sleep()
                
        finally:
            self.cap.release()
            rospy.loginfo("Камера закрыта")

if __name__ == '__main__':
    try:
        camera = CameraPublisher()
        camera.run()
    except rospy.ROSInterruptException:
        pass
    except Exception as e:
        rospy.logerr(f"Критическая ошибка: {e}")
