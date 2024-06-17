#!/bin/bash
# Скрипт для калибровки камеры в ROS
# Использует стандартные параметры шахматной доски 8x6 с размером квадрата 2.4 см

# Проверка, установлен ли пакет для калибровки
if ! command -v cameracalibrator.py &> /dev/null; then
    echo "Пакет camera_calibration не установлен. Установите его командой:"
    echo "sudo apt install ros-noetic-camera-calibration"
    exit 1
fi

# Запуск калибровочного инструмента
echo "Запуск калибровки камеры..."
echo "Пожалуйста, перемещайте шахматную доску перед камерой, чтобы покрыть всё поле зрения."
echo "Для калибровки необходимо собрать данные с разных положений."

# Стандартные параметры:
# --size 8x6 - размеры шахматной доски (8 квадратов по ширине, 6 по высоте)
# --square 0.024 - размер квадрата в метрах (2.4 см)
# image:=/camera/image_raw - топик с изображением от камеры
# camera:=/camera - топик с информацией о камере
rosrun camera_calibration cameracalibrator.py \
    --size 8x6 \
    --square 0.024 \
    image:=/camera/image_raw \
    camera:=/camera
