#!/usr/bin/env fancy_python_path
import rospy
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped

class OdomToPath:
    def __init__(self):
        rospy.init_node('odom_to_path_node')
        self.path = Path()
        self.path.header.frame_id = "odom" # Ваша глобальная система координат
        
        # Подписываемся на вашу одометрию
        rospy.Subscriber('/drone/odom', Odometry, self.odom_callback)
        # Публикуем в формате Path (Траектория), который Foxglove подхватит мгновенно
        self.path_pub = rospy.Publisher('/drone/path', Path, queue_size=10)

    def odom_callback(self, data):
        pose = PoseStamped()
        pose.header = data.header
        pose.pose = data.pose.pose
        
        # Добавляем новую точку в массив траектории
        self.path.poses.append(pose)
        self.path.header.stamp = rospy.Time.now()
        
        # Ограничиваем длину шлейфа траектории до 2000 точек, чтобы не лагало
        if len(self.path.poses) > 2000:
            self.path.poses.pop(0)
            
        self.path_pub.publish(self.path)

if __name__ == '__main__':
    node = OdomToPath()
    rospy.spin()

