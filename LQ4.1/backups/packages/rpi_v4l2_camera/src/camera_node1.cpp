#include <ros/ros.h>
#include <sensor_msgs/Image.h>
#include <cv_bridge/cv_bridge.h>
#include <opencv2/opencv.hpp>
#include <thread>
#include <mutex>

class FastCameraNode {
public:
    FastCameraNode() : run(true) {
        ros::NodeHandle nh("~");
        
        // Параметры со значениями по умолчанию для цветного режима 1280x720 @ 30 FPS
        nh.param<std::string>("camera_device", camera_device_, "/dev/video0");
        nh.param<int>("image_width", image_width_, 640);
        nh.param<int>("image_height", image_height_, 480);
        nh.param<int>("fps", fps_, 10);
        nh.param<std::string>("frame_id", frame_id_, "camera_frame");
        
        // Publisher
        image_pub = nh.advertise<sensor_msgs::Image>("/data", 1);

        // Открываем камеру через V4L2
        ROS_INFO("Opening camera: %s", camera_device_.c_str());
        cap.open(camera_device_, cv::CAP_V4L2);
        
        if (!cap.isOpened()) {
            ROS_ERROR("Could not open camera at %s", camera_device_.c_str());
            return;
        }

        // Принудительно включаем MJPEG, чтобы обойти ограничение USB 2.0 по пропускной способности
        cap.set(cv::CAP_PROP_FOURCC, cv::VideoWriter::fourcc('M', 'J', 'P', 'G'));

        // Устанавливаем геометрические параметры и частоту кадров
        cap.set(cv::CAP_PROP_FRAME_WIDTH, image_width_);
        cap.set(cv::CAP_PROP_FRAME_HEIGHT, image_height_);
        cap.set(cv::CAP_PROP_FPS, fps_);
        
        // Проверяем реальные установленные параметры драйвером камеры
        int actual_width = cap.get(cv::CAP_PROP_FRAME_WIDTH);
        int actual_height = cap.get(cv::CAP_PROP_FRAME_HEIGHT);
        double actual_fps = cap.get(cv::CAP_PROP_FPS);
        
        ROS_INFO("Camera opened successfully:");
        ROS_INFO("  Requested: %dx%d at %d FPS", image_width_, image_height_, fps_);
        ROS_INFO("  Actual: %dx%d at %.0f FPS", actual_width, actual_height, actual_fps);
        ROS_INFO("  Frame ID: %s", frame_id_.c_str());

        // Запуск потока захвата
        capture_thread = std::thread(&FastCameraNode::captureLoop, this);
    }

    ~FastCameraNode() {
        run = false;
        if (capture_thread.joinable()) capture_thread.join();
        if (cap.isOpened()) cap.release();
        ROS_INFO("Camera node stopped");
    }

private:
    void captureLoop() {
        cv::Mat frame;
        while (ros::ok() && run) {
            if (cap.read(frame)) {
                // Код конвертации в оттенки серого удален. 
                // Отправляем полноценный цветной кадр с кодировкой bgr8.
                auto msg = cv_bridge::CvImage(std_msgs::Header(), "bgr8", frame).toImageMsg();
                
                msg->header.stamp = ros::Time::now();
                msg->header.frame_id = frame_id_;
                image_pub.publish(msg);
            } else {
                ROS_WARN_THROTTLE(1, "Failed to grab frame from camera");
            }
        }
    }

    cv::VideoCapture cap;
    ros::Publisher image_pub;
    std::thread capture_thread;
    bool run;
    
    // Параметры
    std::string camera_device_;
    int image_width_;
    int image_height_;
    int fps_;
    std::string frame_id_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "camera_node");
    FastCameraNode node;
    ros::spin();
    return 0;
}

