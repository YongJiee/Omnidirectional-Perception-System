import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from picamera2 import Picamera2
import time

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(Image, '/camera/image_raw', 10)
        self.bridge = CvBridge()
        
        # Initialize Picamera2
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (1920, 1080)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(0.5)  # Autofocus settling time
        
        # Publish at 2 Hz (every 0.5 seconds)
        self.timer = self.create_timer(0.5, self.publish_image)
        self.get_logger().info('Camera Publisher Node Started')

    def publish_image(self):
        # Capture image
        frame = self.picam2.capture_array()
        
        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        # Publish
        self.publisher_.publish(msg)
        self.get_logger().info('Image published')

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
