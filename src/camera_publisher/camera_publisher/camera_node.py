import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from picamera2 import Picamera2
import time
import cv2

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
        
        # Trigger autofocus and wait
        self.picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        time.sleep(2.0)  # Wait 2 seconds for autofocus to settle
        
        self.get_logger().info('Camera Publisher - Capturing ONE image...')
        self.capture_once()

    def capture_once(self):
        # Capture single image
        frame = self.picam2.capture_array()
        
        # Save image to file
        save_path = '/tmp/captured_image.jpg'
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.get_logger().info(f'Image saved to: {save_path}')
        
        # Convert to ROS Image message
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        
        # Publish
        self.publisher_.publish(msg)
        self.get_logger().info('Single image published! Shutting down...')
        
        # Cleanup
        self.picam2.stop()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()