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
        
        # Start overall timing
        self.overall_start = time.time()
        
        # Initialize Picamera2
        init_start = time.time()
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (1920, 1080)}
        )
        self.picam2.configure(config)
        self.picam2.start()
        init_time = time.time() - init_start
        
        # Autofocus timing
        af_start = time.time()
        self.picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        time.sleep(2.0)
        af_time = time.time() - af_start
        
        self.get_logger().info(f'Camera init: {init_time:.3f}s')
        self.get_logger().info(f'Autofocus: {af_time:.3f}s')
        self.get_logger().info('Capturing image...')
        
        self.capture_once()

    def capture_once(self):
        # Capture timing
        capture_start = time.time()
        frame = self.picam2.capture_array()
        capture_time = time.time() - capture_start
        
        # Save timing
        save_start = time.time()
        save_path = '/tmp/captured_image.jpg'
        cv2.imwrite(save_path, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        save_time = time.time() - save_start
        
        # Publish timing
        publish_start = time.time()
        msg = self.bridge.cv2_to_imgmsg(frame, encoding='rgb8')
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = 'camera_frame'
        self.publisher_.publish(msg)
        publish_time = time.time() - publish_start
        
        # Overall timing
        overall_time = time.time() - self.overall_start
        
        # Log timing breakdown
        self.get_logger().info('=== CAMERA NODE TIMING ===')
        self.get_logger().info(f'Capture:     {capture_time:.3f}s')
        self.get_logger().info(f'Save image:  {save_time:.3f}s')
        self.get_logger().info(f'ROS publish: {publish_time:.3f}s')
        self.get_logger().info(f'Overall:     {overall_time:.3f}s')
        self.get_logger().info('==========================')
        
        # Cleanup
        self.picam2.stop()
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    rclpy.spin(node)

if __name__ == '__main__':
    main()