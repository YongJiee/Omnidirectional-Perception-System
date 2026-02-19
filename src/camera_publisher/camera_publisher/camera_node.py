import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from picamera2 import Picamera2
import time
import cv2

class CameraPublisher(Node):
    def __init__(self):
        super().__init__('camera_publisher')
        self.publisher_ = self.create_publisher(CompressedImage, '/camera/image_raw/compressed', 10)
        
        # Start overall timing
        self.overall_start = time.time()
        
        # Initialize Picamera2 with lower resolution
        init_start = time.time()
        self.picam2 = Picamera2()
        config = self.picam2.create_still_configuration(
            main={"size": (1280, 720)}  # Reduced from 1920x1080
        )
        self.picam2.configure(config)
        self.picam2.start()
        init_time = time.time() - init_start
        
        # Autofocus timing
        af_start = time.time()
        self.picam2.set_controls({"AfMode": 2, "AfTrigger": 0})
        time.sleep(2.0)
        af_time = time.time() - af_start
        
        self.get_logger().info(f'Camera init (1280x720): {init_time:.3f}s')
        self.get_logger().info(f'Autofocus: {af_time:.3f}s')
        self.get_logger().info('Capturing image...')
        
        self.capture_once()

    def capture_once(self):
        # Capture timing
        capture_start = time.time()
        frame = self.picam2.capture_array()
        capture_time = time.time() - capture_start
        
        # Convert and save local copy
        save_start = time.time()
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        cv2.imwrite('/tmp/captured_image.jpg', frame_bgr)
        save_time = time.time() - save_start
        
        # Encode as JPEG
        encode_start = time.time()
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
        encode_time = time.time() - encode_start
        
        # Wait for subscriber before publishing
        self.get_logger().info('Waiting for OCR subscriber...')
        while self.publisher_.get_subscription_count() == 0:
            time.sleep(0.1)
        self.get_logger().info('OCR subscriber detected!')
        
        # Create CompressedImage message
        publish_start = time.time()
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'{capture_start}'
        msg.format = "jpeg"
        msg.data = buffer.tobytes()
        
        self.publisher_.publish(msg)
        publish_time = time.time() - publish_start
        
        # Overall timing
        overall_time = time.time() - self.overall_start
        
        # Log timing breakdown
        self.get_logger().info('=== CAMERA NODE TIMING ===')
        self.get_logger().info(f'Capture:     {capture_time:.3f}s')
        self.get_logger().info(f'Save local:  {save_time:.3f}s')
        self.get_logger().info(f'JPEG encode: {encode_time:.3f}s')
        self.get_logger().info(f'ROS publish: {publish_time:.3f}s')
        self.get_logger().info(f'Total:       {overall_time:.3f}s')
        self.get_logger().info('==========================')
        
        # Keep alive briefly after publishing
        time.sleep(1.0)
        self.picam2.stop()

def main(args=None):
    rclpy.init(args=args)
    node = CameraPublisher()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()