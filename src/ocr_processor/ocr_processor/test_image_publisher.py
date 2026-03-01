"""
Test Image Publisher — simulates the Pi camera node
Reads local images from a folder and publishes them to ROS2 topics
so you can test the full WSL pipeline without the Raspberry Pi.

Usage:
    # Use default test_images/ folder, 4 cameras
    ros2 run ocr_processor test_publisher

    # Specify custom image folder and camera count
    ros2 run ocr_processor test_publisher --ros-args \
        -p image_dir:=/home/user/my_images \
        -p num_cameras:=2

Image naming convention (must match):
    camera_0.jpg or camera_0.png
    camera_1.jpg or camera_1.png
    camera_2.jpg or camera_2.png
    camera_3.jpg or camera_3.png
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import os
import time


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')

        # ---------------------------------------------------------------
        # Parameters
        # ---------------------------------------------------------------
        self.declare_parameter('num_cameras', 4)
        self.declare_parameter('image_dir', os.path.expanduser('~/test_images'))

        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.image_dir = self.get_parameter('image_dir').get_parameter_value().string_value

        self.get_logger().info(f'Test Publisher — {self.num_cameras} camera(s)')
        self.get_logger().info(f'Image folder: {self.image_dir}')

        # ---------------------------------------------------------------
        # Publishers — same topics as real Pi node
        # ---------------------------------------------------------------
        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', 10)

        self.batch_pub = self.create_publisher(String, '/batch_complete', 10)

        # ---------------------------------------------------------------
        # Validate images exist before starting
        # ---------------------------------------------------------------
        if not self._check_images():
            return

        # ---------------------------------------------------------------
        # Wait for OCR node subscribers
        # ---------------------------------------------------------------
        self.get_logger().info(f'Waiting for {self.num_cameras} OCR subscriber(s)...')
        self._wait_for_subscribers()

        self.get_logger().info('Subscribers ready — publishing test images...')
        self.publish_all_images()

    # ---------------------------------------------------------------
    # Find image path — checks .jpg first, then .png
    # ---------------------------------------------------------------
    def _find_image(self, camera_id):
        for ext in ['.jpg', '.png']:
            path = os.path.join(self.image_dir, f'camera_{camera_id}{ext}')
            if os.path.exists(path):
                return path
        return None

    # ---------------------------------------------------------------
    # Check all required images exist (.jpg or .png)
    # ---------------------------------------------------------------
    def _check_images(self):
        missing = []
        for i in range(self.num_cameras):
            if self._find_image(i) is None:
                missing.append(f'camera_{i}.jpg or camera_{i}.png')

        if missing:
            self.get_logger().error('Missing image files:')
            for m in missing:
                self.get_logger().error(f'  {m}')
            self.get_logger().error(
                f'\nPlace your test images in: {self.image_dir}\n'
                f'Named as: camera_0.jpg/png, camera_1.jpg/png, ...'
            )
            return False

        self.get_logger().info(f'All {self.num_cameras} test images found ✓')
        return True

    # ---------------------------------------------------------------
    # Wait for WSL OCR node to be ready
    # ---------------------------------------------------------------
    def _wait_for_subscribers(self):
        while True:
            counts = [self.publishers_[i].get_subscription_count() for i in range(self.num_cameras)]
            ready = sum(1 for c in counts if c > 0)
            self.get_logger().info(f'Subscribers ready: {ready}/{self.num_cameras}')
            if ready == self.num_cameras:
                break
            time.sleep(0.5)

    # ---------------------------------------------------------------
    # Publish all images simulating Pi capture sequence
    # ---------------------------------------------------------------
    def publish_all_images(self):
        overall_start = time.time()
        self.get_logger().info(f'=== Starting test publish — {self.num_cameras} images ===')

        for camera_id in range(self.num_cameras):
            image_path = self._find_image(camera_id)
            self._publish_single_image(camera_id, image_path, overall_start)

        # Signal batch complete — same as real Pi node
        batch_msg = String()
        batch_msg.data = f'{self.image_dir},{overall_start}'
        self.batch_pub.publish(batch_msg)
        self.get_logger().info('Batch complete signal sent')

        total = time.time() - overall_start
        self.get_logger().info(f'=== Test publish complete: {total:.3f}s ===')

        time.sleep(1.0)

    # ---------------------------------------------------------------
    # Read and publish a single image
    # ---------------------------------------------------------------
    def _publish_single_image(self, camera_id, image_path, overall_start):
        cam_start = time.time()
        self.get_logger().info(f'Publishing camera_{camera_id} from: {image_path}')

        # Read image
        frame = cv2.imread(image_path)
        if frame is None:
            self.get_logger().error(f'Could not read image: {image_path}')
            return

        # Encode as JPEG — same as real Pi node
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        # Build message — same format as real Pi node
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = f'{overall_start},{cam_start},cam_{camera_id}'
        msg.format = "jpeg"
        msg.data = buffer.tobytes()

        self.publishers_[camera_id].publish(msg)

        elapsed = time.time() - cam_start
        self.get_logger().info(f'Camera {camera_id} published in {elapsed:.3f}s')


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()