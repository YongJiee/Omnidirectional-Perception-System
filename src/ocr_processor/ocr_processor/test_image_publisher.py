"""
Test Image Publisher — simulates the Pi camera node
Reads local images from a folder and publishes them to ROS2 topics
so you can test the full WSL pipeline without the Raspberry Pi.

Usage:
    # Inbound mode (publishes immediately)
    ros2 launch camera_publisher distributed_system.launch.py test_mode:=true scan_mode:=inbound image_dir:=/home/yongjie/test_images

    # Sorting mode (waits for /trigger_capture signal)
    ros2 launch camera_publisher distributed_system.launch.py test_mode:=true scan_mode:=sorting image_dir:=/home/yongjie/test_images
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import os
import time

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class TestImagePublisher(Node):
    def __init__(self):
        super().__init__('test_image_publisher')

        self.declare_parameter('num_cameras', 4)
        self.declare_parameter('image_dir', os.path.expanduser('~/test_images'))
        self.declare_parameter('scan_mode', 'inbound')

        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.image_dir   = self.get_parameter('image_dir').get_parameter_value().string_value
        self.scan_mode   = self.get_parameter('scan_mode').get_parameter_value().string_value
        self._capturing  = False
        self._ready      = False

        self.get_logger().info(f'Test Publisher — {self.num_cameras} camera(s) [{self.scan_mode}]')
        self.get_logger().info(f'Image folder: {self.image_dir}')

        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', RELIABLE_QOS)

        self.batch_pub = self.create_publisher(String, '/batch_complete', RELIABLE_QOS)

        if not self._check_images():
            return

        # Non-blocking subscriber wait using a ROS timer so spin() can run
        self.get_logger().info(f'Waiting for {self.num_cameras} OCR subscriber(s)...')
        self._subscriber_timer = self.create_timer(0.5, self._check_subscribers)

    def _check_subscribers(self):
        if self._ready:
            return

        counts = [self.publishers_[i].get_subscription_count() for i in range(self.num_cameras)]
        ready = sum(1 for c in counts if c > 0)
        self.get_logger().info(f'Subscribers ready: {ready}/{self.num_cameras}')

        if ready == self.num_cameras:
            self._ready = True
            self._subscriber_timer.cancel()

            if self.scan_mode == 'inbound':
                self.get_logger().info('Inbound mode — publishing test images now...')
                self.publish_all_images()
            else:
                self.trigger_sub = self.create_subscription(
                    String, '/trigger_capture', self.on_trigger, RELIABLE_QOS
                )
                self.get_logger().info(
                    'Sorting mode — waiting for /trigger_capture signal...'
                )

    def on_trigger(self, msg):
        if msg.data == 'capture':
            if self._capturing:
                self.get_logger().warn('Capture already in progress — ignoring trigger')
                return
            self.get_logger().info('Trigger received — publishing test images')
            self._capturing = True
            self.publish_all_images()
            self._capturing = False
            self.get_logger().info('Capture complete — ready for next trigger')

    def _find_image(self, camera_id):
        for suffix in ['', '_received']:
            for ext in ['.jpg', '.png']:
                path = os.path.join(self.image_dir, f'camera_{camera_id}{suffix}{ext}')
                if os.path.exists(path):
                    return path
        return None

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

    def publish_all_images(self):
        overall_start = time.time()
        self.get_logger().info(f'=== Starting test publish — {self.num_cameras} images ===')
        for camera_id in range(self.num_cameras):
            image_path = self._find_image(camera_id)
            self._publish_single_image(camera_id, image_path, overall_start)
        total = time.time() - overall_start
        batch_msg = String()
        batch_msg.data = f'{self.image_dir},{overall_start},{total}'
        self.batch_pub.publish(batch_msg)
        self.get_logger().info('Batch complete signal sent')
        self.get_logger().info(f'=== Test publish complete: {total:.3f}s ===')

    def _publish_single_image(self, camera_id, image_path, overall_start):
        cam_start = time.time()
        self.get_logger().info(f'Publishing camera_{camera_id} from: {image_path}')
        frame = cv2.imread(image_path)
        if frame is None:
            self.get_logger().error(f'Could not read image: {image_path}')
            return
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)
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
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()