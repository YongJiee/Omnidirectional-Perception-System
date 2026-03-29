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
from sensor_msgs.msg import CompressedImage  # Message type for publishing camera images
from std_msgs.msg import String               # Message type for trigger and batch signals
import cv2
import os
import time

# ---------------------------------------------------------------------------
# QoS profile shared by all publishers and subscribers in this node.
# RELIABLE ensures no messages are dropped; KEEP_LAST(10) buffers up to 10.
# ---------------------------------------------------------------------------
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class TestImagePublisher(Node):
    """
    Simulates the Raspberry Pi multi-camera node for WSL-side testing.

    Inbound mode  — publishes all camera images immediately once OCR
                    subscribers are detected.
    Sorting mode  — waits for a 'capture' message on /trigger_capture before
                    publishing, mirroring the real trigger flow from the arm.
    """

    def __init__(self):
        super().__init__('test_image_publisher')

        # -----------------------------------------------------------------------
        # ROS2 parameters — can be overridden at launch time:
        #   num_cameras : number of camera topics to simulate (default 4)
        #   image_dir   : folder containing test images named camera_0.jpg, etc.
        #   scan_mode   : 'inbound' or 'sorting'
        # -----------------------------------------------------------------------
        self.declare_parameter('num_cameras', 4)
        self.declare_parameter('image_dir', os.path.expanduser('~/test_images'))
        self.declare_parameter('scan_mode', 'inbound')

        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.image_dir   = self.get_parameter('image_dir').get_parameter_value().string_value
        self.scan_mode   = self.get_parameter('scan_mode').get_parameter_value().string_value

        # Guard flags to prevent concurrent or duplicate captures
        self._capturing  = False  # True while a publish batch is in progress
        self._ready      = False  # True once all OCR subscribers have connected

        self.get_logger().info(f'Test Publisher — {self.num_cameras} camera(s) [{self.scan_mode}]')
        self.get_logger().info(f'Image folder: {self.image_dir}')

        # -----------------------------------------------------------------------
        # Create one CompressedImage publisher per camera, matching the topic
        # names that ocr_node subscribes to on the WSL side.
        # -----------------------------------------------------------------------
        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', RELIABLE_QOS)

        # Publisher that signals ocr_node that all camera images have been sent
        self.batch_pub = self.create_publisher(String, '/batch_complete', RELIABLE_QOS)

        # Abort early if any required test image files are missing
        if not self._check_images():
            return

        # -----------------------------------------------------------------------
        # Poll for OCR subscribers using a non-blocking ROS timer (0.5 s interval)
        # so that rclpy.spin() remains unblocked while waiting.
        # -----------------------------------------------------------------------
        self.get_logger().info(f'Waiting for {self.num_cameras} OCR subscriber(s)...')
        self._subscriber_timer = self.create_timer(0.5, self._check_subscribers)

    def _check_subscribers(self):
        """
        Timer callback — checks whether all camera topics have at least one
        subscriber (i.e. ocr_node is ready). Once all subscribers are present:
          - Inbound : publishes images immediately.
          - Sorting : creates the /trigger_capture subscription and waits.
        Cancels itself after firing to avoid repeated checks.
        """
        if self._ready:
            return  # Already initialised; nothing to do

        # Count how many camera publishers have at least one active subscriber
        counts = [self.publishers_[i].get_subscription_count() for i in range(self.num_cameras)]
        ready = sum(1 for c in counts if c > 0)
        self.get_logger().info(f'Subscribers ready: {ready}/{self.num_cameras}')

        if ready == self.num_cameras:
            self._ready = True
            self._subscriber_timer.cancel()  # Stop polling — all subscribers connected

            if self.scan_mode == 'inbound':
                # Inbound: publish straight away, no trigger needed
                self.get_logger().info('Inbound mode — publishing test images now...')
                self.publish_all_images()
            else:
                # Sorting: arm sends a trigger before each pick; wait for it
                self.trigger_sub = self.create_subscription(
                    String, '/trigger_capture', self.on_trigger, RELIABLE_QOS
                )
                self.get_logger().info(
                    'Sorting mode — waiting for /trigger_capture signal...'
                )

    def on_trigger(self, msg):
        """
        Callback for /trigger_capture (sorting mode only).
        Publishes all camera images when the message data equals 'capture'.
        Ignores duplicate triggers while a capture is already in progress.
        """
        if msg.data == 'capture':
            if self._capturing:
                # Defensive guard: reject overlapping triggers
                self.get_logger().warn('Capture already in progress — ignoring trigger')
                return
            self.get_logger().info('Trigger received — publishing test images')
            self._capturing = True
            self.publish_all_images()
            self._capturing = False
            self.get_logger().info('Capture complete — ready for next trigger')

    def _find_image(self, camera_id):
        """
        Searches for a test image file for the given camera ID.
        Tries two naming conventions and two extensions:
          camera_<id>.jpg / .png          (manually placed test images)
          camera_<id>_received.jpg / .png (images saved by a previous capture)
        Returns the first matching file path, or None if not found.
        """
        for suffix in ['', '_received']:
            for ext in ['.jpg', '.png']:
                path = os.path.join(self.image_dir, f'camera_{camera_id}{suffix}{ext}')
                if os.path.exists(path):
                    return path
        return None

    def _check_images(self):
        """
        Validates that a test image exists for every camera before starting.
        Logs all missing files and returns False if any are absent so the node
        can abort gracefully rather than failing mid-publish.
        """
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
        """
        Publishes one CompressedImage message per camera sequentially.
        After all images are sent, publishes a /batch_complete signal carrying:
          '<image_dir>,<overall_start_timestamp>,<total_elapsed>'
        ocr_node uses this signal to know the full batch has arrived.
        """
        overall_start = time.time()
        self.get_logger().info(f'=== Starting test publish — {self.num_cameras} images ===')

        # Publish each camera image one at a time (sequential, matching Pi behaviour)
        for camera_id in range(self.num_cameras):
            image_path = self._find_image(camera_id)
            self._publish_single_image(camera_id, image_path, overall_start)

        total = time.time() - overall_start

        # Notify ocr_node that all images for this batch have been published
        batch_msg = String()
        batch_msg.data = f'{self.image_dir},{overall_start},{total}'
        self.batch_pub.publish(batch_msg)
        self.get_logger().info('Batch complete signal sent')
        self.get_logger().info(f'=== Test publish complete: {total:.3f}s ===')

    def _publish_single_image(self, camera_id, image_path, overall_start):
        """
        Reads one image from disk, JPEG-encodes it, and publishes it as a
        CompressedImage message on /camera_<id>/image_raw/compressed.

        The message header is reused to carry timing metadata:
          frame_id = '<overall_start>,<cam_start>,cam_<id>'
        ocr_node parses this to compute per-camera and end-to-end latencies.
        """
        cam_start = time.time()
        self.get_logger().info(f'Publishing camera_{camera_id} from: {image_path}')

        # Load image from disk using OpenCV
        frame = cv2.imread(image_path)
        if frame is None:
            self.get_logger().error(f'Could not read image: {image_path}')
            return

        # Encode as JPEG at quality 85 to match real Pi camera output
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
        _, buffer = cv2.imencode('.jpg', frame, encode_param)

        # Build the CompressedImage ROS message
        msg = CompressedImage()
        msg.header.stamp = self.get_clock().now().to_msg()
        # Embed timing info in frame_id so ocr_node can calculate latency without
        # a separate message — format: '<overall_start>,<cam_start>,cam_<id>'
        msg.header.frame_id = f'{overall_start},{cam_start},cam_{camera_id}'
        msg.format = "jpeg"
        msg.data = buffer.tobytes()

        self.publishers_[camera_id].publish(msg)

        elapsed = time.time() - cam_start
        self.get_logger().info(f'Camera {camera_id} published in {elapsed:.3f}s')


def main(args=None):
    rclpy.init(args=args)
    node = TestImagePublisher()
    rclpy.spin(node)          # Keep node alive, processing callbacks
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()