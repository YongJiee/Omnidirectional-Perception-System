import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from picamera2 import Picamera2
import time
import cv2

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class MultiCameraPublisher(Node):
    def __init__(self):
        super().__init__('multi_camera_publisher')

        # ── Parameters ───────────────────────────────────────────────
        self.declare_parameter('num_cameras', 3)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value

        self.declare_parameter('scan_mode', 'inbound')
        self.scan_mode = self.get_parameter('scan_mode').get_parameter_value().string_value

        self.get_logger().info(f'Mode: {self.num_cameras}-camera [{self.scan_mode}]')

        # ── Publishers ───────────────────────────────────────────────
        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', RELIABLE_QOS)
            self.get_logger().info(f'Publisher created: /camera_{i}/image_raw/compressed')

        self.batch_pub = self.create_publisher(String, '/batch_complete', RELIABLE_QOS)

        # ── Wait for WSL OCR node to subscribe ───────────────────────
        self.get_logger().info(f'Waiting for {self.num_cameras} OCR subscriber(s)...')
        self._wait_for_subscribers()
        self.get_logger().info('All subscribers detected!')

        # ── Guard to prevent overlapping captures ────────────────────
        self._capturing = False

        # ── Mode routing ─────────────────────────────────────────────
        if self.scan_mode == 'inbound':
            # Subscribe to /inbound_trigger — wait for "scan" message each time
            self.trigger_sub = self.create_subscription(
                String, '/inbound_trigger', self._on_inbound_trigger, RELIABLE_QOS)
            self.get_logger().info(
                'Inbound mode — waiting for /inbound_trigger (publish "scan" to start)')

        else:
            # Sorting — subscribe to /trigger_capture — wait for "capture" message
            self._capturing = False
            self.trigger_sub = self.create_subscription(
                String, '/trigger_capture', self._on_sorting_trigger, RELIABLE_QOS)
            self.get_logger().info(
                'Sorting mode — waiting for /trigger_capture signal...')

    # ── Inbound trigger callback ─────────────────────────────────────
    def _on_inbound_trigger(self, msg):
        if msg.data.strip().lower() != 'scan':
            return
        if self._capturing:
            self.get_logger().warn('Capture already in progress — ignoring trigger')
            return
        self.get_logger().info('[INBOUND] Trigger received — starting capture')
        self._capturing = True
        self.capture_all_cameras()
        self._capturing = False
        self.get_logger().info('[INBOUND] Capture complete — ready for next trigger')

    # ── Sorting trigger callback ─────────────────────────────────────
    def _on_sorting_trigger(self, msg):
        if msg.data != 'capture':
            return
        if self._capturing:
            self.get_logger().warn('Capture already in progress — ignoring trigger')
            return
        self.get_logger().info('[SORTING] Trigger received — starting capture')
        self._capturing = True
        self.capture_all_cameras()
        self._capturing = False
        self.get_logger().info('[SORTING] Capture complete — ready for next trigger')

    # ── Wait until WSL subscribers are ready ─────────────────────────
    def _wait_for_subscribers(self):
        while True:
            counts = [self.publishers_[i].get_subscription_count()
                      for i in range(self.num_cameras)]
            ready = sum(1 for c in counts if c > 0)
            if ready < self.num_cameras:
                self.get_logger().info(
                    f'Subscribers ready: {ready}/{self.num_cameras}',
                    throttle_duration_sec=1.0)
                time.sleep(0.5)
            else:
                self.get_logger().info(f'Subscribers ready: {ready}/{self.num_cameras}')
                break

    # ── Capture all cameras sequentially ─────────────────────────────
    def capture_all_cameras(self):
        self.overall_start = time.time()
        self.get_logger().info(f'=== Starting {self.num_cameras}-camera cycle ===')

        results = []
        for camera_id in range(self.num_cameras):
            success = self.capture_single_camera(camera_id)
            results.append((camera_id, success))

        success_count = sum(1 for _, s in results if s)
        self.get_logger().info(
            f'Captured {success_count}/{self.num_cameras} cameras successfully')

        # Signal WSL that all images have been published
        batch_msg = String()
        batch_msg.data = f'none,{self.overall_start}'
        self.batch_pub.publish(batch_msg)
        self.get_logger().info('Batch complete signal sent to WSL')

        total = time.time() - self.overall_start
        self.get_logger().info(f'=== Cycle complete: {total:.3f}s ===')

    # ── Capture a single camera ───────────────────────────────────────
    def capture_single_camera(self, camera_id):
        self.get_logger().info(f'--- Capturing camera {camera_id} ---')
        cam_start = time.time()

        try:
            cam = Picamera2(camera_id)
            config = cam.create_still_configuration(
                main={"size": (1280, 720), "format": "RGB888"})
            cam.configure(config)
            cam.set_controls({"AfMode": 0, "LensPosition": 4.0})

            init_time = time.time() - cam_start

            t = time.time()
            cam.start()
            time.sleep(0.05)  # Reduced focus delay — 0.05s is sufficient for fixed focus
            focus_time = time.time() - t

            t = time.time()
            frame = cam.capture_array()
            capture_time = time.time() - t

            cam.stop()
            cam.close()

            # Encode to JPEG
            t = time.time()
            _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            encode_time = time.time() - t

            # Publish to ROS
            t = time.time()
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            # frame_id carries: overall_start | cam_start | camera_id
            msg.header.frame_id = f'{self.overall_start},{cam_start},{camera_id}'
            msg.format = 'jpeg'
            msg.data = buffer.tobytes()
            self.publishers_[camera_id].publish(msg)
            publish_time = time.time() - t

            total_cam = time.time() - cam_start

            self.get_logger().info(f'=== CAMERA {camera_id} TIMING ===')
            self.get_logger().info(f'  Init:        {init_time:.3f}s')
            self.get_logger().info(f'  Fixed focus: {focus_time:.3f}s')
            self.get_logger().info(f'  Capture:     {capture_time:.3f}s')
            self.get_logger().info(f'  JPEG encode: {encode_time:.3f}s')
            self.get_logger().info(f'  ROS publish: {publish_time:.3f}s')
            self.get_logger().info(f'  Total:       {total_cam:.3f}s')

            return True

        except Exception as e:
            self.get_logger().error(f'Camera {camera_id} failed: {str(e)}')
            return False


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPublisher()
    rclpy.spin(node)   # Stay alive — wait for triggers in both modes
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()