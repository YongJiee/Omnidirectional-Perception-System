import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from picamera2 import Picamera2  # Raspberry Pi camera SDK (Pi-only)
import time
import cv2

# ---------------------------------------------------------------------------
# QoS profile — must match ocr_node / test_image_publisher exactly.
# RELIABLE + KEEP_LAST(10) ensures no images are dropped on the
# direct-ethernet link between Pi and WSL.
# ---------------------------------------------------------------------------
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class MultiCameraPublisher(Node):
    """
    Raspberry Pi ROS2 node — drives the 3× IMX708 cameras via the
    Arducam Multi Camera Adapter V2.2 and publishes JPEG-compressed images
    to WSL for OCR processing.

    Camera layout:
      Cam0, Cam2 — 45° angled side-face cameras (Arducam mux, sequential)
      Cam1       — fixed top-down camera
                   Inbound: scans TOP face
                   Sorting: arm holds box underneath, scans BOTTOM face

    Important hardware constraint:
      The Arducam mux shares a single CSI lane — cameras MUST be captured
      sequentially (one at a time). ThreadPoolExecutor for capture is NOT
      possible here; parallelism is only applied on the WSL OCR side.

    Two trigger modes:
      Inbound — listens on /inbound_trigger for a "scan" message
      Sorting — listens on /trigger_capture for a "capture" message
                (sent by database_matcher_node after pos1 signal from arm)
    """

    def __init__(self):
        super().__init__('multi_camera_publisher')

        # ── Parameters ───────────────────────────────────────────────────────
        # num_cameras: number of physical cameras to capture (default 3)
        self.declare_parameter('num_cameras', 3)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value

        # scan_mode: 'inbound' or 'sorting' — determines which trigger topic to subscribe to
        self.declare_parameter('scan_mode', 'inbound')
        self.scan_mode = self.get_parameter('scan_mode').get_parameter_value().string_value

        self.get_logger().info(f'Mode: {self.num_cameras}-camera [{self.scan_mode}]')

        # ── Publishers ───────────────────────────────────────────────────────
        # One CompressedImage publisher per camera, matching the topics
        # that ocr_node subscribes to on the WSL side.
        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', RELIABLE_QOS)
            self.get_logger().info(f'Publisher created: /camera_{i}/image_raw/compressed')

        # Signals ocr_node that all camera images for this batch have been published
        self.batch_pub = self.create_publisher(String, '/batch_complete', RELIABLE_QOS)

        # ── Wait for WSL OCR node to subscribe ───────────────────────────────
        # Blocks until all camera topics have at least one subscriber (ocr_node is ready).
        # Prevents publishing images before ocr_node is listening — images would be dropped.
        self.get_logger().info(f'Waiting for {self.num_cameras} OCR subscriber(s)...')
        self._wait_for_subscribers()
        self.get_logger().info('All subscribers detected!')

        # Guard flag — prevents a second trigger starting a capture while one is running
        self._capturing = False

        # ── Mode routing ─────────────────────────────────────────────────────
        if self.scan_mode == 'inbound':
            # Inbound: operator or conveyor publishes "scan" to /inbound_trigger
            self.trigger_sub = self.create_subscription(
                String, '/inbound_trigger', self._on_inbound_trigger, RELIABLE_QOS)
            self.get_logger().info(
                'Inbound mode — waiting for /inbound_trigger (publish "scan" to start)')

        else:
            # Sorting: database_matcher_node publishes "capture" to /trigger_capture
            # after the robot arm confirms the box is in the camera zone (pos1)
            self._capturing = False
            self.trigger_sub = self.create_subscription(
                String, '/trigger_capture', self._on_sorting_trigger, RELIABLE_QOS)
            self.get_logger().info(
                'Sorting mode — waiting for /trigger_capture signal...')

    # ── Inbound trigger callback ──────────────────────────────────────────────
    def _on_inbound_trigger(self, msg):
        """
        Fires when a message arrives on /inbound_trigger.
        Only acts on the exact string 'scan' (case-insensitive).
        Ignores trigger if a capture is already in progress (_capturing guard).
        """
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

    # ── Sorting trigger callback ──────────────────────────────────────────────
    def _on_sorting_trigger(self, msg):
        """
        Fires when a message arrives on /trigger_capture.
        Only acts on the exact string 'capture'.
        Ignores trigger if a capture is already in progress (_capturing guard).
        """
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

    # ── Wait until WSL subscribers are ready ─────────────────────────────────
    def _wait_for_subscribers(self):
        """
        Polls all camera publisher subscription counts every 0.5s until
        every topic has at least one subscriber. Uses throttle_duration_sec
        to avoid flooding the log while waiting.

        This blocking wait is intentional — there is no point triggering
        a capture before ocr_node is ready to receive the images.
        """
        while True:
            counts = [self.publishers_[i].get_subscription_count()
                      for i in range(self.num_cameras)]
            ready = sum(1 for c in counts if c > 0)
            if ready < self.num_cameras:
                self.get_logger().info(
                    f'Subscribers ready: {ready}/{self.num_cameras}',
                    throttle_duration_sec=1.0)  # Log at most once per second
                time.sleep(0.5)
            else:
                self.get_logger().info(f'Subscribers ready: {ready}/{self.num_cameras}')
                break

    # ── Capture all cameras sequentially ─────────────────────────────────────
    def capture_all_cameras(self):
        """
        Captures images from all cameras one at a time (sequential).

        Sequential capture is a hard requirement of the Arducam mux hardware —
        the adapter shares a single CSI lane so only one camera can be active
        at any moment. Attempting parallel capture would cause hardware errors.

        After all cameras are done, publishes a /batch_complete signal so
        ocr_node knows the full batch has arrived and can begin fusion.

        batch_complete message format: 'none,<overall_start>'
        ('none' placeholder for save_dir — Pi no longer saves images locally)
        """
        self.overall_start = time.time()
        self.get_logger().info(f'=== Starting {self.num_cameras}-camera cycle ===')

        results = []
        for camera_id in range(self.num_cameras):
            success = self.capture_single_camera(camera_id)
            results.append((camera_id, success))

        success_count = sum(1 for _, s in results if s)
        self.get_logger().info(
            f'Captured {success_count}/{self.num_cameras} cameras successfully')

        # Notify ocr_node that all images for this batch have been published
        batch_msg      = String()
        batch_msg.data = f'none,{self.overall_start}'
        self.batch_pub.publish(batch_msg)
        self.get_logger().info('Batch complete signal sent to WSL')

        total = time.time() - self.overall_start
        self.get_logger().info(f'=== Cycle complete: {total:.3f}s ===')

    # ── Capture a single camera ───────────────────────────────────────────────
    def capture_single_camera(self, camera_id):
        """
        Initialises one Picamera2 instance, captures a single still frame,
        JPEG-encodes it, and publishes it as a CompressedImage ROS message.

        Camera settings:
          Resolution:   1280×720 (RGB888 format — grayscale conversion on WSL)
          AfMode 0:     manual focus (no autofocus delay)
          LensPosition: 4.0 diopters — tuned for the fixed working distance
          Focus delay:  0.05s — reduced from 0.2s, saving ~0.6s across 3 cameras
          JPEG quality: 85 — balances image quality vs network transfer size

        Timing metadata in frame_id (parsed by ocr_node):
          '<overall_start>,<cam_start>,<camera_id>'
          overall_start — batch start timestamp (same for all cameras in a cycle)
          cam_start     — per-camera start timestamp (used to compute cam_start_offset)
          camera_id     — camera index (0, 1, or 2)

        Returns True on success, False if the camera raised an exception.
        """
        self.get_logger().info(f'--- Capturing camera {camera_id} ---')
        cam_start = time.time()

        try:
            # Open the camera by index (Arducam mux assigns 0, 1, 2)
            cam    = Picamera2(camera_id)
            config = cam.create_still_configuration(
                main={"size": (1280, 720), "format": "RGB888"})
            cam.configure(config)
            # AfMode 0 = manual focus; LensPosition 4.0 tuned for working distance
            cam.set_controls({"AfMode": 0, "LensPosition": 4.0})

            init_time = time.time() - cam_start

            # Start camera and wait for exposure/gain to stabilise
            t = time.time()
            cam.start()
            time.sleep(0.05)  # 0.05s is sufficient for fixed focus — saves ~0.15s vs 0.2s
            focus_time = time.time() - t

            # Capture single frame as numpy array (RGB888)
            t            = time.time()
            frame        = cam.capture_array()
            capture_time = time.time() - t

            # Release camera immediately — Arducam mux requires this before next camera opens
            cam.stop()
            cam.close()

            # JPEG encode at quality 85 — ocr_node decodes directly to grayscale
            t           = time.time()
            _, buffer   = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            encode_time = time.time() - t

            # Build and publish the CompressedImage ROS message
            t   = time.time()
            msg = CompressedImage()
            msg.header.stamp    = self.get_clock().now().to_msg()
            # Embed timing metadata in frame_id so ocr_node can compute latency
            # without a separate message — format: '<overall_start>,<cam_start>,<camera_id>'
            msg.header.frame_id = f'{self.overall_start},{cam_start},{camera_id}'
            msg.format          = 'jpeg'
            msg.data            = buffer.tobytes()
            self.publishers_[camera_id].publish(msg)
            publish_time = time.time() - t

            total_cam = time.time() - cam_start

            # Per-camera timing breakdown for ≤3s efficiency verification
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
    rclpy.spin(node)   # Keep node alive — waits for inbound or sorting triggers
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()