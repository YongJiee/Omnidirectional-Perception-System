import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
from picamera2 import Picamera2
import time
import cv2
import os
from datetime import datetime


class MultiCameraPublisher(Node):
    def __init__(self):
        super().__init__('multi_camera_publisher')

        # ---------------------------------------------------------------
        # Number of cameras — controlled from launch file
        # ros2 run camera_publisher multi_camera_node --ros-args -p num_cameras:=2
        # ---------------------------------------------------------------
        self.declare_parameter('num_cameras', 4)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.get_logger().info(f'Mode: {self.num_cameras}-camera')

        # Create timestamped folder for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.expanduser(f'~/camera_captures/{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Saving images to: {self.save_dir}')

        # Publishers — one per camera (only create what we need)
        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', 10)
            self.get_logger().info(f'Publisher created: /camera_{i}/image_raw/compressed')

        # Batch complete publisher — signals WSL to start OCR
        self.batch_pub = self.create_publisher(String, '/batch_complete', 10)

        self.get_logger().info(f'Waiting for {self.num_cameras} OCR subscriber(s)...')
        self._wait_for_subscribers()

        self.get_logger().info('All subscribers detected! Starting capture...')
        self.capture_all_cameras()

    # ---------------------------------------------------------------
    # Wait until all camera topics have a subscriber (WSL OCR node)
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
    # Capture a single camera with fixed focus
    # ---------------------------------------------------------------
    def capture_single_camera(self, camera_id):
        self.get_logger().info(f'--- Capturing camera {camera_id} ---')
        cam_start = time.time()

        try:
            # Init
            init_start = time.time()
            cam = Picamera2(camera_id)
            config = cam.create_still_configuration(main={"size": (1280, 720)})
            cam.configure(config)
            cam.start()
            init_time = time.time() - init_start

            # Fixed focus (no autofocus delay)
            focus_start = time.time()
            cam.set_controls({"AfMode": 0, "LensPosition": 4.0})
            time.sleep(0.2)
            focus_time = time.time() - focus_start

            # Capture
            capture_start = time.time()
            frame = cam.capture_array()
            cam.stop()
            cam.close()
            capture_time = time.time() - capture_start

            # Convert BGR
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save locally
            save_start = time.time()
            filename = f'camera_{camera_id}.jpg'
            save_path = os.path.join(self.save_dir, filename)
            cv2.imwrite(save_path, frame_bgr)
            save_time = time.time() - save_start
            self.get_logger().info(f'Saved: {save_path}')

            # Encode JPEG
            encode_start = time.time()
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)
            encode_time = time.time() - encode_start

            # Publish
            publish_start = time.time()
            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            # frame_id carries: overall_start, cam_start, camera_id
            msg.header.frame_id = f'{self.overall_start},{cam_start},cam_{camera_id}'
            msg.format = "jpeg"
            msg.data = buffer.tobytes()
            self.publishers_[camera_id].publish(msg)
            publish_time = time.time() - publish_start

            elapsed = time.time() - cam_start

            # Per-camera timing breakdown
            self.get_logger().info(f'=== CAMERA {camera_id} TIMING ===')
            self.get_logger().info(f'  Init:        {init_time:.3f}s')
            self.get_logger().info(f'  Fixed focus: {focus_time:.3f}s')
            self.get_logger().info(f'  Capture:     {capture_time:.3f}s')
            self.get_logger().info(f'  Save local:  {save_time:.3f}s')
            self.get_logger().info(f'  JPEG encode: {encode_time:.3f}s')
            self.get_logger().info(f'  ROS publish: {publish_time:.3f}s')
            self.get_logger().info(f'  Total:       {elapsed:.3f}s')

            return True

        except Exception as e:
            self.get_logger().error(f'Camera {camera_id} failed: {str(e)}')
            return False

    # ---------------------------------------------------------------
    # Capture all cameras sequentially, then signal batch complete
    # ---------------------------------------------------------------
    def capture_all_cameras(self):
        self.overall_start = time.time()
        self.get_logger().info(f'=== Starting {self.num_cameras}-camera cycle ===')

        results = []
        for camera_id in range(self.num_cameras):
            success = self.capture_single_camera(camera_id)
            results.append((camera_id, success))

        # Summary
        success_count = sum(1 for _, s in results if s)
        self.get_logger().info(f'Captured {success_count}/{self.num_cameras} cameras successfully')

        # Signal WSL that all images have been published
        batch_msg = String()
        batch_msg.data = f'{self.save_dir},{self.overall_start}'
        self.batch_pub.publish(batch_msg)
        self.get_logger().info('Batch complete signal sent to WSL')

        # Overall cycle timing
        total = time.time() - self.overall_start
        self.get_logger().info(f'=== Cycle complete: {total:.3f}s ===')
        self.get_logger().info(f'All images saved to: {self.save_dir}')

        time.sleep(1.0)


def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPublisher()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()