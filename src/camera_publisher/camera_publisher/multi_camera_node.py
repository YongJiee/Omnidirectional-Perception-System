import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from picamera2 import Picamera2
import time
import cv2
import os
from datetime import datetime

class MultiCameraPublisher(Node):
    def __init__(self):
        super().__init__('multi_camera_publisher')
        
        self.num_cameras = 4
        
        # Create timestamped folder for this run
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.save_dir = os.path.expanduser(f'~/camera_captures/{timestamp}')
        os.makedirs(self.save_dir, exist_ok=True)
        self.get_logger().info(f'Saving images to: {self.save_dir}')

        self.publishers_ = {}
        for i in range(self.num_cameras):
            self.publishers_[i] = self.create_publisher(
                CompressedImage, f'/camera_{i}/image_raw/compressed', 10)
        
        self.get_logger().info('Multi-camera node ready, starting capture...')
        self.capture_all_cameras()

    def capture_single_camera(self, camera_id):
        self.get_logger().info(f'--- Capturing camera {camera_id} ---')
        cam_start = time.time()
        try:
            cam = Picamera2(camera_id)
            config = cam.create_still_configuration(main={"size": (1280, 720)})
            cam.configure(config)
            cam.start()

            # Autofocus
            cam.set_controls({"AfMode": 2, "AfTrigger": 0})
            time.sleep(2.0)

            # Capture
            frame = cam.capture_array()
            cam.stop()
            cam.close()

            # Convert
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

            # Save to timestamped folder
            filename = f'camera_{camera_id}.jpg'
            save_path = os.path.join(self.save_dir, filename)
            cv2.imwrite(save_path, frame_bgr)
            self.get_logger().info(f'Saved: {save_path}')

            # Encode and publish
            encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 85]
            _, buffer = cv2.imencode('.jpg', frame_bgr, encode_param)

            msg = CompressedImage()
            msg.header.stamp = self.get_clock().now().to_msg()
            msg.header.frame_id = f'camera_{camera_id}'
            msg.format = "jpeg"
            msg.data = buffer.tobytes()
            self.publishers_[camera_id].publish(msg)

            elapsed = time.time() - cam_start
            self.get_logger().info(f'Camera {camera_id} done in {elapsed:.2f}s')
            return True

        except Exception as e:
            self.get_logger().error(f'Camera {camera_id} failed: {str(e)}')
            return False

    def capture_all_cameras(self):
        cycle_start = time.time()
        self.get_logger().info('=== Starting 4-camera cycle ===')

        for camera_id in range(self.num_cameras):
            self.capture_single_camera(camera_id)

        total = time.time() - cycle_start
        self.get_logger().info(f'=== Cycle complete: {total:.2f}s ===')
        self.get_logger().info(f'All images saved to: {self.save_dir}')
        time.sleep(1.0)

def main(args=None):
    rclpy.init(args=args)
    node = MultiCameraPublisher()
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()