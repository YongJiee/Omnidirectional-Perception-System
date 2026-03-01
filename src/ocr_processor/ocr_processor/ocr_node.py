import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import pytesseract
from pyzbar import pyzbar
import re
import time
import numpy as np
import json


class OCRProcessor(Node):
    def __init__(self):
        super().__init__('ocr_node')

        # ---------------------------------------------------------------
        # Mode selection — default 1 (single), set 4 for multi-camera
        # ros2 run your_pkg ocr_processor --ros-args -p num_cameras:=4
        # ---------------------------------------------------------------
        self.declare_parameter('num_cameras', 1)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.get_logger().info(f'Mode: {self.num_cameras}-camera')

        # Buffer to collect results from all cameras before fusing
        self.camera_results = {}
        self.overall_start = None
        self.batch_save_dir = None
        self.batch_already_processed = False  # prevents duplicate fuse on batch_complete

        # Publisher — fused result after all cameras processed
        self.result_publisher = self.create_publisher(String, '/ocr_results', 10)

        # ---------------------------------------------------------------
        # Subscribe to correct topics based on mode
        # Single: /camera/image_raw/compressed
        # Multi:  /camera_0..N/image_raw/compressed
        # ---------------------------------------------------------------
        self.subscriptions_ = []

        if self.num_cameras == 1:
            # Single camera mode — matches original single camera publisher topic
            sub = self.create_subscription(
                CompressedImage,
                '/camera/image_raw/compressed',
                lambda msg: self.on_image_received(msg, camera_id=0),
                10
            )
            self.subscriptions_.append(sub)
            self.get_logger().info('Subscribed to: /camera/image_raw/compressed')
        else:
            # Multi camera mode — subscribe to each camera topic
            for i in range(self.num_cameras):
                topic = f'/camera_{i}/image_raw/compressed'
                sub = self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, cam_id=i: self.on_image_received(msg, cam_id),
                    10
                )
                self.subscriptions_.append(sub)
                self.get_logger().info(f'Subscribed to: {topic}')

            # Batch complete signal only needed in multi-camera mode
            self.batch_sub = self.create_subscription(
                String, '/batch_complete', self.on_batch_complete, 10)
            self.get_logger().info('Subscribed to: /batch_complete')

        self.get_logger().info(f'OCR Processor ready — waiting for {self.num_cameras} image(s)...')

    # ---------------------------------------------------------------
    # Called when each camera image arrives
    # ---------------------------------------------------------------
    def on_image_received(self, msg, camera_id):
        if camera_id in self.camera_results:
            self.get_logger().warn(f'Camera {camera_id} already processed, skipping duplicate')
            return

        self.get_logger().info(f'Image received from camera {camera_id}')

        # Parse frame_id: overall_start, cam_start, cam_id
        try:
            parts = msg.header.frame_id.split(',')
            if self.overall_start is None:
                self.overall_start = float(parts[0])
            cam_start = float(parts[1]) if len(parts) > 1 else self.overall_start
        except Exception:
            cam_start = time.time()
            if self.overall_start is None:
                self.overall_start = cam_start

        # Process this image
        result = self.process_single_image(msg, camera_id, cam_start)
        self.camera_results[camera_id] = result

        self.get_logger().info(
            f'Camera {camera_id} processed — '
            f'{len(self.camera_results)}/{self.num_cameras} done'
        )

        # Single camera — fuse immediately after first image
        if self.num_cameras == 1 and len(self.camera_results) == 1:
            self.fuse_and_publish()

        # Multi camera — fuse if all images received before batch_complete arrives
        elif self.num_cameras > 1 and len(self.camera_results) == self.num_cameras:
            self.get_logger().info('All camera images received — fusing now')
            self.fuse_and_publish()

    # ---------------------------------------------------------------
    # Called when Pi signals all cameras are done (multi-camera only)
    # ---------------------------------------------------------------
    def on_batch_complete(self, msg):
        # Already fused in on_image_received — ignore this signal
        if self.batch_already_processed:
            self.batch_already_processed = False
            return

        self.get_logger().info('Batch complete signal received from Pi')

        try:
            parts = msg.data.split(',')
            self.batch_save_dir = parts[0]
            if self.overall_start is None and len(parts) > 1:
                self.overall_start = float(parts[1])
        except Exception:
            pass

        # Wait briefly in case last image is still in transit
        time.sleep(0.3)

        received = len(self.camera_results)
        self.get_logger().info(
            f'Batch complete: {received}/{self.num_cameras} images received'
        )

        if received == 0:
            self.get_logger().error('No images received — check Pi publisher')
            return

        # Only fuse here if not already fused in on_image_received
        if received < self.num_cameras:
            self.get_logger().warn(
                f'Only {received}/{self.num_cameras} cameras received — fusing partial results'
            )
            self.fuse_and_publish()

    # ---------------------------------------------------------------
    # Process a single camera image — OCR + barcode
    # ---------------------------------------------------------------
    def process_single_image(self, msg, camera_id, cam_start):
        processing_start = time.time()

        # Decode
        decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decode_time = time.time() - decode_start

        # Save debug image
        cv2.imwrite(f'/tmp/camera_{camera_id}_received.jpg', frame)

        # Grayscale + downscale if needed
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if width > 1280:
            scale = 1280 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Barcode detection
        barcode_start = time.time()
        barcodes = pyzbar.decode(gray)
        if not barcodes:
            barcodes = pyzbar.decode(frame)
        if not barcodes:
            for rotate_flag in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                rotated = cv2.rotate(gray, rotate_flag)
                barcodes = pyzbar.decode(rotated)
                if barcodes:
                    self.get_logger().info(f'Cam {camera_id}: barcode found after rotation')
                    break
        barcode_time = time.time() - barcode_start

        barcode_results = []
        if barcodes:
            for b in barcodes:
                barcode_results.append({
                    'type': b.type,
                    'data': b.data.decode('utf-8')
                })

        # OCR
        ocr_start = time.time()
        cv2.imwrite(f'/tmp/camera_{camera_id}_gray.jpg', gray)
        clean_text = self.extract_clean_text(gray, psm=11, min_conf=75)

        if not clean_text:
            for rotate_flag in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                rotated = cv2.rotate(gray, rotate_flag)
                clean_text = self.extract_clean_text(rotated, psm=11, min_conf=75)
                if clean_text:
                    self.get_logger().info(f'Cam {camera_id}: OCR found after rotation')
                    break

        if clean_text:
            clean_text = self.filter_barcode_text(clean_text)
        ocr_time = time.time() - ocr_start

        processing_time = time.time() - processing_start

        # Per-camera log
        self.get_logger().info(f'=== CAMERA {camera_id} OCR RESULTS ===')
        self.get_logger().info(f'  Barcodes: {barcode_results if barcode_results else "None"}')
        self.get_logger().info(f'  OCR Text: {clean_text if clean_text else "None"}')
        self.get_logger().info(f'  Decode:   {decode_time:.3f}s')
        self.get_logger().info(f'  Barcode:  {barcode_time:.3f}s')
        self.get_logger().info(f'  OCR:      {ocr_time:.3f}s')
        self.get_logger().info(f'  Total:    {processing_time:.3f}s')

        return {
            'camera_id': camera_id,
            'ocr_text': clean_text,
            'barcodes': barcode_results,
            'timing': {
                'decode': decode_time,
                'barcode': barcode_time,
                'ocr': ocr_time,
                'total': processing_time
            }
        }

    # ---------------------------------------------------------------
    # Fuse results from all cameras into one result and publish
    # ---------------------------------------------------------------
    def fuse_and_publish(self):
        fuse_start = time.time()

        # Combine all OCR text across cameras
        all_ocr_parts = []
        for cam_id, result in self.camera_results.items():
            if result['ocr_text']:
                all_ocr_parts.append(result['ocr_text'])

        # Take first barcode found across any camera
        fused_barcode = None
        for cam_id, result in self.camera_results.items():
            if result['barcodes']:
                fused_barcode = result['barcodes'][0]
                self.get_logger().info(f'Barcode from camera {cam_id}: {fused_barcode}')
                break

        fused_ocr = ' '.join(all_ocr_parts) if all_ocr_parts else None
        fuse_time = time.time() - fuse_start
        end_to_end = time.time() - self.overall_start if self.overall_start else None

        # Summary log
        self.get_logger().info('=== FUSED RESULT ===')
        self.get_logger().info(f'  Mode:                   {self.num_cameras}-camera')
        self.get_logger().info(f'  OCR Text (all cameras): {fused_ocr}')
        self.get_logger().info(f'  Barcode:                {fused_barcode}')
        self.get_logger().info(f'  Cameras with OCR:       {len(all_ocr_parts)}/{len(self.camera_results)}')
        self.get_logger().info(f'  Fuse time:              {fuse_time:.3f}s')
        if end_to_end:
            self.get_logger().info(f'  END-TO-END TIME:        {end_to_end:.3f}s')
        self.get_logger().info('====================')

        # Publish fused result
        result_msg = json.dumps({
            'ocr_text': fused_ocr,
            'barcode': fused_barcode,
            'mode': f'{self.num_cameras}-camera',
            'cameras_received': len(self.camera_results),
            'per_camera': self.camera_results,
            'overall_start': self.overall_start,
            'end_to_end_time': end_to_end,
            'save_dir': self.batch_save_dir
        })
        self.result_publisher.publish(String(data=result_msg))
        self.get_logger().info('Fused OCR result published to /ocr_results')

        self.batch_already_processed = True  # tell on_batch_complete to skip

        # Reset buffer for next batch
        self.camera_results = {}
        self.overall_start = None
        self.batch_save_dir = None

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    def extract_clean_text(self, image, psm=11, min_conf=75):
        data = pytesseract.image_to_data(
            image,
            config=f'--oem 1 --psm {psm}',
            output_type=pytesseract.Output.DICT
        )
        clean = []
        for word, conf in zip(data['text'], data['conf']):
            word = word.strip()
            if int(conf) > min_conf and len(word) >= 3:
                clean.append(word)
        return ' '.join(clean)

    def filter_barcode_text(self, text):
        text = re.sub(r'\b[A-Za-z]+-[A-Za-z]+-\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()