import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import pytesseract
from pyzbar import pyzbar
import re
import time
import numpy as np
from std_msgs.msg import String
import json


class OCRProcessor(Node):
    def __init__(self):
        super().__init__('ocr_processor')
        self.result_publisher = self.create_publisher(String, 'ocr_results', 10)
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.process_image,
            10
        )
        self.processed = False
        self.get_logger().info('OCR Processor - Waiting for compressed image...')

    def extract_clean_text(self, image, psm=11, min_conf=75):
        """Extract only high-confidence words from OCR using per-word confidence scores"""
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
        """Remove barcode number patterns from OCR text e.g. ABC-abc-1234"""
        text = re.sub(r'\b[A-Za-z]+-[A-Za-z]+-\d+\b', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def process_image(self, msg):
        if self.processed:
            return

        # Parse both overall_start and capture_start from frame_id
        camera_overall_start = None
        camera_capture_start = None
        try:
            parts = msg.header.frame_id.split(',')
            camera_overall_start = float(parts[0])
            camera_capture_start = float(parts[1]) if len(parts) > 1 else camera_overall_start
        except:
            pass

        processing_start = time.time()

        # Decode compressed image
        decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decode_time = time.time() - decode_start

        # Save raw image for debugging
        save_start = time.time()
        cv2.imwrite('/tmp/received_image.jpg', frame)
        save_time = time.time() - save_start

        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Downscale if needed
        height, width = gray.shape
        if width > 1280:
            scale = 1280 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale)

        # Barcode detection — try gray first, fall back to color, then rotations
        barcode_start = time.time()
        barcodes = pyzbar.decode(gray)
        if not barcodes:
            barcodes = pyzbar.decode(frame)
        if not barcodes:
            for rotate_flag in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                rotated = cv2.rotate(gray, rotate_flag)
                barcodes = pyzbar.decode(rotated)
                if barcodes:
                    self.get_logger().info(f'Barcode found after rotation: {rotate_flag}')
                    break
        barcode_time = time.time() - barcode_start

        barcode_results = []
        if barcodes:
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                barcode_results.append(f'[{barcode.type}] {barcode_data}')

        # OCR — use grayscale directly, no thresholding needed for clean labels
        ocr_start = time.time()

        # Save debug image
        cv2.imwrite('/tmp/preprocessed_binary.jpg', gray)

        # Extract text directly from grayscale
        clean_text = self.extract_clean_text(gray, psm=11, min_conf=75)

        # Try rotations if no text found
        if not clean_text:
            for rotate_flag in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                rotated = cv2.rotate(gray, rotate_flag)
                clean_text = self.extract_clean_text(rotated, psm=11, min_conf=75)
                if clean_text:
                    self.get_logger().info(f'OCR text found after rotation: {rotate_flag}')
                    break

        # Filter out barcode number patterns
        if clean_text:
            clean_text = self.filter_barcode_text(clean_text)

        ocr_time = time.time() - ocr_start
        processing_time = time.time() - processing_start

        # Display results
        self.get_logger().info('\n=== RESULTS ===')
        if barcode_results:
            self.get_logger().info(f'Barcodes: {", ".join(barcode_results)}')
        else:
            self.get_logger().info('Barcodes: None')

        if clean_text:
            self.get_logger().info(f'OCR Text: {clean_text}')
        else:
            self.get_logger().info('OCR Text: None')

        # Log timing
        self.get_logger().info('\n=== TIMING BREAKDOWN ===')
        self.get_logger().info(f'JPEG decode:        {decode_time:.3f}s')
        self.get_logger().info(f'Save image:         {save_time:.3f}s')
        self.get_logger().info(f'Barcode detection:  {barcode_time:.3f}s')
        self.get_logger().info(f'OCR processing:     {ocr_time:.3f}s')
        self.get_logger().info(f'Total processing:   {processing_time:.3f}s')
        self.get_logger().info('====================================')

        result_msg = json.dumps({
            'ocr_text': clean_text,
            'barcode': barcode_results,
            'camera_overall_start': camera_overall_start,
            'camera_capture_start': camera_capture_start
        })
        self.result_publisher.publish(String(data=result_msg))
        self.get_logger().info('Published OCR result to /ocr_results')

        self.processed = True
        rclpy.shutdown()


def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)


if __name__ == '__main__':
    main()