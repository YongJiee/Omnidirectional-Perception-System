import rclpy
from rclpy.node import Node
from sensor_msgs.msg import CompressedImage
import cv2
import pytesseract
from pyzbar import pyzbar
import re
import time
import numpy as np

class OCRProcessor(Node):
    def __init__(self):
        super().__init__('ocr_processor')
        self.subscription = self.create_subscription(
            CompressedImage,
            '/camera/image_raw/compressed',
            self.process_image, 
            10
        )
        self.processed = False
        self.get_logger().info('OCR Processor - Waiting for compressed image...')

    def extract_clean_text(self, text):
        """Extract only meaningful words from OCR text"""
        words = text.split()
        clean = []
        for word in words:
            cleaned = re.sub(r'[^\w]', '', word)
            if len(cleaned) >= 2 and any(c.isalpha() for c in cleaned):
                clean.append(cleaned)
        return ' '.join(clean)

    def process_image(self, msg):
        if self.processed:
            return
        
        # Get capture timestamp from camera
        try:
            camera_capture_time = float(msg.header.frame_id)
        except:
            camera_capture_time = None
        
        processing_start = time.time()
        
        # Decode compressed image
        decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decode_time = time.time() - decode_start
        
        # Save timing
        save_start = time.time()
        save_path = '/tmp/received_image.jpg'
        cv2.imwrite(save_path, frame)
        save_time = time.time() - save_start
        
        # Barcode timing
        barcode_start = time.time()
        barcodes = pyzbar.decode(frame)
        barcode_time = time.time() - barcode_start
        
        barcode_results = []
        if barcodes:
            for barcode in barcodes:
                data = barcode.data.decode('utf-8')
                barcode_results.append(f'[{barcode.type}] {data}')
        
        # OCR timing with IMPROVED preprocessing (from standalone)
        ocr_start = time.time()
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Downscale if needed (OCR works fine at lower res)
        height, width = gray.shape
        if width > 1280:
            scale = 1280 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
        
        # Use Otsu's thresholding (same as standalone)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Save preprocessed image for debugging
        cv2.imwrite('/tmp/preprocessed_binary.jpg', binary)
        
        # Use same Tesseract config as standalone
        # PSM 11 = Sparse text (better for labels/barcodes)
        # OEM 1 = LSTM only (faster)
        custom_config = r'--oem 1 --psm 11'
        
        raw_text = pytesseract.image_to_string(binary, config=custom_config)
        clean_text = self.extract_clean_text(raw_text)
        
        ocr_time = time.time() - ocr_start
        
        processing_time = time.time() - processing_start
        
        # Calculate true end-to-end if we have camera timestamp
        if camera_capture_time:
            true_overall = time.time() - camera_capture_time
        else:
            true_overall = None
        
        # Display results
        self.get_logger().info('\n=== RESULTS ===')
        if barcode_results:
            self.get_logger().info(f'Barcodes: {", ".join(barcode_results)}')
        else:
            self.get_logger().info('Barcodes: None')
        
        if clean_text:
            self.get_logger().info(f'OCR Text: {clean_text}')
            
            # Smart product detection
            if 'raspberry' in clean_text.lower() and 'pi' in clean_text.lower():
                self.get_logger().info('✓ Product: Raspberry Pi')
            elif 'sephora' in clean_text.lower():
                self.get_logger().info('✓ Product: SEPHORA')
            elif 'lego' in clean_text.lower():
                self.get_logger().info('✓ Product: LEGO')
            
            # Model detection
            model_match = re.search(r'Model [A-Z]|\d+|V\d+', clean_text, re.IGNORECASE)
            if model_match:
                self.get_logger().info(f'✓ Model/Version: {model_match.group()}')
        else:
            self.get_logger().info('OCR Text: None')
        
        # Log timing
        self.get_logger().info('\n=== TIMING BREAKDOWN ===')
        self.get_logger().info(f'JPEG decode:        {decode_time:.3f}s')
        self.get_logger().info(f'Save image:         {save_time:.3f}s')
        self.get_logger().info(f'Barcode detection:  {barcode_time:.3f}s')
        self.get_logger().info(f'OCR processing:     {ocr_time:.3f}s')
        self.get_logger().info(f'Total processing:   {processing_time:.3f}s')
        
        if true_overall:
            self.get_logger().info(f'\n=== END-TO-END (Camera→OCR) ===')
            self.get_logger().info(f'Total time:         {true_overall:.3f}s')
            
            if true_overall < 3.0:
                margin = 3.0 - true_overall
                self.get_logger().info(f'✓ PASS: {margin:.3f}s under requirement')
            else:
                exceed = true_overall - 3.0
                self.get_logger().info(f'✗ FAIL: {exceed:.3f}s over requirement')
        
        self.get_logger().info('====================================')
        
        self.processed = True
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)

if __name__ == '__main__':
    main()