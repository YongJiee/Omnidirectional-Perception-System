import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pytesseract
from pyzbar import pyzbar
import re
import time

class OCRProcessor(Node):
    def __init__(self):
        super().__init__('ocr_processor')
        self.subscription = self.create_subscription(
            Image, 
            '/camera/image_raw', 
            self.process_image, 
            10
        )
        self.bridge = CvBridge()
        self.processed = False
        self.start_time = time.time()
        self.get_logger().info('OCR Processor - Waiting for image...')

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
        
        # Image receive timing
        receive_time = time.time() - self.start_time
        processing_start = time.time()
        
        # Convert timing
        convert_start = time.time()
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        convert_time = time.time() - convert_start
        
        # Save timing
        save_start = time.time()
        save_path = '/tmp/received_image.jpg'
        cv2.imwrite(save_path, frame)
        save_time = time.time() - save_start
        
        # Barcode timing
        barcode_start = time.time()
        barcodes = pyzbar.decode(frame)
        barcode_time = time.time() - barcode_start
        
        barcode_data = None
        if barcodes:
            for barcode in barcodes:
                barcode_data = barcode.data.decode('utf-8')
                self.get_logger().info(f'Barcode: [{barcode.type}] {barcode_data}')
        
        # OCR timing
        ocr_start = time.time()
        
        # Preprocessing
        preprocess_start = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        inverted = 255 - gray
        preprocess_time = time.time() - preprocess_start
        
        # OCR execution
        tesseract_start = time.time()
        raw_text = pytesseract.image_to_string(inverted, config='--oem 3 --psm 6')
        tesseract_time = time.time() - tesseract_start
        
        # Text processing
        textprocess_start = time.time()
        clean_text = self.extract_clean_text(raw_text)
        textprocess_time = time.time() - textprocess_start
        
        ocr_time = time.time() - ocr_start
        
        # Display results
        if clean_text:
            self.get_logger().info(f'OCR Text: {clean_text}')
            
            if 'raspberry' in clean_text.lower() and 'pi' in clean_text.lower():
                self.get_logger().info('✓ Product: Raspberry Pi')
            
            model_match = re.search(r'Model [A-Z]|\d+', clean_text, re.IGNORECASE)
            if model_match:
                self.get_logger().info(f'✓ Model: {model_match.group()}')
        
        # Overall timing
        processing_time = time.time() - processing_start
        overall_time = time.time() - self.start_time
        
        # Log comprehensive timing
        self.get_logger().info('\n=== OCR NODE TIMING ===')
        self.get_logger().info(f'Image receive:      {receive_time:.3f}s')
        self.get_logger().info(f'ROS convert:        {convert_time:.3f}s')
        self.get_logger().info(f'Save image:         {save_time:.3f}s')
        self.get_logger().info(f'Barcode detection:  {barcode_time:.3f}s')
        self.get_logger().info(f'OCR preprocessing:  {preprocess_time:.3f}s')
        self.get_logger().info(f'Tesseract OCR:      {tesseract_time:.3f}s')
        self.get_logger().info(f'Text processing:    {textprocess_time:.3f}s')
        self.get_logger().info(f'Total OCR:          {ocr_time:.3f}s')
        self.get_logger().info(f'Processing only:    {processing_time:.3f}s')
        self.get_logger().info(f'Overall (end-to-end): {overall_time:.3f}s')
        self.get_logger().info('========================')
        
        # Check 3-second requirement
        if overall_time < 3.0:
            margin = 3.0 - overall_time
            self.get_logger().info(f'✓ PASS: Under 3s requirement (margin: +{margin:.3f}s)')
        else:
            exceed = overall_time - 3.0
            self.get_logger().info(f'✗ FAIL: Exceeds 3s requirement (over by: {exceed:.3f}s)')
        
        self.processed = True
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)

if __name__ == '__main__':
    main()