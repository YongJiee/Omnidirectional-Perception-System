import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import pytesseract
from pyzbar import pyzbar

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
        self.get_logger().info('OCR Processor - Waiting for ONE image...')

    def process_image(self, msg):
        if self.processed:
            return  # Already processed one image
            
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        save_path = '/tmp/received_image.jpg'
        cv2.imwrite(save_path, frame)
        self.get_logger().info(f'Received image saved to: {save_path}')
        
        # Barcode detection
        barcodes = pyzbar.decode(frame)
        if barcodes:
            for barcode in barcodes:
                data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                self.get_logger().info(f'✓ Barcode [{barcode_type}]: {data}')
        else:
            self.get_logger().info('✗ No barcodes detected')
        
        # OCR processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip():
            self.get_logger().info(f'✓ OCR Text:\n{text.strip()}')
        else:
            self.get_logger().info('✗ No text detected')
        
        self.get_logger().info('=== Processing complete! Shutting down ===')
        self.processed = True
        rclpy.shutdown()

def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)

if __name__ == '__main__':
    main()