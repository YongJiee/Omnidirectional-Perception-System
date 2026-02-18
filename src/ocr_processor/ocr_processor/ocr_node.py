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
        self.get_logger().info('OCR Processor Node Started')

    def process_image(self, msg):
        # Convert ROS Image to OpenCV format
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        
        self.get_logger().info('Processing image...')
        
        # Barcode detection
        barcodes = pyzbar.decode(frame)
        if barcodes:
            for barcode in barcodes:
                data = barcode.data.decode('utf-8')
                barcode_type = barcode.type
                self.get_logger().info(f'Barcode [{barcode_type}]: {data}')
        
        # OCR processing
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        text = pytesseract.image_to_string(gray)
        if text.strip():
            self.get_logger().info(f'OCR Text: {text.strip()}')

def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
