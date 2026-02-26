import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys
import os
import re

sys.path.append(os.path.dirname(__file__))
from smart_match3_vF import SmartMatcher


class DatabaseMatcherNode(Node):
    def __init__(self):
        super().__init__('database_matcher')

        self.subscription = self.create_subscription(
            String,
            'ocr_results',
            self.match_callback,
            10
        )

        self.matcher = SmartMatcher(device_id='PI-001', location='Warehouse')
        self.get_logger().info('Database Matcher Node ready')

    def check_tie(self, ocr_text, products):
        """
        Check if top 2 products score too close to each other.
        Returns True if tie detected, False if clear winner.
        """
        scores = []
        for product in products:
            _, _, score, _ = self.matcher._enhanced_fuzzy_match(ocr_text, None, [product])
            scores.append((product, score))

        # Sort highest score first
        scores.sort(key=lambda x: x[1], reverse=True)

        # Need at least 2 products to have a tie
        if len(scores) >= 2:
            top_score = scores[0][1]
            second_score = scores[1][1]

            # If both scores are meaningful AND too close
            if top_score > 0 and abs(top_score - second_score) < 10:
                self.get_logger().warn(
                    f'TIE DETECTED — '
                    f'{scores[0][0][1]} ({scores[0][1]:.1f}) vs '
                    f'{scores[1][0][1]} ({scores[1][1]:.1f}) — '
                    f'Manual check required'
                )
                return True

        return False

    def match_callback(self, msg):
        data = json.loads(msg.data)

        ocr_text = data.get('ocr_text', '')
        raw_barcode = data.get('barcode', None)

        # Strip ROS format e.g. '[QRCODE] 1234' -> '1234'
        if raw_barcode:
            match = re.search(r'\] (.+)$', raw_barcode)
            barcode = match.group(1) if match else raw_barcode
        else:
            barcode = None

        self.get_logger().info(f'Received — OCR: {ocr_text[:40]} | Barcode: {barcode}')

        # If no barcode, check for tie first
        if not barcode:
            products = self.matcher.db.get_all_products()

            if self.check_tie(ocr_text, products):
                self.get_logger().warn(
                    'Ambiguous match — barcode required for verification'
                )
                return  # Refuse to match, flag for manual check

        # Normal matching — barcode found OR clear OCR winner
        result = self.matcher.match_and_save(
            ocr_text=ocr_text,
            barcode=barcode
        )

        if result and result['matched']:
            self.get_logger().info(
                f'MATCHED: {result["product"][1]} '
                f'({result["accuracy"]:.1f}%)'
            )
        else:
            self.get_logger().warn('NO MATCH FOUND')


def main(args=None):
    rclpy.init(args=args)
    node = DatabaseMatcherNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()