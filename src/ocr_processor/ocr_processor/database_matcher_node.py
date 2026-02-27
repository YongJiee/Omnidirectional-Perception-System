import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys
import os
import re
import time

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

        scores.sort(key=lambda x: x[1], reverse=True)

        if len(scores) >= 2:
            top_score = scores[0][1]
            second_score = scores[1][1]

            if top_score > 0 and abs(top_score - second_score) < 10:
                self.get_logger().warn(
                    f'TIE DETECTED — '
                    f'{scores[0][0][1]} ({scores[0][1]:.1f}) vs '
                    f'{scores[1][0][1]} ({scores[1][1]:.1f}) — '
                    f'Manual check required'
                )
                return True

        return False

    def is_meaningful_ocr(self, ocr_text):
        """
        Check if OCR text contains any recognisable product-related words.
        Prevents garbage/short text from triggering false matches.
        """
        if len(ocr_text.strip()) < 4:
            return False

        products = self.matcher.db.get_all_products()
        ocr_lower = ocr_text.lower()

        for product in products:
            brand = product[2].lower() if product[2] else ''
            product_name = product[1].lower() if product[1] else ''
            keywords = product[4].lower() if product[4] else ''

            all_terms = (brand + ' ' + product_name + ' ' + keywords).split()
            for term in all_terms:
                if len(term) >= 3 and term in ocr_lower:
                    return True

        return False

    def get_ocr_details(self, ocr_text):
        """Run OCR-only fuzzy match and return detail scores."""
        _, _, _, details = self.matcher._enhanced_fuzzy_match(
            ocr_text, None, self.matcher.db.get_all_products()
        )
        return details

    def log_timing(self, callback_start, camera_overall_start, camera_capture_start):
        """Log full timing breakdown including camera init+focus."""
        matcher_time = time.time() - callback_start
        now = time.time()

        self.get_logger().info('\n=== MATCHER TIMING ===')
        self.get_logger().info(f'  Matcher processing: {matcher_time:.3f}s')

        if camera_overall_start and camera_capture_start:
            init_focus_time = camera_capture_start - camera_overall_start
            capture_to_db_time = now - camera_capture_start
            full_end_to_end = now - camera_overall_start

            self.get_logger().info(f'  ── Camera Breakdown ──')
            self.get_logger().info(f'  Init + Focus:       {init_focus_time:.3f}s')
            self.get_logger().info(f'  Capture → Database: {capture_to_db_time:.3f}s')
            self.get_logger().info(f'  ──────────────────────')
            self.get_logger().info(f'  FULL end-to-end:    {full_end_to_end:.3f}s  (Init → OCR → Database)')
            if full_end_to_end < 3.0:
                self.get_logger().info(f'  ✓ PASS: {3.0 - full_end_to_end:.3f}s under requirement')
            else:
                self.get_logger().warn(f'  ✗ FAIL: {full_end_to_end - 3.0:.3f}s over requirement')

        elif camera_overall_start:
            full_end_to_end = now - camera_overall_start
            self.get_logger().info(f'  FULL end-to-end:    {full_end_to_end:.3f}s  (Init → OCR → Database)')
            if full_end_to_end < 3.0:
                self.get_logger().info(f'  ✓ PASS: {3.0 - full_end_to_end:.3f}s under requirement')
            else:
                self.get_logger().warn(f'  ✗ FAIL: {full_end_to_end - 3.0:.3f}s over requirement')

        self.get_logger().info('==================================')

    def match_callback(self, msg):
        callback_start = time.time()

        data = json.loads(msg.data)

        ocr_text = data.get('ocr_text', '')
        camera_overall_start = data.get('camera_overall_start', None)
        camera_capture_start = data.get('camera_capture_start', None)

        # Backwards compatibility with old frame_id key
        if camera_overall_start is None:
            camera_overall_start = data.get('camera_capture_time', None)
            camera_capture_start = camera_overall_start

        # Handle barcode as list (multiple reads) or single string
        raw_barcodes = data.get('barcode', [])
        if isinstance(raw_barcodes, str):
            raw_barcodes = [raw_barcodes]
        elif raw_barcodes is None:
            raw_barcodes = []

        # Strip ROS format e.g. '[CODE128] ABC-abc-1234' -> 'ABC-abc-1234'
        barcodes = []
        for rb in raw_barcodes:
            match = re.search(r'\] (.+)$', rb)
            barcodes.append(match.group(1) if match else rb)

        self.get_logger().info(f'Received — OCR: {ocr_text[:40]} | Barcodes: {barcodes}')

        # Guard: if no barcode and OCR is garbage, skip matching entirely
        if not barcodes and not self.is_meaningful_ocr(ocr_text):
            self.get_logger().warn(f'OCR text unrecognisable: "{ocr_text}" — skipping match')
            self.get_logger().info('\n=== DATABASE MATCHING RESULTS ===')
            self.get_logger().warn('✗ NO MATCH FOUND — insufficient OCR data')
            self.log_timing(callback_start, camera_overall_start, camera_capture_start)
            return

        # If no barcode, check for tie first
        if not barcodes:
            products = self.matcher.db.get_all_products()
            if self.check_tie(ocr_text, products):
                self.get_logger().warn('Ambiguous match — barcode required for verification')
                self.log_timing(callback_start, camera_overall_start, camera_capture_start)
                return

        # Try each barcode until one matches
        result = None
        barcode_used = None
        for barcode in barcodes:
            try:
                result = self.matcher.match_and_save(ocr_text=ocr_text, barcode=barcode)
            except Exception as e:
                self.get_logger().error(f'Exception in match_and_save: {e}')
            if result and result['matched']:
                barcode_used = barcode
                break

        # Fall back to OCR-only match if all barcodes failed
        if not result or not result['matched']:
            result = self.matcher.match_and_save(ocr_text=ocr_text, barcode=None)

        # Display results
        self.get_logger().info('\n=== DATABASE MATCHING RESULTS ===')
        if result and result['matched']:
            self.get_logger().info(f'✓ MATCHED: {result["product"][2]} {result["product"][1]}')
            self.get_logger().info(f'  Overall Accuracy:   {result["accuracy"]:.1f}%')
            self.get_logger().info(f'  Confidence Level:   {result["confidence"]}')
            self.get_logger().info(f'  Verified:           {"Yes" if result["verified"] else "No"}')
            self.get_logger().info(f'  Score Breakdown:')

            if barcode_used:
                self.get_logger().info(f'    Barcode match:      100.0% (exact) [{barcode_used}]')
            else:
                self.get_logger().info(f'    Barcode match:      None')

            details = self.get_ocr_details(ocr_text)
            self.get_logger().info(f'    OCR match:          {result["accuracy"]:.1f}%')
            self.get_logger().info(f'      Brand similarity:   {details["brand_score"]:.1f}%')
            self.get_logger().info(f'      Product similarity: {details["product_score"]:.1f}%')
            self.get_logger().info(f'      Keyword similarity: {details["keyword_score"]:.1f}%')
        else:
            self.get_logger().warn('✗ NO MATCH FOUND')

        self.log_timing(callback_start, camera_overall_start, camera_capture_start)


def main(args=None):
    rclpy.init(args=args)
    node = DatabaseMatcherNode()
    rclpy.spin(node)


if __name__ == '__main__':
    main()