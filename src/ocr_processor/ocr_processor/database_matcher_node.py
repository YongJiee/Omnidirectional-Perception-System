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
            '/ocr_results',   # updated to match new OCR node publisher topic
            self.match_callback,
            10
        )

        self.matcher = SmartMatcher(device_id='PI-001', location='Warehouse')
        self.get_logger().info('Database Matcher Node ready')

    # ---------------------------------------------------------------
    # Parse barcode from new format {'type': ..., 'data': ...}
    # or old format '[CODE128] ABC-123' for backwards compatibility
    # ---------------------------------------------------------------
    def parse_barcodes(self, raw_barcode):
        """
        Accepts:
          - dict:  {'type': 'CODE128', 'data': 'ABC-123'}
          - str:   '[CODE128] ABC-123'
          - list:  [dict, ...] or [str, ...]
          - None
        Returns: list of barcode data strings e.g. ['ABC-123']
        """
        if raw_barcode is None:
            return []

        # Wrap single item in list
        if not isinstance(raw_barcode, list):
            raw_barcode = [raw_barcode]

        barcodes = []
        for rb in raw_barcode:
            if isinstance(rb, dict):
                # New format from updated OCR node
                data = rb.get('data', '').strip()
                if data:
                    barcodes.append(data)
            elif isinstance(rb, str):
                # Old format '[CODE128] ABC-123'
                match = re.search(r'\] (.+)$', rb)
                barcodes.append(match.group(1) if match else rb)

        return barcodes

    # ---------------------------------------------------------------
    # Check if top 2 products score too close (tie detection)
    # ---------------------------------------------------------------
    def check_tie(self, ocr_text, products):
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

    # ---------------------------------------------------------------
    # Guard: check OCR text has recognisable product words
    # ---------------------------------------------------------------
    def is_meaningful_ocr(self, ocr_text):
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
        _, _, _, details = self.matcher._enhanced_fuzzy_match(
            ocr_text, None, self.matcher.db.get_all_products()
        )
        return details

    # ---------------------------------------------------------------
    # Timing log — works with both old and new format
    # ---------------------------------------------------------------
    def log_timing(self, callback_start, overall_start, end_to_end_from_ocr, mode):
        matcher_time = time.time() - callback_start
        now = time.time()

        self.get_logger().info('\n=== MATCHER TIMING ===')
        self.get_logger().info(f'  Mode:               {mode}')
        self.get_logger().info(f'  Matcher processing: {matcher_time:.3f}s')

        if overall_start:
            full_end_to_end = now - overall_start
            self.get_logger().info(f'  FULL end-to-end:    {full_end_to_end:.3f}s  (Pi init → OCR → Database)')
            if full_end_to_end < 3.0:
                self.get_logger().info(f'  ✓ PASS: {3.0 - full_end_to_end:.3f}s under 3s requirement')
            else:
                self.get_logger().warn(f'  ✗ FAIL: {full_end_to_end - 3.0:.3f}s over 3s requirement')

        # Also log OCR node's own end-to-end if available
        if end_to_end_from_ocr:
            self.get_logger().info(f'  OCR node e2e:       {end_to_end_from_ocr:.3f}s')

        self.get_logger().info('==================================')

    # ---------------------------------------------------------------
    # Main callback
    # ---------------------------------------------------------------
    def match_callback(self, msg):
        callback_start = time.time()

        data = json.loads(msg.data)

        ocr_text = data.get('ocr_text', '') or ''
        mode = data.get('mode', '1-camera')
        cameras_received = data.get('cameras_received', 1)
        per_camera = data.get('per_camera', {})

        # Timing — new keys with fallback to old keys
        overall_start = (
            data.get('overall_start') or
            data.get('camera_overall_start') or
            data.get('camera_capture_time')
        )
        end_to_end_from_ocr = data.get('end_to_end_time')

        # Barcode — parse new dict format or old string format
        raw_barcode = data.get('barcode')
        barcodes = self.parse_barcodes(raw_barcode)

        self.get_logger().info(
            f'Received [{mode}] — '
            f'Cameras: {cameras_received} | '
            f'OCR: "{ocr_text[:40]}" | '
            f'Barcodes: {barcodes}'
        )

        # Log per-camera summary if multi-camera
        if per_camera:
            self.get_logger().info('Per-camera OCR summary:')
            for cam_id, cam_result in per_camera.items():
                cam_ocr = cam_result.get('ocr_text', 'None')
                cam_barcodes = cam_result.get('barcodes', [])
                self.get_logger().info(
                    f'  Cam {cam_id}: OCR="{cam_ocr}" | Barcodes={cam_barcodes}'
                )

        # Guard: skip if no barcode and OCR is garbage
        if not barcodes and not self.is_meaningful_ocr(ocr_text):
            self.get_logger().warn(f'OCR text unrecognisable: "{ocr_text}" — skipping match')
            self.get_logger().info('\n=== DATABASE MATCHING RESULTS ===')
            self.get_logger().warn('✗ NO MATCH FOUND — insufficient OCR data')
            self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
            return

        # Tie check if OCR-only
        if not barcodes:
            products = self.matcher.db.get_all_products()
            if self.check_tie(ocr_text, products):
                self.get_logger().warn('Ambiguous match — barcode required for verification')
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
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

        # Fall back to OCR-only if all barcodes failed
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

        self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)


def main(args=None):
    rclpy.init(args=args)
    node = DatabaseMatcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()