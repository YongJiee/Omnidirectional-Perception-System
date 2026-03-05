import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import json
import sys
import os
import re
import time
from fuzzywuzzy import fuzz

sys.path.append(os.path.dirname(__file__))
from smart_match3_vF import SmartMatcher


class DatabaseMatcherNode(Node):
    def __init__(self):
        super().__init__('database_matcher')

        self.subscription = self.create_subscription(
            String,
            '/ocr_results',
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
        if raw_barcode is None:
            return []
        if not isinstance(raw_barcode, list):
            raw_barcode = [raw_barcode]
        barcodes = []
        for rb in raw_barcode:
            if isinstance(rb, dict):
                data = rb.get('data', '').strip()
                if data:
                    barcodes.append(data)
            elif isinstance(rb, (str, int, float)):
                rb = str(rb)
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

    # ---------------------------------------------------------------
    # Unified result display — always same format regardless of outcome
    # ---------------------------------------------------------------
    def log_results(self, matched, product, accuracy, confidence, verified,
                    barcode_used, barcodes, ocr_text, details, reason=None):
        self.get_logger().info('\n=== DATABASE MATCHING RESULTS ===')

        if matched:
            self.get_logger().info(f'✓ MATCHED: {product[2]} {product[1]}')
            self.get_logger().info(f'  Overall Accuracy:   {accuracy:.1f}%')
            self.get_logger().info(f'  Confidence Level:   {confidence}')
            self.get_logger().info(f'  Verified:           {"Yes" if verified else "No"}')
        else:
            self.get_logger().warn(f'✗ NO MATCH FOUND{" — " + reason if reason else ""}')
            self.get_logger().info(f'  Overall Accuracy:   {accuracy:.1f}%')

        # Always show score breakdown
        self.get_logger().info(f'  Score Breakdown:')

        if barcode_used:
            self.get_logger().info(f'    Barcode match:      100.0% (exact) [{barcode_used}]')
        elif barcodes:
            self.get_logger().info(f'    Barcode match:      0.0% (no exact match) {barcodes}')
        else:
            self.get_logger().info(f'    Barcode match:      0.0% (none provided)')

        if ocr_text.strip():
            self.get_logger().info(f'    OCR match:')
            self.get_logger().info(f'      Brand similarity:   {details.get("brand_score", 0.0):.1f}%')
            self.get_logger().info(f'      Product similarity: {details.get("product_score", 0.0):.1f}%')
            self.get_logger().info(f'      Keyword similarity: {details.get("keyword_score", 0.0):.1f}%')
        else:
            self.get_logger().info(f'    OCR match:          N/A (no OCR text)')

    # ---------------------------------------------------------------
    # Timing log
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

        overall_start = (
            data.get('overall_start') or
            data.get('camera_overall_start') or
            data.get('camera_capture_time')
        )
        end_to_end_from_ocr = data.get('end_to_end_time')

        raw_barcode = data.get('barcode')
        barcodes = self.parse_barcodes(raw_barcode)

        self.get_logger().info(
            f'Received [{mode}] — '
            f'Cameras: {cameras_received} | '
            f'OCR: "{ocr_text[:40]}" | '
            f'Barcodes: {barcodes}'
        )

        # ---------------------------------------------------------------
        # Reject if multiple brands detected in OCR text
        # Applies to both inbound and sorting — mixed products = error
        # ---------------------------------------------------------------
        all_products = self.matcher.db.get_all_products()
        brands_found = list({
            p[2] for p in all_products
            if p[2] and fuzz.partial_ratio(p[2].lower(), ocr_text.lower()) >= 85
        })
        if len(brands_found) > 1:
            self.get_logger().warn(
                f'Multiple brands detected {brands_found} — '
                f'possible multiple products in frame'
            )
            self.log_results(
                matched=False, product=None, accuracy=0.0,
                confidence='NO MATCH', verified=False,
                barcode_used=None, barcodes=barcodes,
                ocr_text=ocr_text, details={},
                reason='multiple products detected in frame'
            )
            self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
            return

        # ---------------------------------------------------------------
        # Reject if multiple barcodes match different products in DB
        # e.g. two barcodes from different boxes scanned together
        # ---------------------------------------------------------------
        if len(barcodes) > 1:
            matched_ids = []
            for barcode in barcodes:
                for p in all_products:
                    if p[6] and p[6] == barcode:
                        matched_ids.append((barcode, p[0], p[1]))  # barcode, id, name
            unique_product_ids = list({m[1] for m in matched_ids})
            if len(unique_product_ids) > 1:
                self.get_logger().warn(
                    f'Multiple barcodes matched different products — '
                    f'{[(m[0], m[2]) for m in matched_ids]} — rejecting'
                )
                self.log_results(
                    matched=False, product=None, accuracy=0.0,
                    confidence='NO MATCH', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details={},
                    reason='multiple barcodes matched different products'
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
                return

        if per_camera:
            self.get_logger().info('Per-camera OCR summary:')
            for cam_id, cam_result in per_camera.items():
                cam_ocr = cam_result.get('ocr_text', 'None')
                cam_barcodes = cam_result.get('barcodes', [])
                self.get_logger().info(f'  Cam {cam_id}: OCR="{cam_ocr}" | Barcodes={cam_barcodes}')

        # Guard: skip if no barcode and OCR is garbage
        if not barcodes and not self.is_meaningful_ocr(ocr_text):
            self.get_logger().warn(f'OCR text unrecognisable: "{ocr_text}"')
            self.log_results(
                matched=False, product=None, accuracy=0.0, confidence='NO MATCH',
                verified=False, barcode_used=None, barcodes=barcodes,
                ocr_text=ocr_text, details={}, reason='insufficient OCR data'
            )
            self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
            return

        # Tie check if OCR-only (no barcode provided at all)
        if not barcodes:
            if self.check_tie(ocr_text, all_products):
                _, _, _, details = self.matcher._enhanced_fuzzy_match(ocr_text, None, all_products)
                self.log_results(
                    matched=False, product=None, accuracy=50.0,
                    confidence='AMBIGUOUS', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason='TIE DETECTED, barcode needed'
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
                return

        # Try each barcode until one exactly matches in DB
        result = None
        barcode_used = None
        for barcode in barcodes:
            try:
                result = self.matcher.match_and_save(ocr_text=ocr_text, barcode=barcode)
            except Exception as e:
                self.get_logger().error(f'Exception in match_and_save: {e}')
            if result and result['matched'] and result.get('barcode_matched'):
                barcode_used = barcode
                break

        # Fall back to OCR-only if no barcode exactly matched
        if not result or not result['matched'] or not result.get('barcode_matched'):
            barcode_used = None
            if self.check_tie(ocr_text, all_products):
                details = (result.get('match_details') or {}) if result else {}
                self.log_results(
                    matched=False, product=None, accuracy=50.0,
                    confidence='AMBIGUOUS', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason='TIE DETECTED, barcode needed'
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
                return
            result = self.matcher.match_and_save(ocr_text=ocr_text, barcode=None)

            # Catch ambiguous product (same name across multiple brands)
            if not result or not result['matched']:
                details = (result.get('match_details') or {}) if result else {}
                self.log_results(
                    matched=False, product=None, accuracy=50.0,
                    confidence='AMBIGUOUS', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason='TIE DETECTED, barcode needed'
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)
                return

        # ---------------------------------------------------------------
        # Final result — reject if accuracy below 95% threshold
        # ---------------------------------------------------------------
        details = (result.get('match_details') or {}) if result else {}

        if result and result['matched']:
            if result['accuracy'] < 95.0:
                self.get_logger().warn(
                    f'Accuracy {result["accuracy"]:.1f}% below 95% threshold — rejecting'
                )
                self.log_results(
                    matched=False, product=None, accuracy=result['accuracy'],
                    confidence=result['confidence'], verified=False,
                    barcode_used=barcode_used, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason=f'accuracy {result["accuracy"]:.1f}% below 95% threshold'
                )
            else:
                self.log_results(
                    matched=True,
                    product=result['product'],
                    accuracy=result['accuracy'],
                    confidence=result['confidence'],
                    verified=result['verified'],
                    barcode_used=barcode_used,
                    barcodes=barcodes,
                    ocr_text=ocr_text,
                    details=details
                )
        else:
            self.log_results(
                matched=False, product=None, accuracy=0.0, confidence='NO MATCH',
                verified=False, barcode_used=None, barcodes=barcodes,
                ocr_text=ocr_text, details=details, reason='no product matched'
            )

        self.log_timing(callback_start, overall_start, end_to_end_from_ocr, mode)


def main(args=None):
    rclpy.init(args=args)
    node = DatabaseMatcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()