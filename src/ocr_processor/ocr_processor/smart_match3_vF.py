from database_manager import DatabaseManager
from fuzzywuzzy import fuzz
from datetime import datetime
import time
import os
import sys
import re

# Try to import OCR libraries (optional)
try:
    import cv2
    import pytesseract
    from pyzbar import pyzbar
    OCR_AVAILABLE = True
except ImportError:
    OCR_AVAILABLE = False
    print("⚠ OCR libraries not available. Install: pip3 install opencv-python pytesseract pyzbar")


# ==================== 95%+ ACCURACY CONFIGURATION ====================
ACCURACY_TARGET = 95.0

MIN_BRAND_SCORE = 85
MIN_PRODUCT_SCORE = 80
MIN_KEYWORD_SCORE = 85

HIGH_CONFIDENCE = 85
MEDIUM_CONFIDENCE = 70
LOW_CONFIDENCE = 55

MIN_ACCURACY_FOR_VERIFY = 85
MIN_OCR_CONFIDENCE = 75


# ==================== QUANTITY EXTRACTION ====================

# Ordered from most specific to least specific
QTY_PATTERNS = [
    r'[Qq]uantity[\s:]*(\d+)',            # Quantity: 4
    r'[Qq]\'?ty[\s:\.]*(\d+)',            # Qty: 4, Q'ty: 4, Qty.4
    r'\bCTN\s+OF\s+(\d+)\b',             # CTN OF 12
    r'\b(\d+)\s*[Pp][Cc][Ss]\b',         # 4pcs, 6 PCS
    r'\b[Xx]\s*(\d+)\b',                 # x4, X 6
    r'\b(\d+)\s*[Uu][Nn][Ii][Tt][Ss]?\b', # 4 units, 1 unit
]

def extract_quantity(ocr_text):
    """
    Attempt to extract a quantity value from OCR text.

    Returns:
        (quantity: int or None, source: str)
        - ('ocr', int)     : pattern matched, quantity extracted
        - ('default', 1)   : no pattern found, safe to use 1 (sorting)
        - (None, None)     : caller should decide based on stage
    """
    if not ocr_text:
        return None, None

    for pattern in QTY_PATTERNS:
        match = re.search(pattern, ocr_text)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 9999:
                print(f"  ✓ Quantity extracted: {qty} (pattern: '{pattern}')")
                return qty, 'ocr'

    return None, None


def resolve_quantity(ocr_text, scan_mode):
    """
    Determine final quantity and source based on OCR result and scan stage.

    Rules:
    - OCR found qty → use it, source='ocr'  (both stages)
    - OCR no qty + sorting → default 1, source='default'
    - OCR no qty + inbound → unknown, source='flagged', quantity=None
    """
    qty, source = extract_quantity(ocr_text)

    if source == 'ocr':
        print(f"  Quantity: {qty} (source: ocr)")
        return qty, 'ocr'

    if scan_mode == 'sorting':
        print(f"  ⚠ No quantity found — sorting default: 1")
        return 1, 'default'
    else:
        # inbound — quantity unknown, must be flagged
        print(f"  ⚠ No quantity found — inbound FLAGGED (quantity unknown)")
        return None, 'flagged'


class SmartMatcher:
    def __init__(self, device_id="PI-001", location="Main Location"):
        self.db = DatabaseManager()
        self.device_id = device_id
        self.location = location

        self.min_brand_score = MIN_BRAND_SCORE
        self.min_product_score = MIN_PRODUCT_SCORE
        self.min_keyword_score = MIN_KEYWORD_SCORE

        print(f"\n⚡ 95%+ Accuracy Mode Enabled")
        print(f"   Brand threshold: {MIN_BRAND_SCORE}%")
        print(f"   Product threshold: {MIN_PRODUCT_SCORE}%")
        print(f"   Keyword threshold: {MIN_KEYWORD_SCORE}%")

    # ==================== IMAGE OCR PROCESSING ====================

    def process_image(self, image_path, psm_mode=6, preprocess=True):
        if not OCR_AVAILABLE:
            print("✗ OCR libraries not installed!")
            return None

        if not os.path.exists(image_path):
            print(f"✗ Image not found: {image_path}")
            return None

        print("\n" + "="*70)
        print("ENHANCED OCR PROCESSING (95%+ MODE)")
        print("="*70)
        print(f"Image: {image_path}")

        image = cv2.imread(image_path)
        if image is None:
            print("✗ Could not read image")
            return None

        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        print("\n--- Extracting Text ---")
        start_time = time.time()

        if preprocess:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape
            if width < 800:
                scale = 800 / width
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                print(f"  Upscaled to: {gray.shape[1]}x{gray.shape[0]}")
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 10
            )
        else:
            processed = image

        config = f'--oem 1 --psm {psm_mode}'
        ocr_data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        ocr_text = pytesseract.image_to_string(processed, config=config).strip()

        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) != -1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        ocr_time = time.time() - start_time

        print(f"✓ OCR completed in {ocr_time:.3f}s")
        print(f"  Text length: {len(ocr_text)} characters")
        print(f"  OCR Confidence: {avg_confidence:.1f}%")

        if avg_confidence < MIN_OCR_CONFIDENCE:
            print(f"  ⚠ WARNING: Low OCR confidence ({avg_confidence:.1f}% < {MIN_OCR_CONFIDENCE}%)")

        print("\n--- Detecting Barcodes ---")
        start_time = time.time()
        barcodes = pyzbar.decode(image)
        barcode_time = time.time() - start_time

        barcode_data = None
        if barcodes:
            barcode_data = barcodes[0].data.decode('utf-8')
            print(f"✓ Barcode found: {barcode_data}")
        else:
            print("  No barcode detected")

        print(f"  Barcode detection: {barcode_time:.3f}s")

        return {
            'text': ocr_text,
            'barcode': barcode_data,
            'ocr_confidence': avg_confidence,
            'ocr_time': ocr_time,
            'barcode_time': barcode_time,
            'total_time': ocr_time + barcode_time
        }

    def scan_from_image(self, image_path, psm_mode=6, scan_mode='sorting'):
        ocr_result = self.process_image(image_path, psm_mode)

        if not ocr_result or not ocr_result['text']:
            print("\n✗ No text extracted from image")
            return None

        print("\n" + "="*70)
        print("EXTRACTED TEXT")
        print("="*70)
        print(ocr_result['text'])
        print("="*70)

        result = self.match_and_save(
            ocr_text=ocr_result['text'],
            barcode=ocr_result['barcode'],
            ocr_confidence=ocr_result.get('ocr_confidence'),
            device_id=self.device_id,
            notes=f"Scanned from image: {os.path.basename(image_path)}",
            scan_mode=scan_mode
        )

        return result

    # ==================== DYNAMIC ACCURACY CALCULATION ====================

    def _calculate_accuracy(self, brand_score, product_score, keyword_score,
                             barcode_matched, ocr_text, products, matched_product):
        if barcode_matched:
            return 100.0

        ocr_lower = ocr_text.lower().strip()
        brand_detected = brand_score >= MIN_BRAND_SCORE
        product_detected = product_score >= MIN_PRODUCT_SCORE

        keyword_bonus = min(keyword_score * 0.1, 10.0)

        if brand_detected and product_detected:
            print(f"  Mode: Brand + Product (40/50 + 10 keyword)")
            base = (brand_score * 0.4) + (product_score * 0.5)

        elif product_detected and not brand_detected:
            matched_name = matched_product[1].lower() if matched_product else ''
            same_name_count = sum(1 for p in products if p[1].lower() == matched_name)
            if same_name_count > 1:
                print(f"  Mode: Product only but AMBIGUOUS ({same_name_count} brands) — needs barcode")
                return 0.0
            else:
                print(f"  Mode: Product only, unique to one brand (90 + 10 keyword)")
                base = product_score * 0.9

        elif brand_detected and not product_detected:
            print(f"  Mode: Brand only (60 + 10 keyword)")
            base = brand_score * 0.6

        else:
            print(f"  Mode: No clear brand or product detected")
            return 0.0

        return min(base + keyword_bonus, 100.0)

    # ==================== OCR SCORE HELPER ====================

    def _get_ocr_scores(self, ocr_lower, product):
        brand = product[2].lower() if product[2] else ""
        product_name = product[1].lower() if product[1] else ""
        keywords = product[4] if product[4] else ""

        brand_score = max(
            fuzz.partial_ratio(brand, ocr_lower),
            fuzz.token_sort_ratio(brand, ocr_lower),
            fuzz.token_set_ratio(brand, ocr_lower)
        ) if brand else 0.0

        product_score = max(
            fuzz.partial_ratio(product_name, ocr_lower),
            fuzz.token_sort_ratio(product_name, ocr_lower)
        ) if product_name else 0.0

        keyword_list = [kw.strip() for kw in keywords.split(',')] if keywords else []
        if keyword_list:
            kw_scores = [fuzz.token_set_ratio(kw.lower(), ocr_lower) for kw in keyword_list]
            avg = sum(kw_scores) / len(kw_scores)
            mx = max(kw_scores)
            keyword_score = avg * 0.6 + mx * 0.4
        else:
            keyword_score = 0.0

        return {
            'brand_score': float(brand_score),
            'product_score': float(product_score),
            'keyword_score': float(keyword_score)
        }

    # ==================== ENHANCED MATCHING LOGIC ====================

    def match_and_save(self, ocr_text, barcode=None, ocr_confidence=None,
                       device_id=None, notes=None,
                       session_id=None, scan_mode='sorting'):
        """
        Match OCR text + barcode against products, then save to DB.

        scan_mode   — 'inbound' or 'sorting', controls quantity flagging behaviour:
                      inbound  + no qty → quantity=None, source='flagged'
                      sorting  + no qty → quantity=1,    source='default'
        session_id  — links this scan to a session for stage-level totals
        """
        start_time = time.time()

        print("\n" + "="*70)
        print("SMART PRODUCT MATCHING (95%+ MODE)")
        print("="*70)
        print(f"Device: {device_id or self.device_id}")
        print(f"Location: {self.location}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Scan mode: {scan_mode}")
        if ocr_confidence is not None:
            print(f"OCR Quality: {ocr_confidence:.1f}%")

        # ── Resolve quantity based on OCR text AND scan stage ──
        print("\n--- Quantity Extraction ---")
        quantity, quantity_source = resolve_quantity(ocr_text, scan_mode)

        products = self.db.get_all_products()

        if not products:
            print("\n⚠ No products in database! Add products first.")
            return None

        matched_product, match_confidence, match_score, match_details = self._enhanced_fuzzy_match(
            ocr_text, barcode, products
        )

        barcode_matched = (barcode is not None and matched_product is not None and match_score == 200.0)

        accuracy_percentage = self._calculate_accuracy(
            brand_score=match_details['brand_score'],
            product_score=match_details['product_score'],
            keyword_score=match_details['keyword_score'],
            barcode_matched=barcode_matched,
            ocr_text=ocr_text,
            products=products,
            matched_product=matched_product
        )

        if accuracy_percentage == 0.0 and not barcode_matched:
            matched_product = None
            match_confidence = "NO MATCH"

        if matched_product and accuracy_percentage >= MIN_ACCURACY_FOR_VERIFY:
            verified = 1 if match_confidence in ['HIGH', 'MEDIUM'] else 0
        else:
            verified = 0

        scan_id = self.db.save_scan(
            ocr_text=ocr_text,
            matched_product_id=matched_product[0] if matched_product else None,
            match_confidence=match_confidence,
            match_score=accuracy_percentage,
            barcode=barcode,
            device_id=device_id or self.device_id,
            notes=notes,
            session_id=session_id,
            quantity=quantity if quantity is not None else 1,
            quantity_source=quantity_source,
            scan_mode=scan_mode
        )

        processing_time = time.time() - start_time

        print("\n" + "="*70)
        print("MATCHING RESULTS")
        print("="*70)

        if matched_product:
            status_icon = "✓" if verified else "⚠"
            print(f"{status_icon} MATCH FOUND!")
            print(f"  Status: {'VERIFIED (1)' if verified else 'UNVERIFIED (0)'}")
            print(f"  Product: {matched_product[1]}")
            print(f"  Brand: {matched_product[2]}")
            print(f"  Category: {matched_product[3]}")

            if quantity_source == 'flagged':
                print(f"  Quantity: ⚠ UNKNOWN — flagged for manual resolution")
            else:
                print(f"  Quantity: {quantity} (source: {quantity_source})")

            print(f"\n  Match Accuracy: {accuracy_percentage:.1f}%")
            print(f"  Confidence Level: {match_confidence}")
            print(f"\n  Detailed Breakdown:")
            print(f"    Barcode match:      {'100.0% (exact)' if barcode_matched else '0.0% (no match)'}")
            print(f"    Brand similarity:   {match_details['brand_score']:.1f}%")
            print(f"    Product similarity: {match_details['product_score']:.1f}%")
            print(f"    Keyword similarity: {match_details['keyword_score']:.1f}%")

            if accuracy_percentage >= 95:
                print(f"\n  🎯 EXCELLENT MATCH (95%+)")
            elif accuracy_percentage >= 85:
                print(f"\n  ✓ GOOD MATCH (85-95%)")
            elif accuracy_percentage >= 70:
                print(f"\n  ⚠ ACCEPTABLE MATCH (70-85%)")
            else:
                print(f"\n  ⚠ LOW CONFIDENCE MATCH (<70%)")

            if verified and quantity_source != 'flagged':
                inventory = self.db.get_inventory()
                for item in inventory:
                    if item[0] == matched_product[0]:
                        print(f"\n  Updated Stock: {item[3]} units at {item[4]}")
                        break
        else:
            print(f"✗ NO MATCH FOUND")
            print(f"  Status: UNVERIFIED (0)")
            print(f"  Match Accuracy: {accuracy_percentage:.1f}%")

        print(f"\nScan ID: {scan_id}")
        print(f"Processing Time: {processing_time:.2f}s")
        print("="*70)

        return {
            'scan_id': scan_id,
            'matched': matched_product is not None,
            'verified': verified,
            'product': matched_product,
            'confidence': match_confidence,
            'score': match_score,
            'accuracy': accuracy_percentage,
            'processing_time': processing_time,
            'barcode_matched': barcode_matched,
            'match_details': match_details,
            'quantity': quantity,
            'quantity_source': quantity_source
        }

    def _enhanced_fuzzy_match(self, ocr_text, barcode, products):
        ocr_lower = ocr_text.lower().strip()

        if barcode:
            for product in products:
                if product[6] and product[6] == barcode:
                    print(f"  ✓ Exact barcode match (100% accuracy)")
                    ocr_details = self._get_ocr_scores(ocr_lower, product)
                    return product, "HIGH", 200.0, ocr_details

        best_match = None
        best_score = 0
        match_details = {'brand_score': 0, 'product_score': 0, 'keyword_score': 0}

        for product in products:
            product_name = product[1]
            brand = product[2] if product[2] else ""
            keywords = product[4] if product[4] else ""

            total_score = 0
            details = {'brand_score': 0, 'product_score': 0, 'keyword_score': 0}

            if brand:
                brand_score = max(
                    fuzz.partial_ratio(brand.lower(), ocr_lower),
                    fuzz.token_sort_ratio(brand.lower(), ocr_lower),
                    fuzz.token_set_ratio(brand.lower(), ocr_lower)
                )
                details['brand_score'] = brand_score
                if brand_score >= self.min_brand_score:
                    total_score += brand_score * 0.4

            product_score = max(
                fuzz.partial_ratio(product_name.lower(), ocr_lower),
                fuzz.token_sort_ratio(product_name.lower(), ocr_lower)
            )
            details['product_score'] = product_score
            if product_score >= self.min_product_score:
                total_score += product_score * 0.5

            if keywords:
                keyword_list = [kw.strip() for kw in keywords.split(',')]
                keyword_scores = [fuzz.token_set_ratio(kw.lower(), ocr_lower) for kw in keyword_list]
                avg_keyword_score = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
                max_keyword_score = max(keyword_scores) if keyword_scores else 0
                final_keyword_score = (avg_keyword_score * 0.6 + max_keyword_score * 0.4)
                details['keyword_score'] = final_keyword_score
                if final_keyword_score >= self.min_keyword_score:
                    total_score += final_keyword_score * 0.1

            if total_score > best_score:
                best_score = total_score
                best_match = product
                match_details = details

        if best_score >= 40:
            confidence = "HIGH"
        elif best_score >= 35:
            confidence = "MEDIUM"
        elif best_score >= 25:
            confidence = "LOW"
        else:
            confidence = "NO MATCH"
            best_match = None

        return best_match, confidence, best_score, match_details

    # ==================== QUICK TESTING ====================

    def quick_match(self, text, scan_mode='sorting'):
        products = self.db.get_all_products()
        if not products:
            print("⚠ No products in database!")
            return None

        matched, confidence, score, details = self._enhanced_fuzzy_match(text, None, products)

        barcode_matched = False
        accuracy = self._calculate_accuracy(
            brand_score=details['brand_score'],
            product_score=details['product_score'],
            keyword_score=details['keyword_score'],
            barcode_matched=barcode_matched,
            ocr_text=text,
            products=products,
            matched_product=matched
        )

        quantity, quantity_source = resolve_quantity(text, scan_mode)

        print(f"\nInput: {text}")
        if quantity_source == 'flagged':
            print(f"Quantity: ⚠ UNKNOWN — would be flagged (inbound, no qty found)")
        else:
            print(f"Quantity: {quantity} (source: {quantity_source})")

        if matched and accuracy > 0:
            verified = 1 if (confidence in ['HIGH', 'MEDIUM'] and accuracy >= MIN_ACCURACY_FOR_VERIFY) else 0
            status = "VERIFIED (1)" if verified else "UNVERIFIED (0)"
            print(f"✓ Matched: {matched[1]} ({matched[2]})")
            print(f"  Status: {status}")
            print(f"  Match Accuracy: {accuracy:.1f}%")
            print(f"  Confidence: {confidence} | Raw Score: {score:.1f}")
            print(f"  Brand similarity:   {details['brand_score']:.1f}%")
            print(f"  Product similarity: {details['product_score']:.1f}%")
            print(f"  Keyword similarity: {details['keyword_score']:.1f}%")
            if accuracy >= 95:
                print(f"  🎯 TARGET ACHIEVED (95%+)")
        else:
            print(f"✗ No match")
            print(f"  Match Accuracy: {accuracy:.1f}%")
            print(f"  Score: {score:.1f} | Status: UNVERIFIED (0)")

        return matched


# ==================== INTERACTIVE MENU ====================

def main():
    print("\n" + "="*70)
    print("🎯 SMART MATCHER - 95%+ ACCURACY MODE")
    print("="*70)
    print(f"Target Accuracy: {ACCURACY_TARGET}%")
    print(f"Verification Threshold: {MIN_ACCURACY_FOR_VERIFY}%")

    matcher = SmartMatcher(device_id="PI-95", location="Prod")

    while True:
        print("\n" + "="*70)
        print("MAIN MENU")
        print("="*70)
        print("\n--- Testing Options ---")
        print("1. Quick text match (test only)")
        print("2. Full text scan (match and save)")
        if OCR_AVAILABLE:
            print("3. Scan from image (OCR + match + save)")
        else:
            print("3. [Disabled] Scan from image")

        print("\n--- Database Views ---")
        print("4. View all products")
        print("5. View verified scans")
        print("6. View unverified scans")
        print("7. View scan history")
        print("8. View inventory")
        print("9. View statistics")
        print("10. Check system accuracy")
        print("11. View quantity summary")
        print("12. View flagged scans")
        print("13. Resolve flagged scan")

        print("\n--- Management ---")
        print("14. Add new product")

        print("\n--- Exit ---")
        print("0. Exit")

        choice = input("\nChoose option: ")

        if choice == '1':
            text = input("\nEnter text to match: ")
            mode = input("Scan mode (inbound/sorting) [sorting]: ").strip() or 'sorting'
            if text.strip():
                matcher.quick_match(text, scan_mode=mode)

        elif choice == '2':
            print("\n--- Full Text Scan ---")
            ocr_text = input("Enter OCR text: ")
            barcode = input("Enter barcode (optional): ")
            mode = input("Scan mode (inbound/sorting) [sorting]: ").strip() or 'sorting'
            if ocr_text.strip():
                matcher.match_and_save(
                    ocr_text=ocr_text,
                    barcode=barcode if barcode else None,
                    scan_mode=mode
                )

        elif choice == '3':
            if not OCR_AVAILABLE:
                print("\n⚠ OCR libraries not installed!")
                continue
            print("\n--- Scan from Image ---")
            image_path = input("Enter image path: ")
            if not image_path.strip():
                continue
            image_path = os.path.expanduser(image_path)
            if not os.path.exists(image_path):
                print(f"✗ File not found: {image_path}")
                continue
            psm = input("PSM mode (6=uniform text, 11=sparse): ")
            psm_mode = int(psm) if psm.strip() else 6
            mode = input("Scan mode (inbound/sorting) [sorting]: ").strip() or 'sorting'
            matcher.scan_from_image(image_path, psm_mode, scan_mode=mode)

        elif choice == '4':
            matcher.db.view_all_products()
        elif choice == '5':
            matcher.db.view_verified_scans()
        elif choice == '6':
            matcher.db.view_unverified_scans()
        elif choice == '7':
            limit = input("Show how many scans? (default 10): ")
            limit = int(limit) if limit.strip() else 10
            matcher.db.view_scan_history(limit)
        elif choice == '8':
            matcher.db.view_inventory()
        elif choice == '9':
            matcher.db.view_statistics()

        elif choice == '10':
            verified = matcher.db.get_verified_list()
            unverified = matcher.db.get_unverified_list()
            total = len(verified) + len(unverified)
            if total == 0:
                print("\n⚠ No scans yet!")
            else:
                accuracy = (len(verified) / total) * 100
                print(f"\n{'='*70}")
                print(f"SYSTEM ACCURACY REPORT")
                print(f"{'='*70}")
                print(f"Total Scans: {total}")
                print(f"Verified (✓): {len(verified)}")
                print(f"Unverified (✗): {len(unverified)}")
                print(f"\nCurrent Accuracy: {accuracy:.2f}%")
                print(f"Target: {ACCURACY_TARGET}%")
                print(f"Gap: {ACCURACY_TARGET - accuracy:+.2f}%")
                if accuracy >= ACCURACY_TARGET:
                    print(f"\n🎯 TARGET ACHIEVED! ✓✓✓")
                else:
                    print(f"\n⚠ Need to improve by {ACCURACY_TARGET - accuracy:.2f}%")
                print(f"{'='*70}")

        elif choice == '11':
            matcher.db.view_quantity_summary()

        elif choice == '12':
            matcher.db.view_flagged_scans()

        elif choice == '13':
            matcher.db.view_flagged_scans()
            scan_id = input("\nEnter Scan ID to resolve: ")
            qty = input("Enter actual quantity: ")
            if scan_id.strip() and qty.strip():
                matcher.db.resolve_flagged_scan(int(scan_id), int(qty))

        elif choice == '14':
            print("\n--- Add New Product ---")
            name = input("Product name: ")
            brand = input("Brand: ")
            category = input("Category: ")
            print("\n💡 Keywords tip: Include common typos/OCR errors")
            keywords = input("Keywords (comma-separated): ")
            description = input("Description: ")
            barcode = input("Barcode (optional): ")
            if name and brand:
                matcher.db.add_product(name, brand, category, keywords, description,
                                      barcode if barcode else None)

        elif choice == '0':
            print("\nGoodbye!")
            break
        else:
            print("\n⚠ Invalid option!")


if __name__ == "__main__":
    main()