from database_manager import DatabaseManager
from fuzzywuzzy import fuzz
from datetime import datetime
import time
import os
import sys

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

# Minimum similarity thresholds
MIN_BRAND_SCORE = 85        # Brand must be 85%+ similar
MIN_PRODUCT_SCORE = 80      # Product name must be 80%+ similar
MIN_KEYWORD_SCORE = 85      # Keywords must be 85%+ similar

# Accuracy thresholds (now out of 100, not raw score)
HIGH_CONFIDENCE = 85        # 85%+ accuracy
MEDIUM_CONFIDENCE = 70      # 70-84% accuracy
LOW_CONFIDENCE = 55         # 55-69% accuracy

# Only verify if accuracy is high enough
MIN_ACCURACY_FOR_VERIFY = 85  # Must be 85%+ accurate to mark as verified

# OCR Quality requirements
MIN_OCR_CONFIDENCE = 75     # Reject if OCR confidence < 75%


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

    def scan_from_image(self, image_path, psm_mode=6):
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
            notes=f"Scanned from image: {os.path.basename(image_path)}"
        )

        return result

    # ==================== DYNAMIC ACCURACY CALCULATION ====================

    def _calculate_accuracy(self, brand_score, product_score, keyword_score, barcode_matched, ocr_text, products, matched_product):
        """
        Dynamic accuracy calculation based on what information is available.

        Rules:
        - Barcode exact match → always 100%
        - Brand + Product both detected → 50% brand + 50% product + keyword bonus
        - Product only, unique to one brand → 80% product + 20% brand + keyword bonus
        - Product only, shared across brands → 0% (ambiguous, needs barcode)
        - Brand only → 60% brand + 40% product + keyword bonus
        """
        if barcode_matched:
            return 100.0

        ocr_lower = ocr_text.lower().strip()
        brand_detected = brand_score >= MIN_BRAND_SCORE
        product_detected = product_score >= MIN_PRODUCT_SCORE

        # Keyword bonus (up to 10%)
        keyword_bonus = min(keyword_score * 0.1, 10.0)

        if brand_detected and product_detected:
            # Both brand and product detected — 50/50 split
            print(f"  Mode: Brand + Product (40/50 + 10 keyword)")
            base = (brand_score * 0.4) + (product_score * 0.5)

        elif product_detected and not brand_detected:
            # Product detected but no brand — check if product is unique
            matched_name = matched_product[1].lower() if matched_product else ''
            same_name_count = sum(
                1 for p in products
                if p[1].lower() == matched_name
            )
            if same_name_count > 1:
                # Same product name exists across multiple brands — ambiguous
                print(f"  Mode: Product only but AMBIGUOUS ({same_name_count} brands have this product) — needs barcode")
                return 0.0
            else:
                # Product is unique to one brand
                print(f"  Mode: Product only, unique to one brand (90 + 10 keyword)")
                base = product_score * 0.9  # leaves 10% for keyword bonus

        elif brand_detected and not product_detected:
            # Brand detected but no product name
            print(f"  Mode: Brand only (60 + 10 keyword)")
            base = brand_score * 0.6  # leaves 10% for keyword bonus

        else:
            # Neither brand nor product clearly detected
            print(f"  Mode: No clear brand or product detected")
            return 0.0

        return min(base + keyword_bonus, 100.0)

    # ==================== OCR SCORE HELPER ====================

    def _get_ocr_scores(self, ocr_lower, product):
        """
        Compute real OCR similarity scores independently of barcode.
        Used to show honest OCR breakdown even when barcode matched.
        """
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

    def match_and_save(self, ocr_text, barcode=None, ocr_confidence=None, device_id=None, notes=None):
        start_time = time.time()

        print("\n" + "="*70)
        print("SMART PRODUCT MATCHING (95%+ MODE)")
        print("="*70)
        print(f"Device: {device_id or self.device_id}")
        print(f"Location: {self.location}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if ocr_confidence is not None:
            print(f"OCR Quality: {ocr_confidence:.1f}%")

        products = self.db.get_all_products()

        if not products:
            print("\n⚠ No products in database! Add products first.")
            return None

        matched_product, match_confidence, match_score, match_details = self._enhanced_fuzzy_match(
            ocr_text, barcode, products
        )

        barcode_matched = (barcode is not None and matched_product is not None and match_score == 200.0)

        # Dynamic accuracy calculation
        accuracy_percentage = self._calculate_accuracy(
            brand_score=match_details['brand_score'],
            product_score=match_details['product_score'],
            keyword_score=match_details['keyword_score'],
            barcode_matched=barcode_matched,
            ocr_text=ocr_text,
            products=products,
            matched_product=matched_product
        )

        # If accuracy is 0 due to ambiguity, clear the match
        if accuracy_percentage == 0.0 and not barcode_matched:
            matched_product = None
            match_confidence = "NO MATCH"

        # Verification rule
        if matched_product and accuracy_percentage >= MIN_ACCURACY_FOR_VERIFY:
            verified = 1 if match_confidence in ['HIGH', 'MEDIUM'] else 0
        else:
            verified = 0

        scan_id = self.db.save_scan(
            ocr_text=ocr_text,
            matched_product_id=matched_product[0] if matched_product else None,
            match_confidence=match_confidence,
            match_score=accuracy_percentage,  # save true accuracy, not raw score
            barcode=barcode,
            device_id=device_id or self.device_id,
            notes=notes
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

            if verified:
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
            'match_details': match_details
        }

    def _enhanced_fuzzy_match(self, ocr_text, barcode, products):
        """
        Enhanced fuzzy matching with dynamic weighting.

        Returns:
            tuple: (matched_product, confidence, score, details)
        """
        ocr_lower = ocr_text.lower().strip()

        # Strategy 1: Exact barcode match (100% accuracy)
        if barcode:
            for product in products:
                if product[6] and product[6] == barcode:
                    print(f"  ✓ Exact barcode match (100% accuracy)")
                    # Compute real OCR scores independently so display is honest
                    ocr_details = self._get_ocr_scores(ocr_lower, product)
                    return product, "HIGH", 200.0, ocr_details

        # Strategy 2: Enhanced fuzzy text matching
        best_match = None
        best_score = 0
        match_details = {'brand_score': 0, 'product_score': 0, 'keyword_score': 0}

        for product in products:
            product_name = product[1]
            brand = product[2] if product[2] else ""
            keywords = product[4] if product[4] else ""

            total_score = 0
            details = {'brand_score': 0, 'product_score': 0, 'keyword_score': 0}

            # Brand matching
            if brand:
                brand_score = max(
                    fuzz.partial_ratio(brand.lower(), ocr_lower),
                    fuzz.token_sort_ratio(brand.lower(), ocr_lower),
                    fuzz.token_set_ratio(brand.lower(), ocr_lower)
                )
                details['brand_score'] = brand_score
                if brand_score >= self.min_brand_score:
                    total_score += brand_score * 0.4

            # Product name matching
            product_score = max(
                fuzz.partial_ratio(product_name.lower(), ocr_lower),
                fuzz.token_sort_ratio(product_name.lower(), ocr_lower)
            )
            details['product_score'] = product_score
            if product_score >= self.min_product_score:
                total_score += product_score * 0.5

            # Keyword matching
            if keywords:
                keyword_list = [kw.strip() for kw in keywords.split(',')]
                keyword_scores = []
                for keyword in keyword_list:
                    kw_score = fuzz.token_set_ratio(keyword.lower(), ocr_lower)
                    keyword_scores.append(kw_score)

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

        # Confidence based on raw fuzzy score (used internally for filtering)
        if best_score >= 65:
            confidence = "HIGH"
        elif best_score >= 70:
            confidence = "MEDIUM"
        elif best_score >= 55:
            confidence = "LOW"
        else:
            confidence = "NO MATCH"
            best_match = None

        return best_match, confidence, best_score, match_details

    # ==================== QUICK TESTING ====================

    def quick_match(self, text):
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

        print(f"\nInput: {text}")
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
        print("5. View verified scans (status = 1)")
        print("6. View unverified scans (status = 0)")
        print("7. View scan history")
        print("8. View inventory")
        print("9. View statistics")
        print("10. Check system accuracy")

        print("\n--- Management ---")
        print("11. Add new product")

        print("\n--- Exit ---")
        print("0. Exit")

        choice = input("\nChoose option: ")

        if choice == '1':
            text = input("\nEnter text to match: ")
            if text.strip():
                matcher.quick_match(text)

        elif choice == '2':
            print("\n--- Full Text Scan ---")
            ocr_text = input("Enter OCR text: ")
            barcode = input("Enter barcode (optional): ")

            if ocr_text.strip():
                matcher.match_and_save(
                    ocr_text=ocr_text,
                    barcode=barcode if barcode else None
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

            matcher.scan_from_image(image_path, psm_mode)

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
                    gap = ACCURACY_TARGET - accuracy
                    print(f"\n⚠ Need to improve by {gap:.2f}%")

                print(f"{'='*70}")

        elif choice == '11':
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