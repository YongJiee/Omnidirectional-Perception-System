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

# Stricter thresholds for higher accuracy
MIN_BRAND_SCORE = 85        # Brand must be 85%+ similar
MIN_PRODUCT_SCORE = 80      # Product name must be 80%+ similar  
MIN_KEYWORD_SCORE = 85      # Keywords must be 85%+ similar

# Confidence thresholds (stricter)
HIGH_CONFIDENCE = 180       # 90%+ match accuracy
MEDIUM_CONFIDENCE = 160     # 80-89% match accuracy
LOW_CONFIDENCE = 140        # 70-79% match accuracy

# Only verify if accuracy is high enough
MIN_ACCURACY_FOR_VERIFY = 85  # Must be 85%+ accurate to mark as verified

# OCR Quality requirements
MIN_OCR_CONFIDENCE = 75     # Reject if OCR confidence < 75%


class SmartMatcher:
    def __init__(self, device_id="PI-001", location="Main Location"):
        self.db = DatabaseManager()
        self.device_id = device_id
        self.location = location
        
        # Use strict thresholds
        self.min_brand_score = MIN_BRAND_SCORE
        self.min_product_score = MIN_PRODUCT_SCORE
        self.min_keyword_score = MIN_KEYWORD_SCORE
        
        print(f"\n⚡ 95%+ Accuracy Mode Enabled")
        print(f"   Brand threshold: {MIN_BRAND_SCORE}%")
        print(f"   Product threshold: {MIN_PRODUCT_SCORE}%")
        print(f"   Keyword threshold: {MIN_KEYWORD_SCORE}%")
    
    # ==================== IMAGE OCR PROCESSING ====================
    
    def process_image(self, image_path, psm_mode=6, preprocess=True):
        """
        Process image with enhanced OCR for better accuracy
        
        Args:
            image_path: Path to image file
            psm_mode: Tesseract PSM mode (default 6 for uniform text blocks)
            preprocess: Apply preprocessing
        
        Returns:
            dict with OCR results including confidence
        """
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
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            print("✗ Could not read image")
            return None
        
        print(f"✓ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Extract text with enhanced preprocessing
        print("\n--- Extracting Text ---")
        start_time = time.time()
        
        # Enhanced preprocessing for better OCR
        if preprocess:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Resize if too small
            height, width = gray.shape
            if width < 800:
                scale = 800 / width
                gray = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
                print(f"  Upscaled to: {gray.shape[1]}x{gray.shape[0]}")
            
            # Denoise
            gray = cv2.fastNlMeansDenoising(gray, h=10)
            
            # Adaptive threshold for better text detection
            processed = cv2.adaptiveThreshold(
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
            )
        else:
            processed = image
        
        # OCR with confidence data
        config = f'--oem 1 --psm {psm_mode}'
        
        # Get detailed OCR data
        ocr_data = pytesseract.image_to_data(processed, config=config, output_type=pytesseract.Output.DICT)
        ocr_text = pytesseract.image_to_string(processed, config=config).strip()
        
        # Calculate average confidence
        confidences = [int(conf) for conf in ocr_data['conf'] if int(conf) != -1]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        ocr_time = time.time() - start_time
        
        print(f"✓ OCR completed in {ocr_time:.3f}s")
        print(f"  Text length: {len(ocr_text)} characters")
        print(f"  OCR Confidence: {avg_confidence:.1f}%")
        
        # Warn if OCR quality is low
        if avg_confidence < MIN_OCR_CONFIDENCE:
            print(f"  ⚠ WARNING: Low OCR confidence ({avg_confidence:.1f}% < {MIN_OCR_CONFIDENCE}%)")
            print(f"  ⚠ Consider retaking image with better lighting/focus")
        
        # Extract barcodes
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
        """
        Complete workflow: OCR from image → Match → Save to database
        
        Args:
            image_path: Path to image file
            psm_mode: Tesseract PSM mode (default 6 for labels)
        
        Returns:
            dict with scan results
        """
        # Extract OCR data
        ocr_result = self.process_image(image_path, psm_mode)
        
        if not ocr_result or not ocr_result['text']:
            print("\n✗ No text extracted from image")
            return None
        
        print("\n" + "="*70)
        print("EXTRACTED TEXT")
        print("="*70)
        print(ocr_result['text'])
        print("="*70)
        
        # Match and save
        result = self.match_and_save(
            ocr_text=ocr_result['text'],
            barcode=ocr_result['barcode'],
            ocr_confidence=ocr_result.get('ocr_confidence'),
            device_id=self.device_id,
            notes=f"Scanned from image: {os.path.basename(image_path)}"
        )
        
        return result
    
    # ==================== ENHANCED MATCHING LOGIC ====================
    
    def match_and_save(self, ocr_text, barcode=None, ocr_confidence=None, device_id=None, notes=None):
        """
        Enhanced matching with stricter verification rules
        
        Args:
            ocr_text: Text extracted from OCR
            barcode: Barcode number (if detected)
            ocr_confidence: OCR engine confidence score (0-100)
            device_id: Device identifier
            notes: Additional notes
        
        Returns:
            dict with scan results
        """
        start_time = time.time()
        
        print("\n" + "="*70)
        print("SMART PRODUCT MATCHING (95%+ MODE)")
        print("="*70)
        print(f"Device: {device_id or self.device_id}")
        print(f"Location: {self.location}")
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        if ocr_confidence is not None:
            print(f"OCR Quality: {ocr_confidence:.1f}%")
        
        # Get all products
        products = self.db.get_all_products()
        
        if not products:
            print("\n⚠ No products in database! Add products first.")
            return None
        
        # Try matching
        matched_product, match_confidence, match_score, match_details = self._enhanced_fuzzy_match(
            ocr_text, barcode, products
        )
        
        # Calculate overall accuracy percentage (0-100%)
        accuracy_percentage = min(match_score / 2, 100)
        
        # Stricter verification rule: only verify if accuracy >= threshold
        if matched_product and accuracy_percentage >= MIN_ACCURACY_FOR_VERIFY:
            verified = 1 if match_confidence in ['HIGH', 'MEDIUM'] else 0
        else:
            verified = 0  # Not accurate enough, mark as unverified
        
        # Save to database
        scan_id = self.db.save_scan(
            ocr_text=ocr_text,
            matched_product_id=matched_product[0] if matched_product else None,
            match_confidence=match_confidence,
            match_score=match_score,
            barcode=barcode,
            device_id=device_id or self.device_id,
            notes=notes
        )
        
        processing_time = time.time() - start_time
        
        # Display results
        print("\n" + "="*70)
        print("MATCHING RESULTS")
        print("="*70)
        
        if matched_product:
            status_icon = "✓" if verified else "⚠"
            
            print(f"{status_icon} MATCH FOUND!")
            print(f"  Status: {'VERIFIED (1)' if verified else 'UNVERIFIED (0)'}")
            
            if accuracy_percentage < MIN_ACCURACY_FOR_VERIFY:
                print(f"  ⚠ Accuracy too low for verification ({accuracy_percentage:.1f}% < {MIN_ACCURACY_FOR_VERIFY}%)")
            
            print(f"  Product: {matched_product[1]}")
            print(f"  Brand: {matched_product[2]}")
            print(f"  Category: {matched_product[3]}")
            print(f"\n  Match Accuracy: {accuracy_percentage:.1f}%")
            print(f"  Confidence Level: {match_confidence}")
            print(f"  Match Score: {match_score:.1f}")
            print(f"\n  Detailed Breakdown:")
            print(f"    Brand similarity: {match_details['brand_score']:.1f}%")
            print(f"    Product similarity: {match_details['product_score']:.1f}%")
            print(f"    Keyword similarity: {match_details['keyword_score']:.1f}%")
            
            # Quality assessment
            if accuracy_percentage >= 95:
                print(f"\n  🎯 EXCELLENT MATCH (95%+)")
            elif accuracy_percentage >= 90:
                print(f"\n  ✓ GOOD MATCH (90-95%)")
            elif accuracy_percentage >= 85:
                print(f"\n  ⚠ ACCEPTABLE MATCH (85-90%)")
            else:
                print(f"\n  ⚠ LOW CONFIDENCE MATCH (<85%)")
            
            # Show inventory update only if verified
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
            print(f"  Best score: {match_score:.1f} (threshold: {LOW_CONFIDENCE})")
          
        
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
            'processing_time': processing_time
        }
    
    def _enhanced_fuzzy_match(self, ocr_text, barcode, products):
        """
        Enhanced fuzzy matching with better typo handling
        
        Returns:
            tuple: (matched_product, confidence, score, details)
        """
        ocr_lower = ocr_text.lower().strip()
        
        # Strategy 1: Exact barcode match (100% accuracy)
        if barcode:
            for product in products:
                if product[6] and product[6] == barcode:
                    print(f"  ✓ Exact barcode match (100% accuracy)")
                    return product, "HIGH", 200.0, {
                        'brand_score': 100.0,
                        'product_score': 100.0,
                        'keyword_score': 100.0
                    }
        
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
            
            # Brand matching (weighted x2.5 - brand is most important)
            if brand:
                # Try multiple fuzzy algorithms for better typo handling
                brand_score = max(
                    fuzz.partial_ratio(brand.lower(), ocr_lower),
                    fuzz.token_sort_ratio(brand.lower(), ocr_lower),
                    fuzz.token_set_ratio(brand.lower(), ocr_lower)
                )
                details['brand_score'] = brand_score
                if brand_score >= self.min_brand_score:
                    total_score += brand_score * 2.5  # Higher weight
            
            # Product name matching (weighted x2)
            product_score = max(
                fuzz.partial_ratio(product_name.lower(), ocr_lower),
                fuzz.token_sort_ratio(product_name.lower(), ocr_lower)
            )
            details['product_score'] = product_score
            if product_score >= self.min_product_score:
                total_score += product_score * 2
            
            # Keyword matching (weighted x1.5)
            if keywords:
                keyword_list = [kw.strip() for kw in keywords.split(',')]
                keyword_scores = []
                for keyword in keyword_list:
                    # Use token_set_ratio for better typo tolerance
                    kw_score = fuzz.token_set_ratio(keyword.lower(), ocr_lower)
                    keyword_scores.append(kw_score)
                
                avg_keyword_score = sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0
                max_keyword_score = max(keyword_scores) if keyword_scores else 0
                
                # Use both average and max for better matching
                final_keyword_score = (avg_keyword_score * 0.6 + max_keyword_score * 0.4)
                details['keyword_score'] = final_keyword_score
                
                if final_keyword_score >= self.min_keyword_score:
                    total_score += final_keyword_score * 1.5
            
            # Update best match
            if total_score > best_score:
                best_score = total_score
                best_match = product
                match_details = details
        
        # Stricter confidence determination
        if best_score >= HIGH_CONFIDENCE:
            confidence = "HIGH"
        elif best_score >= MEDIUM_CONFIDENCE:
            confidence = "MEDIUM"
        elif best_score >= LOW_CONFIDENCE:
            confidence = "LOW"
        else:
            confidence = "NO MATCH"
            best_match = None
        
        return best_match, confidence, best_score, match_details
    
    # ==================== QUICK TESTING ====================
    
    def quick_match(self, text):
        """Quick match for testing - shows result without saving"""
        products = self.db.get_all_products()
        if not products:
            print("⚠ No products in database!")
            return None
        
        matched, confidence, score, details = self._enhanced_fuzzy_match(text, None, products)
        
        # Calculate accuracy percentage
        accuracy = min(score / 2, 100)
        
        print(f"\nInput: {text}")
        if matched:
            verified = 1 if (confidence in ['HIGH', 'MEDIUM'] and accuracy >= MIN_ACCURACY_FOR_VERIFY) else 0
            status = "VERIFIED (1)" if verified else "UNVERIFIED (0)"
            print(f"✓ Matched: {matched[1]} ({matched[2]})")
            print(f"  Status: {status}")
            print(f"  Match Accuracy: {accuracy:.1f}%")
            print(f"  Confidence: {confidence} | Score: {score:.1f}")
            
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
        
        # Testing options
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
        
        # Database views
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
            # Check system accuracy
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
                    print(f"💡 Tips:")
                    print(f"   - Add more product variations/keywords")
                    print(f"   - Improve image quality (lighting, focus)")
                    print(f"   - Review unverified scans and update database")
                
                print(f"{'='*70}")
        
        # Management
        elif choice == '11':
            print("\n--- Add New Product ---")
            name = input("Product name: ")
            brand = input("Brand: ")
            category = input("Category: ")
            
            print("\n💡 Keywords tip: Include common typos/OCR errors")
            print("   Example: sephora, 5ephora, seph0ra, sephra")
            keywords = input("Keywords (comma-separated): ")
            
            description = input("Description: ")
            barcode = input("Barcode (optional): ")
            
            if name and brand:
                matcher.db.add_product(name, brand, category, keywords, description, 
                                      barcode if barcode else None)
        
        # Exit
        elif choice == '0':
            print("\nGoodbye!")
            break
        
        else:
            print("\n⚠ Invalid option!")


if __name__ == "__main__":
    main()