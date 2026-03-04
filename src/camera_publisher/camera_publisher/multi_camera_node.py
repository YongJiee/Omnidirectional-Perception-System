import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String
import cv2
import pytesseract
from pyzbar import pyzbar
import re
import time
import numpy as np
import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

# Reliable QoS — must match publisher to prevent message drops
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)


class OCRProcessor(Node):
    def __init__(self):
        super().__init__('ocr_node')
        # ---------------------------------------------------------------
        # Mode selection — default 1 (single), set 4 for multi-camera
        # ---------------------------------------------------------------
        self.declare_parameter('num_cameras', 1)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.get_logger().info(f'Mode: {self.num_cameras}-camera')

        # Buffer to collect results from all cameras before fusing
        self.camera_results = {}
        self.camera_results_lock = threading.Lock()
        self.overall_start = None
        self.batch_save_dir = None
        self.batch_already_processed = False
        self._fuse_triggered = False

        # Publisher
        self.result_publisher = self.create_publisher(String, '/ocr_results', 10)

        # ---------------------------------------------------------------
        # Subscribe to correct topics based on mode
        # ---------------------------------------------------------------
        self.subscriptions_ = []
        if self.num_cameras == 1:
            sub = self.create_subscription(
                CompressedImage,
                '/camera_0/image_raw/compressed',
                lambda msg: self.on_image_received(msg, camera_id=0),
                RELIABLE_QOS
            )
            self.subscriptions_.append(sub)
            self.get_logger().info('Subscribed to: /camera_0/image_raw/compressed')
        else:
            for i in range(self.num_cameras):
                topic = f'/camera_{i}/image_raw/compressed'
                sub = self.create_subscription(
                    CompressedImage,
                    topic,
                    lambda msg, cam_id=i: self.on_image_received(msg, cam_id),
                    RELIABLE_QOS
                )
                self.subscriptions_.append(sub)
                self.get_logger().info(f'Subscribed to: {topic}')
            self.batch_sub = self.create_subscription(
                String, '/batch_complete', self.on_batch_complete, RELIABLE_QOS)
            self.get_logger().info('Subscribed to: /batch_complete')

        self.get_logger().info(f'OCR Processor ready — waiting for {self.num_cameras} image(s)...')

    # ---------------------------------------------------------------
    # Called when each camera image arrives — spawns background thread
    # so multiple cameras process IN PARALLEL
    # ---------------------------------------------------------------
    def on_image_received(self, msg, camera_id):
        with self.camera_results_lock:
            if camera_id in self.camera_results:
                self.get_logger().warn(f'Camera {camera_id} already processed, skipping duplicate')
                return

        self.get_logger().info(f'Image received from camera {camera_id}')

        try:
            parts = msg.header.frame_id.split(',')
            if self.overall_start is None:
                self.overall_start = float(parts[0])
            cam_start = float(parts[1]) if len(parts) > 1 else self.overall_start
        except Exception:
            cam_start = time.time()
            if self.overall_start is None:
                self.overall_start = cam_start

        thread = threading.Thread(
            target=self._process_and_collect,
            args=(msg, camera_id, cam_start),
            daemon=True
        )
        thread.start()

    # ---------------------------------------------------------------
    # Background worker — runs process_single_image per camera
    # ---------------------------------------------------------------
    def _process_and_collect(self, msg, camera_id, cam_start):
        result = self.process_single_image(msg, camera_id, cam_start)

        with self.camera_results_lock:
            self.camera_results[camera_id] = result
            count = len(self.camera_results)
            should_fuse = (
                (self.num_cameras == 1 and count == 1) or
                (self.num_cameras > 1 and count == self.num_cameras)
            )
            if should_fuse and not self._fuse_triggered:
                self._fuse_triggered = True
            else:
                should_fuse = False

        self.get_logger().info(
            f'Camera {camera_id} processed — {count}/{self.num_cameras} done'
        )

        if should_fuse:
            if self.num_cameras > 1:
                self.get_logger().info('All camera images received — fusing now')
            self.fuse_and_publish()

    # ---------------------------------------------------------------
    # Called when Pi signals all cameras done (multi-camera only)
    # Fallback fuse in case not all images arrived via _process_and_collect
    # ---------------------------------------------------------------
    def on_batch_complete(self, msg):
        if self.batch_already_processed:
            self.batch_already_processed = False
            return

        self.get_logger().info('Batch complete signal received from Pi')
        try:
            parts = msg.data.split(',')
            self.batch_save_dir = parts[0]
            if self.overall_start is None and len(parts) > 1:
                self.overall_start = float(parts[1])
        except Exception:
            pass

        # Poll until all cameras are processed or timeout reached
        # This avoids racing against OCR threads still running
        timeout = 3.0   # max seconds to wait for all cameras
        interval = 0.1  # poll every 100ms
        elapsed = 0.0

        while elapsed < timeout:
            with self.camera_results_lock:
                received = len(self.camera_results)
                already_fused = self._fuse_triggered

            if already_fused:
                # _process_and_collect already handled all cameras — do nothing
                self.get_logger().info('Batch complete: already fused by process threads')
                return

            if received == self.num_cameras:
                # All cameras done — fuse will be triggered by _process_and_collect
                self.get_logger().info(f'Batch complete: all {self.num_cameras} cameras received')
                return

            time.sleep(interval)
            elapsed += interval

        # Timeout reached — fuse whatever we have
        with self.camera_results_lock:
            received = len(self.camera_results)
            already_fused = self._fuse_triggered

        self.get_logger().info(f'Batch complete timeout: {received}/{self.num_cameras} images received')

        if already_fused:
            return

        if received == 0:
            self.get_logger().error('No images received — check Pi publisher')
            return

        if received < self.num_cameras:
            self.get_logger().warn(
                f'Only {received}/{self.num_cameras} cameras received after timeout — fusing partial'
            )

        with self.camera_results_lock:
            if not self._fuse_triggered:
                self._fuse_triggered = True
                do_fuse = True
            else:
                do_fuse = False

        if do_fuse:
            self.fuse_and_publish()

    # ---------------------------------------------------------------
    # Detect box corner to split left/right faces
    # ---------------------------------------------------------------
    def find_split_column(self, gray):
        edges = cv2.Canny(gray, 50, 150)
        col_sums = np.sum(edges, axis=0)

        width = gray.shape[1]
        search_start = width // 3
        search_end = 2 * width // 3
        mid_region = col_sums[search_start:search_end]

        split_col = search_start + int(np.argmax(mid_region))

        if split_col < width * 0.25 or split_col > width * 0.75:
            self.get_logger().warn('Split detection uncertain — using center split')
            split_col = width // 2

        self.get_logger().info(f'Detected split column: {split_col} / {width}')
        return split_col

    # ---------------------------------------------------------------
    # Perspective correction
    # SKEW: left=0.15, right=0.25 (right face is steeper angle)
    # ---------------------------------------------------------------
    def correct_perspective(self, face_img, is_left_face=True):
        h, w = face_img.shape[:2]
        SKEW = 0.20 if is_left_face else 0.25

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])

        if is_left_face:
            dst = np.float32([
                [w * SKEW, 0],
                [w,        0],
                [w,        h],
                [w * SKEW, h]
            ])
        else:
            dst = np.float32([
                [0,              0],
                [w * (1 - SKEW), 0],
                [w * (1 - SKEW), h],
                [0,              h]
            ])

        M = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(face_img, M, (w, h))

        if is_left_face:
            corrected = corrected[:, int(w * SKEW):]
        else:
            corrected = corrected[:, :int(w * (1 - SKEW))]

        return corrected

    # ---------------------------------------------------------------
    # Detect ALL code types on an image
    # Pass 1 — pyzbar on original + Otsu  → linear barcodes
    # Pass 2 — QR crop 2x upscale         → corrected face QR
    # Pass 3 — raw face 3x fallback        → raw face QR fallback
    # ---------------------------------------------------------------
    def detect_codes(self, img, label='', raw_img=None):
        results = []
        seen_data = set()

        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Pass 1
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for test_img in [gray, otsu]:
            for b in pyzbar.decode(test_img):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar: {b.type} = {data}')

        # Pass 2: QR crop 2x
        if not any(r['type'] == 'QRCODE' for r in results):
            h, w = gray.shape
            qr_crop = gray[int(h * 0.3):, :int(w * 0.8)]
            large = cv2.resize(qr_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for b in pyzbar.decode(thresh):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar QR-crop 2x: {b.type} = {data}')

        # Pass 3: raw face 3x fallback
        if not any(r['type'] == 'QRCODE' for r in results) and raw_img is not None:
            raw_gray = raw_img if len(raw_img.shape) == 2 else cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            large_raw = cv2.resize(raw_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, thresh_raw = cv2.threshold(large_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for b in pyzbar.decode(thresh_raw):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar raw 3x: {b.type} = {data}')

        cleaned = []
        for r in results:
            if r['type'] == 'QRCODE':
                cleaned.append(r)
            elif re.match(r'^[A-Za-z0-9\-]+$', r['data']):
                cleaned.append(r)

        return cleaned

    # ---------------------------------------------------------------
    # Select best barcode — prefer alphanumeric + dash
    # ---------------------------------------------------------------
    def best_barcode(self, barcodes):
        for b in barcodes:
            if re.match(r'^[A-Za-z0-9\-]+$', b['data']):
                return b
        return barcodes[0] if barcodes else None

    # ---------------------------------------------------------------
    # Process a single face — codes + OCR
    # Extracted so it can be called in parallel via ThreadPoolExecutor
    # ---------------------------------------------------------------
    def process_face(self, camera_id, face_label, full_corrected, raw_face,
                     is_left, ocr_region, global_barcodes, global_qr):

        # Code detection
        code_start = time.time()
        if is_left:
            qr_codes = [c for c in self.detect_codes(
                full_corrected,
                label=f'cam{camera_id}-{face_label}',
                raw_img=raw_face
            ) if c['type'] == 'QRCODE']
            barcode_results = global_barcodes + qr_codes
        else:
            per_face = self.detect_codes(
                full_corrected,
                label=f'cam{camera_id}-{face_label}',
                raw_img=raw_face
            )
            per_face_data = {r['data'] for r in per_face}
            barcode_results = per_face + [q for q in global_qr if q['data'] not in per_face_data]
        code_time = time.time() - code_start

        # OCR — PSM 11 first, fallback to PSM 6, then PSM 3 only if zero words
        ocr_start = time.time()
        clean_text = self.extract_clean_text(ocr_region, psm=11, min_conf=40)

        if not clean_text or len(clean_text.split()) < 2:
            clean_text_psm6 = self.extract_clean_text(ocr_region, psm=6, min_conf=40)
            if clean_text_psm6 and len(clean_text_psm6.split()) > len((clean_text or '').split()):
                clean_text = clean_text_psm6
                self.get_logger().info(f'Cam {camera_id} {face_label}: PSM 6 improved OCR')

        # PSM 3 only if still completely empty
        if not clean_text:
            clean_text_psm3 = self.extract_clean_text(ocr_region, psm=3, min_conf=35)
            if clean_text_psm3:
                clean_text = clean_text_psm3
                self.get_logger().info(f'Cam {camera_id} {face_label}: PSM 3 improved OCR')

        if clean_text:
            clean_text = self.filter_barcode_text(clean_text)

        ocr_time = time.time() - ocr_start

        self.get_logger().info(
            f'  Face [{face_label}] Codes:    '
            f'{barcode_results if barcode_results else "None"} ({code_time:.3f}s)'
        )
        self.get_logger().info(
            f'  Face [{face_label}] OCR:      '
            f'{clean_text if clean_text else "None"} ({ocr_time:.3f}s)'
        )

        return {
            'face': face_label,
            'ocr_text': clean_text,
            'barcodes': barcode_results,
        }

    # ---------------------------------------------------------------
    # Process a single camera image
    # Left + right faces processed IN PARALLEL via ThreadPoolExecutor
    # ---------------------------------------------------------------
    def process_single_image(self, msg, camera_id, cam_start):
        processing_start = time.time()

        # Decode
        decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        decode_time = time.time() - decode_start

        cv2.imwrite(f'/tmp/camera_{camera_id}_received.jpg', frame)

        # Grayscale + downscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        if width > 1280:
            scale = 1280 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            width  = gray.shape[1]
            height = gray.shape[0]

        # Split into left/right faces with 5% overlap
        split_start = time.time()
        split_col = self.find_split_column(gray)
        overlap    = int(width * 0.05)
        left_face  = gray[:, :split_col + overlap]
        right_face = gray[:, split_col:]
        split_time = time.time() - split_start

        # Perspective correction
        left_corrected  = self.correct_perspective(left_face,  is_left_face=True)
        right_corrected = self.correct_perspective(right_face, is_left_face=False)

        # OCR regions
        left_h,  left_w  = left_corrected.shape[:2]
        right_h, right_w = right_corrected.shape[:2]
        left_ocr_region  = left_corrected[:int(left_h  * 0.70), :]
        right_ocr_region = right_corrected[:int(right_h * 0.45), :int(right_w * 0.75)]

        # Save debug images
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_left_corrected.jpg',  left_corrected)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_right_corrected.jpg', right_corrected)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_left_ocr.jpg',        left_ocr_region)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_right_ocr.jpg',       right_ocr_region)

        # Global barcode scan BEFORE warping — warp distorts barcodes
        global_codes    = self.detect_codes(gray, label=f'cam{camera_id}-full')
        global_barcodes = [c for c in global_codes if c['type'] != 'QRCODE']
        global_qr       = [c for c in global_codes if c['type'] == 'QRCODE']
        self.get_logger().info(f'Global barcode scan: {global_barcodes if global_barcodes else "None"}')
        self.get_logger().info(f'Global QR scan: {global_qr if global_qr else "None"}')

        # ---------------------------------------------------------------
        # Process LEFT and RIGHT faces IN PARALLEL
        # Saves ~0.4-0.5s per camera vs sequential face processing
        # ---------------------------------------------------------------
        proc_start = time.time()

        face_configs = [
            ('left',  left_corrected,  left_face,  True,  left_ocr_region),
            ('right', right_corrected, right_face, False, right_ocr_region),
        ]

        face_results = [None, None]

        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_idx = {
                executor.submit(
                    self.process_face,
                    camera_id,
                    face_label,
                    full_corrected,
                    raw_face,
                    is_left,
                    ocr_region,
                    global_barcodes,
                    global_qr
                ): idx
                for idx, (face_label, full_corrected, raw_face, is_left, ocr_region)
                in enumerate(face_configs)
            }
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    face_results[idx] = future.result()
                except Exception as e:
                    face_label = face_configs[idx][0]
                    self.get_logger().error(f'Face {face_label} processing error: {str(e)}')
                    face_results[idx] = {
                        'face': face_configs[idx][0],
                        'ocr_text': None,
                        'barcodes': [],
                    }

        proc_time = time.time() - proc_start
        processing_time = time.time() - processing_start

        # Merge results (left first, then right)
        all_ocr_parts = [f['ocr_text'] for f in face_results if f and f['ocr_text']]
        all_barcodes  = [b for f in face_results if f for b in f['barcodes']]
        merged_ocr    = ' '.join(all_ocr_parts) if all_ocr_parts else None

        self.get_logger().info(f'=== CAMERA {camera_id} OCR RESULTS (2-face + perspective) ===')
        self.get_logger().info(f'  Split col:  {split_col}px  overlap={overlap}px  ({split_time:.3f}s)')
        self.get_logger().info(f'  Codes:      {all_barcodes if all_barcodes else "None"}')
        self.get_logger().info(f'  OCR Text:   {merged_ocr if merged_ocr else "None"}')
        self.get_logger().info(f'  Decode:     {decode_time:.3f}s')
        self.get_logger().info(f'  Split:      {split_time:.3f}s')
        self.get_logger().info(f'  Processing: {proc_time:.3f}s  ← left+right ran in parallel')
        self.get_logger().info(f'  Total:      {processing_time:.3f}s')

        return {
            'camera_id': camera_id,
            'ocr_text': merged_ocr,
            'barcodes': all_barcodes,
            'faces': face_results,
            'split_col': split_col,
            'timing': {
                'decode': decode_time,
                'split': split_time,
                'processing': proc_time,
                'total': processing_time
            }
        }

    # ---------------------------------------------------------------
    # Fuse results from all cameras and publish
    # ---------------------------------------------------------------
    def fuse_and_publish(self):
        fuse_start = time.time()

        with self.camera_results_lock:
            results_snapshot = dict(self.camera_results)

        all_ocr_parts = []
        for cam_id, result in results_snapshot.items():
            if result['ocr_text']:
                all_ocr_parts.append(result['ocr_text'])

        fused_barcode = None
        for cam_id, result in results_snapshot.items():
            if result['barcodes']:
                fused_barcode = self.best_barcode(result['barcodes'])
                self.get_logger().info(f'Best code from camera {cam_id}: {fused_barcode}')
                break

        fused_ocr = ' '.join(all_ocr_parts) if all_ocr_parts else None
        fuse_time = time.time() - fuse_start
        end_to_end = time.time() - self.overall_start if self.overall_start else None

        per_camera_summary = {}
        for cam_id, result in results_snapshot.items():
            per_camera_summary[cam_id] = {
                'ocr_text': result['ocr_text'],
                'barcodes': result['barcodes'],
            }

        self.get_logger().info('=== FUSED RESULT ===')
        self.get_logger().info(f'  Mode:                   {self.num_cameras}-camera')
        self.get_logger().info(f'  OCR Text (all cameras): {fused_ocr}')
        self.get_logger().info(f'  Best code:              {fused_barcode}')
        self.get_logger().info(f'  Cameras with OCR:       {len(all_ocr_parts)}/{len(results_snapshot)}')
        self.get_logger().info(f'  Fuse time:              {fuse_time:.3f}s')
        if end_to_end:
            self.get_logger().info(f'  END-TO-END TIME:        {end_to_end:.3f}s')
        self.get_logger().info('====================')

        result_msg = json.dumps({
            'ocr_text': fused_ocr,
            'barcode': fused_barcode,
            'mode': f'{self.num_cameras}-camera',
            'cameras_received': len(results_snapshot),
            'per_camera': per_camera_summary,
            'overall_start': self.overall_start,
            'end_to_end_time': end_to_end,
            'save_dir': self.batch_save_dir
        })
        self.result_publisher.publish(String(data=result_msg))
        self.get_logger().info('Fused result published to /ocr_results')

        # Reset for next batch
        with self.camera_results_lock:
            self.batch_already_processed = True
            self.camera_results = {}
            self._fuse_triggered = False
        self.overall_start = None
        self.batch_save_dir = None

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    OCR_CORRECTIONS = {
        # Gloss misreads
        'Gass':   'Gloss',
        'Goss':   'Gloss',
        'Gloss':  'Gloss',
        'Gross':  'Gloss',
        # Stain misreads
        'Stam':   'Stain',
        'Stal':   'Stain',
        'Stan':   'Stain',
        'Stain':  'Stain',
        # Cream misreads
        'Crean':  'Cream',
        'Cram':   'Cream',
        'Crear':  'Cream',
        'Crearn': 'Cream',
        # Lip misreads
        'Lin':    'Lip',
        'Lp':     'Lip',
        'Liq':    'Lip',
        # SEPHORA misreads — truncation and mixed case
        'SEPHOR': 'SEPHORA',
        'SEPHOF': 'SEPHORA',
        'SEPHO':  'SEPHORA',
        'sEPHORA':'SEPHORA',
        'sEPHOR': 'SEPHORA',
        'Sephora':'SEPHORA',
        'SEPORA': 'SEPHORA',
        'SEPHORS':'SEPHORA',
    }

    def extract_clean_text(self, image, psm=11, min_conf=40):
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # CLAHE — improves local contrast for Tesseract
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray)

        best_result = []

        for img_variant in [gray, enhanced]:
            data = pytesseract.image_to_data(
                img_variant,
                config=f'--oem 1 --psm {psm}',
                output_type=pytesseract.Output.DICT
            )
            clean = []
            for word, conf in zip(data['text'], data['conf']):
                word = word.strip()
                if int(conf) > min_conf and len(word) >= 2:
                    if not re.match(r'^[A-Za-z0-9]+$', word):
                        continue
                    # Apply correction before length check — fixes short misreads
                    word = self.OCR_CORRECTIONS.get(word, word)
                    # After correction, enforce minimum 3 chars
                    if len(word) < 3:
                        continue
                    clean.append(word)
            if len(clean) > len(best_result):
                best_result = clean

        return ' '.join(best_result)

    # Short noise words Tesseract commonly misreads from edges/tape
    OCR_NOISE_WORDS = {'ill', 'lll', 'lil', 'llI', 'III', 'iil', 'lli'}

    def filter_barcode_text(self, text):
        # Remove barcode-format strings
        text = re.sub(r'\b[A-Za-z]+-[A-Za-z]+-\d+\b', '', text)
        # Remove known noise words
        words = text.split()
        words = [w for w in words if w not in self.OCR_NOISE_WORDS]
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        return text


def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()