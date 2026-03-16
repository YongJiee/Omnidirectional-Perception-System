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

# ---------------------------------------------------------------
# Quantity extraction — raw OCR pass only.
# ocr_node does NOT decide flagging — that is handled in
# database_matcher_node using resolve_quantity(ocr_text, scan_mode).
# ocr_node only reports whether a qty pattern was found or not.
# ---------------------------------------------------------------
QTY_PATTERNS = [
    r'[Qq]uantity[\s:]*(\d+)',
    r'[Qq][Tt][Yy][\s:\.]*(\d+)',
    r'\bCTN\s+OF\s+(\d+)\b',
    r'\b(\d+)\s*[Pp][Cc][Ss]\b',
    r'\b[Xx]\s*(\d+)\b',
    r'\b(\d+)\s*[Uu][Nn][Ii][Tt][Ss]?\b',
    r'\baty[\s:\.]*(\d+)',
]

def extract_quantity_raw(ocr_text):
    """
    Raw quantity extraction — returns (qty, 'ocr') if found, else (None, None).
    Does NOT apply inbound/sorting fallback rules — that is done in matcher node.
    """
    if not ocr_text:
        return None, None
    for pattern in QTY_PATTERNS:
        match = re.search(pattern, ocr_text)
        if match:
            qty = int(match.group(1))
            if 1 <= qty <= 9999:
                return qty, 'ocr'
    return None, None


class OCRProcessor(Node):
    def __init__(self):
        super().__init__('ocr_node')
        # ---------------------------------------------------------------
        # Mode selection
        # ---------------------------------------------------------------
        self.declare_parameter('num_cameras', 1)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.get_logger().info(f'Camera count: {self.num_cameras}-camera')

        self.declare_parameter('scan_mode', 'sorting')
        self.scan_mode = self.get_parameter('scan_mode').get_parameter_value().string_value
        self.get_logger().info(f'Scan mode: {self.scan_mode}')

        self.camera_results = {}
        self.camera_results_lock = threading.Lock()
        self.overall_start = None
        self.batch_save_dir = None
        self.clock_offset = None 
        self.batch_already_processed = False
        self._fuse_triggered = False

        self.result_publisher = self.create_publisher(String, '/ocr_results', 10)

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

        if self.clock_offset is None:
            try:
                pi_time = float(msg.header.frame_id.split(',')[1])
                self.clock_offset = time.time() - pi_time
                self.get_logger().info(f'Clock offset: {self.clock_offset:.3f}s')
            except Exception:
                pass

        thread = threading.Thread(
            target=self._process_and_collect,
            args=(msg, camera_id, cam_start),
            daemon=True
        )
        thread.start()

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

    def on_batch_complete(self, msg):
        if self.batch_already_processed:
            self.get_logger().info('Batch already processed — ignoring duplicate signal')
            return

        self.get_logger().info('Batch complete signal received from Pi')
        try:
            parts = msg.data.split(',')
            self.batch_save_dir = parts[0]
            if self.overall_start is None and len(parts) > 1:
                self.overall_start = float(parts[1])
            self.pi_cycle_time = float(parts[2]) if len(parts) > 2 else None
        except Exception:
            pass

        timeout = max(3.0, self.num_cameras * 1.0)  # 3s for 3 cameras — Pi cycle ~2.5s
        interval = 0.1
        elapsed = 0.0

        while elapsed < timeout:
            with self.camera_results_lock:
                received = len(self.camera_results)
                already_fused = self._fuse_triggered
            if already_fused:
                self.get_logger().info('Batch complete: already fused by process threads')
                return
            if received == self.num_cameras:
                self.get_logger().info(f'Batch complete: all {self.num_cameras} cameras received')
                return
            time.sleep(interval)
            elapsed += interval

        with self.camera_results_lock:
            received = len(self.camera_results)
            already_fused = self._fuse_triggered

        self.get_logger().info(f'Batch complete timeout: {received}/{self.num_cameras} images received')

        if already_fused:
            return
        if received == 0:
            if self.batch_already_processed:
                return
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

    def correct_perspective(self, face_img, is_left_face=True):
        h, w = face_img.shape[:2]
        SKEW = 0.20 if is_left_face else 0.10
        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if is_left_face:
            dst = np.float32([
                [w * SKEW, 0], [w, 0], [w, h], [w * SKEW, h]
            ])
        else:
            dst = np.float32([
                [0, 0], [w * (1 - SKEW), 0], [w * (1 - SKEW), h], [0, h]
            ])
        M = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(face_img, M, (w, h))
        if is_left_face:
            corrected = corrected[:, int(w * SKEW):]
        else:
            corrected = corrected[:, :int(w * (1 - SKEW))]
        return corrected

    def detect_codes(self, img, label='', raw_img=None):
        results = []
        seen_data = set()
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for test_img in [gray, otsu]:
            for b in pyzbar.decode(test_img):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar: {b.type} = {data}')

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

    def best_barcode(self, barcodes):
        for b in barcodes:
            if re.match(r'^[A-Za-z0-9\-]+$', b['data']):
                return b
        return barcodes[0] if barcodes else None

    def process_face(self, camera_id, face_label, full_corrected, raw_face,
                     is_left, ocr_region, global_barcodes, global_qr):

        code_start = time.time()
        if is_left:
            qr_codes = [c for c in self.detect_codes(
                full_corrected, label=f'cam{camera_id}-{face_label}', raw_img=raw_face
            ) if c['type'] == 'QRCODE']
            barcode_results = global_barcodes + qr_codes
        else:
            per_face = self.detect_codes(
                full_corrected, label=f'cam{camera_id}-{face_label}', raw_img=raw_face
            )
            per_face_data = {r['data'] for r in per_face}
            barcode_results = per_face + [q for q in global_qr if q['data'] not in per_face_data]
        code_time = time.time() - code_start

        ocr_start = time.time()
        clean_text = self.extract_clean_text(ocr_region, psm=11, min_conf=40)

        if not clean_text or len(clean_text.split()) < 2:
            clean_text_psm6 = self.extract_clean_text(ocr_region, psm=6, min_conf=40)
            if clean_text_psm6 and len(clean_text_psm6.split()) > len((clean_text or '').split()):
                clean_text = clean_text_psm6
                self.get_logger().info(f'Cam {camera_id} {face_label}: PSM 6 improved OCR')

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

        return {'face': face_label, 'ocr_text': clean_text, 'barcodes': barcode_results}

    def process_single_image(self, msg, camera_id, cam_start):
        processing_start = time.time()
        network_latency = processing_start - cam_start  # time from Pi publish to WSL receive
        # ADD this guard
        if network_latency < 0 or network_latency > 10:
            network_latency = 0.0  # clocks not synced — ignore

        decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        gray = frame
        decode_time = time.time() - decode_start

        cv2.imwrite(f'/tmp/camera_{camera_id}_received.jpg', frame)

        height, width = gray.shape
        if width > 1280:
            scale = 1280 / width
            gray = cv2.resize(gray, None, fx=scale, fy=scale)
            width = gray.shape[1]
            height = gray.shape[0]

        split_start = time.time()
        split_col = self.find_split_column(gray)
        overlap = int(width * 0.05)
        left_face = gray[:, :split_col + overlap]
        right_face = gray[:, split_col:]
        split_time = time.time() - split_start

        left_corrected = self.correct_perspective(left_face, is_left_face=True)
        right_corrected = self.correct_perspective(right_face, is_left_face=False)

        left_h, left_w = left_corrected.shape[:2]
        right_h, right_w = right_corrected.shape[:2]
        left_ocr_region = left_corrected[:int(left_h * 0.70), int(left_w * 0.40):]
        right_ocr_region = right_corrected[:int(right_h * 0.80), :]

        # Full image OCR region — top 65% of full grayscale (for flat-facing cameras)
        full_ocr_region = gray[:int(height * 0.80), :]

        cv2.imwrite(f'/tmp/camera_{camera_id}_face_left_corrected.jpg', left_corrected)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_right_corrected.jpg', right_corrected)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_left_ocr.jpg', left_ocr_region)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_right_ocr.jpg', right_ocr_region)

        global_codes = self.detect_codes(gray, label=f'cam{camera_id}-full')
        global_barcodes = [c for c in global_codes if c['type'] != 'QRCODE']
        global_qr = [c for c in global_codes if c['type'] == 'QRCODE']
        self.get_logger().info(f'Global barcode scan: {global_barcodes if global_barcodes else "None"}')
        self.get_logger().info(f'Global QR scan: {global_qr if global_qr else "None"}')

        proc_start = time.time()
        face_configs = [
            ('left', left_corrected, left_face, True, left_ocr_region),
            ('right', right_corrected, right_face, False, right_ocr_region),
        ]
        face_results = [None, None]

        # ── Run left face, right face, and full image OCR in parallel ──
        def run_full_ocr(region):
            text = self.extract_clean_text(region, psm=11, min_conf=40)
            if not text or len(text.split()) < 2:
                text = self.extract_clean_text(region, psm=6, min_conf=40)
            if text:
                text = self.filter_barcode_text(text)
            return text

        with ThreadPoolExecutor(max_workers=3) as executor:
            future_to_idx = {
                executor.submit(
                    self.process_face, camera_id, face_label, full_corrected,
                    raw_face, is_left, ocr_region, global_barcodes, global_qr
                ): idx
                for idx, (face_label, full_corrected, raw_face, is_left, ocr_region)
                in enumerate(face_configs)
            }
            full_future = executor.submit(run_full_ocr, full_ocr_region)

            for future in as_completed(list(future_to_idx.keys()) + [full_future]):
                if future is full_future:
                    continue
                idx = future_to_idx[future]
                try:
                    face_results[idx] = future.result()
                except Exception as e:
                    face_label = face_configs[idx][0]
                    self.get_logger().error(f'Face {face_label} processing error: {str(e)}')
                    face_results[idx] = {'face': face_configs[idx][0], 'ocr_text': None, 'barcodes': []}

            try:
                full_ocr_text = full_future.result()
            except Exception:
                full_ocr_text = None

        proc_time = time.time() - proc_start
        processing_time = time.time() - processing_start

        all_ocr_parts = [f['ocr_text'] for f in face_results if f and f['ocr_text']]
        all_barcodes = [b for f in face_results if f for b in f['barcodes']]
        face_merged = ' '.join(all_ocr_parts) if all_ocr_parts else None

        # Score OCR candidates — prefer the one with more meaningful words
        # (avoids face split "winning" due to noise tokens like 'IRA', 'MTT')
        def ocr_score(text):
            if not text:
                return 0
            words = text.split()
            # Count only words >=3 chars (filters single-char noise and 2-char fragments)
            meaningful = [w for w in words if len(w) >= 3]
            return len(meaningful)

        full_score = ocr_score(full_ocr_text)
        face_score = ocr_score(face_merged)

        if full_ocr_text and full_score >= face_score:
            merged_ocr = full_ocr_text
            self.get_logger().info(f'  Full OCR:   {full_ocr_text} ← used (score {full_score} >= face {face_score})')
        else:
            merged_ocr = face_merged
            if full_ocr_text:
                self.get_logger().info(f'  Full OCR:   {full_ocr_text} (score {full_score} < face {face_score})')

        self.get_logger().info(f'=== CAMERA {camera_id} OCR RESULTS (2-face + full parallel) ===')
        self.get_logger().info(f'  Split col:  {split_col}px  overlap={overlap}px  ({split_time:.3f}s)')
        self.get_logger().info(f'  Codes:      {all_barcodes if all_barcodes else "None"}')
        self.get_logger().info(f'  OCR Text:   {merged_ocr if merged_ocr else "None"}')
        self.get_logger().info(f'  Decode:     {decode_time:.3f}s')
        self.get_logger().info(f'  Split:      {split_time:.3f}s')
        self.get_logger().info(f'  Processing: {proc_time:.3f}s  ← left+right+full ran in parallel')
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
                'total': processing_time,
                'network_latency': network_latency,
                'cam_start_offset': max(0.0, cam_start - (self.overall_start or cam_start)),
            }
        }

    # ---------------------------------------------------------------
    # Fuse results from all cameras and publish.
    # Quantity is extracted raw here — inbound/sorting flagging rules
    # are applied in database_matcher_node via resolve_quantity().
    # ---------------------------------------------------------------
    def fuse_and_publish(self):
        fuse_start = time.time()

        with self.camera_results_lock:
            results_snapshot = dict(self.camera_results)

        all_ocr_parts = []
        for cam_id, result in results_snapshot.items():
            if result['ocr_text']:
                all_ocr_parts.append(result['ocr_text'])

        all_barcodes_by_cam = {}
        for cam_id, result in results_snapshot.items():
            if result['barcodes']:
                all_barcodes_by_cam[cam_id] = result['barcodes']

        fused_ocr = ' '.join(all_ocr_parts) if all_ocr_parts else None
        fuse_time = time.time() - fuse_start
        if self.overall_start and self.clock_offset:
            end_to_end = time.time() - (self.overall_start + self.clock_offset)
        else:
            end_to_end = None
        # ADD this guard
        if end_to_end and end_to_end > 30:
            end_to_end = None  # clock mismatch — cannot measure cross-device e2e
            self.get_logger().warn('End-to-end timing unavailable — Pi/WSL clocks not synced')

        # ── Raw quantity extraction — no stage rules applied here ──
        qty_raw, qty_source_raw = extract_quantity_raw(fused_ocr or '')
        if qty_raw:
            self.get_logger().info(f'  Quantity (raw):         {qty_raw} (source: ocr)')
        else:
            self.get_logger().info(
                f'  Quantity (raw):         not found — '
                f'matcher will apply {"flagged" if self.scan_mode == "inbound" else "default=1"} rule'
            )

        per_camera_summary = {}
        for cam_id, result in results_snapshot.items():
            t = result.get('timing', {})
            per_camera_summary[cam_id] = {
                'ocr_text': result['ocr_text'],
                'barcodes': result['barcodes'],
                'timing': {
                    'total': t.get('total', 0.0),
                    'network_latency': t.get('network_latency', 0.0),
                    'cam_start_offset': t.get('cam_start_offset', 0.0),
                }
            }

        fused_barcode = None
        if self.scan_mode == 'sorting' and all_barcodes_by_cam:
            for cam_id, barcodes in all_barcodes_by_cam.items():
                fused_barcode = self.best_barcode(barcodes)
                if fused_barcode:
                    self.get_logger().info(
                        f'  Best code (sorting):    {fused_barcode} from cam {cam_id}'
                    )
                    break

        self.get_logger().info('=== FUSED RESULT ===')
        self.get_logger().info(f'  Scan mode:              {self.scan_mode}')
        self.get_logger().info(f'  Camera count:           {self.num_cameras}-camera')
        self.get_logger().info(f'  OCR Text (all cameras): {fused_ocr}')
        self.get_logger().info(f'  Best code:              {fused_barcode}')
        self.get_logger().info(f'  Barcodes by camera:     {all_barcodes_by_cam}')
        self.get_logger().info(f'  Cameras with OCR:       {len(all_ocr_parts)}/{len(results_snapshot)}')
        self.get_logger().info(f'  Fuse time:              {fuse_time:.3f}s')
        if end_to_end:
            self.get_logger().info(f'  END-TO-END TIME:        {end_to_end:.3f}s')
        self.get_logger().info('====================')

        # ── Pipeline timing breakdown ─────────────────────────────────
        self.get_logger().info('')
        self.get_logger().info('=== PIPELINE TIMING BREAKDOWN ===')
        self.get_logger().info('  [Pi -- multi_camera_publisher]  (capture timing: see Pi log)')
        for cam_id in sorted(results_snapshot.keys()):
            t = results_snapshot[cam_id].get('timing', {})
            offset = t.get('cam_start_offset', 0.0)
            net = t.get('network_latency', 0.0)
            self.get_logger().info(
                f'    Cam {cam_id}: published at +{offset:.3f}s  '
                f'| network delay to WSL: {net:.3f}s'
            )
        self.get_logger().info('')
        self.get_logger().info('  [WSL -- ocr_node  (cameras processed in parallel)]')
        slowest_ocr = 0.0
        for cam_id in sorted(results_snapshot.keys()):
            t = results_snapshot[cam_id].get('timing', {})
            ocr_t = t.get('total', 0.0)
            slowest_ocr = max(slowest_ocr, ocr_t)
            has_bc = 'BC:yes' if results_snapshot[cam_id].get('barcodes') else 'BC:no'
            has_ocr = 'OCR:yes' if results_snapshot[cam_id].get('ocr_text') else 'OCR:no'
            self.get_logger().info(
                f'    Cam {cam_id} OCR+barcode: {ocr_t:.3f}s  [{has_ocr}  {has_bc}]'
            )
        self.get_logger().info(f'    ocr_node wall-clock (slowest cam): {slowest_ocr:.3f}s')
        self.get_logger().info('')
        self.get_logger().info('  [WSL -- database_matcher]')
        self.get_logger().info('    Matcher processing: ~0.022s  (see matcher log for exact)')
        self.get_logger().info('')
        if end_to_end:
            status = 'PASS' if end_to_end <= 3.0 else f'FAIL  (+{end_to_end - 3.0:.3f}s over target)'
            self.get_logger().info(f'  TOTAL end-to-end: {end_to_end:.3f}s  [{status}]')
        self.get_logger().info('=================================')

        result_msg = json.dumps({
            'ocr_text': fused_ocr,
            'barcode': fused_barcode,
            'all_barcodes_by_cam': {str(k): v for k, v in all_barcodes_by_cam.items()},
            'scan_mode': self.scan_mode,
            'mode': f'{self.num_cameras}-camera',
            'cameras_received': len(results_snapshot),
            'per_camera': per_camera_summary,
            'overall_start': self.overall_start,
            'end_to_end_time': end_to_end,
            'save_dir': self.batch_save_dir,
            'quantity_raw': qty_raw,           # raw OCR value, may be None
            'quantity_source_raw': qty_source_raw,  # 'ocr' or None
            'clock_offset': self.clock_offset,
            'pi_cycle_time': self.pi_cycle_time,
        })
        self.result_publisher.publish(String(data=result_msg))
        self.get_logger().info('Fused result published to /ocr_results')

        with self.camera_results_lock:
            self.batch_already_processed = True
            self.camera_results = {}
            self._fuse_triggered = False
        self.overall_start = None
        self.batch_save_dir = None
        self.clock_offset = None 
        self.pi_cycle_time = None
        self.pi_cycle_time = None

    # ---------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------
    OCR_CORRECTIONS = {
        'Gass': 'Gloss', 'Goss': 'Gloss', 'Gloss': 'Gloss', 'Gross': 'Gloss',
        'Glosg': 'Gloss', 'Glogs': 'Gloss',
        'Stam': 'Stain', 'Stal': 'Stain', 'Stan': 'Stain', 'Stain': 'Stain',
        'Crean': 'Cream', 'Cram': 'Cream', 'Crear': 'Cream', 'Crearn': 'Cream',
        'Team': 'Cream', 'TEAM': 'Cream', 'Tearn': 'Cream', 'ream': 'Cream',
        'Lin': 'Lip', 'Lp': 'Lip', 'Liq': 'Lip', 'Li': 'Lip',
        'Crea': 'Cream',
        'SEPHOR': 'SEPHORA', 'SEPHOF': 'SEPHORA', 'SEPHO': 'SEPHORA',
        'sEPHORA': 'SEPHORA', 'sEPHOR': 'SEPHORA', 'Sephora': 'SEPHORA',
        'SEPORA': 'SEPHORA', 'SEPHORS': 'SEPHORA',
        'aty:': 'Qty:', 'aty': 'Qty', 'Oty:': 'Qty:', 'Oty': 'Qty',
        'qty:': 'Qty:', 'QTY:': 'Qty:',
        'FRiDays': 'FRIDAYS','FRiDaYs': 'FRIDAYS','Fridays': 'FRIDAYS',
        'FRIDAS': 'FRIDAYS',
    }

    def extract_clean_text(self, image, psm=11, min_conf=40):
        gray = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
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
                if int(conf) > min_conf and (len(word) >= 2 or word.isdigit()):
                    if not re.match(r'^[A-Za-z0-9:]+$', word):
                        continue
                    word = self.OCR_CORRECTIONS.get(word, word)
                    if len(word) < 3 and not word.isdigit():
                        continue
                    clean.append(word)
            if len(clean) > len(best_result):
                best_result = clean
        return ' '.join(best_result)

    OCR_NOISE_WORDS = {
    'ill', 'lll', 'lil', 'llI', 'III', 'iil', 'lli',
    'ban', 'say', 'Fal', 'iif', 'cae', 'wil', 'eam',
    'aty', 'mtt', 'MTT', 'IRA', 'Lee', 'aig', 'ait',
    'Bee', 'ges', 'pig', 'wer', 'Ake', 'ant', 'bal',
    'pad', 'fig',
}

    def filter_barcode_text(self, text):
        text = re.sub(r'\b[A-Za-z]+-[A-Za-z]+-\d+\b', '', text)
        words = text.split()
        words = [w for w in words if w not in self.OCR_NOISE_WORDS]
        text = ' '.join(words)
        text = re.sub(r'\s+', ' ', text).strip()
        # Drop result if nothing meaningful remains (all short tokens)
        meaningful = [w for w in text.split() if len(w) >= 3 and not w.isdigit()]
        if not meaningful:
            return None
        return text if text else None


def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()