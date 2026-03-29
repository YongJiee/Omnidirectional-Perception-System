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

# ---------------------------------------------------------------------------
# QoS profile — must match the publisher (Pi / test_image_publisher) exactly.
# RELIABLE + KEEP_LAST(10) prevents image messages being dropped on the
# direct-ethernet link between Pi and WSL.
# ---------------------------------------------------------------------------
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# ---------------------------------------------------------------------------
# Raw quantity extraction — used only for reporting in the JSON payload.
# ocr_node does NOT decide 'flagged' vs 'default' — that is handled in
# database_matcher_node using resolve_quantity(ocr_text, scan_mode).
# This keeps stage-specific business logic out of the OCR layer.
# ---------------------------------------------------------------------------
QTY_PATTERNS = [
    r'[Qq]uantity[\s:]*(\d+)',
    r'[Qq][Tt][Yy][\s:\.]*(\d+)',
    r'\bCTN\s+OF\s+(\d+)\b',
    r'\b(\d+)\s*[Pp][Cc][Ss]\b',
    r'\b[Xx]\s*(\d+)\b',
    r'\b(\d+)\s*[Uu][Nn][Ii][Tt][Ss]?\b',
    r'\baty[\s:\.]*(\d+)',   # Common OCR misread of 'Qty'
]

def extract_quantity_raw(ocr_text):
    """
    Scans fused OCR text for a quantity value using QTY_PATTERNS.
    Returns (qty, 'ocr') if a valid number is found, else (None, None).
    Stage-specific fallback rules (flagged / default) are NOT applied here.
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
    """
    ROS2 node that receives compressed images from the Pi cameras,
    runs OCR and barcode detection, then publishes a fused JSON result
    to /ocr_results for database_matcher_node to consume.

    Processing pipeline per camera image:
      1. Decode JPEG → grayscale
      2. Find natural split column (Canny edge detection)
      3. Split into left / right face regions with overlap
      4. Correct perspective skew on each face
      5. Run OCR + barcode detection on left face, right face, and full
         image in parallel (ThreadPoolExecutor, 3 workers)
      6. Select best OCR text (face-merged vs full-image, scored by
         meaningful word count)

    After all cameras complete:
      7. Fuse all camera OCR texts into one string
      8. Extract raw quantity (reporting only)
      9. Publish JSON payload to /ocr_results

    Supports 1-camera mode (no batch_complete needed) and
    multi-camera mode (waits for /batch_complete signal).
    """

    def __init__(self):
        super().__init__('ocr_node')

        # ── Parameters ─────────────────────────────────────────────────────
        # num_cameras: how many camera topics to subscribe to (1 or 3+)
        self.declare_parameter('num_cameras', 1)
        self.num_cameras = self.get_parameter('num_cameras').get_parameter_value().integer_value
        self.get_logger().info(f'Camera count: {self.num_cameras}-camera')

        # scan_mode: 'inbound' or 'sorting' — passed through to matcher payload
        self.declare_parameter('scan_mode', 'sorting')
        self.scan_mode = self.get_parameter('scan_mode').get_parameter_value().string_value
        self.get_logger().info(f'Scan mode: {self.scan_mode}')

        # ── Batch state — shared across camera callbacks ────────────────────
        self.camera_results         = {}        # {camera_id: result_dict}
        self.camera_results_lock    = threading.Lock()  # Guards camera_results across threads
        self.overall_start          = None      # Pi-side batch start timestamp (from frame_id)
        self.batch_save_dir         = None      # Image folder path (from /batch_complete)
        self.clock_offset           = None      # Pi↔WSL clock difference, measured on first image
        self.batch_already_processed = False    # Prevents double-publish for the same batch
        self._fuse_triggered        = False     # Ensures fuse_and_publish() runs exactly once

        # ── Publisher — sends fused result to database_matcher_node ────────
        self.result_publisher = self.create_publisher(String, '/ocr_results', 10)

        # ── Subscribers — one per camera topic ─────────────────────────────
        self.subscriptions_ = []
        if self.num_cameras == 1:
            # Single-camera mode: subscribe to camera_0 only, no batch signal needed
            sub = self.create_subscription(
                CompressedImage,
                '/camera_0/image_raw/compressed',
                lambda msg: self.on_image_received(msg, camera_id=0),
                RELIABLE_QOS
            )
            self.subscriptions_.append(sub)
            self.get_logger().info('Subscribed to: /camera_0/image_raw/compressed')
        else:
            # Multi-camera mode: subscribe to all camera topics + /batch_complete
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
            # /batch_complete signals that the Pi has finished publishing all images
            self.batch_sub = self.create_subscription(
                String, '/batch_complete', self.on_batch_complete, RELIABLE_QOS)
            self.get_logger().info('Subscribed to: /batch_complete')

        self.get_logger().info(f'OCR Processor ready — waiting for {self.num_cameras} image(s)...')

    def on_image_received(self, msg, camera_id):
        """
        ROS2 callback — fires when a CompressedImage arrives on a camera topic.
        Extracts timing metadata from msg.header.frame_id, measures the
        Pi↔WSL clock offset on the first image of each batch, then spawns
        a daemon thread to process the image without blocking the ROS executor.

        frame_id format (set by multi_camera_publisher / test_image_publisher):
            '<overall_start>,<cam_start>,cam_<id>'
        """
        with self.camera_results_lock:
            if camera_id in self.camera_results:
                # Duplicate arrival (e.g. QoS retry) — skip to avoid double-processing
                self.get_logger().warn(f'Camera {camera_id} already processed, skipping duplicate')
                return

        self.get_logger().info(f'Image received from camera {camera_id}')

        # Parse timing metadata embedded in the frame_id header field
        try:
            parts = msg.header.frame_id.split(',')
            if self.overall_start is None:
                self.overall_start = float(parts[0])  # First camera sets overall_start
            cam_start = float(parts[1]) if len(parts) > 1 else self.overall_start
        except Exception:
            # Fallback: use current WSL time if frame_id is malformed
            cam_start = time.time()
            if self.overall_start is None:
                self.overall_start = cam_start

        # Measure Pi↔WSL clock offset on the first image of the batch.
        # offset = WSL_receive_time − Pi_cam_start_time
        # Applied in fuse_and_publish() to correct end-to-end timing.
        if self.clock_offset is None:
            try:
                pi_time = float(msg.header.frame_id.split(',')[1])
                self.clock_offset = time.time() - pi_time
                self.get_logger().info(f'Clock offset: {self.clock_offset:.3f}s')
            except Exception:
                pass  # Clock offset stays None — timing correction skipped

        # Spawn a daemon thread so heavy OCR work does not block the ROS spin loop
        thread = threading.Thread(
            target=self._process_and_collect,
            args=(msg, camera_id, cam_start),
            daemon=True
        )
        thread.start()

    def _process_and_collect(self, msg, camera_id, cam_start):
        """
        Runs in a background thread.
        Calls process_single_image(), stores the result, then checks
        whether all cameras have completed. If so, triggers fuse_and_publish()
        exactly once using the _fuse_triggered flag.
        """
        result = self.process_single_image(msg, camera_id, cam_start)

        with self.camera_results_lock:
            self.camera_results[camera_id] = result
            count = len(self.camera_results)
            # Trigger fuse when: single-camera mode and 1 result, OR all cameras done
            should_fuse = (
                (self.num_cameras == 1 and count == 1) or
                (self.num_cameras > 1 and count == self.num_cameras)
            )
            if should_fuse and not self._fuse_triggered:
                self._fuse_triggered = True  # Claim the fuse trigger (only one thread wins)
            else:
                should_fuse = False  # Another thread already triggered fuse

        self.get_logger().info(
            f'Camera {camera_id} processed — {count}/{self.num_cameras} done'
        )

        if should_fuse:
            if self.num_cameras > 1:
                self.get_logger().info('All camera images received — fusing now')
            self.fuse_and_publish()

    def on_batch_complete(self, msg):
        """
        /batch_complete callback (multi-camera mode only).
        Parses the batch save directory and Pi cycle time from the message,
        then waits up to `timeout` seconds for all camera results to arrive.

        This is a safety net — in normal operation, _process_and_collect()
        triggers fuse before this callback completes its wait loop.
        Falls back to partial fuse if not all cameras arrive within timeout.

        Message format: '<save_dir>,<overall_start>,<pi_cycle_time>'
        """
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

        # Wait for all camera processing threads to finish.
        # Timeout = 3s minimum, scaled by camera count (Pi cycle ~2.5s for 3 cameras)
        timeout  = max(3.0, self.num_cameras * 1.0)
        interval = 0.1
        elapsed  = 0.0

        while elapsed < timeout:
            with self.camera_results_lock:
                received      = len(self.camera_results)
                already_fused = self._fuse_triggered
            if already_fused:
                # Processing threads already triggered fuse — nothing to do
                self.get_logger().info('Batch complete: already fused by process threads')
                return
            if received == self.num_cameras:
                self.get_logger().info(f'Batch complete: all {self.num_cameras} cameras received')
                return
            time.sleep(interval)
            elapsed += interval

        # Timeout reached — check final state
        with self.camera_results_lock:
            received      = len(self.camera_results)
            already_fused = self._fuse_triggered

        self.get_logger().info(f'Batch complete timeout: {received}/{self.num_cameras} images received')

        if already_fused:
            return  # Fuse was triggered while we were checking
        if received == 0:
            if self.batch_already_processed:
                return
            self.get_logger().error('No images received — check Pi publisher')
            return
        if received < self.num_cameras:
            # Partial result — fuse what we have rather than silently dropping the scan
            self.get_logger().warn(
                f'Only {received}/{self.num_cameras} cameras received after timeout — fusing partial'
            )

        # Claim the fuse trigger and run (only if not already triggered)
        with self.camera_results_lock:
            if not self._fuse_triggered:
                self._fuse_triggered = True
                do_fuse = True
            else:
                do_fuse = False

        if do_fuse:
            self.fuse_and_publish()

    def find_split_column(self, gray):
        """
        Detects the natural vertical split between the two angled faces on a
        Cam0/Cam2 image using Canny edge detection.

        Searches only the middle third of the image width to avoid false splits
        at the frame edges. Falls back to the centre column if the detected
        split is outside the 25–75% width range (unreliable detection).

        Returns the split column index (pixel x-coordinate).
        """
        edges    = cv2.Canny(gray, 50, 150)
        col_sums = np.sum(edges, axis=0)
        width    = gray.shape[1]

        # Restrict search to middle third — edges near the frame border are noise
        search_start = width // 3
        search_end   = 2 * width // 3
        mid_region   = col_sums[search_start:search_end]
        split_col    = search_start + int(np.argmax(mid_region))

        if split_col < width * 0.25 or split_col > width * 0.75:
            self.get_logger().warn('Split detection uncertain — using center split')
            split_col = width // 2

        self.get_logger().info(f'Detected split column: {split_col} / {width}')
        return split_col

    def correct_perspective(self, face_img, is_left_face=True):
        """
        Corrects the perspective skew introduced by the 45° camera angle
        on Cam0 and Cam2. Uses cv2.getPerspectiveTransform to warp the
        face region toward a more front-on view.

        SKEW values were tuned empirically:
          Left face:  0.20 (steeper angle from camera mounting)
          Right face: 0.10 (shallower angle)

        The warped padding column is cropped out after transformation so
        the output image has no black borders.
        """
        h, w = face_img.shape[:2]
        SKEW = 0.20 if is_left_face else 0.10

        src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
        if is_left_face:
            # Shift left edge inward to de-skew the left face
            dst = np.float32([
                [w * SKEW, 0], [w, 0], [w, h], [w * SKEW, h]
            ])
        else:
            # Shift right edge inward to de-skew the right face
            dst = np.float32([
                [0, 0], [w * (1 - SKEW), 0], [w * (1 - SKEW), h], [0, h]
            ])

        M         = cv2.getPerspectiveTransform(src, dst)
        corrected = cv2.warpPerspective(face_img, M, (w, h))

        # Crop out the warped-in blank column introduced by the transform
        if is_left_face:
            corrected = corrected[:, int(w * SKEW):]
        else:
            corrected = corrected[:, :int(w * (1 - SKEW))]
        return corrected

    def detect_codes(self, img, label='', raw_img=None):
        """
        Attempts barcode and QR code detection on an image using pyzbar.
        Runs three passes to maximise detection rate on real warehouse images:

        Pass 1 — raw grayscale + Otsu threshold (standard)
        Pass 2 — QR-focused: crop bottom 70% × left 80%, upscale 2×, re-threshold
                 (handles small or partially obscured QR codes)
        Pass 3 — raw_img at 3× scale (only if QR still not found and raw image provided)
                 (fallback for very low-resolution captures)

        Deduplicates results by barcode data string.
        Filters out results with non-alphanumeric characters (noise rejection).

        Returns a list of {'type': ..., 'data': ...} dicts.
        """
        results   = []
        seen_data = set()
        gray = img if len(img.shape) == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Pass 1 — standard detection on grayscale and Otsu-thresholded image
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        for test_img in [gray, otsu]:
            for b in pyzbar.decode(test_img):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar: {b.type} = {data}')

        # Pass 2 — QR-focused crop at 2× scale (only if no QR found yet)
        if not any(r['type'] == 'QRCODE' for r in results):
            h, w        = gray.shape
            qr_crop     = gray[int(h * 0.3):, :int(w * 0.8)]
            large       = cv2.resize(qr_crop, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
            _, thresh   = cv2.threshold(large, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for b in pyzbar.decode(thresh):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar QR-crop 2x: {b.type} = {data}')

        # Pass 3 — raw image at 3× scale (fallback for low-res captures)
        if not any(r['type'] == 'QRCODE' for r in results) and raw_img is not None:
            raw_gray    = raw_img if len(raw_img.shape) == 2 else cv2.cvtColor(raw_img, cv2.COLOR_BGR2GRAY)
            large_raw   = cv2.resize(raw_gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
            _, thresh_raw = cv2.threshold(large_raw, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            for b in pyzbar.decode(thresh_raw):
                data = b.data.decode('utf-8')
                if data not in seen_data:
                    results.append({'type': b.type, 'data': data})
                    seen_data.add(data)
                    self.get_logger().info(f'  [{label}] pyzbar raw 3x: {b.type} = {data}')

        # Filter: keep QR codes always; keep barcodes only if alphanumeric-hyphen
        cleaned = []
        for r in results:
            if r['type'] == 'QRCODE':
                cleaned.append(r)
            elif re.match(r'^[A-Za-z0-9\-]+$', r['data']):
                cleaned.append(r)
        return cleaned

    def best_barcode(self, barcodes):
        """
        Selects the most trustworthy barcode from a list.
        Prefers barcodes whose data is purely alphanumeric-hyphen (clean reads).
        Falls back to the first entry if none match the clean pattern.
        Used in sorting mode to pick a single representative barcode.
        """
        for b in barcodes:
            if re.match(r'^[A-Za-z0-9\-]+$', b['data']):
                return b
        return barcodes[0] if barcodes else None

    def process_face(self, camera_id, face_label, full_corrected, raw_face,
                     is_left, ocr_region, global_barcodes, global_qr):
        """
        Processes one face (left or right) of a split camera image.
        Runs barcode detection and OCR on the face region, using global
        barcode results from the full-image scan to supplement per-face results.

        Left face strategy:
          - Only uses QR codes from per-face detection (barcodes already in global_barcodes)
          - Combines global_barcodes + per-face QR codes
        Right face strategy:
          - Runs full per-face detection (may catch barcodes missed globally)
          - Merges with global QR codes, deduplicating by data string

        OCR cascade (tries progressively broader PSM modes):
          1. PSM 11 (sparse text) — best for labels with scattered text
          2. PSM 6  (uniform block) — if PSM 11 gives < 2 words
          3. PSM 3  (auto)         — last resort if both above fail

        Returns {'face': label, 'ocr_text': str|None, 'barcodes': list}
        """
        code_start = time.time()
        if is_left:
            # Left face: global_barcodes already covers standard barcodes;
            # only add QR codes found specifically on this face
            qr_codes       = [c for c in self.detect_codes(
                full_corrected, label=f'cam{camera_id}-{face_label}', raw_img=raw_face
            ) if c['type'] == 'QRCODE']
            barcode_results = global_barcodes + qr_codes
        else:
            # Right face: run full detection, then supplement with any global QR not yet seen
            per_face      = self.detect_codes(
                full_corrected, label=f'cam{camera_id}-{face_label}', raw_img=raw_face
            )
            per_face_data = {r['data'] for r in per_face}
            barcode_results = per_face + [q for q in global_qr if q['data'] not in per_face_data]
        code_time = time.time() - code_start

        # OCR cascade — try PSM 11 first, fall back to 6 then 3
        ocr_start  = time.time()
        clean_text = self.extract_with_rotation(ocr_region, psm=11, min_conf=40)

        if not clean_text or len(clean_text.split()) < 2:
            clean_text_psm6 = self.extract_with_rotation(ocr_region, psm=6, min_conf=40)
            if clean_text_psm6 and len(clean_text_psm6.split()) > len((clean_text or '').split()):
                clean_text = clean_text_psm6
                self.get_logger().info(f'Cam {camera_id} {face_label}: PSM 6 improved OCR')

        if not clean_text:
            clean_text_psm3 = self.extract_with_rotation(ocr_region, psm=3, min_conf=35)
            if clean_text_psm3:
                clean_text = clean_text_psm3
                self.get_logger().info(f'Cam {camera_id} {face_label}: PSM 3 improved OCR')

        # Remove barcode-style strings from OCR text (format: ABC-abc-12345)
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
        """
        Full processing pipeline for one camera image.

        Steps:
          1. JPEG decode → grayscale (direct, no RGB→BGR conversion needed)
          2. Downscale to max 1280px wide if oversized
          3. Find vertical split column (Canny edge detection)
          4. Extract left/right face regions with 15% overlap to avoid edge cutoff
          5. Correct perspective skew on each face
          6. Define OCR regions (cropped sub-rectangles of each face + full image top 80%)
          7. Save debug images to /tmp/ for inspection
          8. Run global barcode scan on full grayscale image
          9. Process left face, right face, and full image OCR in parallel
             (ThreadPoolExecutor, 3 workers)
         10. Score and select best OCR result (face-merged vs full-image)
         11. Return result dict with OCR text, barcodes, and timing breakdown

        Debug images written to /tmp/ (intentional — aids tuning without disk overhead):
          camera_<id>_received.jpg
          camera_<id>_face_left_corrected.jpg
          camera_<id>_face_right_corrected.jpg
          camera_<id>_face_left_ocr.jpg
          camera_<id>_face_right_ocr.jpg
        """
        processing_start = time.time()
        # Network latency = time from Pi publish (cam_start) to WSL receive
        network_latency  = processing_start - cam_start
        # Clamp to 0 if clocks are not synced or latency looks unrealistic
        if network_latency < 0 or network_latency > 10:
            network_latency = 0.0

        # Step 1 — Decode JPEG bytes directly to grayscale
        # Pi sends RGB; decoding straight to GRAY avoids a wasted BGR conversion
        decode_start = time.time()
        np_arr = np.frombuffer(msg.data, np.uint8)
        frame  = cv2.imdecode(np_arr, cv2.IMREAD_GRAYSCALE)
        gray   = frame
        decode_time = time.time() - decode_start

        # Save received image for debug inspection (intentional /tmp/ write)
        cv2.imwrite(f'/tmp/camera_{camera_id}_received.jpg', frame)

        # Step 2 — Downscale oversized images to cap memory and processing time
        height, width = gray.shape
        if width > 1280:
            scale  = 1280 / width
            gray   = cv2.resize(gray, None, fx=scale, fy=scale)
            width  = gray.shape[1]
            height = gray.shape[0]

        # Step 3 — Find the vertical ridge between the two box faces
        split_start = time.time()
        split_col   = self.find_split_column(gray)

        # Step 4 — Split with 15% overlap to ensure labels near the boundary
        # are captured by both face regions (avoids text being cut in half)
        overlap    = int(width * 0.15)
        left_face  = gray[:, :split_col + overlap]
        right_face = gray[:, split_col:]
        split_time = time.time() - split_start

        # Step 5 — Correct perspective skew from 45° camera angle
        left_corrected  = self.correct_perspective(left_face,  is_left_face=True)
        right_corrected = self.correct_perspective(right_face, is_left_face=False)

        # Step 6 — Define OCR sub-regions
        left_h,  left_w  = left_corrected.shape[:2]
        right_h, right_w = right_corrected.shape[:2]
        # Left OCR: top 70% height, right 60% width (avoids blank left-side padding)
        left_ocr_region  = left_corrected[:int(left_h * 0.70), int(left_w * 0.40):]
        # Right OCR: top 80% height, full width
        right_ocr_region = right_corrected[:int(right_h * 0.80), :]
        # Full image OCR: top 80% of full grayscale (for flat-facing cameras like Cam1)
        full_ocr_region  = gray[:int(height * 0.80), :]

        # Step 7 — Save corrected and cropped regions to /tmp/ for visual debugging
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_left_corrected.jpg',  left_corrected)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_right_corrected.jpg', right_corrected)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_left_ocr.jpg',        left_ocr_region)
        cv2.imwrite(f'/tmp/camera_{camera_id}_face_right_ocr.jpg',       right_ocr_region)

        # Step 8 — Global barcode scan on full image before face-level processing
        global_codes    = self.detect_codes(gray, label=f'cam{camera_id}-full')
        global_barcodes = [c for c in global_codes if c['type'] != 'QRCODE']
        global_qr       = [c for c in global_codes if c['type'] == 'QRCODE']
        self.get_logger().info(f'Global barcode scan: {global_barcodes if global_barcodes else "None"}')
        self.get_logger().info(f'Global QR scan: {global_qr if global_qr else "None"}')

        # Step 9 — Parallel processing of left face, right face, and full image OCR
        proc_start   = time.time()
        face_configs = [
            ('left',  left_corrected,  left_face,  True,  left_ocr_region),
            ('right', right_corrected, right_face, False, right_ocr_region),
        ]
        face_results = [None, None]

        # Full-image OCR helper — tries PSM 11 then PSM 6 if result is weak
        def run_full_ocr(region):
            text = self.extract_with_rotation(region, psm=11, min_conf=40)
            if not text or len(text.split()) < 2:
                text = self.extract_with_rotation(region, psm=6, min_conf=40)
            if text:
                text = self.filter_barcode_text(text)
            return text

        # Run left face, right face, and full image OCR concurrently
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
                    continue  # Collect full_future result separately below
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

        proc_time       = time.time() - proc_start
        processing_time = time.time() - processing_start

        # Step 10 — Select the best OCR result
        all_ocr_parts = [f['ocr_text'] for f in face_results if f and f['ocr_text']]
        all_barcodes  = [b for f in face_results if f for b in f['barcodes']]
        face_merged   = ' '.join(all_ocr_parts) if all_ocr_parts else None

        # Score by meaningful word count (words ≥3 chars) to avoid noise tokens
        # winning over real text from face-split processing
        def ocr_score(text):
            if not text:
                return 0
            words       = text.split()
            # Exclude single/double-char fragments which are often OCR noise
            meaningful  = [w for w in words if len(w) >= 3]
            return len(meaningful)

        full_score = ocr_score(full_ocr_text)
        face_score = ocr_score(face_merged)

        if full_ocr_text and full_score >= face_score:
            # Full-image OCR won — typically happens for flat-facing Cam1 scans
            merged_ocr = full_ocr_text
            self.get_logger().info(f'  Full OCR:   {full_ocr_text} ← used (score {full_score} >= face {face_score})')
        else:
            # Face-split OCR won — typical for angled Cam0/Cam2 scans
            merged_ocr = face_merged
            if full_ocr_text:
                self.get_logger().info(f'  Full OCR:   {full_ocr_text} (score {full_score} < face {face_score})')

        # Step 11 — Log results and return
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
            'ocr_text':  merged_ocr,
            'barcodes':  all_barcodes,
            'faces':     face_results,
            'split_col': split_col,
            'timing': {
                'decode':           decode_time,
                'split':            split_time,
                'processing':       proc_time,
                'total':            processing_time,
                'network_latency':  network_latency,
                # cam_start_offset: time from overall batch start to this camera's publish
                'cam_start_offset': max(0.0, cam_start - (self.overall_start or cam_start)),
            }
        }

    # ---------------------------------------------------------------
    # Fuse results from all cameras and publish to /ocr_results.
    # Merges OCR text and barcodes from all camera results into a single
    # JSON payload for database_matcher_node.
    # Quantity is extracted raw (reporting only) — flagging rules applied
    # downstream in database_matcher_node via resolve_quantity().
    # Resets all batch state after publishing so the node is ready for
    # the next scan trigger.
    # ---------------------------------------------------------------
    def fuse_and_publish(self):
        fuse_start = time.time()

        # Snapshot results under lock so we don't read while another thread writes
        with self.camera_results_lock:
            results_snapshot = dict(self.camera_results)

        # Collect OCR text from all cameras that produced a result
        all_ocr_parts = []
        for cam_id, result in results_snapshot.items():
            if result['ocr_text']:
                all_ocr_parts.append(result['ocr_text'])

        # Build per-camera barcode map for cross-camera conflict detection in matcher
        all_barcodes_by_cam = {}
        for cam_id, result in results_snapshot.items():
            if result['barcodes']:
                all_barcodes_by_cam[cam_id] = result['barcodes']

        # Simple space-join — database_matcher_node's SmartMatcher handles deduplication
        fused_ocr = ' '.join(all_ocr_parts) if all_ocr_parts else None
        fuse_time = time.time() - fuse_start

        # Compute end-to-end time using clock offset to bridge Pi↔WSL timestamps
        if self.overall_start and self.clock_offset:
            end_to_end = time.time() - (self.overall_start + self.clock_offset)
        else:
            end_to_end = None
        # Sanity check: >30s means clocks are not synced — discard the measurement
        if end_to_end and end_to_end > 30:
            end_to_end = None
            self.get_logger().warn('End-to-end timing unavailable — Pi/WSL clocks not synced')

        # Raw quantity extraction — reporting only, no stage rules applied here
        qty_raw, qty_source_raw = extract_quantity_raw(fused_ocr or '')
        if qty_raw:
            self.get_logger().info(f'  Quantity (raw):         {qty_raw} (source: ocr)')
        else:
            self.get_logger().info(
                f'  Quantity (raw):         not found — '
                f'matcher will apply {"flagged" if self.scan_mode == "inbound" else "default=1"} rule'
            )

        # Build per-camera summary for timing breakdown in matcher log
        per_camera_summary = {}
        for cam_id, result in results_snapshot.items():
            t = result.get('timing', {})
            per_camera_summary[cam_id] = {
                'ocr_text': result['ocr_text'],
                'barcodes': result['barcodes'],
                'timing': {
                    'total':            t.get('total', 0.0),
                    'network_latency':  t.get('network_latency', 0.0),
                    'cam_start_offset': t.get('cam_start_offset', 0.0),
                }
            }

        # In sorting mode, pick the single best barcode across all cameras
        # (one item at a time — first clean barcode found wins)
        fused_barcode = None
        if self.scan_mode == 'sorting' and all_barcodes_by_cam:
            for cam_id, barcodes in all_barcodes_by_cam.items():
                fused_barcode = self.best_barcode(barcodes)
                if fused_barcode:
                    self.get_logger().info(
                        f'  Best code (sorting):    {fused_barcode} from cam {cam_id}'
                    )
                    break

        # ── Log fused result summary ──────────────────────────────────────
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

        # ── Pipeline timing breakdown (mirrors what matcher will log) ─────
        self.get_logger().info('')
        self.get_logger().info('=== PIPELINE TIMING BREAKDOWN ===')
        self.get_logger().info('  [Pi -- multi_camera_publisher]  (capture timing: see Pi log)')
        for cam_id in sorted(results_snapshot.keys()):
            t      = results_snapshot[cam_id].get('timing', {})
            offset = t.get('cam_start_offset', 0.0)
            net    = t.get('network_latency', 0.0)
            self.get_logger().info(
                f'    Cam {cam_id}: published at +{offset:.3f}s  '
                f'| network delay to WSL: {net:.3f}s'
            )
        self.get_logger().info('')
        self.get_logger().info('  [WSL -- ocr_node  (cameras processed in parallel)]')
        slowest_ocr = 0.0
        for cam_id in sorted(results_snapshot.keys()):
            t         = results_snapshot[cam_id].get('timing', {})
            ocr_t     = t.get('total', 0.0)
            slowest_ocr = max(slowest_ocr, ocr_t)
            has_bc    = 'BC:yes' if results_snapshot[cam_id].get('barcodes') else 'BC:no'
            has_ocr   = 'OCR:yes' if results_snapshot[cam_id].get('ocr_text') else 'OCR:no'
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

        # ── Build and publish JSON payload ────────────────────────────────
        # all_barcodes_by_cam keys are cast to str for JSON serialisation
        result_msg = json.dumps({
            'ocr_text':             fused_ocr,
            'barcode':              fused_barcode,           # Best single barcode (sorting) or None
            'all_barcodes_by_cam':  {str(k): v for k, v in all_barcodes_by_cam.items()},
            'scan_mode':            self.scan_mode,
            'mode':                 f'{self.num_cameras}-camera',
            'cameras_received':     len(results_snapshot),
            'per_camera':           per_camera_summary,
            'overall_start':        self.overall_start,      # Pi batch start timestamp
            'end_to_end_time':      end_to_end,              # Clock-corrected e2e, or None
            'save_dir':             self.batch_save_dir,
            'quantity_raw':         qty_raw,                 # Raw OCR qty value, may be None
            'quantity_source_raw':  qty_source_raw,          # 'ocr' or None
            'clock_offset':         self.clock_offset,       # Pi↔WSL clock difference
            'pi_cycle_time':        self.pi_cycle_time,      # Total Pi capture cycle duration
        })
        self.result_publisher.publish(String(data=result_msg))
        self.get_logger().info('Fused result published to /ocr_results')

        # ── Reset batch state for next scan ──────────────────────────────
        with self.camera_results_lock:
            self.batch_already_processed = True
            self.camera_results          = {}   # Clear results for next batch
            self._fuse_triggered         = False
        self.overall_start  = None
        self.batch_save_dir = None
        self.clock_offset   = None
        self.pi_cycle_time  = None

    # ---------------------------------------------------------------
    # OCR correction dictionary — maps common Tesseract misreads on
    # corrugated cardboard backgrounds to their correct values.
    # Applied word-by-word in extract_clean_text() after confidence filtering.
    # Entries were added iteratively during resilience testing (Groups A–F).
    # ---------------------------------------------------------------
    OCR_CORRECTIONS = {
        'Gass': 'Gloss', 'Goss': 'Gloss', 'Gloss': 'Gloss', 'Gross': 'Gloss',
        'Glosg': 'Gloss', 'Glogs': 'Gloss','Glo': 'Gloss', 'Glos': 'Gloss',
        'Stam': 'Stain', 'Stal': 'Stain', 'Stan': 'Stain', 'Stain': 'Stain',
        'Crean': 'Cream', 'Cram': 'Cream', 'Crear': 'Cream', 'Crearn': 'Cream',
        'Team': 'Cream', 'TEAM': 'Cream', 'Tearn': 'Cream', 'ream': 'Cream',
        'Lin': 'Lip', 'Lp': 'Lip', 'Liq': 'Lip', 'Li': 'Lip',
        'Crea': 'Cream',
        'Painter': 'Painter', 'ainter': 'Painter', 'alnter': 'Painter', 
        'Palnter': 'Painter', 'Paint': 'Painter',
        'SEPHOR': 'SEPHORA', 'SEPHOF': 'SEPHORA', 'SEPHO': 'SEPHORA',
        'sEPHORA': 'SEPHORA', 'sEPHOR': 'SEPHORA', 'Sephora': 'SEPHORA',
        'SEPORA': 'SEPHORA', 'SEPHORS': 'SEPHORA', 'SEPH': 'SEPHORA',
        'aty:': 'Qty:', 'aty': 'Qty', 'Oty:': 'Qty:', 'Oty': 'Qty',
        'qty:': 'Qty:', 'QTY:': 'Qty:',
        'SUMME': 'SUMMER', 'UMMER': 'SUMMER', 'SUMM': 'SUMMER', 'SUM': 'SUMMER',
        'FRiDays': 'FRIDAYS','FRiDaYs': 'FRIDAYS','Fridays': 'FRIDAYS',
        'FRIDAS': 'FRIDAYS','FRIAYS': 'FRIDAYS', 'FRiAYS': 'FRIDAYS',
        'FRiAYS': 'FRIDAYS','RIDAYS': 'FRIDAYS', 'RIDAY': 'FRIDAYS',
        'IDAYS': 'FRIDAYS', 'RIDAYS': 'FRIDAYS', 'FRipays': 'FRIDAYS',
    }

    def extract_with_rotation(self, image, psm=11, min_conf=40):
        """
        Runs OCR at 0° first. If the result has fewer than 1 meaningful word
        (words ≥3 chars, not in OCR_NOISE_WORDS), tries 180° rotation.
        Returns the orientation that produced more meaningful words.

        Short-circuits at ≥1 meaningful word to avoid the overhead of a
        second Tesseract call when 0° already gave a usable result.
        """
        text_0 = self.extract_clean_text(image, psm=psm, min_conf=min_conf)

        def score(t):
            if not t:
                return 0
            # Exclude noise words when scoring — pure noise shouldn't trigger short-circuit
            return len([w for w in t.split() 
                        if len(w) >= 3 and w not in self.OCR_NOISE_WORDS])

        # If 0° already has meaningful content, skip the 180° attempt
        if score(text_0) >= 2:
            return text_0

        # Try 180° — handles upside-down labels common on sorted packages
        rotated_180 = cv2.rotate(image, cv2.ROTATE_180)
        text_180    = self.extract_clean_text(rotated_180, psm=psm, min_conf=min_conf)

        if score(text_180) > score(text_0):
            self.get_logger().info(f'  Rotation 180° improved OCR: "{text_0}" → "{text_180}"')
            return text_180
        return text_0

    def extract_clean_text(self, image, psm=11, min_conf=40):
        """
        Core OCR method — runs Tesseract on the image and returns a clean
        string of high-confidence words.

        Two-pass approach:
          1. Raw grayscale
          2. CLAHE-enhanced grayscale (improves contrast on dark/faded labels)
        Returns the pass with more words.

        Per-word filtering pipeline:
          - Skip words with Tesseract confidence ≤ min_conf
          - Skip single-char tokens unless purely numeric
          - Skip words containing non-alphanumeric characters (punctuation noise)
          - Apply OCR_CORRECTIONS dictionary (brand/product typo fixes)
          - Skip corrected words still under 3 chars (unless numeric)
        """
        gray     = image if len(image.shape) == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # CLAHE: contrast-limited adaptive histogram equalisation — improves faded text
        clahe    = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
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
                    # Reject words containing special characters (usually noise)
                    if not re.match(r'^[A-Za-z0-9:]+$', word):
                        continue
                    # Apply brand/product OCR correction dictionary
                    word = self.OCR_CORRECTIONS.get(word, word)
                    # Drop words that are still too short after correction
                    if len(word) < 3 and not word.isdigit():
                        continue
                    clean.append(word)
            # Keep the pass that produced more words
            if len(clean) > len(best_result):
                best_result = clean
        return ' '.join(best_result)

    # ---------------------------------------------------------------
    # Known OCR noise words — single tokens that Tesseract commonly
    # produces from corrugated cardboard texture, shadow lines, and
    # label edge artefacts. Added iteratively during resilience testing.
    # These are excluded from OCR scoring and stripped from final text.
    # ---------------------------------------------------------------
    OCR_NOISE_WORDS = {
    'ill', 'lll', 'lil', 'llI', 'III', 'iil', 'lli',
    'ban', 'say', 'Fal', 'iif', 'cae', 'wil', 'eam',
    'aty', 'mtt', 'MTT', 'IRA', 'Lee', 'aig', 'ait',
    'Bee', 'ges', 'pig', 'wer', 'Ake', 'ant', 'bal',
    'pad', 'fig', 'mss', 'ers', 'wii', 'IWNS', 'iwns', 
    'Wii', 'ales', 'ime', 'INS', 'see', 'eel', 'SAvalyy',
    'IONS', 'WAY', 'AYS', 'lig', 'sin', 'bys', 'ees',
    'bet', 'Nes', 'ays', 'hore', 'mje', 'ipa', 'bes',
    'Wes', 'Pr:', 'iad'
}

    def filter_barcode_text(self, text):
        """
        Post-processes OCR text to remove barcode-style strings and noise tokens.

        Steps:
          1. Strip barcode-format substrings (e.g. 'ABC-abc-12345')
          2. Remove known OCR noise words from OCR_NOISE_WORDS
          3. Collapse multiple spaces
          4. Return None if no meaningful words (≥3 chars, non-numeric) remain
             — signals to the caller that this camera produced no useful OCR
        """
        # Remove barcode-format strings (letters-letters-digits) that Tesseract reads as text
        text  = re.sub(r'\b[A-Za-z]+-[A-Za-z]+-\d+\b', '', text)
        words = text.split()
        words = [w for w in words if w not in self.OCR_NOISE_WORDS]
        text  = ' '.join(words)
        text  = re.sub(r'\s+', ' ', text).strip()

        # Drop the entire result if nothing meaningful remains
        meaningful = [w for w in text.split() if len(w) >= 3 and not w.isdigit()]
        if not meaningful:
            return None
        return text if text else None


def main(args=None):
    rclpy.init(args=args)
    node = OCRProcessor()
    rclpy.spin(node)       # Keep node alive, processing subscribed topic callbacks
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()