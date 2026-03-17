import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from std_msgs.msg import String
import json
import sys
import os
import re
import time
import threading
from fuzzywuzzy import fuzz

sys.path.append(os.path.dirname(__file__))
from smart_match3_vF import SmartMatcher, resolve_quantity

RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# ---------------------------------------------------------------
# CamelCase splitter
# "CreamLipGloss"   →  "Cream Lip Gloss"
# "SEPHORALipStain" →  "SEPHORA Lip Stain"
# ---------------------------------------------------------------
def split_camel_case(name: str) -> str:
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.strip()


class DatabaseMatcherNode(Node):
    def __init__(self):
        super().__init__('database_matcher')

        # ── Existing subscription: OCR results from ocr_node ──
        self.subscription = self.create_subscription(
            String,
            '/ocr_results',
            self.match_callback,
            10
        )

        # ── NEW: Robot arm integration ─────────────────────────
        # Subscribe to robot arm messages (product name + pos1)
        self.robot_sub = self.create_subscription(
            String,
            '/robot_data',
            self.on_robot_data,
            10
        )
        # Publish pass/fail back to robot arm
        self.robot_cmd_pub = self.create_publisher(String, '/robot_command', 10)
        # Trigger Pi cameras on demand (sorting mode)
        self.trigger_pub = self.create_publisher(String, '/trigger_capture', RELIABLE_QOS)

        # ── Robot state machine ────────────────────────────────
        # 'idle'       — waiting for product name from robot
        # 'await_pos1' — name matched <95%, waiting for pos1
        # 'await_ocr'  — cameras triggered, waiting for OCR result
        self._robot_state      = 'idle'
        self._robot_state_lock = threading.Lock()
        self._pending_robot_name   = None  # raw e.g. "CreamLipGloss"
        self._pending_ocr_text     = None  # split e.g. "Cream Lip Gloss"
        self._pending_session_id   = None
        self._pending_name_result  = None
        self._pending_name_start   = None
        self._pos1_timer           = None
        self._ocr_timer            = None

        # Timeouts
        self.POS1_TIMEOUT = 30.0   # seconds to wait for pos1
        self.OCR_TIMEOUT  = 10.0   # seconds to wait for OCR after cameras triggered

        self.matcher = SmartMatcher(device_id='PI-001', location='Warehouse')

        self.current_session_id = None
        self.current_session_stage = None

        self.get_logger().info('Database Matcher Node ready')
        self.get_logger().info('Robot arm integration active — listening on /robot_data')

    # ===========================================================
    # ROBOT ARM INTEGRATION
    # ===========================================================

    # ---------------------------------------------------------------
    # /robot_data callback — receives product name and pos1
    # ---------------------------------------------------------------
    def on_robot_data(self, msg):
        data = msg.data.strip()
        self.get_logger().info(f'[/robot_data] received: "{data}"')

        with self._robot_state_lock:
            state = self._robot_state

        # ── pos1 signal ────────────────────────────────────────
        if data == 'pos1':
            with self._robot_state_lock:
                if self._robot_state != 'await_pos1':
                    self.get_logger().warn(
                        f'pos1 received but robot state is "{self._robot_state}" — ignoring'
                    )
                    return
                name_result = self._pending_name_result
                robot_name  = self._pending_robot_name
                ocr_text    = self._pending_ocr_text
                self._robot_state = 'idle' if name_result is not None else 'await_ocr'

            self._cancel_pos1_timer()

            if name_result is not None:
                # ── ≥95% name match confirmed by pos1 — send pass ──
                self.get_logger().info(
                    'pos1 received — name ≥95% confirmed, sending pass'
                )
                product = name_result['product']
                details = name_result.get('match_details') or {}
                self.log_results(
                    matched=True,
                    product=product,
                    accuracy=name_result['accuracy'],
                    confidence=name_result.get('confidence', 'HIGH'),
                    verified=name_result.get('verified', True),
                    barcode_used=None,
                    barcodes=[],
                    ocr_text=ocr_text,
                    details=details,
                    quantity=name_result.get('quantity'),
                    quantity_source=name_result.get('quantity_source', 'default')
                )
                self.get_logger().info('\n=== PIPELINE TIMING (ROBOT NAME MATCH) ===')
                self.get_logger().info(f'  Source:    robot arm end-effector OCR')
                self.get_logger().info(f'  Input:     "{robot_name}" → "{ocr_text}"')
                self.get_logger().info(f'  Result:    PASS — name ≥95%, confirmed by pos1')
                self.get_logger().info('')
                self.get_logger().info('  [WSL -- database_matcher]')
                self.get_logger().info(f'    Matcher processing: {self._pending_name_start:.3f}s')
                self.get_logger().info('==========================================')
                self._publish_robot_command('pass')
                self._robot_reset_state()
            else:
                # ── <95% — trigger cameras ─────────────────────────
                self.get_logger().info(
                    'pos1 received — box in camera zone, triggering Pi cameras'
                )
                self._trigger_cameras()
            return

        # ── Product name from robot ────────────────────────────
        if state != 'idle':
            self.get_logger().warn(
                f'Product name "{data}" received while robot state="{state}" — resetting'
            )
            self._robot_reset_state()

        self._handle_robot_name(data)

    # ---------------------------------------------------------------
    # Step 1 — DB lookup on robot product name
    # ---------------------------------------------------------------
    def _handle_robot_name(self, robot_name: str):
        name_start = time.time()
        ocr_text   = split_camel_case(robot_name)
        session_id = self.get_or_create_session('sorting')

        self.get_logger().info(
            f'[ROBOT] name: "{robot_name}" → split: "{ocr_text}"'
        )

        quantity, quantity_source = resolve_quantity(ocr_text, 'sorting')

        try:
            result = self.matcher.match_and_save(
                ocr_text=ocr_text,
                barcode=None,
                session_id=session_id,
                scan_mode='sorting'
            )
            db_time = time.time() - name_start
        except Exception as e:
            self.get_logger().error(f'[ROBOT] SmartMatcher error: {e}')
            result = None
            db_time = 0.0

        accuracy = result['accuracy'] if result and result.get('matched') else 0.0
        matched  = result and result.get('matched') and accuracy >= 95.0

        self.get_logger().info(
            f'[ROBOT] DB lookup: accuracy={accuracy:.1f}%  matched={matched}'
        )

        if matched:
            # ── ≥95% — store result, wait for pos1 before sending pass ─
            self.get_logger().info(
                f'[ROBOT] name ≥95% matched — waiting for pos1 before sending pass'
            )
            with self._robot_state_lock:
                self._robot_state         = 'await_pos1'
                self._pending_robot_name  = robot_name
                self._pending_ocr_text    = ocr_text
                self._pending_session_id  = session_id
                self._pending_name_result = result  # stored for logging when pos1 arrives
                self._pending_name_start  = db_time

            self._pos1_timer = threading.Timer(
                self.POS1_TIMEOUT, self._on_pos1_timeout
            )
            self._pos1_timer.daemon = True
            self._pos1_timer.start()

        else:
            # ── <95% — wait for pos1 to trigger cameras ───────
            self.get_logger().info(
                f'[ROBOT] accuracy {accuracy:.1f}% < 95% — '
                f'waiting for pos1 to trigger camera capture'
            )

            if result and result.get('scan_id'):
                self.matcher.db.update_scan(
                    scan_id=result['scan_id'],
                    match_confidence='LOW_CONFIDENCE',
                    verified=0,
                    notes=f'Robot name match {accuracy:.1f}% — awaiting camera verification'
                )

            with self._robot_state_lock:
                self._robot_state        = 'await_pos1'
                self._pending_robot_name = robot_name
                self._pending_ocr_text   = ocr_text
                self._pending_session_id = session_id

            # Watchdog — send fail if pos1 never arrives
            self._pos1_timer = threading.Timer(
                self.POS1_TIMEOUT, self._on_pos1_timeout
            )
            self._pos1_timer.daemon = True
            self._pos1_timer.start()

    # ---------------------------------------------------------------
    # Step 2 — Trigger Pi cameras
    # ---------------------------------------------------------------
    def _trigger_cameras(self):
        trigger_msg = String()
        trigger_msg.data = 'capture'
        self.trigger_pub.publish(trigger_msg)
        self.get_logger().info('[ROBOT] Capture trigger sent to /trigger_capture')

        # Watchdog — send fail if OCR never arrives
        self._ocr_timer = threading.Timer(
            self.OCR_TIMEOUT, self._on_ocr_timeout
        )
        self._ocr_timer.daemon = True
        self._ocr_timer.start()

    # ---------------------------------------------------------------
    # Publish pass/fail to robot
    # ---------------------------------------------------------------
    def _publish_robot_command(self, command: str):
        msg = String()
        msg.data = command
        self.robot_cmd_pub.publish(msg)
        self.get_logger().info(f'[ROBOT] → /robot_command: "{command}"')

    # ---------------------------------------------------------------
    # Timeout handlers
    # ---------------------------------------------------------------
    def _on_pos1_timeout(self):
        with self._robot_state_lock:
            if self._robot_state != 'await_pos1':
                return
            self._robot_state = 'idle'
        self.get_logger().warn(
            f'[ROBOT] pos1 timeout after {self.POS1_TIMEOUT}s '
            f'— sending fail for "{self._pending_robot_name}"'
        )
        self._publish_robot_command('fail')
        self._robot_reset_state()

    def _on_ocr_timeout(self):
        with self._robot_state_lock:
            if self._robot_state != 'await_ocr':
                return
            self._robot_state = 'idle'
        self.get_logger().warn(
            f'[ROBOT] OCR timeout after {self.OCR_TIMEOUT}s — sending fail'
        )
        self._publish_robot_command('fail')
        self._robot_reset_state()

    # ---------------------------------------------------------------
    # Reset robot state
    # ---------------------------------------------------------------
    def _cancel_pos1_timer(self):
        if self._pos1_timer:
            self._pos1_timer.cancel()
            self._pos1_timer = None

    def _cancel_ocr_timer(self):
        if self._ocr_timer:
            self._ocr_timer.cancel()
            self._ocr_timer = None

    def _robot_reset_state(self):
        with self._robot_state_lock:
            self._robot_state        = 'idle'
            self._pending_robot_name = None
            self._pending_ocr_text   = None
            self._pending_session_id = None
            self._pending_name_result = None
            self._pending_name_start = None
        self._cancel_pos1_timer()
        self._cancel_ocr_timer()

    # ===========================================================
    # EXISTING CODE — unchanged below
    # ===========================================================

    # ---------------------------------------------------------------
    # Session management
    # ---------------------------------------------------------------
    def get_or_create_session(self, stage):
        if self.current_session_id is not None and self.current_session_stage == stage:
            return self.current_session_id

        if self.current_session_id is not None:
            self.get_logger().info(
                f'Stage changed {self.current_session_stage} -> {stage} '
                f'— closing session {self.current_session_id}'
            )
            self.matcher.db.close_session(self.current_session_id)

        self.current_session_id = self.matcher.db.create_session(stage=stage)
        self.current_session_stage = stage
        self.get_logger().info(
            f'New session created: ID={self.current_session_id} stage={stage}'
        )
        return self.current_session_id

    def reset_session(self):
        if self.current_session_id is not None:
            self.matcher.db.close_session(self.current_session_id)
            self.get_logger().info(f'Session {self.current_session_id} reset')
            self.current_session_id = None
            self.current_session_stage = None

    # ---------------------------------------------------------------
    # Parse barcode
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
    # Tie detection
    # ---------------------------------------------------------------
    def check_tie(self, ocr_text, products):
        scores = []
        for product in products:
            _, _, score, _ = self.matcher._enhanced_fuzzy_match(ocr_text, None, [product])
            scores.append((product, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        non_zero = [s for s in scores if s[1] > 0]
        if len(non_zero) <= 1:
            return False

        if len(scores) >= 2:
            top_score = scores[0][1]
            second_score = scores[1][1]
            if top_score > 0 and abs(top_score - second_score) < 5:
                self.get_logger().warn(
                    f'TIE DETECTED — '
                    f'{scores[0][0][1]} ({scores[0][1]:.1f}) vs '
                    f'{scores[1][0][1]} ({scores[1][1]:.1f}) — '
                    f'Manual check required'
                )
                return True
        return False

    # ---------------------------------------------------------------
    # Guard: OCR meaningfulness check
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
    # Unified result display
    # ---------------------------------------------------------------
    def log_results(self, matched, product, accuracy, confidence, verified,
                    barcode_used, barcodes, ocr_text, details, reason=None,
                    quantity=None, quantity_source='default'):
        self.get_logger().info('\n=== DATABASE MATCHING RESULTS ===')

        if matched:
            self.get_logger().info(f'✓ MATCHED: {product[2]} {product[1]}')
            self.get_logger().info(f'  Overall Accuracy:   {accuracy:.1f}%')
            self.get_logger().info(f'  Confidence Level:   {confidence}')
            self.get_logger().info(f'  Verified:           {"Yes" if verified else "No"}')

            if quantity_source == 'flagged':
                self.get_logger().warn(
                    f'  Quantity:           WARNING: UNKNOWN — flagged for manual resolution'
                )
            elif quantity_source == 'default':
                self.get_logger().info(
                    f'  Quantity:           {quantity} (source: default — sorting fallback)'
                )
            else:
                self.get_logger().info(
                    f'  Quantity:           {quantity} (source: {quantity_source})'
                )
        else:
            self.get_logger().warn(f'NO MATCH FOUND{" — " + reason if reason else ""}')
            self.get_logger().info(f'  Overall Accuracy:   {accuracy:.1f}%')

        self.get_logger().info(f'  Score Breakdown:')

        if barcode_used and confidence == 'CONFLICT':
            self.get_logger().warn(f'    Barcode match:      CONFLICT — barcode matched [{barcode_used}] but contradicts OCR')
        elif barcode_used:
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
    # Timing log — full pipeline breakdown
    # ---------------------------------------------------------------
    def log_timing(self, callback_start, overall_start, end_to_end_from_ocr,
               mode, scan_mode='sorting', per_camera=None,
               clock_offset=0.0, pi_cycle_time=None):
        matcher_time = time.time() - callback_start
        now = time.time()
        full_end_to_end = now - (overall_start + clock_offset) if overall_start else None

        self.get_logger().info('\n=== PIPELINE TIMING BREAKDOWN ===')

        self.get_logger().info('  [Pi -- multi_camera_publisher]  (capture times: see Pi log)')
        if per_camera:
            for cam_id in sorted(per_camera.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                t = per_camera.get(cam_id, {}).get('timing', {})
                offset = t.get('cam_start_offset', 0.0)
                net    = t.get('network_latency', 0.0)
                self.get_logger().info(
                    f'    Cam {cam_id}: published at +{offset:.3f}s  '
                    f'| network delay to WSL: {net:.3f}s'
                )

        self.get_logger().info('')
        self.get_logger().info('  [WSL -- ocr_node  (cameras processed in parallel)]')
        if per_camera:
            slowest = 0.0
            for cam_id in sorted(per_camera.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                t      = per_camera.get(cam_id, {}).get('timing', {})
                ocr_t  = t.get('total', 0.0)
                slowest = max(slowest, ocr_t)
                has_bc  = 'BC:yes' if per_camera[cam_id].get('barcodes') else 'BC:no '
                has_ocr = 'OCR:yes' if per_camera[cam_id].get('ocr_text') else 'OCR:no '
                self.get_logger().info(
                    f'    Cam {cam_id} OCR+barcode: {ocr_t:.3f}s  [{has_ocr}  {has_bc}]'
                )
            self.get_logger().info(
                f'    ocr_node wall-clock (slowest cam): {slowest:.3f}s'
            )
        elif end_to_end_from_ocr:
            self.get_logger().info(f'    ocr_node e2e: {end_to_end_from_ocr:.3f}s')

        self.get_logger().info('')
        self.get_logger().info('  [WSL -- database_matcher]')
        self.get_logger().info(f'    Matcher processing: {matcher_time:.3f}s')

        self.get_logger().info('')
        self.get_logger().info(f'  Mode: {mode} [{scan_mode}]')

        if pi_cycle_time and per_camera:
            latest_finish = 0.0
            for cam_id in per_camera:
                t = per_camera[cam_id].get('timing', {})
                offset = t.get('cam_start_offset', 0.0)
                network = t.get('network_latency', 0.0)
                ocr = t.get('total', 0.0)
                finish = offset + network + ocr
                latest_finish = max(latest_finish, finish)

            true_e2e = latest_finish + matcher_time
            status = 'PASS' if true_e2e <= 3.0 else f'FAIL  (+{true_e2e - 3.0:.3f}s over target)'
            self.get_logger().info(f'  Pi capture cycle:   {pi_cycle_time:.3f}s')
            self.get_logger().info(f'  Last cam finish:    {latest_finish:.3f}s')
            self.get_logger().info(f'  Matcher:            {matcher_time:.3f}s')
            self.get_logger().info(
                f'  TOTAL end-to-end:   {true_e2e:.3f}s  [{status}]'
            )
        elif full_end_to_end is not None and 0 < full_end_to_end < 30:
            status = 'PASS' if full_end_to_end <= 3.0 else f'FAIL  (+{full_end_to_end - 3.0:.3f}s over target)'
            self.get_logger().info(
                f'  TOTAL end-to-end (Pi init -> OCR -> DB): {full_end_to_end:.3f}s  [{status}]'
            )
        else:
            self.get_logger().info(
                '  TOTAL end-to-end: unavailable (Pi/WSL clocks not synced)'
            )
        self.get_logger().info('=================================')

    # ---------------------------------------------------------------
    # Main callback — OCR results from ocr_node
    # ---------------------------------------------------------------
    def match_callback(self, msg):
        callback_start = time.time()

        data = json.loads(msg.data)
        ocr_text            = data.get('ocr_text', '') or ''
        mode                = data.get('mode', '1-camera')
        scan_mode           = data.get('scan_mode', 'sorting')
        cameras_received    = data.get('cameras_received', 1)
        per_camera          = data.get('per_camera', {})
        all_barcodes_by_cam = data.get('all_barcodes_by_cam', {})

        overall_start = (
            data.get('overall_start') or
            data.get('camera_overall_start') or
            data.get('camera_capture_time')
        )
        end_to_end_from_ocr = data.get('end_to_end_time')
        clock_offset = data.get('clock_offset') or 0.0
        pi_cycle_time = data.get('pi_cycle_time')

        raw_barcode = data.get('barcode')
        barcodes = self.parse_barcodes(raw_barcode)

        quantity, quantity_source = resolve_quantity(ocr_text, scan_mode)

        self.get_logger().info(
            f'Received [{mode}] [{scan_mode}] — '
            f'Cameras: {cameras_received} | '
            f'OCR: "{ocr_text[:40]}" | '
            f'Barcodes: {barcodes} | '
            f'Qty: {"FLAGGED" if quantity_source == "flagged" else f"{quantity} ({quantity_source})"}'
        )

        if quantity_source == 'flagged':
            self.get_logger().warn(
                f'[INBOUND] Quantity not found in OCR text — '
                f'scan will be FLAGGED for manual resolution'
            )

        # ── Robot sorting: if we triggered this OCR, handle pass/fail ──
        with self._robot_state_lock:
            robot_waiting = self._robot_state == 'await_ocr'
            pending_session = self._pending_session_id

        if robot_waiting:
            self._cancel_ocr_timer()
            self.get_logger().info('[ROBOT] OCR result received — evaluating for pass/fail')
            robot_barcode = barcodes[0] if barcodes else None

            try:
                result = self.matcher.match_and_save(
                    ocr_text=ocr_text,
                    barcode=robot_barcode,
                    session_id=pending_session or self.get_or_create_session('sorting'),
                    scan_mode='sorting'
                )
            except Exception as e:
                self.get_logger().error(f'[ROBOT] SmartMatcher error: {e}')
                result = None

            accuracy = result['accuracy'] if result and result.get('matched') else 0.0
            matched  = result and result.get('matched') and accuracy >= 95.0

            details = (result.get('match_details') or {}) if result else {}
            result_qty     = result.get('quantity') if result else None
            result_qty_src = result.get('quantity_source', 'default') if result else 'default'

            if matched:
                product = result['product']
                self.log_results(
                    matched=True,
                    product=product,
                    accuracy=accuracy,
                    confidence=result.get('confidence', 'HIGH'),
                    verified=result.get('verified', True),
                    barcode_used=robot_barcode,
                    barcodes=[],
                    ocr_text=ocr_text,
                    details=details,
                    quantity=result_qty,
                    quantity_source=result_qty_src
                )
            else:
                if result and result.get('scan_id'):
                    self.matcher.db.update_scan(
                        scan_id=result['scan_id'],
                        match_confidence='LOW_CONFIDENCE',
                        verified=0,
                        notes=f'Camera match {accuracy:.1f}% — below 95% threshold'
                    )
                self.log_results(
                    matched=False,
                    product=None,
                    accuracy=accuracy,
                    confidence='LOW_CONFIDENCE',
                    verified=False,
                    barcode_used=None,
                    barcodes=[],
                    ocr_text=ocr_text,
                    details=details,
                    reason=f'camera match {accuracy:.1f}% below 95% threshold',
                    quantity=result_qty,
                    quantity_source=result_qty_src
                )

            self._publish_robot_command('pass' if matched else 'fail')
            self._robot_reset_state()
            self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                            mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
            return

        # ── Normal inbound/sorting OCR path (unchanged) ───────────
        session_id = self.get_or_create_session(scan_mode)
        all_products = self.matcher.db.get_all_products()

        # ── Inbound cross-camera barcode conflict detection ───────
        if scan_mode == 'inbound' and all_barcodes_by_cam:
            cam_matches = {}
            for cam_id, cam_barcodes in all_barcodes_by_cam.items():
                for b in cam_barcodes:
                    barcode_data = b.get('data', '') if isinstance(b, dict) else str(b)
                    for p in all_products:
                        if p[6] and p[6] == barcode_data:
                            cam_matches[cam_id] = (barcode_data, p[0], p[1])

            unique_products = {v[1] for v in cam_matches.values()}

            if len(unique_products) > 1:
                conflict_barcodes = [v[0] for v in cam_matches.values()]
                self.get_logger().warn(
                    f'[INBOUND] Cross-camera barcode CONFLICT: '
                    f'{[(cam, v[0], v[2]) for cam, v in cam_matches.items()]}'
                )
                self.matcher.db.save_scan(
                    ocr_text=ocr_text,
                    matched_product_id=None,
                    match_confidence='CONFLICT',
                    match_score=0.0,
                    barcode=str(conflict_barcodes),
                    device_id='PI-001',
                    session_id=session_id,
                    quantity=quantity,
                    quantity_source=quantity_source,
                    scan_mode=scan_mode
                )
                self.log_results(
                    matched=False, product=None, accuracy=0.0,
                    confidence='NO MATCH', verified=False,
                    barcode_used=None, barcodes=conflict_barcodes,
                    ocr_text=ocr_text, details={},
                    reason='inbound conflict — cameras see different products',
                    quantity=quantity, quantity_source=quantity_source
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                                mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
                return

            elif len(unique_products) == 1:
                winning_cam = next(iter(cam_matches))
                barcodes = [cam_matches[winning_cam][0]]
                self.get_logger().info(
                    f'[INBOUND] All cameras agree on product ID {list(unique_products)[0]} '
                    f'— proceeding with barcode {barcodes[0]}'
                )
            else:
                self.get_logger().info('[INBOUND] No barcode DB matches — falling through to OCR')

        # ── Reject if multiple brands detected ───────────────────
        brands_found = list({
            p[2] for p in all_products
            if p[2] and fuzz.partial_ratio(p[2].lower(), ocr_text.lower()) >= 85
        })
        if len(brands_found) > 1:
            self.get_logger().warn(
                f'Multiple brands detected {brands_found} — possible multiple products in frame'
            )
            self.log_results(
                matched=False, product=None, accuracy=0.0,
                confidence='NO MATCH', verified=False,
                barcode_used=None, barcodes=barcodes,
                ocr_text=ocr_text, details={},
                reason='multiple products detected in frame',
                quantity=quantity, quantity_source=quantity_source
            )
            self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                            mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
            return

        # ── Reject if multiple barcodes match different products ──
        if len(barcodes) > 1:
            matched_ids = []
            for barcode in barcodes:
                for p in all_products:
                    if p[6] and p[6] == barcode:
                        matched_ids.append((barcode, p[0], p[1]))
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
                    reason='multiple barcodes matched different products',
                    quantity=quantity, quantity_source=quantity_source
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                                mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
                return

        if per_camera:
            self.get_logger().info('Per-camera OCR summary:')
            for cam_id, cam_result in per_camera.items():
                cam_ocr      = cam_result.get('ocr_text', 'None')
                cam_barcodes = cam_result.get('barcodes', [])
                self.get_logger().info(
                    f'  Cam {cam_id}: OCR="{cam_ocr}" | Barcodes={cam_barcodes}'
                )

        # ── Guard: skip if no barcode and OCR is garbage ─────────
        if not barcodes and not self.is_meaningful_ocr(ocr_text):
            self.get_logger().warn(f'OCR text unrecognisable: "{ocr_text}"')
            self.log_results(
                matched=False, product=None, accuracy=0.0, confidence='NO MATCH',
                verified=False, barcode_used=None, barcodes=barcodes,
                ocr_text=ocr_text, details={}, reason='insufficient OCR data',
                quantity=quantity, quantity_source=quantity_source
            )
            self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                            mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
            return

        # ── Tie check if OCR-only ─────────────────────────────────
        if not barcodes:
            if self.check_tie(ocr_text, all_products):
                _, _, _, details = self.matcher._enhanced_fuzzy_match(ocr_text, None, all_products)
                self.matcher.db.save_scan(
                    ocr_text=ocr_text,
                    matched_product_id=None,
                    match_confidence='TIE',
                    match_score=50.0,
                    barcode=None,
                    device_id='PI-001',
                    session_id=session_id,
                    quantity=quantity,
                    quantity_source=quantity_source,
                    scan_mode=scan_mode
                )
                self.log_results(
                    matched=False, product=None, accuracy=50.0,
                    confidence='AMBIGUOUS', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason='TIE DETECTED, barcode needed',
                    quantity=quantity, quantity_source=quantity_source
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                                mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
                return

        # ── Try each barcode until exact DB match ─────────────────
        result = None
        barcode_used = None
        for barcode in barcodes:
            try:
                result = self.matcher.match_and_save(
                    ocr_text=ocr_text,
                    barcode=barcode,
                    session_id=session_id,
                    scan_mode=scan_mode
                )
            except Exception as e:
                self.get_logger().error(f'Exception in match_and_save: {e}')
            if result and result['matched'] and result.get('barcode_matched'):
                barcode_product = result['product']
                _, _, ocr_score, _ = self.matcher._enhanced_fuzzy_match(
                    ocr_text, None, [barcode_product]
                )
                for p in all_products:
                    if p[0] == barcode_product[0]:
                        continue
                    _, _, other_score, _ = self.matcher._enhanced_fuzzy_match(
                        ocr_text, None, [p]
                    )
                    if other_score > ocr_score and other_score >= 40:
                        self.get_logger().warn(
                            f'[CONFLICT] Barcode says "{barcode_product[1]}" '
                            f'but OCR strongly suggests "{p[1]}" '
                            f'(barcode_ocr={ocr_score:.1f} vs ocr_only={other_score:.1f})'
                        )
                        if result and result.get('scan_id'):
                            self.matcher.db.delete_scan(result['scan_id'])
                        self.matcher.db.save_scan(
                            ocr_text=ocr_text,
                            matched_product_id=None,
                            match_confidence='CONFLICT',
                            match_score=50.0,
                            barcode=barcode,
                            device_id='PI-001',
                            session_id=session_id,
                            quantity=quantity,
                            quantity_source=quantity_source,
                            scan_mode=scan_mode
                        )
                        _, _, _, conflict_details = self.matcher._enhanced_fuzzy_match(
                            ocr_text, None, all_products
                        )
                        self.log_results(
                            matched=False, product=None, accuracy=50.0,
                            confidence='CONFLICT', verified=False,
                            barcode_used=barcode, barcodes=barcodes,
                            ocr_text=ocr_text, details=conflict_details,
                            reason=f'CONFLICT — barcode={barcode_product[1]} vs OCR={p[1]}',
                            quantity=quantity, quantity_source=quantity_source
                        )
                        self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                                        mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
                        return
                barcode_used = barcode
                break

        # ── Fall back to OCR-only ─────────────────────────────────
        if not result or not result['matched'] or not result.get('barcode_matched'):
            barcode_used = None
            if self.check_tie(ocr_text, all_products):
                details = (result.get('match_details') or {}) if result else {}
                self.matcher.db.save_scan(
                    ocr_text=ocr_text,
                    matched_product_id=None,
                    match_confidence='TIE',
                    match_score=50.0,
                    barcode=None,
                    device_id='PI-001',
                    session_id=session_id,
                    quantity=quantity,
                    quantity_source=quantity_source,
                    scan_mode=scan_mode
                )
                self.log_results(
                    matched=False, product=None, accuracy=50.0,
                    confidence='AMBIGUOUS', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason='TIE DETECTED, barcode needed',
                    quantity=quantity, quantity_source=quantity_source
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                                mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
                return

            result = self.matcher.match_and_save(
                ocr_text=ocr_text,
                barcode=None,
                session_id=session_id,
                scan_mode=scan_mode
            )

            if not result or not result['matched']:
                details = (result.get('match_details') or {}) if result else {}
                self.log_results(
                    matched=False, product=None, accuracy=0.0,
                    confidence='NO MATCH', verified=False,
                    barcode_used=None, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason='OCR score too low — barcode needed',
                    quantity=quantity, quantity_source=quantity_source
                )
                self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                                mode, scan_mode, per_camera, clock_offset, pi_cycle_time)
                return

        # ── Final result ──────────────────────────────────────────
        details          = (result.get('match_details') or {}) if result else {}
        result_qty       = result.get('quantity')
        result_qty_src   = result.get('quantity_source', quantity_source)

        if result and result['matched']:
            if result['accuracy'] < 95.0:
                self.get_logger().warn(
                    f'Accuracy {result["accuracy"]:.1f}% below 95% threshold — flagging'
                )
                self.matcher.db.update_scan(
                    scan_id=result['scan_id'],
                    match_confidence='LOW_CONFIDENCE',
                    verified=0,
                    notes=f'Accuracy {result["accuracy"]:.1f}% below 95% threshold'
                )
                self.log_results(
                    matched=False, product=None, accuracy=result['accuracy'],
                    confidence='LOW_CONFIDENCE', verified=False,
                    barcode_used=barcode_used, barcodes=barcodes,
                    ocr_text=ocr_text, details=details,
                    reason=f'accuracy {result["accuracy"]:.1f}% below 95% threshold — flagged',
                    quantity=result_qty, quantity_source=result_qty_src
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
                    details=details,
                    quantity=result_qty,
                    quantity_source=result_qty_src
                )

                summary = self.matcher.db.get_quantity_by_stage(session_id=session_id)
                self.get_logger().info(f'  Session {session_id} running totals (known qty):')
                if summary:
                    for stage, product_name, brand, total in summary:
                        self.get_logger().info(
                            f'    [{stage}] {brand} {product_name}: {total} units'
                        )
                else:
                    self.get_logger().info(f'    No confirmed quantities yet')
        else:
            self.log_results(
                matched=False, product=None, accuracy=0.0, confidence='NO MATCH',
                verified=False, barcode_used=None, barcodes=barcodes,
                ocr_text=ocr_text, details=details, reason='no product matched',
                quantity=result_qty, quantity_source=result_qty_src
            )

        self.log_timing(callback_start, overall_start, end_to_end_from_ocr,
                        mode, scan_mode, per_camera, clock_offset, pi_cycle_time)


def main(args=None):
    rclpy.init(args=args)
    node = DatabaseMatcherNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()