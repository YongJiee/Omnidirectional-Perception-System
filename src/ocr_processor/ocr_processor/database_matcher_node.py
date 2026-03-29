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

# ---------------------------------------------------------------------------
# Add the current directory to sys.path so that smart_match3_vF.py can be
# imported regardless of how the ROS2 package is launched.
# ---------------------------------------------------------------------------
sys.path.append(os.path.dirname(__file__))
from smart_match3_vF import SmartMatcher, resolve_quantity

# ---------------------------------------------------------------------------
# QoS profile used for topics that must not drop messages.
# RELIABLE + KEEP_LAST(10) ensures trigger and batch signals are delivered.
# ---------------------------------------------------------------------------
RELIABLE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10
)

# ---------------------------------------------------------------------------
# CamelCase splitter — converts the robot arm's compact product name strings
# into space-separated words that SmartMatcher can tokenise properly.
#   "CreamLipGloss"   →  "Cream Lip Gloss"
#   "SEPHORALipStain" →  "SEPHORA Lip Stain"
# ---------------------------------------------------------------------------
def split_camel_case(name: str) -> str:
    # Insert space between a lowercase letter followed by an uppercase letter
    s = re.sub(r'([a-z])([A-Z])', r'\1 \2', name)
    # Insert space between a run of uppercase letters and the start of a new word
    s = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1 \2', s)
    return s.strip()


class DatabaseMatcherNode(Node):
    """
    Central matching and decision node for the OPS system.

    Responsibilities:
      1. Receive merged OCR results from ocr_node on /ocr_results.
      2. Run SmartMatcher to identify the product and assess confidence.
      3. In sorting mode, coordinate a 3-step state machine with the robot arm:
           robot name → (optional) camera capture → pass/fail command.
      4. Detect and flag conflicts (cross-camera barcode, multi-brand OCR,
         barcode-vs-OCR disagreement).
      5. Log full pipeline timing breakdowns for performance verification.
    """

    def __init__(self):
        super().__init__('database_matcher')

        # ── Subscribe to merged OCR results published by ocr_node ──────────
        self.subscription = self.create_subscription(
            String,
            '/ocr_results',
            self.match_callback,
            10
        )

        # ── Robot arm integration ───────────────────────────────────────────
        # Receive product name + position signals from the arm controller
        self.robot_sub = self.create_subscription(
            String,
            '/robot_data',
            self.on_robot_data,
            10
        )
        # Send pass/fail decisions back to the arm controller
        self.robot_cmd_pub = self.create_publisher(String, '/robot_command', 10)
        # Tell the Pi cameras to capture (sorting mode only)
        self.trigger_pub = self.create_publisher(String, '/trigger_capture', RELIABLE_QOS)

        # ── Robot arm state machine ─────────────────────────────────────────
        # Tracks where in the sorting handshake the system currently is:
        #   'idle'       — no active package; waiting for robot name message
        #   'await_pos1' — name received; waiting for pos1 (box in camera zone)
        #   'await_ocr'  — cameras triggered; waiting for OCR result to arrive
        self._robot_state      = 'idle'
        self._robot_state_lock = threading.Lock()  # Guards state across ROS callbacks

        # Pending data held between state transitions
        self._pending_robot_name   = None  # Raw CamelCase name from arm e.g. "CreamLipGloss"
        self._pending_ocr_text     = None  # Split version e.g. "Cream Lip Gloss"
        self._pending_session_id   = None  # Session to link the scan to
        self._pending_name_result  = None  # SmartMatcher result if name scored ≥95%
        self._pending_name_start   = None  # DB lookup duration for timing log

        # Watchdog timers — send fail if expected signal never arrives
        self._pos1_timer = None  # Fires if pos1 not received within POS1_TIMEOUT
        self._ocr_timer  = None  # Fires if OCR not received within OCR_TIMEOUT

        # Timeout durations (seconds)
        self.POS1_TIMEOUT = 30.0   # Allow time for arm to move box to camera zone
        self.OCR_TIMEOUT  = 10.0   # Allow time for Pi capture + OCR processing

        # SmartMatcher instance — handles DB lookup, fuzzy scoring, scan saving
        self.matcher = SmartMatcher(device_id='PI-001', location='Warehouse')

        # Track the active session so all scans in a batch share the same session_id
        self.current_session_id    = None
        self.current_session_stage = None

        self.get_logger().info('Database Matcher Node ready')
        self.get_logger().info('Robot arm integration active — listening on /robot_data')

    # ===========================================================
    # ROBOT ARM INTEGRATION
    # ===========================================================

    # ---------------------------------------------------------------
    # /robot_data callback — entry point for all robot arm messages.
    # Two message types are handled:
    #   'pos1'        — arm signals the box has reached the camera zone
    #   anything else — treated as a product name (CamelCase string)
    # ---------------------------------------------------------------
    def on_robot_data(self, msg):
        data = msg.data.strip()
        self.get_logger().info(f'[/robot_data] received: "{data}"')

        # Read current state without holding lock across the entire handler
        with self._robot_state_lock:
            state = self._robot_state

        # ── Handle pos1 signal ─────────────────────────────────────────────
        if data == 'pos1':
            with self._robot_state_lock:
                if self._robot_state != 'await_pos1':
                    # pos1 arrived at an unexpected time — ignore it
                    self.get_logger().warn(
                        f'pos1 received but robot state is "{self._robot_state}" — ignoring'
                    )
                    return
                # Snapshot pending data before releasing lock
                name_result = self._pending_name_result
                robot_name  = self._pending_robot_name
                ocr_text    = self._pending_ocr_text
                # Transition: if name already matched ≥95% go idle, else wait for OCR
                self._robot_state = 'idle' if name_result is not None else 'await_ocr'

            self._cancel_pos1_timer()  # Stop the pos1 watchdog

            if name_result is not None:
                # ── Name scored ≥95% and pos1 confirms box is present → PASS ──
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
                # ── Name scored <95% — box is now in camera zone, trigger cameras ──
                self.get_logger().info(
                    'pos1 received — box in camera zone, triggering Pi cameras'
                )
                self._trigger_cameras()
            return

        # ── Handle product name from robot ─────────────────────────────────
        if state != 'idle':
            # A new name arrived before the previous handshake completed — reset
            self.get_logger().warn(
                f'Product name "{data}" received while robot state="{state}" — resetting'
            )
            self._robot_reset_state()

        self._handle_robot_name(data)

    # ---------------------------------------------------------------
    # Step 1 — DB lookup on the robot arm's product name.
    # If the name scores ≥95% the result is stored and we wait for pos1
    # to physically confirm the box is present before sending pass.
    # If <95% we still wait for pos1 (to know the box is in camera zone)
    # but will then trigger the Pi cameras for a full OCR scan.
    # ---------------------------------------------------------------
    def _handle_robot_name(self, robot_name: str):
        name_start = time.time()
        # Convert CamelCase to space-separated for SmartMatcher tokenisation
        ocr_text   = split_camel_case(robot_name)
        session_id = self.get_or_create_session('sorting')

        self.get_logger().info(
            f'[ROBOT] name: "{robot_name}" → split: "{ocr_text}"'
        )

        # Resolve quantity from the name string (usually 'default' for sorting)
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
        # Only treat as matched if accuracy meets the 95% success criterion
        matched  = result and result.get('matched') and accuracy >= 95.0

        self.get_logger().info(
            f'[ROBOT] DB lookup: accuracy={accuracy:.1f}%  matched={matched}'
        )

        if matched:
            # ── ≥95% match — store result and wait for pos1 confirmation ──
            self.get_logger().info(
                f'[ROBOT] name ≥95% matched — waiting for pos1 before sending pass'
            )
            with self._robot_state_lock:
                self._robot_state         = 'await_pos1'
                self._pending_robot_name  = robot_name
                self._pending_ocr_text    = ocr_text
                self._pending_session_id  = session_id
                self._pending_name_result = result   # Passed to log_results when pos1 arrives
                self._pending_name_start  = db_time  # DB lookup duration for timing log

            # Watchdog: if pos1 never arrives, send fail after POS1_TIMEOUT seconds
            self._pos1_timer = threading.Timer(
                self.POS1_TIMEOUT, self._on_pos1_timeout
            )
            self._pos1_timer.daemon = True
            self._pos1_timer.start()

        else:
            # ── <95% — wait for pos1 so we know the box is in camera range ──
            self.get_logger().info(
                f'[ROBOT] accuracy {accuracy:.1f}% < 95% — '
                f'waiting for pos1 to trigger camera capture'
            )

            # Mark the preliminary scan as low confidence pending camera verification
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
                # _pending_name_result left as None — signals cameras are needed

            # Watchdog: send fail if pos1 never arrives
            self._pos1_timer = threading.Timer(
                self.POS1_TIMEOUT, self._on_pos1_timeout
            )
            self._pos1_timer.daemon = True
            self._pos1_timer.start()

    # ---------------------------------------------------------------
    # Step 2 — Publish capture trigger to Pi cameras.
    # Called when pos1 arrives and name confidence was <95%.
    # Also starts an OCR watchdog in case the Pi never responds.
    # ---------------------------------------------------------------
    def _trigger_cameras(self):
        trigger_msg = String()
        trigger_msg.data = 'capture'
        self.trigger_pub.publish(trigger_msg)
        self.get_logger().info('[ROBOT] Capture trigger sent to /trigger_capture')

        # Watchdog: send fail if OCR result never arrives within OCR_TIMEOUT
        self._ocr_timer = threading.Timer(
            self.OCR_TIMEOUT, self._on_ocr_timeout
        )
        self._ocr_timer.daemon = True
        self._ocr_timer.start()

    # ---------------------------------------------------------------
    # Publish a pass or fail command to the robot arm on /robot_command.
    # ---------------------------------------------------------------
    def _publish_robot_command(self, command: str):
        msg = String()
        msg.data = command
        self.robot_cmd_pub.publish(msg)
        self.get_logger().info(f'[ROBOT] → /robot_command: "{command}"')

    # ---------------------------------------------------------------
    # Timeout handlers — fire if expected signals never arrive.
    # Both send fail and reset state so the system can handle the next box.
    # ---------------------------------------------------------------
    def _on_pos1_timeout(self):
        with self._robot_state_lock:
            if self._robot_state != 'await_pos1':
                return  # State changed before timeout fired — nothing to do
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
                return  # State changed before timeout fired — nothing to do
            self._robot_state = 'idle'
        self.get_logger().warn(
            f'[ROBOT] OCR timeout after {self.OCR_TIMEOUT}s — sending fail'
        )
        self._publish_robot_command('fail')
        self._robot_reset_state()

    # ---------------------------------------------------------------
    # Timer cancellation helpers — always check for None before cancelling
    # since timers may not have been started yet.
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
        """
        Clears all pending robot state and cancels any active watchdog timers.
        Called after every terminal state (pass sent, fail sent, or error).
        """
        with self._robot_state_lock:
            self._robot_state         = 'idle'
            self._pending_robot_name  = None
            self._pending_ocr_text    = None
            self._pending_session_id  = None
            self._pending_name_result = None
            self._pending_name_start  = None
        self._cancel_pos1_timer()
        self._cancel_ocr_timer()

    # ===========================================================
    # EXISTING CODE — unchanged below
    # ===========================================================

    # ---------------------------------------------------------------
    # Session management — reuses an existing session if the stage
    # hasn't changed, otherwise closes the old session and opens a new one.
    # This keeps all scans in a single warehouse operation grouped together.
    # ---------------------------------------------------------------
    def get_or_create_session(self, stage):
        if self.current_session_id is not None and self.current_session_stage == stage:
            return self.current_session_id  # Reuse active session for same stage

        if self.current_session_id is not None:
            # Stage changed (e.g. inbound → sorting) — close the previous session
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
        """Manually closes the current session and clears session state."""
        if self.current_session_id is not None:
            self.matcher.db.close_session(self.current_session_id)
            self.get_logger().info(f'Session {self.current_session_id} reset')
            self.current_session_id    = None
            self.current_session_stage = None

    # ---------------------------------------------------------------
    # Normalise the barcode field from the OCR payload.
    # Handles three formats that ocr_node may produce:
    #   - dict with a 'data' key  e.g. {'data': 'ABC123', 'type': 'EAN13'}
    #   - raw string/int/float
    #   - ROS-style string e.g. "[EAN13] ABC123"
    # Always returns a plain list of barcode strings.
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
                # Strip "[TYPE] " prefix if present
                match = re.search(r'\] (.+)$', rb)
                barcodes.append(match.group(1) if match else rb)
        return barcodes

    # ---------------------------------------------------------------
    # Tie detection — checks whether two or more products score within
    # 5 points of each other on the fuzzy matcher.
    # A tie means the OCR text is ambiguous and a barcode is needed to
    # break the deadlock; the scan is saved as 'TIE' / 'AMBIGUOUS'.
    # Returns False immediately if only one product scores above zero
    # (avoids false ties for unique products).
    # ---------------------------------------------------------------
    def check_tie(self, ocr_text, products):
        scores = []
        for product in products:
            _, _, score, _ = self.matcher._enhanced_fuzzy_match(ocr_text, None, [product])
            scores.append((product, score))
        scores.sort(key=lambda x: x[1], reverse=True)

        # No tie possible if only one product has any score at all
        non_zero = [s for s in scores if s[1] > 0]
        if len(non_zero) <= 1:
            return False

        if len(scores) >= 2:
            top_score    = scores[0][1]
            second_score = scores[1][1]
            # Flag as tie if top two scores are within 5 points of each other
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
    # Meaningfulness guard — prevents the pipeline from running a full
    # DB match on pure noise (e.g. blank or random characters from OCR).
    # Returns True if the OCR text contains at least one known term
    # (brand, product name token, or keyword) of 3+ characters.
    # ---------------------------------------------------------------
    def is_meaningful_ocr(self, ocr_text):
        if len(ocr_text.strip()) < 4:
            return False  # Too short to be useful
        products  = self.matcher.db.get_all_products()
        ocr_lower = ocr_text.lower()
        for product in products:
            brand        = product[2].lower() if product[2] else ''
            product_name = product[1].lower() if product[1] else ''
            keywords     = product[4].lower() if product[4] else ''
            all_terms    = (brand + ' ' + product_name + ' ' + keywords).split()
            for term in all_terms:
                if len(term) >= 3 and term in ocr_lower:
                    return True  # Found a recognisable term — OCR is meaningful
        return False

    # ---------------------------------------------------------------
    # Unified result logger — prints a consistent summary block to the
    # ROS2 logger for every scan outcome (matched, no match, conflict, tie).
    # Used by both the robot arm path and the normal OCR path.
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

            # Quantity display varies by source to make flagged items immediately visible
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

        # Show barcode contribution — differentiate conflict from normal match
        if barcode_used and confidence == 'CONFLICT':
            self.get_logger().warn(f'    Barcode match:      CONFLICT — barcode matched [{barcode_used}] but contradicts OCR')
        elif barcode_used:
            self.get_logger().info(f'    Barcode match:      100.0% (exact) [{barcode_used}]')
        elif barcodes:
            self.get_logger().info(f'    Barcode match:      0.0% (no exact match) {barcodes}')
        else:
            self.get_logger().info(f'    Barcode match:      0.0% (none provided)')

        # Show OCR sub-scores only when OCR text was actually present
        if ocr_text.strip():
            self.get_logger().info(f'    OCR match:')
            self.get_logger().info(f'      Brand similarity:   {details.get("brand_score", 0.0):.1f}%')
            self.get_logger().info(f'      Product similarity: {details.get("product_score", 0.0):.1f}%')
            self.get_logger().info(f'      Keyword similarity: {details.get("keyword_score", 0.0):.1f}%')
        else:
            self.get_logger().info(f'    OCR match:          N/A (no OCR text)')

    # ---------------------------------------------------------------
    # Pipeline timing breakdown — logs latency at each stage:
    #   Pi capture → network → ocr_node → database_matcher
    # Used to verify the ≤3s end-to-end efficiency success criterion.
    # clock_offset corrects for Pi↔WSL clock drift on direct ethernet
    # (no NTP available), measured autonomously at session start.
    # ---------------------------------------------------------------
    def log_timing(self, callback_start, overall_start, end_to_end_from_ocr,
               mode, scan_mode='sorting', per_camera=None,
               clock_offset=0.0, pi_cycle_time=None):
        matcher_time = time.time() - callback_start
        now          = time.time()
        # Apply clock offset to align Pi timestamp with WSL wall clock
        full_end_to_end = now - (overall_start + clock_offset) if overall_start else None

        self.get_logger().info('\n=== PIPELINE TIMING BREAKDOWN ===')

        # Pi capture stage — per-camera publish offset and network latency
        self.get_logger().info('  [Pi -- multi_camera_publisher]  (capture times: see Pi log)')
        if per_camera:
            for cam_id in sorted(per_camera.keys(), key=lambda x: int(x) if str(x).isdigit() else x):
                t      = per_camera.get(cam_id, {}).get('timing', {})
                offset = t.get('cam_start_offset', 0.0)  # Time from batch start to this camera
                net    = t.get('network_latency', 0.0)    # Time from Pi publish to WSL receive
                self.get_logger().info(
                    f'    Cam {cam_id}: published at +{offset:.3f}s  '
                    f'| network delay to WSL: {net:.3f}s'
                )

        self.get_logger().info('')
        # ocr_node stage — cameras processed in parallel via ThreadPoolExecutor
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
            # Wall-clock for ocr_node is determined by the slowest camera (parallel execution)
            self.get_logger().info(
                f'    ocr_node wall-clock (slowest cam): {slowest:.3f}s'
            )
        elif end_to_end_from_ocr:
            self.get_logger().info(f'    ocr_node e2e: {end_to_end_from_ocr:.3f}s')

        self.get_logger().info('')
        # database_matcher stage
        self.get_logger().info('  [WSL -- database_matcher]')
        self.get_logger().info(f'    Matcher processing: {matcher_time:.3f}s')

        self.get_logger().info('')
        self.get_logger().info(f'  Mode: {mode} [{scan_mode}]')

        if pi_cycle_time and per_camera:
            # Compute true end-to-end: latest camera finish time + matcher time
            latest_finish = 0.0
            for cam_id in per_camera:
                t       = per_camera[cam_id].get('timing', {})
                offset  = t.get('cam_start_offset', 0.0)
                network = t.get('network_latency', 0.0)
                ocr     = t.get('total', 0.0)
                finish  = offset + network + ocr
                latest_finish = max(latest_finish, finish)

            true_e2e = latest_finish + matcher_time
            status = 'PASS' if true_e2e <= 3.0 else f'FAIL  (+{true_e2e - 3.0:.3f}s over target)'
            self.get_logger().info(f'  Pi capture cycle:   {pi_cycle_time:.3f}s')
            self.get_logger().info(f'  Last cam finish:    {latest_finish:.3f}s')
            self.get_logger().info(f'  Matcher:            {matcher_time:.3f}s')
            self.get_logger().info(
                f'  TOTAL end-to-end:   {true_e2e:.3f}s  [{status}]'
            )
        elif full_end_to_end is not None and 0 < full_end_to_end < 600:
            # Fallback: use clock-offset-corrected wall time
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
    # Main OCR callback — handles every message from /ocr_results.
    #
    # Two distinct execution paths:
    #   A) robot_waiting == True  — this OCR was triggered by the arm;
    #      evaluate pass/fail and publish to /robot_command, then return.
    #   B) normal path — inbound or standalone sorting scan;
    #      run full conflict detection, tie check, and SmartMatcher.
    # ---------------------------------------------------------------
    def match_callback(self, msg):
        callback_start = time.time()  # Start timing immediately on arrival

        # Parse the JSON payload produced by ocr_node
        data = json.loads(msg.data)
        ocr_text            = data.get('ocr_text', '') or ''
        mode                = data.get('mode', '1-camera')
        scan_mode           = data.get('scan_mode', 'sorting')
        cameras_received    = data.get('cameras_received', 1)
        per_camera          = data.get('per_camera', {})
        all_barcodes_by_cam = data.get('all_barcodes_by_cam', {})  # {cam_id: [barcode_dicts]}

        # overall_start is the Pi-side timestamp when the batch began;
        # try multiple field names for backward compatibility with older payloads
        overall_start = (
            data.get('overall_start') or
            data.get('camera_overall_start') or
            data.get('camera_capture_time')
        )
        # Injected scans have no Pi timestamp — use the current WSL time instead
        if data.get('injected'):
            overall_start = callback_start

        end_to_end_from_ocr = data.get('end_to_end_time')
        clock_offset  = data.get('clock_offset') or 0.0  # Pi↔WSL offset measured at session start
        pi_cycle_time = data.get('pi_cycle_time')         # Total Pi capture cycle duration

        # Normalise the barcode field into a plain list of strings
        raw_barcode = data.get('barcode')
        barcodes    = self.parse_barcodes(raw_barcode)

        # Resolve quantity from OCR text; source will be 'flagged' if not found (inbound)
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

        # ── Path A: Robot arm triggered this OCR — evaluate pass/fail ──────
        with self._robot_state_lock:
            robot_waiting   = self._robot_state == 'await_ocr'
            pending_session = self._pending_session_id

        if robot_waiting:
            self._cancel_ocr_timer()  # Stop the OCR watchdog — result arrived in time
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

            details        = (result.get('match_details') or {}) if result else {}
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
                # Camera OCR still below threshold — mark as low confidence
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
            return  # Robot path complete — do not fall through to normal path

        # ── Path B: Normal inbound / standalone sorting scan ────────────────
        session_id   = self.get_or_create_session(scan_mode)
        all_products = self.matcher.db.get_all_products()

        # ── Conflict check 1: Cross-camera barcode conflict (inbound only) ──
        # If different cameras see barcodes that match different products,
        # it means two different items are in the scan zone simultaneously.
        # Saved as CONFLICT; no match attempted.
        if scan_mode == 'inbound' and all_barcodes_by_cam:
            cam_matches = {}
            for cam_id, cam_barcodes in all_barcodes_by_cam.items():
                for b in cam_barcodes:
                    barcode_data = b.get('data', '') if isinstance(b, dict) else str(b)
                    for p in all_products:
                        if p[6] and p[6] == barcode_data:
                            cam_matches[cam_id] = (barcode_data, p[0], p[1])

            # Build set of unique product IDs seen across all cameras
            unique_products = {v[1] for v in cam_matches.values()}

            if len(unique_products) > 1:
                # Multiple cameras agree on different products → definite conflict
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
                # All cameras that saw a barcode agree on the same product — use it
                winning_cam = next(iter(cam_matches))
                barcodes = [cam_matches[winning_cam][0]]
                self.get_logger().info(
                    f'[INBOUND] All cameras agree on product ID {list(unique_products)[0]} '
                    f'— proceeding with barcode {barcodes[0]}'
                )
            else:
                # No camera detected a barcode that matches the DB — fall through to OCR
                self.get_logger().info('[INBOUND] No barcode DB matches — falling through to OCR')

        # ── Conflict check 2: Multiple brands in OCR text ───────────────────
        # If the merged OCR text strongly matches more than one brand name,
        # it likely means labels from two different products are visible.
        # Logged as a warning; no match saved.
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

        # ── Conflict check 3: Multiple barcodes from different products ──────
        # If multiple barcodes were scanned and they match different DB products,
        # the result is ambiguous — reject rather than guess.
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

        # Log per-camera OCR summary for debugging
        if per_camera:
            self.get_logger().info('Per-camera OCR summary:')
            for cam_id, cam_result in per_camera.items():
                cam_ocr      = cam_result.get('ocr_text', 'None')
                cam_barcodes = cam_result.get('barcodes', [])
                self.get_logger().info(
                    f'  Cam {cam_id}: OCR="{cam_ocr}" | Barcodes={cam_barcodes}'
                )

        # ── Guard: reject if no barcode and OCR is noise ─────────────────────
        # Prevents wasted DB lookups and false low-confidence saves on blank scans.
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

        # ── Tie check (OCR-only, no barcode) ─────────────────────────────────
        # If two products score within 5 points, a barcode is required to
        # break the tie — save as TIE/AMBIGUOUS and return.
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

        # ── Try each barcode until an exact DB match is found ────────────────
        # For each matching barcode, also check OCR consistency:
        # if OCR strongly suggests a different product than the barcode, flag CONFLICT.
        result      = None
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
                # Score OCR against the barcode-matched product
                _, _, ocr_score, _ = self.matcher._enhanced_fuzzy_match(
                    ocr_text, None, [barcode_product]
                )
                # Compare against every other product to detect OCR-barcode disagreement
                for p in all_products:
                    if p[0] == barcode_product[0]:
                        continue  # Skip the barcode-matched product itself
                    _, _, other_score, _ = self.matcher._enhanced_fuzzy_match(
                        ocr_text, None, [p]
                    )
                    if other_score > ocr_score and other_score >= 40:
                        # OCR favours a different product than the barcode — CONFLICT
                        self.get_logger().warn(
                            f'[CONFLICT] Barcode says "{barcode_product[1]}" '
                            f'but OCR strongly suggests "{p[1]}" '
                            f'(barcode_ocr={ocr_score:.1f} vs ocr_only={other_score:.1f})'
                        )
                        # Delete the matched scan and replace with a CONFLICT record
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
                break  # Barcode matched cleanly — stop trying other barcodes

        # ── Fall back to OCR-only if no barcode matched cleanly ──────────────
        if not result or not result['matched'] or not result.get('barcode_matched'):
            barcode_used = None

            # Check for tie again before attempting OCR-only match
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

            # Final attempt: pure OCR match with no barcode assistance
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

        # ── Final result — matched product found ─────────────────────────────
        details        = (result.get('match_details') or {}) if result else {}
        result_qty     = result.get('quantity')
        result_qty_src = result.get('quantity_source', quantity_source)

        if result and result['matched']:
            if result['accuracy'] < 95.0:
                # Match found but below the 95% success criterion — flag as low confidence
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
                # ≥95% accuracy — verified match, log result and update session totals
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

                # Print running quantity totals for this session after every confirmed match
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
    rclpy.spin(node)       # Keep node alive, processing all subscribed topic callbacks
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()