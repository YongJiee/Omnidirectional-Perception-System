#!/bin/bash
# =============================================================
# OPS Live Demo — OCR Word Injector
# =============================================================
# Bypasses the Pi cameras and ROS2 image pipeline entirely.
# Publishes a hand-crafted JSON payload directly to /ocr_results,
# which database_matcher_node consumes exactly as if ocr_node
# had produced it — useful for assessor demos and resilience tests.
#
# Usage:
#   ./inject.sh                        (inbound, no barcode)
#   ./inject.sh sorting                (sorting mode)
#   ./inject.sh inbound ABC-abc-12345  (inbound + barcode)
# =============================================================

# Default to 'inbound' if no mode argument is supplied
MODE=${1:-inbound}

# Default to 'null' (JSON null) if no barcode argument is supplied
BARCODE=${2:-null}

# Wrap barcode value in JSON double-quotes if one was provided;
# leave as bare null so the JSON payload stays schema-compliant
if [ "$BARCODE" != "null" ]; then
    BARCODE="\"$BARCODE\""
fi

# Print a clear header so the operator knows what is about to be injected
echo ""
echo "========================================"
echo "  OPS Live Demo — OCR Injector"
echo "  Mode    : $MODE"
echo "  Barcode : $BARCODE"
echo "========================================"
echo ""

# Prompt the operator to type the OCR words to inject
# (e.g. brand name, product name, or any text that would appear on the box)
echo -n "  Enter OCR words: "
read WORDS

# Abort if the operator pressed Enter without typing anything
if [ -z "$WORDS" ]; then
    echo "  No words entered. Exiting."
    exit 1
fi

# Generate a Unix timestamp (float) to fill overall_start.
# database_matcher_node uses this to compute end-to-end latency,
# so it should be as close to "now" as possible.
TIMESTAMP=$(python3 -c "import time; print(time.time())")

# -----------------------------------------------------------------------
# Publish one message to /ocr_results using the same JSON schema that
# ocr_node produces, so database_matcher_node needs no special handling.
#
# Key fields:
#   ocr_text        — words entered by the operator (space-separated)
#   barcode         — scanned barcode string, or null if absent
#   scan_mode       — 'inbound' or 'sorting', controls DB match logic
#   overall_start   — timestamp used to measure end-to-end latency
#   num_cameras     — 0 signals this is an injected (not real) capture
#   end_to_end_time — null; matcher will compute it from overall_start
#   clock_offset    — 0.0; no Pi↔WSL clock difference in injection mode
#   pi_cycle_time   — null; no real Pi capture occurred
#   batch_save_dir  — null; no images were saved to disk
#   per_camera      — empty dict; no per-camera breakdown available
#   cameras_with_ocr— 0; no real cameras involved
#   injected        — true; flags this record as a manual injection in DB
# -----------------------------------------------------------------------
ros2 topic pub --once /ocr_results std_msgs/msg/String \
  "data: '{\"ocr_text\": \"$WORDS\", \"barcode\": $BARCODE, \"scan_mode\": \"$MODE\", \"overall_start\": $TIMESTAMP, \"num_cameras\": 0, \"end_to_end_time\": null, \"clock_offset\": 0.0, \"pi_cycle_time\": null, \"batch_save_dir\": null, \"per_camera\": {}, \"cameras_with_ocr\": 0, \"injected\": true}'" \
  2>/dev/null

# Confirm injection to the operator and direct them to the right terminal
echo ""
echo "  ✅ Injected: \"$WORDS\""
echo "  → Check database_matcher terminal for result"
echo ""