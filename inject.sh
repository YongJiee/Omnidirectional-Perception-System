#!/bin/bash
# =============================================================
# OPS Live Demo — OCR Word Injector
# =============================================================
# Usage:
#   ./inject.sh                        (inbound, no barcode)
#   ./inject.sh sorting                (sorting mode)
#   ./inject.sh inbound ABC-abc-12345  (inbound + barcode)
# =============================================================

MODE=${1:-inbound}
BARCODE=${2:-null}

# Wrap barcode in quotes if provided, else keep as null
if [ "$BARCODE" != "null" ]; then
    BARCODE="\"$BARCODE\""
fi

echo ""
echo "========================================"
echo "  OPS Live Demo — OCR Injector"
echo "  Mode    : $MODE"
echo "  Barcode : $BARCODE"
echo "========================================"
echo ""
echo -n "  Enter OCR words: "
read WORDS

if [ -z "$WORDS" ]; then
    echo "  No words entered. Exiting."
    exit 1
fi

TIMESTAMP=$(python3 -c "import time; print(time.time())")

ros2 topic pub --once /ocr_results std_msgs/msg/String \
  "data: '{\"ocr_text\": \"$WORDS\", \"barcode\": $BARCODE, \"scan_mode\": \"$MODE\", \"overall_start\": $TIMESTAMP, \"num_cameras\": 0, \"end_to_end_time\": null, \"clock_offset\": 0.0, \"pi_cycle_time\": null, \"batch_save_dir\": null, \"per_camera\": {}, \"cameras_with_ocr\": 0, \"injected\": true}'" \
  2>/dev/null

echo ""
echo "  ✅ Injected: \"$WORDS\""
echo "  → Check database_matcher terminal for result"
echo ""