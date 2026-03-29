# Omnidirectional Perception System (OPS)

[![ROS2](https://img.shields.io/badge/ROS2-Humble-blue)](https://docs.ros.org/en/humble/)
[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![Platform](https://img.shields.io/badge/Platform-Raspberry%20Pi%204-red)](https://www.raspberrypi.com/)

> Multi-camera warehouse scanning system that captures all 6 faces of a package, extracts text and barcodes via OCR, and generates a Universal Product Passport — even when the primary barcode is missing or damaged.

**Built for:** RSE2109-Project 4 [2025/26 T2]  
**Team:** JIACHENG (System Engineer) · ALBERT (Project Manager) · YONG ZHUO (Mechanical) · YONG JIE (Technical Lead – Software)  
**Framework:** ROS2 Humble · Raspberry Pi 4 · WSL Ubuntu

---

## 🎯 Overview

Current warehouse inbound operations rely on single-point barcode scanning — if a barcode is obscured, damaged, or misoriented, the item's identity is lost. OPS eliminates this single point of failure by scanning all 6 faces of a package simultaneously using a multi-camera system, fusing OCR text and barcode data to identify any product regardless of label condition. Built in collaboration with an industry partner.

### Success Criteria

| Criteria | Target | Result |
|----------|--------|--------|
| **Efficiency** | ≤ 3 seconds end-to-end | ✅ 2.5–2.7s achieved |
| **Accuracy** | ≥ 95% match rate | ✅ 100% on test products |
| **Resilience** | Works without barcode | ✅ Single keyword sufficient for unique products |

---

## ✨ Features

### Core Capabilities
- **6-Face Coverage** — Cam0 + Cam2 at 45° (4 side faces) + Cam1 top-down (TOP in inbound / BOTTOM in sorting) + robot arm end-effector camera (TOP face before pick-up in sorting mode)
- **Dual Scan Modes** — Inbound mode (pallet validation with conflict detection) and Sorting mode (robotic arm integration with fast-path matching)
- **OCR + Barcode Fusion** — Tesseract PSM 11 / OEM 1 with pyzbar multi-pass detection (1×/2×/3× upscale + Otsu)
- **Perspective Correction** — OpenCV `getPerspectiveTransform` + `warpPerspective` to correct 45° angular distortion per face
- **Fuzzy Matching** — Weighted brand/product/keyword scoring via thefuzz with tie detection
- **Quantity Tracking** — OCR-extracted quantities with session-based inbound/sorting totals
- **Robotic Arm Integration** — `/robot_data` trigger-based capture with fast-path (≥95% name match) and camera-path (<95%)

### Technical Highlights
- Parallel OCR via `ThreadPoolExecutor` — wall time = max(cameras), not sum
- Autonomous Pi/WSL clock offset measurement — NTP-free, self-calibrates per session
- Cross-camera barcode conflict detection for inbound mixed-pallet rejection
- Trigger-based capture — `/inbound_trigger` for inbound, `/trigger_capture` for sorting
- SQLite database with session tracking, quantity flagging, and scan history
- Live demo injector (`inject.sh`) — bypasses cameras for custom-word testing

---

## 🏗️ System Architecture

```
Robot Arm WSL          Raspberry Pi 4              Your WSL
──────────────         ──────────────────          ──────────────────────────
robot_publisher   →    multi_camera_node      →    ocr_node
  /robot_data           Picamera2 SDK               Face split + perspective
  (product name,        3× IMX708 cameras           correction + Tesseract OCR
   pos1)                Cam0+Cam2: 45° sides        + pyzbar barcode detection
                        Cam1: top-down              ThreadPoolExecutor parallel
                        Sequential capture           /ocr_results ↓
                         /trigger_capture ←         database_matcher_node
                                                      SmartMatcher fuzzy scoring
                                                      SQLite cartoon_products.db
  /robot_command ←                                    pass / fail decision

  [Demo / Testing]
  inject.sh        →   /ocr_results (direct)  →   database_matcher_node
  test_image_publisher → /camera_N/...        →   ocr_node (no Pi needed)
```

---

## 🚀 Quick Start

### Prerequisites
- ROS2 Humble installed on WSL (Ubuntu)
- Python 3.10
- Dependencies: `pip install pytesseract pyzbar opencv-python thefuzz`
- Tesseract OCR: `sudo apt install tesseract-ocr`

### Build (required after fresh clone)
```bash
cd ~/Project4_ws
colcon build --packages-select ocr_processor camera_publisher
source install/setup.bash
```

> ⚠️ `cartoon_products.db` must be present in `src/ocr_processor/data/` before building.

---

### Launch — Test Mode (no Pi needed, 1 terminal)
Runs the full pipeline using local test images — no Pi or robotic arm required.
```bash
ros2 launch camera_publisher distributed_system.launch.py \
  test_mode:=true scan_mode:=inbound num_cameras:=3 \
  image_dir:=/home/yongjie/test_images
```

---

### Launch — Live Demo Injector (bypass cameras, 2 terminals)

**Terminal 1 — Start matcher node:**
```bash
cd ~/Project4_ws
source install/setup.bash
ros2 run ocr_processor database_matcher_node
```

**Terminal 2 — Run injector:**
```bash
cd ~/Project4_ws
source install/setup.bash
./inject.sh
```

**Inject with barcode:**
```bash
./inject.sh inbound ABC-abc-12345
```

**Inject in sorting mode:**
```bash
./inject.sh sorting
```

---

### Launch — Inbound Mode (real Pi)
```bash
ros2 launch camera_publisher distributed_system.launch.py \
  num_cameras:=3 \
  scan_mode:=inbound
```

### Launch — Sorting Mode (robotic arm)
```bash
ros2 launch camera_publisher distributed_system.launch.py \
  num_cameras:=3 \
  scan_mode:=sorting
```

### Trigger Inbound Scan Manually
```bash
ros2 topic pub --once /inbound_trigger std_msgs/msg/String "data: 'scan'"
```

---

## ⚙️ Configuration

### Launch Arguments

| Parameter | Default | Description |
|-----------|---------|-------------|
| `num_cameras` | `3` | Number of cameras (Pi side: 1–3) |
| `scan_mode` | `sorting` | `inbound` or `sorting` |
| `test_mode` | `false` | `true` = use local test images, no Pi needed |
| `image_dir` | `~/test_images` | Folder with test images (test mode only) |

### Network Setup

```bash
# Pi (192.168.10.2) ↔ WSL (192.168.10.1) — direct ethernet
export ROS_DOMAIN_ID=30
export ROS_LOCALHOST_ONLY=0
export ROS_IP=192.168.10.1   # set on WSL side

# For multi-machine (robot arm on WiFi + Pi on ethernet simultaneously)
export FASTRTPS_DEFAULT_PROFILES_FILE=~/fastdds_multicast.xml
```

### Camera Config (Pi `/boot/firmware/config.txt`)

```
camera_auto_detect=0
dtoverlay=camera-mux-4port,cam0-imx708,cam1-imx708,cam2-imx708,cam3-imx708
```

---

## 📂 Project Structure

```
Omnidirectional-Perception-System/
├── camera_publisher/                  # Pi-side ROS2 package
│   ├── launch/
│   │   └── distributed_system.launch.py
│   └── camera_publisher/
│       ├── multi_camera_node.py       # Picamera2 capture + ROS2 publish (sequential, Arducam mux)
│       └── camera_node.py             # Single camera node
├── ocr_processor/                     # WSL-side ROS2 package
│   ├── data/
│   │   └── cartoon_products.db        # Product catalogue (committed via git add -f)
│   └── ocr_processor/
│       ├── ocr_node.py                # Face split, perspective correction, OCR, barcode
│       ├── database_matcher_node.py   # Fuzzy matching, robot arm integration, DB save
│       ├── smart_match3_vF.py         # SmartMatcher — scoring, tie detection, quantity
│       ├── database_manager.py        # SQLite operations, sessions, flagged scans
│       └── test_image_publisher.py    # Simulates Pi cameras using local images (no Pi needed)
├── inject.sh                          # Live demo OCR word injector (bypasses cameras)
└── README.md
```

---

## 🧠 How It Works

### Inbound Mode
1. `/inbound_trigger` receives `"scan"` → Pi captures 3 cameras sequentially
2. Images sent to WSL as `CompressedImage` (JPEG Q85, 1280×720)
3. Each image split into left/right faces + perspective corrected
4. Tesseract OCR + pyzbar barcode run in parallel across all cameras
5. Cross-camera barcode conflict detection — rejects mismatched pallets
6. SmartMatcher accuracy formula:
   - **Barcode exact match** → 100% (overrides OCR scores)
   - **Brand + Product detected** → `(brand × 0.4) + (product × 0.5) + (keyword bonus, max 10%)`
   - **Product only, unique brand** → `product × 0.9 + keyword bonus`
   - **Brand only** → `brand × 0.6 + keyword bonus`
7. ≥95% → MATCHED, saved to DB with quantity and session ID

### Sorting Mode
1. Robot arm scans top face → publishes product name to `/robot_data`
2. DB pre-match on name: ≥95% → wait for `pos1` → publish `"pass"` to `/robot_command`
3. <95% → wait for `pos1` → publish `/trigger_capture` → Pi cameras capture
4. Camera OCR/barcode evaluated → `"pass"` or `"fail"` published to `/robot_command`

### Test Mode
- `test_image_publisher.py` replaces the Pi — reads images from a local folder and publishes them to the same ROS2 topics, so the full WSL pipeline runs without any hardware.

### Demo Injector (`inject.sh`)
- Skips cameras entirely — publishes a hand-crafted JSON payload directly to `/ocr_results`, allowing assessors to type any OCR words and see the matcher respond in real time.

---

## 🔧 Common Commands

```bash
# Fix ROS2 node discovery
ros2 daemon stop && ros2 daemon start && ros2 topic list

# Kill camera if device busy
sudo killall -9 cam  # or restart the Pi

# Check DB contents
cd ~/Project4_ws/src/ocr_processor/data
python3 -c "
import sqlite3
conn = sqlite3.connect('cartoon_products.db')
c = conn.cursor()
c.execute('SELECT id, product_name, brand FROM products')
[print(r) for r in c.fetchall()]
conn.close()
"

# Clear test scans
python3 -c "
import sqlite3
conn = sqlite3.connect('cartoon_products.db')
conn.execute('DELETE FROM scans')
conn.commit()
conn.close()
print('Scans cleared')
"
```

---

## 🙏 Acknowledgments

- **ROS2 Community** — Documentation and support
- **Tesseract OCR** — Open source OCR engine
- **pyzbar / ZBar** — Barcode and QR code detection
- **OpenCV** — Computer vision processing
- **Arducam** — Multi-camera adapter hardware