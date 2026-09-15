# CHRONOS — Quickstart & Execution Guide

This document contains all the commands to run, test, and benchmark the **CHRONOS** Reference Integrity Track Limits project.

---

## 1. Environment Setup

The repository uses a pre-configured Python 3.11 virtual environment in `.venv/`.

You can either prefix commands with `.venv/bin/python` or activate the virtual environment:

```bash
# Option A: Activate virtual environment
source .venv/bin/activate
python chronos.py

# Option B: Run directly with venv Python (no activation needed)
.venv/bin/python chronos.py
```

> **Note**: If you ever need to re-install dependencies:
> ```bash
> uv pip install --python .venv/bin/python numpy opencv-python scipy matplotlib PyQt6 ultralytics
> ```

---

## 2. Launch the Interactive GUI Console

The main application is an interactive PyQt6 console showing live track visualization, car contact patches, boundary integrity metrics, degradation sliders, wheel state timeline, and real-time verdicts.

### Default Synthetic Scenario (Violation)
```bash
.venv/bin/python chronos.py
```

### Different Synthetic Driving Paths
Choose how the car travels through the corner using `--path`:

```bash
# Clean lap - car stays completely within track limits
.venv/bin/python chronos.py --path clean

# Legal brush - tyre touches white line (legal under FIA rules)
.venv/bin/python chronos.py --path brush

# One wheel on line
.venv/bin/python chronos.py --path one_wheel_on_line

# All four wheels outside (violation event)
.venv/bin/python chronos.py --path violation

# Straight wide corner exit
.venv/bin/python chronos.py --path straight_wide
```

### Custom Footage & Real Video
Run the console against real race clips, GIFs, or image sequences:

```bash
# Real broadcast GIF sample
.venv/bin/python chronos.py --video "data/real/WhatsApp_GIF_2026-09-12_at_23.06.02.gif"

# Real still frame
.venv/bin/python chronos.py --video "data/real/WhatsApp_Image_2026-09-12_at_22.56.47_1_.jpeg"

# Any custom MP4 / video clip
.venv/bin/python chronos.py --video path/to/your/clip.mp4
```

### Visual Themes
Choose between three custom themes with `--theme`:

```bash
# F1 broadcast livery (default)
.venv/bin/python chronos.py --theme f1

# Original dark instrumentation console
.venv/bin/python chronos.py --theme instrument

# High-contrast brutalist styling
.venv/bin/python chronos.py --theme brutal
```

### Reference Degradation Slider Types
Select which degradation kind is controlled by the live slider with `--kind`:

```bash
# Rubber buildup (marbles / tyre marks, default)
.venv/bin/python chronos.py --kind rubber

# Dust / sand contamination
.venv/bin/python chronos.py --kind dust

# Wet track surface reflections
.venv/bin/python chronos.py --kind wet

# Faded line paint
.venv/bin/python chronos.py --kind fade

# Sun glare
.venv/bin/python chronos.py --kind glare

# Heavy cast shadows
.venv/bin/python chronos.py --kind shadow
```

---

## 3. Console Keyboard Shortcuts & Controls

When the console window is active:

| Key / Control | Action |
| :--- | :--- |
| `O` | Open file dialog to load video, image, or folder |
| `Space` | Play / Pause playback |
| `S` | Toggle diagnostic statistics panel |
| `I` | Cycle integrity sampling frequency (`1:1` → `1:5` → `1:25`) |
| `E` | Export audit report & violation frames to `./output/` |
| `Slider` | Adjust degradation intensity on the fly |
| `Timeline` | Scrub through frames |

---

## 4. Benchmark & Data Generation Tools

Run individual analytical modules and generate synthetic benchmarks:

```bash
# Generate a single synthetic corner frame and ground-truth JSON
.venv/bin/python -m benchmark.generate --out data/frame.jpg

# Test classical boundary detection on an image against ground truth
.venv/bin/python -m chronos.boundary --image data/frame.jpg --gt data/frame_gt.json

# Run degradation sweep across all 17 corner scenes (generates comparison plots)
.venv/bin/python -m benchmark.sweep

# Run real-footage plausibility report
.venv/bin/python -m tools.real_run

# Run real-footage quick validation
.venv/bin/python -m tools.real_test
```

---

## 5. Running Tests

CHRONOS includes 39 synthetic ground-truth unit tests and 7 real-footage regression checks:

```bash
# Boundary detection tests (8/8)
.venv/bin/python tests/test_boundary.py

# Boundary integrity engine & calibration tests (12/12)
.venv/bin/python tests/test_integrity.py

# Spatiotemporal, car tracking, wheel states & decision tests (19/19)
.venv/bin/python tests/test_temporal.py

# Real broadcast footage regression checks (7/7)
.venv/bin/python tests/real_regression.py

# Run all tests in sequence
.venv/bin/python tests/test_boundary.py && \
.venv/bin/python tests/test_integrity.py && \
.venv/bin/python tests/test_temporal.py && \
.venv/bin/python tests/real_regression.py
```
