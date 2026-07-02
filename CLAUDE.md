# CLAUDE.md

## Project
**GripVT** — Marker displacement tracker for the GripVT visuotactile forearm sensor.

Two apps (both in this repo):
- `viewer.py` — live feed + baseline capture + session recording. **Current priority.**
- `test_app.py` — structured test suite with a main menu (not yet implemented).

The pipeline (threshold + connected-component detection → Kalman tracking → Hungarian assignment) was ported from `mdm-kalman` (phantom slab testing). Core logic is the same; geometry and camera differ.

## Stack
- **Language:** Python 3.10+
- **GUI:** CustomTkinter
- **Vision:** OpenCV (`cv2`), NumPy
- **Tracking:** SciPy (`linear_sum_assignment`, `linalg`)
- **Output:** CSV (stdlib), PNG (OpenCV)
- **Hardware:** HX711 load cell + Ender 3 V2 rig (test app only)
- **Deploy target:** Local desktop (Windows / macOS / Linux)

## Camera
16:9 aspect ratio camera. **No fisheye undistortion.** Camera is loaded raw via `cv2.VideoCapture`.
When calibration is eventually added, it goes in `core/calibration.py`, precomputed once at startup — never inside the frame loop.

## Marker geometry
11×11 grid = **121 markers** on a curved cylindrical forearm surface.

**Important:** `GATE_PX` (default 100 px) and the marker blob area bounds (`min_area`/`max_area`) in `detector.py` were tuned for a fisheye phantom camera and **must be retuned** once real GripVT footage is available. Expect the first session to need parameter adjustment.

`zdisplacement.py` is present but its curved-surface geometry adaptation is **pending**. The flat-plane formula from the phantom is a placeholder.

## Structure
```
GripVT/
  viewer.py               # Viewer app entry point
  test_app.py             # Test app entry point (future)
  requirements.txt
  core/
    detector.py           # Threshold + connected-component marker detection
    kalman.py             # Kalman state management (predict / correct)
    hungarian.py          # Assignment via scipy.optimize.linear_sum_assignment
    tracker.py            # Orchestrates predict -> assign -> correct per frame
    zdisplacement.py      # Z-displacement estimation (curved geometry pending)
  ui/
    viewer_window.py      # CTk main window for viewer app
    overlay.py            # Frame annotation drawing
  output/
    writer.py             # Base CSV writer
    checkpoint.py         # Baseline checkpoint (save / load)
    sessions/             # Timestamped session folders (CSV + overlay PNG)
  ender/
    jog_control.py        # Ender 3 V2 jog control (test app only)
  loadcell_calibration/
    loadcell_calibration.ino
```

## Planned test app structure (not yet built)
Main menu → user selects:
1. **Sensitivity, Uniformity & Repeatability** — dx/dy vs applied load; spatial uniformity across the sensor surface
2. **Marker Stability** — drift over time under sustained load
3. **Hysteresis** — load/unload cycles, hysteresis index

Test-specific output writers (`sensitivity_writer.py`, `stability_writer.py`, `hysteresis_writer.py`) should be added to `output/` only when the test app is being actively built.

## Commands
```bash
# Run viewer app
python viewer.py

# Install dependencies
pip install -r requirements.txt
```

## Conventions
- **Displacement is always baseline-relative.** `dx = x_current - x_baseline`. Never frame[k] − frame[k-1].
- **Kalman states are initialised once at Capture Baseline and never added or removed during a session.** Markers are fixed on the sensor.
- **No undistortion.** Camera loads raw. Do not add `initUndistortRectifyMap` anywhere in the pipeline until a calibration module is explicitly introduced.
- **All session output goes to `output/sessions/<timestamp>/`.** Never write CSV or PNG to the project root.

## Do
- State "Guillaume" during the beginning of every response, may be varied but always answer with my name during the start of every response.
- When asked to commit changes, make sure to follow conventional commits, do not go above 100 characters, do not include "edited by Claude Code" or the likes. Just a simple commit would do. no need to write who wrote.

## Don't
| Anti-pattern | Do this instead |
|---|---|
| Spawn new Kalman states mid-session | Initialise all states at Capture Baseline only |
| Compute undistortion per frame | Wait for `core/calibration.py`; precompute once at startup |
| Use optical flow for tracking | Threshold + connected-component centroid + Hungarian assignment only |
| Set `GATE_PX` below 80 px | Default 100 px; retune for GripVT footage but keep ≥ 80 |
| Write output to project root | Write to `output/sessions/<timestamp>/` always |
| Add test writers before test app is being built | Keep `output/` minimal during viewer phase |
| Use the flat-plane z-displacement formula as authoritative | It is a placeholder; curved-surface adaptation is pending |
