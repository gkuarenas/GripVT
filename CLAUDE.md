# CLAUDE.md

## Project
**GripVT** — Marker displacement tracker for the GripVT visuotactile forearm sensor.

`viewer.py` — live feed + baseline capture + session recording — is the sole app for v1.
Bench characterization (and its planned `test_app.py` test suite) was descoped by the
adviser on 2026-07-17; scope is now human data collection only.

The pipeline (threshold + connected-component detection → Kalman tracking → Hungarian assignment) was ported from `mdm-kalman` (phantom slab testing). Core logic is the same; geometry and camera differ.

## Stack
- **Language:** Python 3.10+
- **GUI:** CustomTkinter
- **Vision:** OpenCV (`cv2`), NumPy
- **Tracking:** SciPy (`linear_sum_assignment`, `linalg`)
- **Output:** CSV (stdlib), PNG (OpenCV)
- **Deploy target:** Local desktop (Windows / macOS / Linux)

## Camera
4:3 aspect ratio camera, native 1024×768 (GXIVISION S8M01, fixed-focus M12 lens). Camera is loaded raw via `cv2.VideoCapture`.
Intrinsics are calibrated (fisheye model, near-rectilinear `D`) at the sensor's working distance; see `camera_calibration/results/<timestamp>/calibration.json`.
Undistortion maps are precomputed **once at startup** in `core/calibration.py` from `K, D` — never inside the frame loop. See the undistortion policy under Conventions.

## Marker geometry
9×11 grid = **99 markers** on a curved cylindrical forearm surface.

**Important:** `GATE_PX` (default 100 px) and the marker blob area bounds (`min_area`/`max_area`) in `detector.py` were tuned for a fisheye phantom camera and **must be retuned** once real GripVT footage is available. Expect the first session to need parameter adjustment.

## Structure
```
GripVT/
  viewer.py               # Viewer app entry point
  requirements.txt
  core/
    detector.py           # Threshold + connected-component marker detection
    kalman.py             # Kalman state management (predict / correct)
    hungarian.py          # Assignment via scipy.optimize.linear_sum_assignment
    tracker.py            # Orchestrates predict -> assign -> correct per frame
  ui/
    viewer_window.py      # CTk main window for viewer app
    overlay.py            # Frame annotation drawing
  output/
    writer.py             # Base CSV writer
    sessions/             # Timestamped session folders (CSV + overlay PNG)
```

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
- **Undistortion = points for math, remap for display only.** Camera loads raw. Undistort the ~99 marker centroids per frame with `cv2.fisheye.undistortPoints` (cheap, points only) for all measurement/geometry math. Full-frame `remap` is for the **live display only** — never in the detection/tracking math path. Maps are precomputed once from `K, D` in `core/calibration.py`; never recompute them inside the frame loop.
- **All session output goes to `output/sessions/<timestamp>/`.** Never write CSV or PNG to the project root.

## Do
- State "Guillaume" during the beginning of every response, may be varied but always answer with my name during the start of every response.
- When asked to commit changes, make sure to follow conventional commits, do not go above 100 characters, do not include "edited by Claude Code" or the likes. Just a simple commit would do. no need to write who wrote.

## Don't
| Anti-pattern | Do this instead |
|---|---|
| Spawn new Kalman states mid-session | Initialise all states at Capture Baseline only |
| Full-frame `remap` in the math path, or recomputing maps per frame | Undistort **points** (`cv2.fisheye.undistortPoints`) per frame for math; `remap` for display only; precompute maps once at startup |
| Use optical flow for tracking | Threshold + connected-component centroid + Hungarian assignment only |
| Set `GATE_PX` below 80 px | Default 100 px; retune for GripVT footage but keep ≥ 80 |
| Write output to project root | Write to `output/sessions/<timestamp>/` always |
