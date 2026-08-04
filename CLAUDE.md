# CLAUDE.md

## Project
**GripVT** — Marker displacement tracker for the GripVT visuotactile forearm sensor.

`main.py` is the sole app entry point for v1, launching the flow
LandingWindow → NewParticipantWindow → CalibrationWindow → SessionWindow (live feed,
baseline capture, session recording). Bench characterization (and its planned `test_app.py`
test suite) was descoped by the adviser on 2026-07-17; scope is now human data collection only.

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
Intrinsics are calibrated (fisheye model, near-rectilinear `D`) at the sensor's working distance; see `calibration/intrinsics/results/<timestamp>/calibration.json`.
Undistortion maps are precomputed **once at startup** in `core/calibration.py` from `K, D` — never inside the frame loop. See the undistortion policy under Conventions.

## Marker geometry
9×11 grid = **99 markers** on a curved cylindrical forearm surface.

**Important:** `GATE_PX` (default 100 px) and the marker blob area bounds (`min_area`/`max_area`) in `detector.py` were tuned for a fisheye phantom camera and **must be retuned** once real GripVT footage is available. Expect the first session to need parameter adjustment.

## Structure
```
GripVT/
  main.py                  # App entry point (LandingWindow -> NewParticipantWindow ->
                            # CalibrationWindow -> SessionWindow)
  requirements.txt
  core/
    detector.py           # Threshold + connected-component marker detection
    kalman.py             # Kalman state management (predict / correct)
    hungarian.py          # Assignment via scipy.optimize.linear_sum_assignment
    tracker.py            # Orchestrates predict -> assign -> correct per frame
  ui/
    session/              # SessionWindow split into mixins
      window.py           # SessionWindow scaffold + shared __init__
      progress.py         # Session/trial/target progress blocks
      baseline.py         # Baseline capture state machine
      recording.py        # Trial recording, timers, finalize
      modals.py           # Modal/toast plumbing
      camera_feed.py      # Camera polling + tracked-marker indicator
    overlay.py             # Frame annotation drawing
  calibration/
    intrinsics/            # Fisheye camera calibration (K, D)
    detector_tuning/       # Detector param tuning tool
    roi/                   # ROI selection tool (run once per camera mount)
  output/
    writer.py             # Base CSV writer
    participants.py       # De-identified participant persistence + external name key
  recordings/             # Per-participant data tree (gitignored)
    participantNNN_<ts>/
      participant.json    # De-identified: coded ID + anthropometrics + metadata (NO name)
      sessionN_<ts>/      # Session CSV, overlay PNG, baseline artifacts, grid_layout.json
```
Participant names live only in an external key file outside the repo:
`C:\Users\Guillaume\Documents\DLSU\THESIS\GripVT_keys\participant_key.json`
(`participantNNN -> {name, created_at}`), per URERB Form 6A.

## Commands
```bash
# Run app
python main.py

# Install dependencies
pip install -r requirements.txt
```

## Conventions
- **Displacement is always baseline-relative.** `dx = x_current - x_baseline`. Never frame[k] − frame[k-1].
- **Kalman states are initialised once at Capture Baseline and never added or removed during a session.** Markers are fixed on the sensor.
- **Detection runs on the undistorted frame, not raw.** Camera loads raw. `core/tracker.py::process_frame` applies the precomputed fisheye `remap` to the full frame *first*, then detects marker blobs on that rectified image — so `x`, `y`, and `area` (and everything derived from them: `dx`, `dy`, `dA`, `magnitude`) are already geometrically corrected at the point of measurement. This corrects blob *area* as well as position, which a points-only undistortion (`cv2.fisheye.undistortPoints` on raw-frame detections) would not, since fisheye distortion stretches area non-uniformly across the frame. One remap serves both this math path and the live display — never two. Maps are precomputed once from `K, D` in `core/calibration.py`; never recompute them inside the frame loop.
  (Changed 2026-08-03, see `docs/logs/20260731_152249_framerate_spike_plan.md` — this de-duplicated a redundant per-frame undistort, and detection has run on the rectified frame since. This note replaces an earlier points-for-math/remap-for-display split that no longer reflects the code.)
- **All participant/session output goes to `recordings/participantNNN_<ts>/sessionN_<ts>/`.** Never write to the project root. (`output/sessions/` is retired; `output/writer.make_session_dir` still points there and is retargeted in the session recording pass.)
- **De-identify at the source.** Never write a participant's name into `recordings/`; it belongs only in the external key file. See `output/participants.py`.
- **`marker_id` is derived from real grid geometry**, not sort order: `marker_id = row * n_cols + col`, row-major, origin top-left, 9 columns × 11 rows. `grid_layout.json` (per session) is authoritative for each marker's `(row, col)` — but `source == "fallback_sort"` means the ids are a bare `(y, x)` sort with no geometric meaning, so check `source` before trusting `layout`.

## Do
- State "Guillaume" during the beginning of every response, may be varied but always answer with my name during the start of every response.
- When asked to commit changes, make sure to follow conventional commits, do not go above 100 characters, do not include "edited by Claude Code" or the likes. Just a simple commit would do. no need to write who wrote.
- When asked to explain code and write notes, prioritize readability from a human standpoint. Avoid too much technicality, prioritize what would be important to digest, and make sure it could be understood by a junior developer with little experience.

## Don't
| Anti-pattern | Do this instead |
|---|---|
| Spawn new Kalman states mid-session | Initialise all states at Capture Baseline only |
| Recomputing undistortion maps per frame, or a second redundant `remap` call per frame | One full-frame `remap` per frame, precomputed maps from startup, shared by detection and display |
| Use optical flow for tracking | Threshold + connected-component centroid + Hungarian assignment only |
| Set `GATE_PX` below 80 px | Default 100 px; retune for GripVT footage but keep ≥ 80 |
| Write output to project root | Write to `recordings/participantNNN_<ts>/…` always |
| Store a participant's name alongside their data | Name → external key file only; `recordings/` stays de-identified |
