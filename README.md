# GripVT

GripVT is a marker-displacement visuotactile sensor that estimates grip strength from musculotendon-induced skin deformation on the anterior wrist. This repository is the tracking software: it reads a 9x11 grid of 99 markers printed on the forearm, follows how each one moves relative to a captured baseline, and records that displacement per session.

<!-- TODO: screenshot of SessionWindow live feed with tracked markers -->

## How tracking works

Each frame goes through three stages, run in `core/tracker.py`:

1. **Detection** (`core/detector.py`) - threshold the frame and find marker blobs as connected components.
2. **Prediction** (`core/kalman.py`) - a Kalman filter predicts where each marker should be, one state per marker, initialized once at baseline capture and never added to or removed mid-session.
3. **Assignment** (`core/hungarian.py`) - detections are matched to predicted states with `scipy.optimize.linear_sum_assignment`, gated at 100 px so a marker can't jump to a neighbor's position.

Displacement is always measured against the baseline frame (`dx = x_current - x_baseline`), not frame to frame. Marker centroids are undistorted as points (`cv2.fisheye.undistortPoints`) for this math; a separate full-frame remap exists only for the live display.

<!-- TODO: diagram of detect -> predict -> assign -> correct loop -->

## App flow

```
LandingWindow -> NewParticipantWindow -> CalibrationWindow -> SessionWindow
```

`SessionWindow` covers the live camera feed, baseline capture, and trial recording, split into mixins under `ui/session/` (`baseline.py`, `recording.py`, `camera_feed.py`, `progress.py`, `modals.py`).

## Running it

Requires Python 3.10+ and a connected camera (built for a fixed-focus 1024x768 sensor; see `calibration/intrinsics/` for the fisheye calibration used to undistort it).

```bash
pip install -r requirements.txt
python main.py
```

Session output is written to `recordings/participantNNN_<timestamp>/sessionN_<timestamp>/` as CSV and PNG; nothing is written to the project root.

## Status

- Bench characterization and its planned test suite were descoped; this is now a human-data-collection tool only.
- Detection parameters (`GATE_PX`, blob area bounds in `core/detector.py`) were tuned on a fisheye phantom-slab rig, not on real GripVT footage, so the first recording sessions will likely need retuning.
- The marker grid geometry (pitch, orientation) has not yet been fully verified against real footage.
