# Wire Tracker into Viewer — live tracked-marker overlay

**Date:** 2026-07-02
**Branch:** `worktree-wire-tracker-viewer` (off `feat/marker-displacement-v1`)
**Goal:** Replace the raw undistorted+cropped display in `ui/viewer_window.py`
with live tracked-marker output driven by `core/tracker.py::Tracker` — a
Capture Baseline flow that locks the 121 Kalman states, a per-frame overlay of
tracked marker IDs / displacement, and a Start/Stop Session recording flow.

## Decisions (confirmed with Guillaume, 2026-07-02)

1. **Undistort path:** Route `Tracker` through `core/calibration.py::CameraCalibration`
   (the same alpha/balance path `roi.json` was authored against). Remove
   Tracker's private `K→K` fisheye undistort so there is **one** undistort path.
2. **Baseline grab:** Async over ~15 `_poll()` ticks (non-blocking), then lock.
3. **Display:** Draw overlay on the full undistorted frame, then crop to the ROI
   box for display (keeps today's zoomed-in cropped view).
4. **Scope:** Include Start/Stop Session recording (displacement CSV + overlay
   PNG to `output/sessions/<timestamp>/`).

## Key constraints (CLAUDE.md — must hold)

- Displacement always baseline-relative (`dx = x - x_baseline`). ✔ enforced in Tracker.
- Kalman states locked once at Capture Baseline, never added/removed mid-session. ✔
- `gate_px` stays ≥ 80px (currently 100). ✔ do not lower.
- No new undistortion code paths — we **unify** onto `CameraCalibration`, not add a second.
- All session output → `output/sessions/<timestamp>/`. Never project root.

## Subtleties surfaced during investigation

- **`calibration.json` is absent at repo root** → `Tracker()`'s default
  `calib_path` would crash. Fisheye K/D lives at
  `camera_calibration/results/20260701_180005_calibration/` (the `model:"fisheye"`,
  4-coeff folder; the older `174717` is *standard* 5-coeff and won't work with
  `cv2.fisheye`). Routing through `CameraCalibration` (which loads the latest
  folder) removes this dependency entirely.
- **mm-math intrinsics change under the unified path.** `CameraCalibration`
  remaps to a *new* camera matrix `K_new` (fisheye `estimateNewCameraMatrix`
  at the chosen `balance`), so undistorted pixel coords live in `K_new` space,
  not raw `K`. Tracker's px→mm conversion (`H_MM/fx`, `cx`/`cy`) must therefore
  use `K_new`, not raw `K`. `CameraCalibration` must expose `K_new`.
- **HUD collision.** `overlay.draw_overlay` writes its HUD at full-frame
  `(8,20)`, which is outside the ROI crop `(307,113,…)` → vanishes after crop.
  Viewer draws its own status text on the cropped display instead.
- **Alpha lock.** The alpha toggle remaps pixels; changing it after baseline
  invalidates the locked baseline. Disable the toggle once baseline is locked.
- **Session CSV reuse.** `output/writer.py::CSVWriter` already writes exactly the
  `MarkerRecord` fields via a window-based API. Reuse it (buffer per frame, flush
  once at stop as a single `window_type="session"`, `rep=0`, `force_n=0`) rather
  than adding a viewer-specific writer — respects "no new test writers yet".

---

## Phase 0 — Expose `K_new` from `CameraCalibration`

- [x] In `core/calibration.py`, store the computed new camera matrix (`K_new`)
      for both the `fisheye` and `standard` branches.
- [x] Expose convenience intrinsics: `K_new` (3×3) plus `fx`, `fy`, `cx`, `cy`
      properties derived from it. Leave `undistort()` and `image_size` unchanged
      so existing consumers (`calibrate_detector.py`, current viewer) are unaffected.

## Phase 1 — Refactor `Tracker` onto `CameraCalibration`

- [x] Change `Tracker.__init__` signature to accept an injected
      `CameraCalibration` (and optional `ROICrop`), dropping `calib_path` and the
      private `initUndistortRectifyMap`/`map1`/`map2`.
- [x] Use `calibration.fx/fy/cx/cy` (`K_new`) for `_fx/_fy/_cx/_cy` used by the
      baseline mm map and by `zdisplacement.compute`.
- [x] `Tracker.undistort()` delegates to `calibration.undistort()`, preserving the
      resize-to-`image_size` guard for mismatched frame sizes.
- [x] Keep `gate_px = 100.0`, `load_params()`, baseline clustering, and
      `process_frame` logic unchanged otherwise.
- [x] Update the stale NOTE comment (lines ~102–110) — the two-calibration-path
      mismatch is now resolved by unification.

## Phase 2 — Viewer: shared calibration + Tracker wiring

- [x] In `ui/viewer_window.py`, construct **one** `CameraCalibration(alpha=roi.alpha)`
      and pass it into `Tracker(calibration, roi)`; the same instance also serves
      the pre-baseline display crop.
- [x] Keep the raw `cap.read()` frame; pass it **unmodified** to the tracker
      (tracker owns undistort + crop + mask). Do not pre-undistort/crop for tracking.
- [x] `_poll()` display states:
  - **Pre-baseline:** undistort (shared cal) + `roi.crop` + show, no overlay.
  - **Baseline captured, idle:** `tracker.process_frame(raw)` → `draw_overlay(
    last_undistorted, records, session_active=False)` → `roi.crop` → show.
  - **Recording:** same but `session_active=True` (arrows) + buffer CSV frame.
- [x] Draw a viewer-owned HUD on the cropped display (baseline status / frame idx /
      marker count / REC indicator), since `draw_overlay`'s HUD is cropped out.
- Implementation note: `_on_alpha_change` now also rebuilds `self._tracker`
  (not just `self._cal`) with the new `CameraCalibration`, since Tracker
  caches `fx/fy/cx/cy`/image size at construction. This is safe pre-baseline
  (no Kalman state to lose) and the alpha control is disabled the instant a
  baseline capture starts (see Phase 3), so this rebuild never runs once
  states exist.

## Phase 3 — Capture Baseline (async burst)

- [x] Add a "Capture Baseline" button to the control bar.
- [x] Import `BASELINE_FRAMES` from `core.tracker`; on click set
      `self._baseline_capturing = True`, `self._baseline_frames = []`.
- [x] In `_poll()`, while capturing: append the raw frame; show `Capturing
      baseline... n/15` status. At `len == BASELINE_FRAMES`, call
      `tracker.capture_baseline(frames)`, set baseline flag, clear capturing.
- [x] Surface the locked marker count (Tracker's own off-target warning already
      prints to console) so a bad capture is visible; allow re-capture (button
      re-enables as "Re-capture Baseline", disabled again only during session
      recording and mid-burst).
- [x] Lock/disable the alpha toggle once baseline is locked. Refinement beyond
      the original plan: also disabled the instant a capture *starts* (not just
      once it completes) — protects the burst itself from a mid-capture alpha
      change mixing pixel spaces across frames in the same baseline.

## Phase 4 — Start/Stop Session recording

- [x] Add a "Start Session" / "Stop Session" toggle button (enabled only after
      baseline is captured).
- [x] On start: `session_dir = make_session_dir()`; `CSVWriter(session_dir)`;
      set `session_active = True`; record wall-clock t0 for `timestamp_ms`.
- [x] Per recording frame: `writer.buffer_frame(records, tracker.frame_index,
      timestamp_ms)`.
- [x] On stop: `writer.write_window(rep=0, force_n=0.0, window_type="session")`;
      `cv2.imwrite(writer.png_path, last_overlay_frame)`; `writer.close()`; set
      `session_active = False`.
- [x] Ensure `_on_close()` stops an active session cleanly (flush + close writer,
      release camera).

## Phase 5 — Verify

- [x] `python -c "import ui.viewer_window"` — clean. Also verified
      `core.calibration`/`core.tracker` import and construct standalone.
- [x] No camera attached in this environment, so `python viewer.py` itself
      wasn't run live. Instead exercised the full pipeline headlessly with
      synthetic frames: `CameraCalibration(alpha=1.0)` construction (confirms
      `K_new` differs materially from raw `K` — fx 274.8 vs 488.1, validating
      the intrinsics-mismatch concern was real) → `Tracker(cal, roi)` →
      `capture_baseline([...15 frames])` → `process_frame()` →
      `draw_overlay()` → `roi.crop()` → `CSVWriter` buffer/write/close, all
      run without exceptions. **Live-camera run with a real GripVT feed is
      still outstanding** — flagged for Guillaume to verify on the rig.
- [x] Confirmed no second undistortion path exists: `core/tracker.py` no
      longer calls `initUndistortRectifyMap`/`cv2.fisheye.*` anywhere; the only
      remaining hits repo-wide are `core/calibration.py` (the canonical path)
      and `camera_calibration/calibrate_app.py` (a separate standalone
      calibration tool, out of scope).
- [x] Confirmed `gate_px` still `100.0` (≥ 80px). Confirmed
      `kalman.states.clear()`/`init_state()` only run inside
      `Tracker.capture_baseline()`, which the viewer only calls from
      `_finish_baseline_capture()` — itself gated by
      `_on_capture_baseline_click()` refusing to start a new burst while
      `_session_active` is `True`. So states cannot be re-initialised while a
      session is recording.
- [x] `mypy` clean on all four touched files (`core/calibration.py`,
      `core/tracker.py`, `ui/viewer_window.py`, `viewer.py`), including
      `--check-untyped-defs`. The one remaining mypy error
      (`output/writer.py:82`, `cv2.VideoWriter_fourcc` stub gap) is
      pre-existing and unrelated to this change (verified via git stash).

## Out of scope (explicitly deferred)

- Curved-surface z-displacement adaptation (`zdisplacement.py` stays a placeholder).
- Detector param / `gate_px` retuning for real GripVT footage.
- Reconciling test-app force/rep columns — session CSV writes them as 0/blank.
