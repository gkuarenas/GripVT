"""Throwaway dev tool: quantify how much the blur pre-processing knob (see
core/detector.py's denoise()) reduces per-marker centroid jitter.

Captures N frames once (from a video file or the live camera), then runs
detection over that identical frame buffer twice -- once with blur forced
off, once with detector_params.json's currently-saved ("tuned") params --
so both passes see exactly the same noise/motion instants and only the
blur setting differs. Detections are clustered across frames the same way
Tracker.capture_baseline does (gated nearest-neighbour via
core.hungarian.assign(), since the sensor doesn't move), then each
cluster's centroid std-dev is computed and averaged.

Not wired into the app; nothing written to output/ or the project root.

Usage:
    python tools/measure_jitter.py --video path/to/clip.mp4
    python tools/measure_jitter.py --camera 1 --frames 90
"""
import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))
from core.calibration import CameraCalibration  # noqa: E402
from core.detector import detect, load_params, preprocess  # noqa: E402
from core.roi import DEFAULT_ALPHA, ROICrop  # noqa: E402
from core.tracker import BASELINE_CLUSTER_RADIUS_PX, BASELINE_PRESENCE_FRACTION, _cluster_across_frames  # noqa: E402

DEFAULT_CAMERA_INDEX = 1
DEFAULT_FRAME_COUNT = 60


def _capture_gray_frames(
    cap: cv2.VideoCapture, cal: CameraCalibration, roi: ROICrop, n_frames: int
) -> list[np.ndarray]:
    """Same undistort -> rotate -> crop -> greyscale pipeline as
    calibrate_detector.py's _read_processed_gray(), buffered once so both
    param configurations run against identical frames."""
    frames: list[np.ndarray] = []
    for _ in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        frame = cal.undistort(frame)
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = roi.crop(frame)
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    return frames


def _detect_all(frames: list[np.ndarray], roi: ROICrop, params: dict) -> list[list[tuple[float, float, float]]]:
    per_frame_dets: list[list[tuple[float, float, float]]] = []
    for gray in frames:
        proc = preprocess(gray, params)
        mask = roi.sensor_mask(proc.shape)
        if mask is not None:
            proc = cv2.bitwise_and(proc, mask)
        per_frame_dets.append(detect(gray, proc, params))
    return per_frame_dets


def mean_centroid_std_dev(per_frame_dets: list[list[tuple[float, float, float]]], n_frames: int) -> tuple[float, int]:
    """Clusters detections into per-marker tracks, keeps only clusters seen
    in >= BASELINE_PRESENCE_FRACTION of frames (a track seen in one or two
    frames is a transient false positive, not a marker, and its std-dev is
    meaningless), and returns the mean radial centroid std-dev in px plus
    how many marker tracks that mean was computed over."""
    clusters = _cluster_across_frames(per_frame_dets, BASELINE_CLUSTER_RADIUS_PX)
    jitters: list[float] = []
    for xs, ys, _areas in clusters:
        if len(xs) < BASELINE_PRESENCE_FRACTION * n_frames:
            continue
        jitters.append(float(np.hypot(np.std(xs), np.std(ys))))
    if not jitters:
        return 0.0, 0
    return float(np.mean(jitters)), len(jitters)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", type=str, default=None, help="Path to a recorded clip; live camera if omitted.")
    parser.add_argument("--camera", type=int, default=DEFAULT_CAMERA_INDEX, help="Camera index if --video is omitted.")
    parser.add_argument("--frames", type=int, default=DEFAULT_FRAME_COUNT, help="Number of frames to buffer and compare.")
    args = parser.parse_args()

    roi = ROICrop()
    roi_config = roi.config
    alpha = roi_config.alpha if roi_config is not None else DEFAULT_ALPHA
    cal = CameraCalibration(alpha=alpha)

    source = args.video if args.video is not None else args.camera
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")
    if args.video is None:
        cal_w, cal_h = cal.image_size
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)

    frames = _capture_gray_frames(cap, cal, roi, args.frames)
    cap.release()
    if len(frames) < 2:
        raise RuntimeError(f"Only captured {len(frames)} frame(s); need at least 2 to measure jitter.")
    print(f"Captured {len(frames)} frames from {'video ' + args.video if args.video else f'camera {args.camera}'}.")

    params_tuned = load_params()
    params_none = dict(params_tuned)
    params_none["blur"] = "none"

    std_none, n_none = mean_centroid_std_dev(_detect_all(frames, roi, params_none), len(frames))
    std_tuned, n_tuned = mean_centroid_std_dev(_detect_all(frames, roi, params_tuned), len(frames))

    print(f"blur=none    : mean centroid std-dev = {std_none:.3f} px  ({n_none} marker tracks)")
    print(
        f"blur={params_tuned['blur']} (tuned): mean centroid std-dev = {std_tuned:.3f} px  "
        f"(ksize={params_tuned['blur_ksize']}, sigma={params_tuned['blur_sigma']})  ({n_tuned} marker tracks)"
    )
    if std_none > 0:
        delta = std_none - std_tuned
        pct = 100.0 * delta / std_none
        print(f"delta        : {delta:.3f} px ({pct:.1f}% reduction)")


if __name__ == "__main__":
    main()
