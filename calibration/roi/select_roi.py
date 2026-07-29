"""Interactive ROI selector — run once after mounting the camera.

Captures a single frame from the camera, undistorts and rotates it to match
the viewer's display orientation, lets you draw a rectangle around the
elastomer surface, and saves the box to roi.json at the project root
alongside the alpha it was drawn under. Consumers (viewer, calibrate_detector)
read that alpha back and undistort+rotate with it before cropping, so the
box always lines up with the frame it's applied to — undistorting with a
different alpha, or cropping before rotating, remaps pixel positions and
misaligns the crop.

After the rectangle, optionally trace a polygon around the sensor's
actual (curved) outline within that crop — this is what lets detection
exclude the dark margins a bounding rectangle can't avoid on a
cylindrical sensor. Skipping it (ENTER with no points) keeps the old
rectangle-only behavior.

Usage:
    python calibration/roi/select_roi.py
    python calibration/roi/select_roi.py --camera 1   # if camera index is not 0
    python calibration/roi/select_roi.py --alpha 0    # select under "crop" undistort mode instead
"""
import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).parent.parent.parent
ROI_PATH = Path(__file__).parent / "roi.json"

sys.path.insert(0, str(ROOT))
from core.calibration import CameraCalibration  # noqa: E402
from core.roi import DEFAULT_ALPHA  # noqa: E402

MASK_WINDOW_NAME = "Trace sensor outline (optional)"


def _select_mask_polygon(cropped: np.ndarray) -> list[tuple[int, int]]:
    """Click points around the sensor outline; ENTER to finish, BACKSPACE to
    undo the last point, 's' to skip masking entirely. Returns [] if skipped
    or fewer than 3 points were placed (not enough to form a polygon)."""
    points: list[tuple[int, int]] = []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(MASK_WINDOW_NAME)
    cv2.setMouseCallback(MASK_WINDOW_NAME, on_mouse)

    print("Trace the sensor outline (optional):")
    print("  left-click  — add a point")
    print("  BACKSPACE   — undo last point")
    print("  ENTER       — finish (needs >= 3 points)")
    print("  s           — skip, keep rectangle-only crop")

    while True:
        display = cropped.copy()
        for pt in points:
            cv2.circle(display, pt, 3, (0, 255, 0), -1)
        if len(points) >= 2:
            cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 1)
        cv2.imshow(MASK_WINDOW_NAME, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == ord("s"):
            points = []
            break
        elif key == 8:  # BACKSPACE
            if points:
                points.pop()

    cv2.destroyWindow(MASK_WINDOW_NAME)
    if len(points) < 3:
        if points:
            print("Fewer than 3 points placed — skipping mask.")
        return []
    return points


def select_roi(camera_index: int, alpha: float) -> None:
    cal = CameraCalibration(alpha=alpha)
    cal_w, cal_h = cal.image_size

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    # Capture resolution must match the calibration's image_size, or the
    # undistort maps (built for that resolution) misalign with the frame.
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)

    print("Capturing frame...")
    for _ in range(5):  # flush a few frames so auto-exposure settles
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture frame from camera")

    frame = cal.undistort(frame)
    # Rotate to match the viewer's display orientation before drawing the box,
    # so the saved x/y/w/h are in the same (rotated) coordinate space the
    # viewer crops from -- otherwise margins tight on one edge here end up
    # tight on a different edge once the viewer rotates the frame.
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    print("Draw a rectangle around the elastomer surface.")
    print("  ENTER or SPACE — confirm selection")
    print("  C              — cancel")

    roi = cv2.selectROI("Select elastomer ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No ROI selected — roi.json not updated.")
        return

    x, y, w, h = (int(v) for v in roi)
    cropped = frame[y : y + h, x : x + w]

    mask_points = _select_mask_polygon(cropped)

    config: dict = {"x": x, "y": y, "w": w, "h": h, "alpha": alpha}
    if mask_points:
        config["mask_points"] = [list(p) for p in mask_points]

    with open(ROI_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"ROI saved to {ROI_PATH}")
    print(f"  x={x}, y={y}, w={w}, h={h}, alpha={alpha}")
    print(f"  mask_points={'none (rectangle only)' if not mask_points else len(mask_points)}")

    preview = cropped.copy()
    if mask_points:
        cv2.polylines(preview, [np.array(mask_points, dtype=np.int32)], True, (0, 255, 0), 1)
    cv2.imshow("ROI preview — press any key to close", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and save elastomer ROI")
    parser.add_argument("--camera", type=int, default=1, help="Camera index (default: 0)")
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"Undistort alpha/balance, saved into roi.json alongside the box (default: {DEFAULT_ALPHA})",
    )
    args = parser.parse_args()
    select_roi(args.camera, args.alpha)


if __name__ == "__main__":
    main()
