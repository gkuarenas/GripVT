"""Interactive ROI selector — run once after mounting the camera.

Captures a single frame from the camera, undistorts it using the latest
camera calibration, lets you draw a rectangle around the elastomer
surface, and saves the box to roi.json at the project root alongside
the alpha it was drawn under. Consumers (viewer, calibrate_detector)
read that alpha back and undistort with it before cropping, so the
box always lines up with the frame it's applied to — undistorting
with a different alpha remaps pixel positions and misaligns the crop.

Usage:
    python tools/select_roi.py
    python tools/select_roi.py --camera 1   # if camera index is not 0
    python tools/select_roi.py --alpha 0    # select under "crop" undistort mode instead
"""
import argparse
import json
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
ROI_PATH = ROOT / "roi.json"

sys.path.insert(0, str(ROOT))
from core.calibration import CameraCalibration  # noqa: E402
from core.roi import DEFAULT_ALPHA  # noqa: E402


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

    print("Draw a rectangle around the elastomer surface.")
    print("  ENTER or SPACE — confirm selection")
    print("  C              — cancel")

    roi = cv2.selectROI("Select elastomer ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No ROI selected — roi.json not updated.")
        return

    x, y, w, h = (int(v) for v in roi)
    config = {"x": x, "y": y, "w": w, "h": h, "alpha": alpha}

    with open(ROI_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"ROI saved to {ROI_PATH}")
    print(f"  x={x}, y={y}, w={w}, h={h}, alpha={alpha}")

    preview = frame[y : y + h, x : x + w]
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
