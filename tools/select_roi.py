"""Interactive ROI selector — run once after mounting the camera.

Captures a single frame from the camera, lets you draw a rectangle
around the elastomer surface, and saves the result to roi.json at
the project root. The viewer app loads this on startup and crops
every frame to this region before passing it to the MDM pipeline.

Usage:
    python tools/select_roi.py
    python tools/select_roi.py --camera 1   # if camera index is not 0
"""
import argparse
import json
from pathlib import Path

import cv2

ROOT = Path(__file__).parent.parent
ROI_PATH = ROOT / "roi.json"


def select_roi(camera_index: int) -> None:
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {camera_index}")

    print("Capturing frame...")
    for _ in range(5):  # flush a few frames so auto-exposure settles
        cap.read()
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise RuntimeError("Failed to capture frame from camera")

    print("Draw a rectangle around the elastomer surface.")
    print("  ENTER or SPACE — confirm selection")
    print("  C              — cancel")

    roi = cv2.selectROI("Select elastomer ROI", frame, showCrosshair=True, fromCenter=False)
    cv2.destroyAllWindows()

    if roi == (0, 0, 0, 0):
        print("No ROI selected — roi.json not updated.")
        return

    x, y, w, h = (int(v) for v in roi)
    config = {"x": x, "y": y, "w": w, "h": h}

    with open(ROI_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"ROI saved to {ROI_PATH}")
    print(f"  x={x}, y={y}, w={w}, h={h}")

    preview = frame[y : y + h, x : x + w]
    cv2.imshow("ROI preview — press any key to close", preview)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Select and save elastomer ROI")
    parser.add_argument("--camera", type=int, default=0, help="Camera index (default: 0)")
    args = parser.parse_args()
    select_roi(args.camera)


if __name__ == "__main__":
    main()
