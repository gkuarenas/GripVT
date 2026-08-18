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

After the outline, optionally trace one or more holes — small reflection
spots inside the outline that must not count as markers. Each hole is
punched to 0 out of the (otherwise 255) mask by core.roi.sensor_mask().
Stop tracing holes by pressing 's' with no points placed.

Usage:
    python calibration/roi/select_roi.py
    python calibration/roi/select_roi.py --camera 1   # if camera index is not 0
    python calibration/roi/select_roi.py --alpha 0    # select under "crop" undistort mode instead
    python calibration/roi/select_roi.py --holes-only # repaint holes only, keep the saved rect/outline
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
HOLE_WINDOW_NAME = "Trace hole polygon (optional)"

OUTLINE_COLOR = (0, 255, 0)
HOLE_COLOR = (0, 0, 255)


def _select_mask_polygon(
    cropped: np.ndarray,
    window_title: str,
    context_polygons: list[tuple[list[tuple[int, int]], tuple[int, int, int]]] | None = None,
) -> list[tuple[int, int]]:
    """Click points around a polygon; ENTER to finish, BACKSPACE to undo the
    last point, 's' to skip. Returns [] if skipped or fewer than 3 points
    were placed (not enough to form a polygon). context_polygons are drawn
    underneath the in-progress polygon, each as (points, bgr_color), so a
    caller can show already-committed shapes (the outline, prior holes)."""
    points: list[tuple[int, int]] = []
    context_polygons = context_polygons or []

    def on_mouse(event: int, x: int, y: int, _flags: int, _param: None) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))

    cv2.namedWindow(window_title)
    cv2.setMouseCallback(window_title, on_mouse)

    print(f"{window_title}:")
    print("  left-click  — add a point")
    print("  BACKSPACE   — undo last point")
    print("  ENTER       — finish (needs >= 3 points)")
    print("  s           — skip")

    while True:
        display = cropped.copy()
        for poly, color in context_polygons:
            cv2.polylines(display, [np.array(poly, dtype=np.int32)], True, color, 1)
        for pt in points:
            cv2.circle(display, pt, 3, (0, 255, 0), -1)
        if len(points) >= 2:
            cv2.polylines(display, [np.array(points, dtype=np.int32)], False, (0, 255, 0), 1)
        cv2.imshow(window_title, display)

        key = cv2.waitKey(20) & 0xFF
        if key == 13:  # ENTER
            break
        elif key == ord("s"):
            points = []
            break
        elif key == 8:  # BACKSPACE
            if points:
                points.pop()

    cv2.destroyWindow(window_title)
    if len(points) < 3:
        if points:
            print("Fewer than 3 points placed — skipping.")
        return []
    return points


def _select_mask_holes(
    cropped: np.ndarray,
    outline: list[tuple[int, int]],
    existing_holes: list[list[tuple[int, int]]] | None = None,
) -> list[list[tuple[int, int]]]:
    """Loop the polygon picker to collect reflection-hole polygons. Draws the
    outline (green) and every already-committed hole (red, both existing_holes
    from a prior save and holes committed earlier in this same loop) for
    context. Stops when the operator presses 's' with no points placed."""
    existing_holes = existing_holes or []
    holes: list[list[tuple[int, int]]] = []
    print("Trace reflection holes (optional). Press 's' with no points to stop.")

    while True:
        context: list[tuple[list[tuple[int, int]], tuple[int, int, int]]] = []
        if outline:
            context.append((outline, OUTLINE_COLOR))
        for hole in existing_holes + holes:
            context.append((hole, HOLE_COLOR))
        hole = _select_mask_polygon(cropped, HOLE_WINDOW_NAME, context)
        if not hole:
            break
        holes.append(hole)

    return holes


def _capture_frame(camera_index: int, cal: CameraCalibration) -> np.ndarray:
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
    return frame


def select_roi(camera_index: int, alpha: float) -> None:
    cal = CameraCalibration(alpha=alpha)
    frame = _capture_frame(camera_index, cal)

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

    mask_points = _select_mask_polygon(cropped, MASK_WINDOW_NAME)
    holes = _select_mask_holes(cropped, mask_points)

    config: dict = {"x": x, "y": y, "w": w, "h": h, "alpha": alpha}
    if mask_points:
        config["mask_points"] = [list(p) for p in mask_points]
    if holes:
        config["mask_holes"] = [[list(p) for p in hole] for hole in holes]

    with open(ROI_PATH, "w") as f:
        json.dump(config, f, indent=2)

    print(f"ROI saved to {ROI_PATH}")
    print(f"  x={x}, y={y}, w={w}, h={h}, alpha={alpha}")
    print(f"  mask_points={'none (rectangle only)' if not mask_points else len(mask_points)}")
    print(f"  mask_holes={len(holes)}")

    _show_preview(cropped, mask_points, holes)


def select_holes_only(camera_index: int) -> None:
    if not ROI_PATH.exists():
        raise RuntimeError(f"{ROI_PATH} does not exist — run select_roi.py without --holes-only first.")

    raw = json.loads(ROI_PATH.read_text())
    alpha = float(raw.get("alpha", DEFAULT_ALPHA))
    cal = CameraCalibration(alpha=alpha)
    frame = _capture_frame(camera_index, cal)

    frame = cal.undistort(frame)
    frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)

    x, y, w, h = int(raw["x"]), int(raw["y"]), int(raw["w"]), int(raw["h"])
    cropped = frame[y : y + h, x : x + w]

    outline = [(int(p[0]), int(p[1])) for p in raw.get("mask_points", [])]
    existing_holes = [
        [(int(p[0]), int(p[1])) for p in hole] for hole in raw.get("mask_holes", [])
    ]

    new_holes = _select_mask_holes(cropped, outline, existing_holes)
    merged_holes = existing_holes + new_holes

    if merged_holes:
        raw["mask_holes"] = [[list(p) for p in hole] for hole in merged_holes]
    else:
        raw.pop("mask_holes", None)

    with open(ROI_PATH, "w") as f:
        json.dump(raw, f, indent=2)

    print(f"ROI saved to {ROI_PATH} (rectangle and outline unchanged)")
    print(f"  mask_holes={len(merged_holes)} ({len(new_holes)} new)")

    _show_preview(cropped, outline, merged_holes)


def _show_preview(
    cropped: np.ndarray,
    outline: list[tuple[int, int]],
    holes: list[list[tuple[int, int]]],
) -> None:
    preview = cropped.copy()
    if outline:
        cv2.polylines(preview, [np.array(outline, dtype=np.int32)], True, OUTLINE_COLOR, 1)
    for hole in holes:
        cv2.polylines(preview, [np.array(hole, dtype=np.int32)], True, HOLE_COLOR, 1)
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
    parser.add_argument(
        "--holes-only",
        action="store_true",
        help="Repaint reflection holes only, against the existing rectangle and outline in roi.json",
    )
    args = parser.parse_args()
    if args.holes_only:
        select_holes_only(args.camera)
    else:
        select_roi(args.camera, args.alpha)


if __name__ == "__main__":
    main()
