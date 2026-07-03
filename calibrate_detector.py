"""Interactive detector calibration for GripVT footage.

Freeze a live frame (SPACE), tune threshold/area/morphology params with the
trackbars against the known curved 11x11 marker grid, then save the winning
parameters with 's'.
"""
import json
import time
from datetime import datetime

import cv2
import numpy as np

from core.calibration import CameraCalibration
from core.detector import _PARAMS_PATH, count_oversized_components, detect, load_params, preprocess
from core.roi import DEFAULT_ALPHA, ROICrop

CAMERA_INDEX = 1
WINDOW_NAME = "GripVT Detector Calibration"
CONTROLS_WINDOW_NAME = "Detector Controls"
OUTPUT_PATH = _PARAMS_PATH

# WIN32UI trackbars emit no hover events, so a literal tooltip is impossible --
# nudging a slider surfaces its description on the HUD instead, fading out
# after HINT_FADE_SECONDS so it doesn't permanently occupy the frame.
HINT_FADE_SECONDS = 2.0
PARAM_DESCRIPTIONS: dict[str, str] = {
    "thresh": "Global binary cutoff -- pixels above this become foreground. "
              "Lower catches dim markers, higher rejects noise.",
    "min_area": "Smallest blob area (px) kept as a marker -- filters tiny noise specks.",
    "max_area": "Largest blob area (px) kept as a marker -- filters merged/oversized blobs.",
    "erode": "Shrinks foreground blobs by 1px -- strips thin noise, may erode small markers.",
    "open": "Erode then dilate -- removes small noise specks without shrinking real blobs.",
    "dilate": "Grows foreground blobs by 1px -- fills small gaps, may merge close markers.",
    "clahe": "Local contrast boost (tiled histogram eq.) -- lifts dim edge markers, can amplify noise.",
    "clahe_clip": "CLAHE contrast-amplification limit -- higher lifts dim areas more but adds noise.",
    "adaptive": "Per-pixel local threshold instead of global thresh -- beats uneven lighting, but noisier cutoff.",
    "adaptive_C": "Adaptive threshold strictness -- more negative requires a pixel to stand out more from its neighborhood.",
    "tophat": "Subtracts a smoothed background estimate -- flattens uneven illumination before thresholding.",
    "tophat_ksize": "Top-hat structuring element size -- must be larger than a marker or the marker vanishes too.",
    "blur": "Smooths raw grey before thresholding (none/gaussian/median) -- reduces centroid jitter from pixel noise.",
    "blur_ksize": "Blur kernel size (odd) -- larger smooths more but can erode small/thin blob detail.",
    "blur_sigma": "Gaussian blur sigma -- 0 lets OpenCV derive it from kernel size; higher = stronger smoothing.",
}

# Live view docks left, trackbars dock right in their own window, both sized
# to fit the screen -- 11+ trackbars stacked above the video used to run off
# the bottom of the screen.
WINDOW_MARGIN = 30
CONTROLS_WIDTH = 380
WINDOW_GAP = 12
TASKBAR_ALLOWANCE = 90


def _get_screen_size() -> tuple[int, int]:
    try:
        import ctypes
        user32 = ctypes.windll.user32
        return int(user32.GetSystemMetrics(0)), int(user32.GetSystemMetrics(1))
    except Exception:
        return 1920, 1080


_BLUR_MODES = ["none", "gaussian", "median"]

GRID_ROWS = 11      # circumferential (around the forearm) -- edges may curl away
ROW_ALLOWANCE = 2   # circumferential rows allowed to be missing at the curved edges
# Band diagnostic only, purely informational -- top/bottom bands are sized to
# roughly the allowed-missing row count.
EDGE_BAND_FRACTION = ROW_ALLOWANCE / GRID_ROWS
_BAND_COLORS = {"top": (255, 255, 0), "middle": (0, 255, 0), "bottom": (255, 0, 255)}


def _band_for_y(y: float, height: int) -> str:
    if height <= 0:
        return "middle"
    frac = y / height
    if frac < EDGE_BAND_FRACTION:
        return "top"
    if frac > 1.0 - EDGE_BAND_FRACTION:
        return "bottom"
    return "middle"


def band_counts(detections: list[tuple[float, float, float]], height: int) -> dict[str, int]:
    counts = {"top": 0, "middle": 0, "bottom": 0}
    for _x, y, _area in detections:
        counts[_band_for_y(y, height)] += 1
    return counts


class CalibratorApp:
    def __init__(self) -> None:
        self._roi = ROICrop()
        # Undistort with whatever alpha the ROI was selected under (see
        # tools/select_roi.py) -- a mismatch here remaps pixel positions and
        # misaligns the crop with the marker grid.
        roi_config = self._roi.config
        alpha = roi_config.alpha if roi_config is not None else DEFAULT_ALPHA
        self._cal = CameraCalibration(alpha=alpha)
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")
        cal_w, cal_h = self._cal.image_size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)
        self._cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self._cap.set(cv2.CAP_PROP_EXPOSURE, -6)

        self.params: dict = load_params()
        self.frozen_gray: np.ndarray | None = None
        self.last_oversized_count: int | None = None
        self.live: bool = True
        self._last_touched: str | None = None
        self._last_touched_at: float = 0.0

        self._build_windows()
        self._build_trackbars()

    def _touch(self, name: str) -> None:
        self._last_touched = name
        self._last_touched_at = time.monotonic()

    def _build_windows(self) -> None:
        screen_w, screen_h = _get_screen_size()
        usable_h = max(240, screen_h - TASKBAR_ALLOWANCE)
        avail_video_w = max(320, screen_w - CONTROLS_WIDTH - WINDOW_GAP - 2 * WINDOW_MARGIN)

        cal_w, cal_h = self._cal.image_size
        aspect = cal_w / cal_h if cal_h else 16 / 9
        video_h = usable_h
        video_w = int(video_h * aspect)
        if video_w > avail_video_w:
            video_w = avail_video_w
            video_h = int(video_w / aspect)

        video_pos = (WINDOW_MARGIN, WINDOW_MARGIN)
        controls_pos = (WINDOW_MARGIN + video_w + WINDOW_GAP, WINDOW_MARGIN)

        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(WINDOW_NAME, np.zeros((cal_h, cal_w, 3), dtype=np.uint8))
        cv2.resizeWindow(WINDOW_NAME, video_w, video_h)
        cv2.moveWindow(WINDOW_NAME, *video_pos)

        cv2.namedWindow(CONTROLS_WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.imshow(CONTROLS_WINDOW_NAME, np.zeros((10, CONTROLS_WIDTH, 3), dtype=np.uint8))
        cv2.resizeWindow(CONTROLS_WINDOW_NAME, CONTROLS_WIDTH, usable_h)
        cv2.moveWindow(CONTROLS_WINDOW_NAME, *controls_pos)

    def _build_trackbars(self) -> None:
        cv2.createTrackbar("thresh", CONTROLS_WINDOW_NAME, self.params["thresh"], 254, self._on_thresh)
        cv2.createTrackbar("min_area", CONTROLS_WINDOW_NAME, int(self.params["min_area"]), 500, self._on_min_area)
        cv2.createTrackbar("max_area", CONTROLS_WINDOW_NAME, int(self.params["max_area"]), 5000, self._on_max_area)
        cv2.createTrackbar("erode", CONTROLS_WINDOW_NAME, int(self.params["erode"]), 1, self._on_erode)
        cv2.createTrackbar("open", CONTROLS_WINDOW_NAME, int(self.params["open"]), 1, self._on_open)
        cv2.createTrackbar("dilate", CONTROLS_WINDOW_NAME, int(self.params["dilate"]), 1, self._on_dilate)
        cv2.createTrackbar("clahe", CONTROLS_WINDOW_NAME, int(self.params["clahe"]), 1, self._on_clahe)
        cv2.createTrackbar("clahe_clip x10", CONTROLS_WINDOW_NAME, int(self.params["clahe_clip"] * 10), 100, self._on_clahe_clip)
        cv2.createTrackbar("adaptive", CONTROLS_WINDOW_NAME, int(self.params["adaptive"]), 1, self._on_adaptive)
        cv2.createTrackbar("adaptive_C +20", CONTROLS_WINDOW_NAME, self.params["adaptive_C"] + 20, 40, self._on_adaptive_c)
        cv2.createTrackbar("tophat", CONTROLS_WINDOW_NAME, int(self.params["tophat"]), 1, self._on_tophat)
        cv2.createTrackbar("tophat_ksize", CONTROLS_WINDOW_NAME, self.params["tophat_ksize"], 101, self._on_tophat_ksize)
        blur_index = _BLUR_MODES.index(self.params["blur"]) if self.params["blur"] in _BLUR_MODES else 0
        cv2.createTrackbar("blur", CONTROLS_WINDOW_NAME, blur_index, len(_BLUR_MODES) - 1, self._on_blur)
        cv2.createTrackbar("blur_ksize", CONTROLS_WINDOW_NAME, self.params["blur_ksize"], 15, self._on_blur_ksize)
        cv2.createTrackbar("blur_sigma x10", CONTROLS_WINDOW_NAME, int(self.params["blur_sigma"] * 10), 50, self._on_blur_sigma)

    def _on_thresh(self, value: int) -> None:
        self.params["thresh"] = value
        self._touch("thresh")

    def _on_min_area(self, value: int) -> None:
        self.params["min_area"] = float(value)
        self._touch("min_area")

    def _on_max_area(self, value: int) -> None:
        self.params["max_area"] = float(max(value, 1))
        self._touch("max_area")

    def _on_erode(self, value: int) -> None:
        self.params["erode"] = bool(value)
        self._touch("erode")

    def _on_open(self, value: int) -> None:
        self.params["open"] = bool(value)
        self._touch("open")

    def _on_dilate(self, value: int) -> None:
        self.params["dilate"] = bool(value)
        self._touch("dilate")

    def _on_clahe(self, value: int) -> None:
        self.params["clahe"] = bool(value)
        self._touch("clahe")

    def _on_clahe_clip(self, value: int) -> None:
        self.params["clahe_clip"] = max(0.1, value / 10.0)
        self._touch("clahe_clip")

    def _on_adaptive(self, value: int) -> None:
        self.params["adaptive"] = bool(value)
        self._touch("adaptive")

    def _on_adaptive_c(self, value: int) -> None:
        self.params["adaptive_C"] = value - 20
        self._touch("adaptive_C")

    def _on_tophat(self, value: int) -> None:
        self.params["tophat"] = bool(value)
        self._touch("tophat")

    def _on_tophat_ksize(self, value: int) -> None:
        self.params["tophat_ksize"] = max(3, value | 1)
        self._touch("tophat_ksize")

    def _on_blur(self, value: int) -> None:
        self.params["blur"] = _BLUR_MODES[value]
        self._touch("blur")

    def _on_blur_ksize(self, value: int) -> None:
        self.params["blur_ksize"] = max(3, value | 1)
        self._touch("blur_ksize")

    def _on_blur_sigma(self, value: int) -> None:
        self.params["blur_sigma"] = value / 10.0
        self._touch("blur_sigma")

    def _read_processed_gray(self) -> np.ndarray | None:
        if not self.live:
            return self.frozen_gray
        ok, frame = self._cap.read()
        if not ok:
            return None
        frame = self._cal.undistort(frame)
        # Rotate before cropping to match the viewer/select_roi.py pipeline --
        # roi.json's box is defined in the rotated coordinate space.
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        frame = self._roi.crop(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _detect(self, gray: np.ndarray) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
        proc = preprocess(gray, self.params)
        mask = self._roi.sensor_mask(proc.shape)
        if mask is not None:
            proc = cv2.bitwise_and(proc, mask)
        return proc, detect(gray, proc, self.params)

    def _freeze(self) -> None:
        gray = self._read_processed_gray()
        if gray is None:
            return
        self.frozen_gray = gray.copy()
        self.live = False
        print("Frame frozen. Adjust trackbars, 's' to save.")
        proc, dets = self._detect(gray)
        self.last_oversized_count = count_oversized_components(gray, proc, self.params)
        _print_band_counts(dets, gray.shape[0], self.last_oversized_count)

    def _unfreeze(self) -> None:
        self.live = True
        self.frozen_gray = None
        self.last_oversized_count = None

    def _save(self, dets: list[tuple[float, float, float]], height: int) -> None:
        counts = band_counts(dets, height)
        roi_config = self._roi.config
        payload = {
            "params": self.params,
            "band_counts": counts,
            "n_detected": len(dets),
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "calibration_image_size": list(self._cal.image_size),
            "roi": (
                {"x": roi_config.x, "y": roi_config.y,
                 "w": roi_config.w, "h": roi_config.h,
                 "alpha": roi_config.alpha}
                if roi_config is not None else None
            ),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
        print(f"Saved parameters to {OUTPUT_PATH}")

    def run(self) -> None:
        print("SPACE freeze | 'u' unfreeze/live | 'b' band report | 's' save | 'q' quit")
        while True:
            gray = self._read_processed_gray()
            if gray is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            proc, dets = self._detect(gray)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            hint = self._last_touched
            if hint is not None and time.monotonic() - self._last_touched_at > HINT_FADE_SECONDS:
                hint = None
            _draw_overlay(display, dets, self.live, self.last_oversized_count, hint)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                self._freeze()
            elif key == ord("u"):
                self._unfreeze()
            elif key == ord("b"):
                self.last_oversized_count = count_oversized_components(gray, proc, self.params)
                _print_band_counts(dets, gray.shape[0], self.last_oversized_count)
            elif key == ord("s"):
                self._save(dets, gray.shape[0])

        self._cap.release()
        cv2.destroyAllWindows()


def _wrap_text(text: str, font: int, scale: float, thickness: int, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        candidate = f"{current} {word}".strip()
        (width, _), _ = cv2.getTextSize(candidate, font, scale, thickness)
        if width > max_width and current:
            lines.append(current)
            current = word
        else:
            current = candidate
    if current:
        lines.append(current)
    return lines


def _draw_overlay(
    display: np.ndarray,
    dets: list[tuple[float, float, float]],
    live: bool,
    oversized_count: int | None,
    hint: str | None,
) -> None:
    height = display.shape[0]
    counts = band_counts(dets, height)
    for x, y, _area in dets:
        color = _BAND_COLORS[_band_for_y(y, height)]
        cv2.circle(display, (int(x), int(y)), 4, color, 1)
    state = "LIVE" if live else "FROZEN"
    oversized_str = "?" if oversized_count is None else str(oversized_count)
    cv2.putText(
        display,
        f"{state}  markers: {len(dets)}  "
        f"top:{counts['top']} mid:{counts['middle']} bot:{counts['bottom']}  oversized:{oversized_str}",
        (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1,
    )
    if hint is not None:
        font = cv2.FONT_HERSHEY_PLAIN
        scale = 1.0
        max_width = max(display.shape[1] - 16, 50)
        text = f"{hint}: {PARAM_DESCRIPTIONS.get(hint, '')}"
        for i, line in enumerate(_wrap_text(text, font, scale, 1, max_width)):
            cv2.putText(display, line, (8, 40 + i * 16), font, scale, (255, 255, 255), 1)


def _print_band_counts(dets: list[tuple[float, float, float]], height: int, oversized_count: int) -> None:
    counts = band_counts(dets, height)
    print(
        f"  band counts: top={counts['top']} middle={counts['middle']} bottom={counts['bottom']}  "
        f"oversized_components={oversized_count}"
    )


def main() -> None:
    app = CalibratorApp()
    app.run()


if __name__ == "__main__":
    main()
