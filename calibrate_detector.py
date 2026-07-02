"""Interactive + auto detector calibration for GripVT footage.

Freeze a live frame (SPACE), then either let the optimizer search parameters
('a') against the known curved 11x11 marker grid, or fine-tune with the
fallback trackbars. Save the winning parameters with 's'.
"""
import json
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from core.calibration import CameraCalibration
from core.detector import count_merged_components, default_params, detect, preprocess
from core.roi import ROICrop

CAMERA_INDEX = 1
WINDOW_NAME = "GripVT Detector Calibration"
CONTROLS_WINDOW_NAME = "Detector Controls"
OUTPUT_PATH = Path(__file__).parent / "detector_params.json"

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

GRID_COLUMNS = 11   # axial (along the forearm) -- required, full coverage
GRID_ROWS = 11      # circumferential (around the forearm) -- edges may curl away
ROW_ALLOWANCE = 2   # circumferential rows allowed to be missing at the curved edges
_ROW_REQUIREMENT = GRID_ROWS - ROW_ALLOWANCE
TARGET_COUNT = 121

# Same tolerance as ROW_ALLOWANCE above: the top/bottom edge bands are sized
# to roughly the allowed-missing row count, so the diagnostic bands line up
# with what the scoring already treats as the curved, best-effort zone.
EDGE_BAND_FRACTION = ROW_ALLOWANCE / GRID_ROWS
_BAND_COLORS = {"top": (255, 255, 0), "middle": (0, 255, 0), "bottom": (255, 0, 255)}

W_MESH = 3.0
W_AXIAL = 3.0
W_CIRC = 1.5
W_COUNT = 1.0
W_DUP = 2.0

THRESH_SEARCH = [50, 80, 110, 140, 170, 200]
# White top-hat output has a very different dynamic range from raw grayscale
# (it isolates local contrast, not absolute brightness), so it needs its own,
# much lower threshold search range.
TOPHAT_THRESH_SEARCH = [10, 20, 35, 50, 65]
SIGMA_SEARCH = [6.0, 9.0, 12.0, 15.0, 18.0]
ADAPTIVE_BLOCK_SEARCH = [31, 51, 71]
ADAPTIVE_C_SEARCH = [-15, -5]
CLAHE_CLIP_SEARCH = [2.0, 4.0]


# ---------------------------------------------------------------------------
# Scoring — deformation-invariant so a curved, foreshortened grid still scores
# correctly. Never bins on fixed pixel widths.
# ---------------------------------------------------------------------------

def _estimate_spacing(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    np.fill_diagonal(dist, np.inf)
    return float(np.median(dist.min(axis=1)))


def _mesh_coherence(points: np.ndarray, spacing: float) -> tuple[float, int]:
    n = len(points)
    if n < 2 or spacing <= 0:
        return 0.0, 0
    diff = points[:, None, :] - points[None, :, :]
    dist = np.linalg.norm(diff, axis=2)
    duplicate_count = int(np.sum((dist > 1e-6) & (dist < spacing * 0.5)) // 2)
    adjacency = (dist >= spacing * 0.4) & (dist <= spacing * 1.8)
    n_components, labels = connected_components(csr_matrix(adjacency), directed=False)
    if n_components == 0:
        return 0.0, duplicate_count
    largest = int(np.max(np.bincount(labels)))
    return largest / n, duplicate_count


def _sequential_clusters(values: np.ndarray, split_gap: float) -> list[np.ndarray]:
    """Group sorted 1D values, starting a new cluster whenever the gap to the
    previous value exceeds split_gap. split_gap must come from the
    independently-estimated marker-to-marker spacing (_estimate_spacing),
    not from these values' own gap statistics -- a perfectly aligned grid
    has within-cluster gaps of exactly 0, which would make a self-derived
    median threshold degenerate and never split anything."""
    if len(values) == 0:
        return []
    order = np.argsort(values)
    sorted_vals = values[order]
    if len(sorted_vals) == 1 or split_gap <= 0:
        return [order]

    diffs = np.diff(sorted_vals)
    clusters: list[list[int]] = [[int(order[0])]]
    for i in range(1, len(sorted_vals)):
        if diffs[i - 1] > split_gap:
            clusters.append([])
        clusters[-1].append(int(order[i]))
    return [np.array(c) for c in clusters]


def _clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_detections(detections: list[tuple[float, float, float]]) -> dict:
    if not detections:
        return {
            "total": 0.0, "mesh": 0.0, "axial": 0.0, "circumferential": 0.0,
            "count": 0.0, "duplicate_fraction": 0.0, "n_detected": 0,
            "n_columns": 0, "n_rows": 0, "spacing_px": 0.0,
        }

    points = np.array([(d[0], d[1]) for d in detections], dtype=float)
    n = len(points)

    spacing = _estimate_spacing(points)
    mesh_frac, dup_count = _mesh_coherence(points, spacing)
    dup_fraction = dup_count / n

    split_gap = spacing * 0.6
    n_columns = len(_sequential_clusters(points[:, 0], split_gap))
    n_rows = len(_sequential_clusters(points[:, 1], split_gap))

    # Axial (x) columns run along the forearm and don't curve away from the
    # camera, so any missing column means bad params -- full coverage is
    # required. Circumferential (y) rows wrap around the cylinder, so the
    # outermost rows may legitimately curl out of view.
    axial_score = _clip01(1.0 - abs(GRID_COLUMNS - n_columns) / GRID_COLUMNS)
    circ_score = _clip01(1.0 - max(0, _ROW_REQUIREMENT - n_rows) / _ROW_REQUIREMENT)

    if n <= TARGET_COUNT:
        count_score = n / TARGET_COUNT
    else:
        count_score = max(0.0, 1.0 - (n - TARGET_COUNT) / TARGET_COUNT)

    total = (
        W_MESH * mesh_frac
        + W_AXIAL * axial_score
        + W_CIRC * circ_score
        + W_COUNT * count_score
        - W_DUP * dup_fraction
    )

    return {
        "total": total, "mesh": mesh_frac, "axial": axial_score,
        "circumferential": circ_score, "count": count_score,
        "duplicate_fraction": dup_fraction, "n_detected": n,
        "n_columns": n_columns, "n_rows": n_rows, "spacing_px": spacing,
    }


# ---------------------------------------------------------------------------
# Per-band diagnostic -- reports where detections land (top/bottom curved
# edge bands vs middle), separate from and never fed back into the scoring.
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Auto-calibration search
# ---------------------------------------------------------------------------

def _derive_ksize(sigma: float) -> int:
    return max(3, int(3.2 * sigma) | 1)


def _estimate_tophat_ksize(gray: np.ndarray, base_params: dict) -> int:
    """Size the top-hat structuring element from the real marker diameter on
    this frame -- it must be a bit LARGER than a marker or the marker gets
    erased along with the background it's meant to isolate from."""
    probe_params = dict(base_params, tophat=False, adaptive=False)
    proc = preprocess(gray, probe_params)
    dets = detect(gray, proc, probe_params)
    areas = [d[2] for d in dets if d[2] > 0]
    if not areas:
        return 15
    median_area = float(np.median(areas))
    diameter = 2.0 * np.sqrt(median_area / np.pi)
    ksize = int(round(diameter * 1.75)) | 1
    return max(5, min(101, ksize))


def _evaluate(gray: np.ndarray, params: dict) -> dict:
    proc = preprocess(gray, params)
    dets = detect(gray, proc, params)
    result = score_detections(dets)
    result["params"] = dict(params)
    result["detections"] = dets
    return result


def _refine(gray: np.ndarray, best: dict) -> dict:
    params = best["params"]
    candidates = [best]

    if params.get("adaptive", False):
        block = params["adaptive_block"]
        for delta in (-10, -6, 6, 10):
            candidate_block = max(11, block + delta) | 1
            candidates.append(_evaluate(gray, dict(params, adaptive_block=candidate_block)))
        c = params["adaptive_C"]
        for delta in (-6, -3, 3, 6):
            candidates.append(_evaluate(gray, dict(params, adaptive_C=c + delta)))
    else:
        thresh = params["thresh"]
        for delta in (-15, -8, 8, 15):
            candidate_thresh = int(np.clip(thresh + delta, 1, 254))
            candidates.append(_evaluate(gray, dict(params, thresh=candidate_thresh)))

    if params.get("clahe", False):
        clip = params["clahe_clip"]
        for delta in (-1.0, -0.5, 0.5, 1.0):
            candidate_clip = max(0.1, clip + delta)
            candidates.append(_evaluate(gray, dict(params, clahe_clip=candidate_clip)))

    if params.get("tophat", False):
        k = params["tophat_ksize"]
        for delta in (-6, -3, 3, 6):
            candidate_k = max(5, (k + delta) | 1)
            candidates.append(_evaluate(gray, dict(params, tophat_ksize=candidate_k)))

    sigma = params["log_sigma"]
    for delta in (-2.0, -1.0, 1.0, 2.0):
        candidate_sigma = max(2.0, sigma + delta)
        candidates.append(_evaluate(
            gray, dict(params, log_sigma=candidate_sigma, log_ksize=_derive_ksize(candidate_sigma))
        ))

    return max(candidates, key=lambda r: r["total"])


def _threshold_branch_candidates(gray: np.ndarray, shared: dict, thresh_values: list[int]) -> list[dict]:
    candidates = []
    for thresh in thresh_values:
        candidates.append(_evaluate(gray, dict(shared, thresh=thresh, adaptive=False)))
    for block in ADAPTIVE_BLOCK_SEARCH:
        for c in ADAPTIVE_C_SEARCH:
            candidates.append(_evaluate(
                gray, dict(shared, adaptive=True, adaptive_block=block, adaptive_C=c)
            ))
    return candidates


def auto_calibrate(gray: np.ndarray, base_params: dict) -> dict:
    tophat_ksize = _estimate_tophat_ksize(gray, base_params)

    candidates: list[dict] = []
    for sigma in SIGMA_SEARCH:
        ksize = _derive_ksize(sigma)
        for tophat_on in (False, True):
            thresh_values = TOPHAT_THRESH_SEARCH if tophat_on else THRESH_SEARCH
            tophat_kwargs = {"tophat": True, "tophat_ksize": tophat_ksize} if tophat_on else {"tophat": False}
            no_clahe = dict(base_params, log_sigma=sigma, log_ksize=ksize, clahe=False, **tophat_kwargs)
            candidates.extend(_threshold_branch_candidates(gray, no_clahe, thresh_values))
            for clip in CLAHE_CLIP_SEARCH:
                with_clahe = dict(no_clahe, clahe=True, clahe_clip=clip)
                candidates.extend(_threshold_branch_candidates(gray, with_clahe, thresh_values))

    best = max(candidates, key=lambda r: r["total"])
    return _refine(gray, best)


# ---------------------------------------------------------------------------
# Interactive app
# ---------------------------------------------------------------------------

class CalibratorApp:
    def __init__(self) -> None:
        self._cal = CameraCalibration(alpha=0.0)
        self._roi = ROICrop()
        self._cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera at index {CAMERA_INDEX}")
        cal_w, cal_h = self._cal.image_size
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, cal_w)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cal_h)

        self.params: dict = default_params()
        self.frozen_gray: np.ndarray | None = None
        self.last_score: dict | None = None
        self.last_merged_count: int | None = None
        self.live: bool = True

        self._build_windows()
        self._build_trackbars()

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
        cv2.createTrackbar("log_sigma x10", CONTROLS_WINDOW_NAME, int(self.params["log_sigma"] * 10), 400, self._on_sigma)
        cv2.createTrackbar("erode", CONTROLS_WINDOW_NAME, int(self.params["erode"]), 1, self._on_erode)
        cv2.createTrackbar("open", CONTROLS_WINDOW_NAME, int(self.params["open"]), 1, self._on_open)
        cv2.createTrackbar("dilate", CONTROLS_WINDOW_NAME, int(self.params["dilate"]), 1, self._on_dilate)
        cv2.createTrackbar("clahe", CONTROLS_WINDOW_NAME, int(self.params["clahe"]), 1, self._on_clahe)
        cv2.createTrackbar("clahe_clip x10", CONTROLS_WINDOW_NAME, int(self.params["clahe_clip"] * 10), 100, self._on_clahe_clip)
        cv2.createTrackbar("adaptive", CONTROLS_WINDOW_NAME, int(self.params["adaptive"]), 1, self._on_adaptive)
        cv2.createTrackbar("adaptive_C +20", CONTROLS_WINDOW_NAME, self.params["adaptive_C"] + 20, 40, self._on_adaptive_c)
        cv2.createTrackbar("tophat", CONTROLS_WINDOW_NAME, int(self.params["tophat"]), 1, self._on_tophat)
        cv2.createTrackbar("tophat_ksize", CONTROLS_WINDOW_NAME, self.params["tophat_ksize"], 101, self._on_tophat_ksize)

    def _on_thresh(self, value: int) -> None:
        self.params["thresh"] = value

    def _on_sigma(self, value: int) -> None:
        sigma = max(2.0, value / 10.0)
        self.params["log_sigma"] = sigma
        self.params["log_ksize"] = _derive_ksize(sigma)

    def _on_erode(self, value: int) -> None:
        self.params["erode"] = bool(value)

    def _on_open(self, value: int) -> None:
        self.params["open"] = bool(value)

    def _on_dilate(self, value: int) -> None:
        self.params["dilate"] = bool(value)

    def _on_clahe(self, value: int) -> None:
        self.params["clahe"] = bool(value)

    def _on_clahe_clip(self, value: int) -> None:
        self.params["clahe_clip"] = max(0.1, value / 10.0)

    def _on_adaptive(self, value: int) -> None:
        self.params["adaptive"] = bool(value)

    def _on_adaptive_c(self, value: int) -> None:
        self.params["adaptive_C"] = value - 20

    def _on_tophat(self, value: int) -> None:
        self.params["tophat"] = bool(value)

    def _on_tophat_ksize(self, value: int) -> None:
        self.params["tophat_ksize"] = max(3, value | 1)

    def _read_processed_gray(self) -> np.ndarray | None:
        if not self.live:
            return self.frozen_gray
        ok, frame = self._cap.read()
        if not ok:
            return None
        frame = self._cal.undistort(frame)
        frame = self._roi.crop(frame)
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    def _freeze(self) -> None:
        gray = self._read_processed_gray()
        if gray is None:
            return
        self.frozen_gray = gray.copy()
        self.live = False
        print("Frame frozen. Press 'a' to auto-calibrate or adjust trackbars.")
        proc = preprocess(gray, self.params)
        dets = detect(gray, proc, self.params)
        self.last_merged_count = count_merged_components(gray, proc, self.params)
        _print_band_counts(dets, gray.shape[0], self.last_merged_count)

    def _unfreeze(self) -> None:
        self.live = True
        self.frozen_gray = None
        self.last_score = None
        self.last_merged_count = None

    def _run_auto_calibrate(self) -> None:
        if self.frozen_gray is None:
            print("Freeze a frame first (SPACE) before auto-calibrating.")
            return
        print("Auto-calibrating... this may take a few seconds.")
        start = time.time()
        result = auto_calibrate(self.frozen_gray, self.params)
        elapsed = time.time() - start
        self.params = result["params"]
        self.last_score = result
        proc = preprocess(self.frozen_gray, self.params)
        self.last_merged_count = count_merged_components(self.frozen_gray, proc, self.params)
        self._sync_trackbars()
        print(f"Auto-calibration done in {elapsed:.1f}s")
        _print_score(result, self.frozen_gray.shape[0], self.last_merged_count)

    def _sync_trackbars(self) -> None:
        cv2.setTrackbarPos("thresh", CONTROLS_WINDOW_NAME, int(self.params["thresh"]))
        cv2.setTrackbarPos("log_sigma x10", CONTROLS_WINDOW_NAME, int(self.params["log_sigma"] * 10))
        cv2.setTrackbarPos("erode", CONTROLS_WINDOW_NAME, int(self.params["erode"]))
        cv2.setTrackbarPos("open", CONTROLS_WINDOW_NAME, int(self.params["open"]))
        cv2.setTrackbarPos("dilate", CONTROLS_WINDOW_NAME, int(self.params["dilate"]))
        cv2.setTrackbarPos("clahe", CONTROLS_WINDOW_NAME, int(self.params["clahe"]))
        cv2.setTrackbarPos("clahe_clip x10", CONTROLS_WINDOW_NAME, int(self.params["clahe_clip"] * 10))
        cv2.setTrackbarPos("adaptive", CONTROLS_WINDOW_NAME, int(self.params["adaptive"]))
        cv2.setTrackbarPos("adaptive_C +20", CONTROLS_WINDOW_NAME, self.params["adaptive_C"] + 20)
        cv2.setTrackbarPos("tophat", CONTROLS_WINDOW_NAME, int(self.params["tophat"]))
        cv2.setTrackbarPos("tophat_ksize", CONTROLS_WINDOW_NAME, self.params["tophat_ksize"])

    def _save(self, dets: list[tuple[float, float, float]]) -> None:
        score = self.last_score or score_detections(dets)
        payload = {
            "params": self.params,
            "score": {k: v for k, v in score.items() if k not in ("params", "detections")},
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "calibration_image_size": list(self._cal.image_size),
            "roi": (
                {"x": self._roi.config.x, "y": self._roi.config.y,
                 "w": self._roi.config.w, "h": self._roi.config.h}
                if self._roi.is_set else None
            ),
        }
        OUTPUT_PATH.write_text(json.dumps(payload, indent=2))
        print(f"Saved parameters to {OUTPUT_PATH}")

    def run(self) -> None:
        print("SPACE freeze | 'a' auto-calibrate | 'u' unfreeze/live | 'b' band report | 's' save | 'q' quit")
        while True:
            gray = self._read_processed_gray()
            if gray is None:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
                continue

            proc = preprocess(gray, self.params)
            dets = detect(gray, proc, self.params)
            display = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
            _draw_overlay(display, dets, self.live, self.last_merged_count)
            cv2.imshow(WINDOW_NAME, display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            elif key == ord(" "):
                self._freeze()
            elif key == ord("u"):
                self._unfreeze()
            elif key == ord("a"):
                self._run_auto_calibrate()
            elif key == ord("b"):
                self.last_merged_count = count_merged_components(gray, proc, self.params)
                _print_band_counts(dets, gray.shape[0], self.last_merged_count)
            elif key == ord("s"):
                self._save(dets)

        self._cap.release()
        cv2.destroyAllWindows()


def _draw_overlay(
    display: np.ndarray,
    dets: list[tuple[float, float, float]],
    live: bool,
    merged_count: int | None,
) -> None:
    height = display.shape[0]
    counts = band_counts(dets, height)
    for x, y, _area in dets:
        color = _BAND_COLORS[_band_for_y(y, height)]
        cv2.circle(display, (int(x), int(y)), 4, color, 1)
    state = "LIVE" if live else "FROZEN"
    merged_str = "?" if merged_count is None else str(merged_count)
    cv2.putText(
        display,
        f"{state}  markers: {len(dets)}  "
        f"top:{counts['top']} mid:{counts['middle']} bot:{counts['bottom']}  merged:{merged_str}",
        (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 255, 255), 1,
    )


def _print_band_counts(dets: list[tuple[float, float, float]], height: int, merged_count: int) -> None:
    counts = band_counts(dets, height)
    print(
        f"  band counts: top={counts['top']} middle={counts['middle']} bottom={counts['bottom']}  "
        f"merged_components={merged_count}"
    )


def _print_score(result: dict, height: int, merged_count: int) -> None:
    print(
        f"  total={result['total']:.2f}  n={result['n_detected']}  "
        f"columns={result['n_columns']}/{GRID_COLUMNS}  rows={result['n_rows']}/{GRID_ROWS}  "
        f"mesh={result['mesh']:.2f}  axial={result['axial']:.2f}  "
        f"circ={result['circumferential']:.2f}  dup_frac={result['duplicate_fraction']:.2f}"
    )
    print(f"  params={result['params']}")
    _print_band_counts(result["detections"], height, merged_count)


def main() -> None:
    app = CalibratorApp()
    app.run()


if __name__ == "__main__":
    main()
