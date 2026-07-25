"""Three-pose detector parameter search for per-wear sensor calibration.

Scores one candidate ``thresh``/``min_area`` pair against the sensor captured at
three hand poses at once and keeps the set that survives the *worst* pose, so a
dim oblique edge marker cannot fall under the threshold mid-recording.

Pure and rig-free: takes already-undistorted/cropped grey frames, returns params
plus per-pose diagnostics. No tkinter, no cv2 GUI calls -- safe to run on a
worker thread and testable on synthetic frames.
"""
from dataclasses import dataclass

import numpy as np

from core.detector import (
    binarize,
    count_oversized_components,
    denoise,
    detect,
    flatten_illumination,
)

TARGET_MARKERS = 99
MIN_ACCEPTABLE = 94
GRID_SHAPE = (9, 11)
BAND_LINES = 2
CORNER_LINES = 3

W_DROP = 3.0
W_OVER = 1.0
W_OVERSIZED = 2.0
W_CORNER = 2.0

THRESH_RANGE = (30, 60, 2)
MIN_AREA_RANGE = (4, 40, 2)

POSE_KEYS = ("relaxed", "full_open", "max_grip")

Detection = tuple[float, float, float]


class AnchorError(Exception):
    """The relaxed pose is unusable as the geometry anchor, so every downstream
    number would be meaningless. Carries a UI-ready message."""


@dataclass(frozen=True)
class Geometry:
    x_lines: int
    y_lines: int
    x_corner: tuple[float, float]
    y_corner: tuple[float, float]
    y_band: tuple[float, float]


@dataclass(frozen=True)
class PoseDiagnostics:
    n_detected: int
    bands: dict[str, int]
    corners: dict[str, int]
    oversized: int
    gaussian_fallbacks: int
    cost: float


@dataclass(frozen=True)
class SearchResult:
    params: dict
    poses: dict[str, PoseDiagnostics]
    detections: dict[str, list[Detection]]
    geometry: Geometry
    grid_shape: tuple[int, int]
    reference_bands: dict[str, int]
    reference_corners: dict[str, int]
    worst_pose: str
    worst_count: int
    passed: bool
    bounds: dict[str, list[int]]


def grid_lines(coords: np.ndarray) -> list[float]:
    """Cluster 1-D marker coordinates into grid lines, returning each line's mean.

    Markers sharing a line sit within centroid jitter of each other while the step
    to the next line is a full pitch, so the sorted gaps form two additively
    well-separated populations. The cut is placed midway across the largest jump
    between consecutive sorted gaps. A fixed fraction of the median gap would not
    work: the median is dominated by the many near-zero within-line gaps.
    """
    ordered = np.sort(np.asarray(coords, dtype=float))
    if ordered.size == 0:
        return []
    if ordered.size == 1:
        return [float(ordered[0])]

    gaps = np.diff(ordered)
    if gaps.size < 2:
        cut = float(gaps[0]) * 0.5
    else:
        ranked = np.sort(gaps)
        i = int(np.argmax(np.diff(ranked)))
        cut = 0.5 * float(ranked[i] + ranked[i + 1])

    lines: list[float] = []
    start = 0
    for idx, gap in enumerate(gaps):
        if gap > cut:
            lines.append(float(ordered[start : idx + 1].mean()))
            start = idx + 1
    lines.append(float(ordered[start:].mean()))
    return lines


def _midpoints(lines: list[float], depth: int) -> tuple[float, float]:
    """Boundaries enclosing the outer `depth` lines at each end, placed midway
    between lines so every cell edge clears the nearest marker by a half-pitch."""
    lo = 0.5 * (lines[depth - 1] + lines[depth])
    hi = 0.5 * (lines[-depth] + lines[-depth - 1])
    return lo, hi


def geometry_from(detections: list[Detection]) -> Geometry:
    xs = np.array([d[0] for d in detections], dtype=float)
    ys = np.array([d[1] for d in detections], dtype=float)
    x_lines = grid_lines(xs)
    y_lines = grid_lines(ys)
    if len(x_lines) <= CORNER_LINES or len(y_lines) <= CORNER_LINES:
        raise AnchorError(
            f"Detected only {len(x_lines)}x{len(y_lines)} grid lines — "
            "check the ROI and lighting, then recapture."
        )
    return Geometry(
        x_lines=len(x_lines),
        y_lines=len(y_lines),
        x_corner=_midpoints(x_lines, CORNER_LINES),
        y_corner=_midpoints(y_lines, CORNER_LINES),
        y_band=_midpoints(y_lines, BAND_LINES),
    )


def region_counts(detections: list[Detection], geom: Geometry) -> dict[str, dict[str, int]]:
    bands = {"top": 0, "middle": 0, "bottom": 0}
    corners = {"tl": 0, "tr": 0, "bl": 0, "br": 0}
    band_lo, band_hi = geom.y_band
    x_lo, x_hi = geom.x_corner
    y_lo, y_hi = geom.y_corner

    for x, y, _size in detections:
        if y < band_lo:
            bands["top"] += 1
        elif y > band_hi:
            bands["bottom"] += 1
        else:
            bands["middle"] += 1

        vertical = "t" if y < y_lo else "b" if y > y_hi else None
        horizontal = "l" if x < x_lo else "r" if x > x_hi else None
        if vertical is not None and horizontal is not None:
            corners[vertical + horizontal] += 1

    return {"bands": bands, "corners": corners}


def pose_cost(
    n_detected: int, oversized: int, corners: dict[str, int], reference: dict[str, int]
) -> float:
    drop = max(0, TARGET_MARKERS - n_detected)
    over = max(0, n_detected - TARGET_MARKERS)
    deficit = sum(max(0, reference[k] - corners[k]) for k in reference)
    return W_DROP * drop + W_OVER * over + W_OVERSIZED * oversized + W_CORNER * deficit


def _prepare(gray: np.ndarray, base_params: dict, mask: np.ndarray | None) -> tuple:
    """Hoist everything the swept levers do not affect out of the candidate loop.

    CLAHE + a large top-hat structuring element dominate the per-evaluation cost
    and depend only on params held fixed here, so they are computed once per pose.
    """
    denoised = denoise(gray, base_params)
    return denoised, flatten_illumination(denoised, base_params), mask


def _evaluate(prepared: tuple, params: dict) -> tuple[list[Detection], int, int]:
    denoised, flattened, mask = prepared
    binary = binarize(flattened, params)
    if mask is not None:
        binary = np.bitwise_and(binary, mask)
    fallbacks: list[int] = []
    dets = detect(denoised, binary, params, fallback_count=fallbacks)
    oversized = count_oversized_components(denoised, binary, params)
    return dets, oversized, len(fallbacks)


def _candidates(base_params: dict) -> list[tuple[int, float]]:
    t_lo, t_hi, t_step = THRESH_RANGE
    a_lo, a_hi, a_step = MIN_AREA_RANGE
    grid = [
        (t, float(a))
        for t in range(t_lo, t_hi + 1, t_step)
        for a in range(a_lo, a_hi + 1, a_step)
    ]
    seed = (int(base_params["thresh"]), float(base_params["min_area"]))
    if seed not in grid:
        grid.append(seed)
    return grid


def _anchor_detections(prepared: tuple, base_params: dict) -> list[Detection]:
    """Best available view of the relaxed pose, used to fix the scoring geometry.

    Swept rather than taken from the seed: a per-wear recalibration exists
    precisely because the seed may no longer fit this mount, so anchoring on the
    seed would block calibration exactly when it is most needed. Over-detection
    is penalised equally, so a flooded frame cannot become the anchor.
    """
    best_key: tuple | None = None
    best: list[Detection] = []
    for thresh, min_area in _candidates(base_params):
        candidate = dict(base_params, thresh=thresh, min_area=min_area, size_method="cc")
        dets, _oversized, _fallbacks = _evaluate(prepared, candidate)
        key = (abs(len(dets) - TARGET_MARKERS), -thresh, -min_area)
        if best_key is None or key < best_key:
            best_key = key
            best = dets
    return best


def search(
    pose_frames: dict[str, np.ndarray],
    base_params: dict,
    mask: np.ndarray | None = None,
) -> SearchResult:
    missing = [k for k in POSE_KEYS if k not in pose_frames]
    if missing:
        raise ValueError(f"missing pose frames: {', '.join(missing)}")

    prepared = {k: _prepare(pose_frames[k], base_params, mask) for k in POSE_KEYS}

    scoring_base = dict(base_params, size_method="cc")
    anchor_dets = _anchor_detections(prepared["relaxed"], base_params)
    if len(anchor_dets) < MIN_ACCEPTABLE:
        raise AnchorError(
            f"Relaxed pose detected only {len(anchor_dets)} markers — reseat the "
            "sensor or check lighting, then recapture."
        )
    geom = geometry_from(anchor_dets)
    if sorted((geom.x_lines, geom.y_lines)) != sorted(GRID_SHAPE):
        raise AnchorError(
            f"Expected a 9x11 grid, detected {geom.x_lines}x{geom.y_lines} — "
            "check the ROI and lighting, then recapture."
        )

    reference = region_counts(anchor_dets, geom)
    ref_corners = reference["corners"]

    best_key: tuple | None = None
    best_pair: tuple[int, float] = (int(base_params["thresh"]), float(base_params["min_area"]))
    # size_method="cc" scores identically to "gaussian" -- both accept the same
    # blobs from the same binary image, and only the reported centroid/size
    # differs -- so this skips ~99 Gaussian fits per evaluation.
    for thresh, min_area in _candidates(base_params):
        candidate = dict(scoring_base, thresh=thresh, min_area=min_area)
        worst = 0.0
        total = 0.0
        for key in POSE_KEYS:
            dets, oversized, _fallbacks = _evaluate(prepared[key], candidate)
            cost = pose_cost(
                len(dets), oversized, region_counts(dets, geom)["corners"], ref_corners
            )
            worst = max(worst, cost)
            total += cost
        sort_key = (worst, total, -thresh, -min_area)
        if best_key is None or sort_key < best_key:
            best_key = sort_key
            best_pair = (thresh, min_area)

    winner = dict(
        base_params,
        thresh=best_pair[0],
        min_area=best_pair[1],
        size_method=base_params.get("size_method", "gaussian"),
    )

    poses: dict[str, PoseDiagnostics] = {}
    detections: dict[str, list[Detection]] = {}
    for key in POSE_KEYS:
        dets, oversized, fallbacks = _evaluate(prepared[key], winner)
        counts = region_counts(dets, geom)
        poses[key] = PoseDiagnostics(
            n_detected=len(dets),
            bands=counts["bands"],
            corners=counts["corners"],
            oversized=oversized,
            gaussian_fallbacks=fallbacks,
            cost=pose_cost(len(dets), oversized, counts["corners"], ref_corners),
        )
        detections[key] = dets

    worst_pose = max(POSE_KEYS, key=lambda k: (poses[k].cost, -poses[k].n_detected))
    worst_count = min(poses[k].n_detected for k in POSE_KEYS)

    return SearchResult(
        params=winner,
        poses=poses,
        detections=detections,
        geometry=geom,
        grid_shape=(geom.x_lines, geom.y_lines),
        reference_bands=reference["bands"],
        reference_corners=ref_corners,
        worst_pose=worst_pose,
        worst_count=worst_count,
        passed=worst_count >= MIN_ACCEPTABLE,
        bounds={"thresh": list(THRESH_RANGE), "min_area": list(MIN_AREA_RANGE)},
    )
