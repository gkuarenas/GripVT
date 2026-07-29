import json
from pathlib import Path

import cv2
import numpy as np

from core.blobfit import estimate_background, fit_gaussian_logparabola, fit_moments

_PARAMS_PATH = Path(__file__).parent.parent / "calibration" / "detector_tuning" / "detector_params.json"


def default_params() -> dict:
    return {
        "thresh": 75,
        "erode": True,
        "open": True,
        "dilate": True,
        "min_area": 5.0,
        "max_area": 3000.0,
        "clahe": False,
        "clahe_clip": 3.0,
        "clahe_grid": 8,
        "adaptive": False,
        "adaptive_block": 51,
        "adaptive_C": -10,
        "tophat": False,
        "tophat_ksize": 15,
        "blur": "none",
        "blur_ksize": 3,
        "blur_sigma": 0.0,
        "size_method": "gaussian",
        "blob_margin_px": 4,
        "log_size_crosscheck": False,
    }


def load_params() -> dict:
    """default_params(), overlaid with a previously-saved calibration (see
    calibrate_detector.py's 's' key) if one exists. Merges onto the current
    schema's defaults rather than trusting the file wholesale, so a saved
    file predating a params-schema change (missing/extra keys) can't crash
    anything -- missing keys keep today's default, extra stale keys are
    carried along harmlessly."""
    params = default_params()
    if _PARAMS_PATH.exists():
        saved = json.loads(_PARAMS_PATH.read_text())
        params.update(saved.get("params", {}))
    return params


def denoise(frame_gray: np.ndarray, params: dict) -> np.ndarray:
    """First stage of preprocess(), run on the raw grey before illumination
    flattening. detect()'s centroid is a binary centroid (mean of
    above-threshold pixel coordinates, not intensity-weighted), so sensor
    noise flickering boundary pixels in/out of threshold wobbles it every
    frame, even at rest -- smoothing here, before CLAHE would amplify that
    same noise and before threshold locks the flicker in, stabilizes which
    pixels clear the cutoff. Gaussian suits Gaussian sensor noise; median
    kills salt-and-pepper speckle without smearing blob edges (the usual
    pick for centroid stability) at the cost of eroding thin structure if
    the kernel is too large relative to the blob."""
    mode = params.get("blur", "none")
    k = params.get("blur_ksize", 3) | 1
    if mode == "gaussian":
        return cv2.GaussianBlur(frame_gray, (k, k), params.get("blur_sigma", 0.0))
    if mode == "median":
        return cv2.medianBlur(frame_gray, k)
    return frame_gray


def flatten_illumination(frame_gray: np.ndarray, params: dict) -> np.ndarray:
    """CLAHE + white top-hat stage of preprocess(), split out so a caller
    sweeping many threshold/adaptive candidates against the same illumination
    settings (e.g. calibrate_detector's search) can reuse one flattened image
    instead of repeating the expensive top-hat morphology per candidate."""
    img_gray = frame_gray
    if params.get("clahe", False):
        clahe = cv2.createCLAHE(
            clipLimit=params.get("clahe_clip", 3.0),
            tileGridSize=(params.get("clahe_grid", 8), params.get("clahe_grid", 8)),
        )
        img_gray = clahe.apply(img_gray)

    if params.get("tophat", False):
        # White top-hat = image - opening(image): erases bright features
        # smaller than the structuring element (the markers) and keeps the
        # smooth background, so subtracting it flattens uneven illumination
        # (dim edge markers vs bright centre) before thresholding. The SE
        # must be a bit LARGER than a marker -- too small and the marker
        # itself gets erased away by the opening and vanishes here too.
        k = params.get("tophat_ksize", 15) | 1
        se = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        img_gray = cv2.morphologyEx(img_gray, cv2.MORPH_TOPHAT, se)

    return img_gray


def binarize(flattened_gray: np.ndarray, params: dict) -> np.ndarray:
    """Threshold + cleanup morphology stage of preprocess(), operating on an
    already illumination-flattened image (see flatten_illumination())."""
    if params.get("adaptive", False):
        block = params.get("adaptive_block", 51) | 1  # must be odd
        img = cv2.adaptiveThreshold(
            flattened_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            block, params.get("adaptive_C", -10),
        )
    else:
        _, img = cv2.threshold(flattened_gray, params["thresh"], 255, cv2.THRESH_BINARY)

    kernel = np.ones((3, 3), dtype=np.uint8)
    if params["erode"]:
        img = cv2.erode(img, kernel, iterations=1)
    if params["open"]:
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    if params["dilate"]:
        img = cv2.dilate(img, kernel, iterations=1)
    return img


def preprocess(frame_gray: np.ndarray, params: dict) -> np.ndarray:
    return binarize(flatten_illumination(denoise(frame_gray, params), params), params)


def _labeled_components(preprocessed: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Connected-component pass shared by detect(), detection_labels(), and
    count_oversized_components() so each caller doesn't repeat the analysis."""
    _, labels, stats, centroids = cv2.connectedComponentsWithStats(preprocessed, connectivity=8)
    return labels, stats, centroids


def _area_accept(stats: np.ndarray, params: dict) -> np.ndarray:
    min_area = params.get("min_area", 5.0)
    max_area = params.get("max_area", 3000.0)
    areas = stats[:, cv2.CC_STAT_AREA].astype(float)
    accept = (areas >= min_area) & (areas <= max_area)
    accept[0] = False  # label 0 is background
    return accept


def detection_labels(
    _gray: np.ndarray,
    preprocessed: np.ndarray,
    params: dict,
) -> tuple[np.ndarray, set[int]]:
    labels, stats, _ = _labeled_components(preprocessed)
    accept = _area_accept(stats, params)
    accepted = {label for label in range(1, stats.shape[0]) if accept[label]}
    return labels, accepted


def _blob_window(
    gray: np.ndarray,
    labels: np.ndarray,
    label: int,
    stats_row: np.ndarray,
    margin: int,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """CC bounding box dilated by margin (clipped to image bounds), plus a
    mask excluding pixels claimed by a DIFFERENT component's label -- pixels
    below threshold (label 0, the halo the Gaussian fit needs) stay
    included; only a neighboring marker's own labeled footprint is masked
    out, guarding against bleed if two blobs ever get close enough to share
    a dilated window."""
    h, w = gray.shape
    left = int(stats_row[cv2.CC_STAT_LEFT])
    top = int(stats_row[cv2.CC_STAT_TOP])
    width = int(stats_row[cv2.CC_STAT_WIDTH])
    height = int(stats_row[cv2.CC_STAT_HEIGHT])

    x0 = max(0, left - margin)
    y0 = max(0, top - margin)
    x1 = min(w, left + width + margin)
    y1 = min(h, top + height + margin)

    window = gray[y0:y1, x0:x1].astype(float)
    label_window = labels[y0:y1, x0:x1]
    neighbor_mask = (label_window == 0) | (label_window == label)
    return window, neighbor_mask, x0, y0


def detect(
    gray: np.ndarray,
    preprocessed: np.ndarray,
    params: dict,
    *,
    fallback_count: list[int] | None = None,
    moments_out: list[float] | None = None,
) -> list[tuple[float, float, float]]:
    """moments_out, when passed as an (initially empty) list, is appended
    with one intensity-weighted-moments size estimate per returned
    detection (NaN on a degenerate window) -- a free cross-check against
    whichever size_method actually drives the returned (x, y, size). Left
    None by default so callers not asking for it pay zero extra cost."""
    labels, stats, centroids = _labeled_components(preprocessed)
    accept = _area_accept(stats, params)
    size_method = params.get("size_method", "gaussian")
    margin = int(params.get("blob_margin_px", 4))
    need_window = size_method == "gaussian" or moments_out is not None

    detections: list[tuple[float, float, float]] = []
    for label in range(1, stats.shape[0]):
        if not accept[label]:
            continue
        # Connected-component centroid/area is the shape-agnostic fallback --
        # stays stable even when the marker is squished/elliptical near the
        # curved edges, and is used directly when size_method == "cc" or
        # whenever the Gaussian fit is invalid for this blob.
        cx, cy = centroids[label]
        cc_area = float(stats[label, cv2.CC_STAT_AREA])

        window: np.ndarray | None = None
        neighbor_mask: np.ndarray | None = None
        wx0 = wy0 = 0
        background = 0.0
        if need_window:
            window, neighbor_mask, wx0, wy0 = _blob_window(gray, labels, label, stats[label], margin)
            background = estimate_background(window)

        if moments_out is not None:
            assert window is not None and neighbor_mask is not None
            moments = fit_moments(window, background, mask=neighbor_mask)
            moments_out.append(moments.size if moments is not None else float("nan"))

        if size_method != "gaussian":
            detections.append((float(cx), float(cy), cc_area))
            continue

        assert window is not None and neighbor_mask is not None
        fit = fit_gaussian_logparabola(window, background, mask=neighbor_mask)
        if fit is None:
            if fallback_count is not None:
                fallback_count.append(1)
            detections.append((float(cx), float(cy), cc_area))
        else:
            detections.append((fit.x0 + wx0, fit.y0 + wy0, fit.size))

    return detections


def count_oversized_components(
    _gray: np.ndarray,
    preprocessed: np.ndarray,
    params: dict,
    oversize_factor: float = 1.8,
) -> int:
    """Diagnostic: components whose area exceeds oversize_factor times the
    median accepted area -- a canary for markers whose blobs have merged."""
    _, stats, _ = _labeled_components(preprocessed)
    accept = _area_accept(stats, params)
    areas = stats[accept, cv2.CC_STAT_AREA].astype(float)
    if areas.size == 0:
        return 0
    median_area = float(np.median(areas))
    if median_area <= 0:
        return 0
    return int(np.sum(areas > oversize_factor * median_area))
