import json
from pathlib import Path

import cv2
import numpy as np

_PARAMS_PATH = Path(__file__).parent.parent / "detector_params.json"


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


def detect(
    _gray: np.ndarray,
    preprocessed: np.ndarray,
    params: dict,
) -> list[tuple[float, float, float]]:
    _, stats, centroids = _labeled_components(preprocessed)
    accept = _area_accept(stats, params)

    detections: list[tuple[float, float, float]] = []
    for label in range(1, stats.shape[0]):
        if not accept[label]:
            continue
        # Connected-component centroid is shape-agnostic and stays stable
        # even when the marker is squished/elliptical near the curved edges.
        cx, cy = centroids[label]
        detections.append((float(cx), float(cy), float(stats[label, cv2.CC_STAT_AREA])))

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
