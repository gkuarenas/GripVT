"""ROI crop — loads roi.json and applies the saved crop to every frame.

If roi.json is absent, crop() returns the frame unchanged.

Note: all pixel coordinates produced by the MDM pipeline are relative
to the ROI origin. Add (roi.x, roi.y) back before any back-projection
to 3D cylinder coordinates.

The ROI box was drawn on a frame undistorted with a specific alpha
(see calibration/roi/select_roi.py). Undistorting with a different alpha remaps
pixel positions, so the saved box only lines up with the marker grid
when consumers undistort with the same alpha it was selected under —
hence alpha travels with x/y/w/h instead of being chosen independently
downstream.

The rectangular crop alone still includes the dark curved margins of
the cylindrical sensor (its outline isn't a rectangle), which is where
stray blobs get falsely detected as markers. sensor_mask() gates
detection to a polygon traced around the sensor outline, authored by
calibration/roi/select_roi.py and stored crop-relative so it doesn't need to
know the rect's origin.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_ROI_PATH = Path(__file__).parent.parent / "calibration" / "roi" / "roi.json"
DEFAULT_ALPHA = 1.0


@dataclass
class ROIConfig:
    x: int
    y: int
    w: int
    h: int
    alpha: float = DEFAULT_ALPHA
    mask_points: Optional[list[tuple[int, int]]] = None


class ROICrop:
    def __init__(self, roi_path: Path = _ROI_PATH):
        self._config: Optional[ROIConfig] = None
        self._mask_cache: Optional[np.ndarray] = None
        self._mask_cache_shape: Optional[tuple[int, int]] = None
        if roi_path.exists():
            raw = json.loads(roi_path.read_text())
            raw_points = raw.get("mask_points")
            mask_points = (
                [(int(p[0]), int(p[1])) for p in raw_points] if raw_points else None
            )
            self._config = ROIConfig(
                x=int(raw["x"]), y=int(raw["y"]), w=int(raw["w"]), h=int(raw["h"]),
                alpha=float(raw.get("alpha", DEFAULT_ALPHA)),
                mask_points=mask_points,
            )

    @property
    def is_set(self) -> bool:
        return self._config is not None

    @property
    def config(self) -> Optional[ROIConfig]:
        return self._config

    def crop(self, frame: np.ndarray) -> np.ndarray:
        if self._config is None:
            return frame
        c = self._config
        return frame[c.y : c.y + c.h, c.x : c.x + c.w]

    def sensor_mask(self, shape: tuple[int, int]) -> Optional[np.ndarray]:
        """Binary mask (255 inside the sensor outline polygon, 0 outside),
        sized to `shape` (h, w) of an already-cropped frame. None if no
        polygon was authored (rectangle-only roi.json) — callers should
        skip masking entirely in that case, not treat it as mask-everything.
        """
        config = self._config
        if config is None or not config.mask_points:
            return None
        if self._mask_cache is not None and self._mask_cache_shape == shape:
            return self._mask_cache
        mask = np.zeros(shape, dtype=np.uint8)
        points = np.array(config.mask_points, dtype=np.int32)
        cv2.fillPoly(mask, [points], 255)
        self._mask_cache = mask
        self._mask_cache_shape = shape
        return mask

    def to_full_frame(self, x: float, y: float) -> tuple[float, float]:
        """Convert ROI-relative pixel coords back to full-frame coords."""
        if self._config is None:
            return x, y
        return x + self._config.x, y + self._config.y

    def to_crop(self, x: float, y: float) -> tuple[float, float]:
        """Inverse of to_full_frame -- full-frame coords to ROI-relative."""
        if self._config is None:
            return x, y
        return x - self._config.x, y - self._config.y
