"""ROI crop — loads roi.json and applies the saved crop to every frame.

If roi.json is absent, crop() returns the frame unchanged.

Note: all pixel coordinates produced by the MDM pipeline are relative
to the ROI origin. Add (roi.x, roi.y) back before any back-projection
to 3D cylinder coordinates.

The ROI box was drawn on a frame undistorted with a specific alpha
(see tools/select_roi.py). Undistorting with a different alpha remaps
pixel positions, so the saved box only lines up with the marker grid
when consumers undistort with the same alpha it was selected under —
hence alpha travels with x/y/w/h instead of being chosen independently
downstream.
"""
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np

_ROI_PATH = Path(__file__).parent.parent / "roi.json"
DEFAULT_ALPHA = 1.0


@dataclass
class ROIConfig:
    x: int
    y: int
    w: int
    h: int
    alpha: float = DEFAULT_ALPHA


class ROICrop:
    def __init__(self, roi_path: Path = _ROI_PATH):
        self._config: Optional[ROIConfig] = None
        if roi_path.exists():
            raw = json.loads(roi_path.read_text())
            self._config = ROIConfig(
                x=int(raw["x"]), y=int(raw["y"]), w=int(raw["w"]), h=int(raw["h"]),
                alpha=float(raw.get("alpha", DEFAULT_ALPHA)),
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

    def to_full_frame(self, x: float, y: float) -> tuple[float, float]:
        """Convert ROI-relative pixel coords back to full-frame coords."""
        if self._config is None:
            return x, y
        return x + self._config.x, y + self._config.y
