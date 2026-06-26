"""Camera calibration — loads the latest calibration.json and precomputes
undistortion maps once at startup for use in the frame loop via remap().
"""
import json
from pathlib import Path

import cv2
import numpy as np

_RESULTS_DIR = Path(__file__).parent.parent / "camera_calibration" / "results"


def _latest_calibration_path() -> Path:
    folders = sorted(_RESULTS_DIR.iterdir())
    if not folders:
        raise FileNotFoundError(f"No calibration results found in {_RESULTS_DIR}")
    return folders[-1] / "calibration.json"


class CameraCalibration:
    def __init__(self, cal_path: Path | None = None):
        path = cal_path or _latest_calibration_path()
        raw = json.loads(path.read_text())
        K = np.array(raw["K"], dtype=np.float64)
        D = np.array(raw["D"], dtype=np.float64)
        w, h = raw["image_size"]
        # alpha=1.0 retains all source pixels — no edge cropping
        K_new, _ = cv2.getOptimalNewCameraMatrix(K, D, (w, h), alpha=1.0)
        self._map1, self._map2 = cv2.initUndistortRectifyMap(
            K, D, None, K_new, (w, h), cv2.CV_16SC2
        )

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        return cv2.remap(frame, self._map1, self._map2, cv2.INTER_LINEAR)
