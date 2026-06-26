import json
from dataclasses import dataclass

import cv2
import numpy as np

from core.detector import default_params, preprocess, detect, detection_labels
from core.kalman import KalmanManager
from core.hungarian import assign
from core import zdisplacement


@dataclass
class MarkerRecord:
    marker_id: int
    x: float
    y: float
    area: float
    dx: float
    dy: float
    dA: float
    magnitude: float
    dx_mm: float
    dy_mm: float
    delta_z_mm: float
    magnitude_mm: float
    predicted_x: float
    predicted_y: float
    autofilled: bool


class Tracker:
    def __init__(self, calib_path: str = "calibration.json") -> None:
        with open(calib_path, "r") as f:
            data = json.load(f)

        K = np.array(data["K"], dtype=float)
        D = np.array(data["D"], dtype=float)    # shape (4,1) — already correct for cv2.fisheye
        w, h = data["image_size"]               # [width, height] — width is index 0
        self._fx: float = float(K[0, 0])
        self._fy: float = float(K[1, 1])
        self._cx: float = float(K[0, 2])
        self._cy: float = float(K[1, 2])

        self.map1, self.map2 = cv2.fisheye.initUndistortRectifyMap(
            K, D, np.eye(3), K, (w, h), cv2.CV_16SC2
        )
        self._calib_w: int = w
        self._calib_h: int = h

        self.kalman = KalmanManager()
        self.gate_px: float = 280.0
        self.params: dict = default_params()
        self.baseline_set: bool = False
        self.baseline_positions_mm: dict[int, tuple[float, float]] = {}
        self.frame_index: int = 0
        self._last_baseline_gray: np.ndarray | None = None
        self._last_baseline_binary: np.ndarray | None = None
        self._last_undistorted: np.ndarray | None = None

    @property
    def last_undistorted(self) -> np.ndarray | None:
        """Undistorted frame from the most recent process_frame() call —
        overlay drawing must use this so marker coordinates line up."""
        return self._last_undistorted

    def undistort(self, frame: np.ndarray) -> np.ndarray:
        h, w = frame.shape[:2]
        if w != self._calib_w or h != self._calib_h:
            frame = cv2.resize(frame, (self._calib_w, self._calib_h), interpolation=cv2.INTER_LINEAR)
        return cv2.remap(frame, self.map1, self.map2, cv2.INTER_LINEAR)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        return self.undistort(frame)

    def capture_baseline(self, raw_frame: np.ndarray) -> int:
        undistorted = self._undistort(raw_frame)
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        self._last_baseline_gray = gray.copy()
        proc = preprocess(gray, self.params)
        self._last_baseline_binary = proc.copy()
        dets = detect(gray, proc, self.params)
        dets.sort(key=lambda d: (d[1], d[0]))  # deterministic: top→bottom, left→right

        self.kalman.states.clear()
        for i, (x, y, area) in enumerate(dets):
            self.kalman.init_state(i, x, y, area)

        # Baseline positions converted to mm, ASSUMING the camera principal
        # point (cx, cy) coincides with the G92 machine origin (X0 Y0) — both
        # are physically the slab centre by design. Uses the same per-axis
        # px->mm scale factor (H_MM / f) the pipeline already applies to
        # deltas (see core/zdisplacement.compute). The X term is negated:
        # empirically confirmed via sensitivity_analysis.ipynb's spatial
        # cross-check that the camera's pixel-X axis is mirrored relative to
        # the G92 machine X-axis (Y is not mirrored).
        self.baseline_positions_mm = {
            sid: (-(s.baseline_pos[0] - self._cx) * (zdisplacement.H_MM / self._fx),
                   (s.baseline_pos[1] - self._cy) * (zdisplacement.H_MM / self._fy))
            for sid, s in self.kalman.states.items()
        }

        self.baseline_set = True
        count = len(dets)
        if count < 100:
            print(f"Warning: only {count} markers detected at baseline (expected ~154)")
        return count

    def build_detection_diagnostic(self) -> np.ndarray | None:
        if self._last_baseline_gray is None or self._last_baseline_binary is None:
            return None
        labels, accepted = detection_labels(
            self._last_baseline_gray, self._last_baseline_binary, self.params
        )
        vis = np.zeros((*self._last_baseline_gray.shape, 3), dtype=np.uint8)
        for lbl in np.unique(labels).tolist():
            if lbl == 0:
                continue
            mask = labels == lbl
            vis[mask] = (0, 255, 0) if lbl in accepted else (0, 0, 255)
        return vis

    def process_frame(self, raw_frame: np.ndarray) -> list[MarkerRecord]:
        if not self.baseline_set:
            raise RuntimeError("Call capture_baseline() before process_frame()")

        undistorted = self._undistort(raw_frame)
        self._last_undistorted = undistorted
        gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        proc = preprocess(gray, self.params)
        dets = detect(gray, proc, self.params)

        priors = self.kalman.predict_all()
        matches, unmatched = assign(priors, dets, self.gate_px)

        records: list[MarkerRecord] = []
        for sid, s in self.kalman.states.items():
            predicted_xy = priors[sid]

            if sid in matches:
                det = dets[matches[sid]]
                self.kalman.correct(sid, np.array([det[0], det[1]], dtype=float))
                x, y, area = det[0], det[1], det[2]
            else:
                self.kalman.mark_autofilled(sid)
                x, y, area = float(s.x[0]), float(s.x[1]), s.baseline_area

            bx, by = s.baseline_pos
            dx_px = x - bx
            dy_px = y - by
            dx_mm, dy_mm, delta_z_mm = zdisplacement.compute(
                dx_px, dy_px, s.baseline_area, area, self._fx, self._fy
            )
            records.append(MarkerRecord(
                marker_id=sid,
                x=x,
                y=y,
                area=area,
                dx=dx_px,
                dy=dy_px,
                dA=area - s.baseline_area,
                magnitude=float(np.hypot(dx_px, dy_px)),
                dx_mm=dx_mm,
                dy_mm=dy_mm,
                delta_z_mm=delta_z_mm,
                magnitude_mm=float(np.sqrt(dx_mm**2 + dy_mm**2 + delta_z_mm**2)),
                predicted_x=float(predicted_xy[0]),
                predicted_y=float(predicted_xy[1]),
                autofilled=s.autofilled,
            ))

        self.frame_index += 1
        return records
