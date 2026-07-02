import json
from dataclasses import dataclass

import cv2
import numpy as np

from core.detector import load_params, preprocess, detect, detection_labels
from core.kalman import KalmanManager
from core.hungarian import assign
from core.roi import ROICrop
from core import zdisplacement

TARGET_MARKER_COUNT = 121
BASELINE_FRAMES = 15
# A cluster must be seen in at least this fraction of sampled baseline
# frames to be locked -- a detection appearing in only one or two frames is
# more likely a transient false positive than a real marker.
BASELINE_PRESENCE_FRACTION = 0.6
# Max distance between detections across frames to count as the same
# marker. The sensor doesn't move during baseline capture, so this only
# needs to absorb centroid jitter -- well under half the ~60-90px marker
# pitch, so adjacent markers can't get merged into one cluster.
BASELINE_CLUSTER_RADIUS_PX = 15.0
BASELINE_COUNT_TOLERANCE = 5


def _cluster_across_frames(
    per_frame_dets: list[list[tuple[float, float, float]]],
    radius_px: float,
) -> list[tuple[list[float], list[float], list[float]]]:
    """Associate detections across frames into per-marker clusters by
    nearest-neighbour proximity -- reuses the same gated Hungarian
    assignment as frame-to-frame tracking, since a static sensor needs no
    motion model. Returns, per cluster, the x/y/area samples collected
    across the frames it was seen in."""
    cluster_centroids: dict[int, np.ndarray] = {}
    cluster_samples: dict[int, tuple[list[float], list[float], list[float]]] = {}
    next_id = 0

    for dets in per_frame_dets:
        if not dets:
            continue
        matches, _unmatched = assign(cluster_centroids, dets, radius_px)
        matched_det_indices = set(matches.values())

        for cluster_id, det_idx in matches.items():
            x, y, area = dets[det_idx]
            xs, ys, areas = cluster_samples[cluster_id]
            xs.append(x)
            ys.append(y)
            areas.append(area)
            cluster_centroids[cluster_id] = np.array([np.mean(xs), np.mean(ys)])

        for det_idx, (x, y, area) in enumerate(dets):
            if det_idx in matched_det_indices:
                continue
            cluster_samples[next_id] = ([x], [y], [area])
            cluster_centroids[next_id] = np.array([x, y], dtype=float)
            next_id += 1

    return list(cluster_samples.values())


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

        # NOTE: this undistort path is independent of core/calibration.py's
        # CameraCalibration (which viewer_window.py and calibrate_detector.py
        # use, and which tools/select_roi.py draws the ROI against with a
        # chosen alpha/balance). This class builds its own map directly from
        # K/D with no balance parameter, so roi.json's saved alpha may not
        # correspond to pixel positions here -- the crop below assumes it
        # does. Flagged, not fixed: reconciling the two calibration paths is
        # a separate change from detector reliability and isn't verifiable
        # without the real rig.
        self._roi = ROICrop()

        self.kalman = KalmanManager()
        # gate_px bounds how far a single-frame detection may sit from a
        # state's Kalman-predicted position (~33ms of motion at 30fps) to
        # still be treated as the same marker -- it is NOT a limit on total
        # baseline-relative displacement, which is unbounded over a session.
        # 280px let a stray detection anywhere in roughly a third of the
        # frame capture a state; with ROI+mask now removing border false
        # positives at the source, CLAUDE.md's documented default (100px,
        # never below 80px) is the safer gate -- real per-frame motion at
        # 30fps is far under this for any physically plausible press/release.
        self.gate_px: float = 100.0
        self.params: dict = load_params()
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

    def _detect_full_frame(
        self, gray_full: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float]]]:
        """Crop to the ROI and gate to the sensor-outline mask (if authored)
        before detecting, so stray blobs in the dark curved margins can't
        become detections at all. Detected coordinates are translated back
        to full-frame pixel space before returning -- Kalman state
        positions, the mm conversion (_cx/_cy), and overlay drawing (on
        last_undistorted) all operate in full-frame space. Returns
        (cropped_gray, masked_binary, full_frame_detections)."""
        cropped_gray = self._roi.crop(gray_full)
        proc = preprocess(cropped_gray, self.params)
        mask = self._roi.sensor_mask(proc.shape)
        if mask is not None:
            proc = cv2.bitwise_and(proc, mask)
        dets_cropped = detect(cropped_gray, proc, self.params)
        dets = [(*self._roi.to_full_frame(x, y), area) for x, y, area in dets_cropped]
        return cropped_gray, proc, dets

    def capture_baseline(self, raw_frames: list[np.ndarray]) -> int:
        """Locks the 121 Kalman states from a burst of frames (the sensor
        must be at rest) rather than a single grab -- since these states are
        never re-created for the rest of the session, a baseline captured
        from one noisy/jittery frame permanently poisons every displacement
        computed against it. Each marker's locked position is the per-frame
        median across every frame it was detected in, which is far less
        sensitive to a single bad frame than either a single-frame read or a
        running mean."""
        if not raw_frames:
            raise ValueError("capture_baseline() requires at least one frame")

        per_frame_dets: list[list[tuple[float, float, float]]] = []
        cropped_gray = proc = None
        for raw_frame in raw_frames:
            undistorted = self._undistort(raw_frame)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            cropped_gray, proc, dets = self._detect_full_frame(gray)
            per_frame_dets.append(dets)

        # Diagnostic view reflects the most recently sampled frame.
        assert cropped_gray is not None and proc is not None
        self._last_baseline_gray = cropped_gray.copy()
        self._last_baseline_binary = proc.copy()

        clusters = _cluster_across_frames(per_frame_dets, BASELINE_CLUSTER_RADIUS_PX)
        min_presence = max(1, int(round(len(raw_frames) * BASELINE_PRESENCE_FRACTION)))
        locked = [
            (float(np.median(xs)), float(np.median(ys)), float(np.median(areas)))
            for xs, ys, areas in clusters
            if len(xs) >= min_presence
        ]
        locked.sort(key=lambda d: (d[1], d[0]))  # deterministic: top→bottom, left→right

        self.kalman.states.clear()
        for i, (x, y, area) in enumerate(locked):
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
        count = len(locked)
        if abs(count - TARGET_MARKER_COUNT) > BASELINE_COUNT_TOLERANCE:
            print(
                f"Warning: baseline locked {count} markers (target {TARGET_MARKER_COUNT}) "
                f"-- consider retrying capture_baseline()."
            )
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
        _, _, dets = self._detect_full_frame(gray)

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
