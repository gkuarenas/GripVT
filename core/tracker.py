from dataclasses import dataclass

import cv2
import numpy as np

from core.calibration import CameraCalibration
from core.detector import load_params, preprocess, denoise, detect, detection_labels
from core.detector_search import AnchorError, geometry_from, grid_positions
from core.kalman import KalmanManager
from core.hungarian import assign
from core.profiling import stage
from core.roi import ROICrop

TARGET_MARKER_COUNT = 99
BASELINE_FRAMES = 15
# A cluster must be seen in at least this fraction of sampled baseline
# frames to be locked -- a detection appearing in only one or two frames is
# more likely a transient false positive than a real marker.
BASELINE_PRESENCE_FRACTION = 0.6
# Max distance between detections across frames to count as the same
# marker. The sensor doesn't move during baseline capture, so this only
# needs to absorb centroid jitter -- well under half the 30-51px
# rig-measured marker pitch, so adjacent markers can't get merged into
# one cluster.
BASELINE_CLUSTER_RADIUS_PX = 12.0
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
    predicted_x: float
    predicted_y: float
    autofilled: bool
    size_moments: float


class Tracker:
    def __init__(self, calibration: CameraCalibration, roi: ROICrop | None = None) -> None:
        # Undistorts through the same CameraCalibration + alpha/balance path
        # roi.json was authored against (see calibration/roi/select_roi.py), so the ROI
        # crop below lands on the marker grid correctly.
        self._calibration = calibration
        calib_w, calib_h = calibration.image_size
        # calib_w/calib_h stay un-rotated -- they gate the resize below,
        # which runs on the raw (pre-undistort, pre-rotate) capture frame.
        self._calib_w, self._calib_h = calib_w, calib_h

        self._roi = roi if roi is not None else ROICrop()

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
        # Populated by capture_baseline() from real grid geometry (see
        # docs/logs/20260804_161615_marker_grid_lookup_plan.md); grid_layout
        # stays None whenever geometry can't be resolved and marker_id falls
        # back to the naive (y, x) sort -- check grid_layout_source before
        # trusting grid_layout.
        self.grid_layout: dict[int, tuple[int, int]] | None = None
        self.grid_layout_source: str = "none"
        self.grid_layout_reason: str | None = None
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
        undistorted = self._calibration.undistort(frame)
        # Rotate to match the viewer's display orientation -- roi.json's box
        # (see calibration/roi/select_roi.py) is defined in this rotated coordinate
        # space, so cropping must happen after this, not before.
        return cv2.rotate(undistorted, cv2.ROTATE_90_CLOCKWISE)

    def _undistort(self, frame: np.ndarray) -> np.ndarray:
        return self.undistort(frame)

    def _detect_full_frame(
        self, gray_full: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, list[tuple[float, float, float]], list[float] | None]:
        """Crop to the ROI and gate to the sensor-outline mask (if authored)
        before detecting, so stray blobs in the dark curved margins can't
        become detections at all. Detected coordinates are translated back
        to full-frame pixel space before returning -- Kalman state
        positions and overlay drawing (on last_undistorted) all operate in
        full-frame space. Returns
        (cropped_gray, masked_binary, full_frame_detections, size_moments).
        size_moments is None unless params['log_size_crosscheck'] is set (kept
        off by default so callers not asking for the cross-check pay zero
        extra cost); when present it's aligned 1:1 with full_frame_detections."""
        cropped_gray = self._roi.crop(gray_full)
        proc = preprocess(cropped_gray, self.params)
        mask = self._roi.sensor_mask(proc.shape)
        if mask is not None:
            proc = cv2.bitwise_and(proc, mask)
        # Gaussian fitting needs the photometric profile intact -- CLAHE/
        # top-hat (inside preprocess()) are nonlinear and would distort it,
        # so the fit runs against denoise()'s output, not the fully
        # preprocessed one. proc still supplies blob location/labels.
        denoised = denoise(cropped_gray, self.params)
        moments_out: list[float] | None = [] if self.params.get("log_size_crosscheck", False) else None
        dets_cropped = detect(denoised, proc, self.params, moments_out=moments_out)
        dets = [(*self._roi.to_full_frame(x, y), area) for x, y, area in dets_cropped]
        return cropped_gray, proc, dets, moments_out

    def capture_baseline(self, raw_frames: list[np.ndarray]) -> int:
        """Locks the 99 Kalman states from a burst of frames (the sensor
        must be at rest) rather than a single grab -- since these states are
        never re-created for the rest of the session, a baseline captured
        from one noisy/jittery frame permanently poisons every displacement
        computed against it. Each marker's locked position is the per-frame
        median across every frame it was detected in, which is far less
        sensitive to a single bad frame than either a single-frame read or a
        running mean."""
        if not raw_frames:
            raise ValueError("capture_baseline() requires at least one frame")

        # A re-capture must not leave a stale layout from a previous attempt
        # visible if this attempt takes the fallback path.
        self.grid_layout = None
        self.grid_layout_source = "none"
        self.grid_layout_reason = None

        per_frame_dets: list[list[tuple[float, float, float]]] = []
        cropped_gray = proc = None
        for raw_frame in raw_frames:
            undistorted = self._undistort(raw_frame)
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
            cropped_gray, proc, dets, _moments = self._detect_full_frame(gray)
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
        # Fallback ordering only -- used when geometry can't be resolved
        # below. In practice this sorts by y alone: the x tiebreak never
        # fires for real float centroids, so it is not actually
        # top-to-bottom-then-left-to-right (see plan §1 Mode A).
        locked.sort(key=lambda d: (d[1], d[0]))

        try:
            geom = geometry_from(locked)
            positions = grid_positions(locked, geom)
            ids = [r * geom.x_lines + c for r, c in positions]
            self.grid_layout = {mid: (row, col) for mid, (row, col) in zip(ids, positions)}
            self.grid_layout_source = "geometry"
        except AnchorError as exc:
            ids = list(range(len(locked)))
            self.grid_layout = None
            self.grid_layout_source = "fallback_sort"
            self.grid_layout_reason = str(exc)
            print(
                f"Warning: baseline grid geometry unresolved ({exc}) -- marker_id "
                "fell back to the flat (y, x) sort; no geometric row/col available "
                "for this session."
            )

        self.kalman.states.clear()
        for mid, (x, y, area) in sorted(zip(ids, locked), key=lambda pair: pair[0]):
            self.kalman.init_state(mid, x, y, area)

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

        if self.params.get("size_method", "gaussian") == "gaussian":
            # Overlay both estimators on the same accepted blobs so drift
            # between the CC centroid and the Gaussian fit is visible while
            # tuning -- cc_params forces the "cc" path on an otherwise
            # identical params dict, so the only difference between the two
            # detect() calls is the estimator itself.
            cc_params = dict(self.params, size_method="cc")
            denoised = denoise(self._last_baseline_gray, self.params)
            cc_dets = detect(self._last_baseline_gray, self._last_baseline_binary, cc_params)
            gaussian_dets = detect(denoised, self._last_baseline_binary, self.params)
            for (cx, cy, _cc_area), (gx, gy, _g_size) in zip(cc_dets, gaussian_dets):
                cv2.drawMarker(
                    vis, (int(round(cx)), int(round(cy))), (0, 0, 255),
                    markerType=cv2.MARKER_CROSS, markerSize=6, thickness=1,
                )
                cv2.drawMarker(
                    vis, (int(round(gx)), int(round(gy))), (255, 255, 0),
                    markerType=cv2.MARKER_TILTED_CROSS, markerSize=6, thickness=1,
                )
        return vis

    def process_frame(
        self, raw_frame: np.ndarray, undistorted: np.ndarray | None = None,
    ) -> list[MarkerRecord]:
        """undistorted (Phase 3, framerate spike plan D4/D5): pass the
        already-undistorted frame when a caller (today, _poll()) has
        already computed it for the display path, so the identical
        full-frame undistort isn't repeated here. None (the default)
        preserves today's behaviour exactly -- capture_baseline() and any
        other caller that hasn't pre-undistorted keep working unchanged."""
        if not self.baseline_set:
            raise RuntimeError("Call capture_baseline() before process_frame()")

        # Phase 1 (framerate spike, see docs/logs/20260731_152249_
        # framerate_spike_plan.md): stage() is a no-op unless
        # GRIPVT_PROFILE=1 -- this splits the previously opaque "track"
        # blob into undistort/gray/detect without changing any math.
        if undistorted is None:
            with stage("undistort_math"):
                undistorted = self._undistort(raw_frame)
        self._last_undistorted = undistorted
        with stage("gray"):
            gray = cv2.cvtColor(undistorted, cv2.COLOR_BGR2GRAY)
        with stage("detect"):
            _, _, dets, moments = self._detect_full_frame(gray)

        priors = self.kalman.predict_all()
        matches, unmatched = assign(priors, dets, self.gate_px)

        records: list[MarkerRecord] = []
        for sid, s in self.kalman.states.items():
            predicted_xy = priors[sid]

            if sid in matches:
                det_idx = matches[sid]
                det = dets[det_idx]
                self.kalman.correct(sid, np.array([det[0], det[1]], dtype=float))
                x, y, area = det[0], det[1], det[2]
                size_moments = moments[det_idx] if moments is not None else float("nan")
            else:
                self.kalman.mark_autofilled(sid)
                x, y, area = float(s.x[0]), float(s.x[1]), s.baseline_area
                size_moments = float("nan")

            bx, by = s.baseline_pos
            dx_px = x - bx
            dy_px = y - by
            records.append(MarkerRecord(
                marker_id=sid,
                x=x,
                y=y,
                area=area,
                dx=dx_px,
                dy=dy_px,
                dA=area - s.baseline_area,
                magnitude=float(np.hypot(dx_px, dy_px)),
                size_moments=size_moments,
                predicted_x=float(predicted_xy[0]),
                predicted_y=float(predicted_xy[1]),
                autofilled=s.autofilled,
            ))

        self.frame_index += 1
        return records
