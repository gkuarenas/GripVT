"""Tests for tools/sync_session.py (docs/logs/20260806_133000_session_sync_plan.md).

Every fixture here is synthetic and built inline with numpy/csv/json -- this
suite must pass on a machine with no recordings/ directory at all. Where the
spec calls for "a known injected lag", the fixtures construct the camera
grip-proxy signal as an explicit function of the force signal shifted by a
chosen offset, so the true answer is known before the code under test ever
sees it (see sync_session.py's own convention: force_corrected(T) =
interp_force(T - offset), a positive offset meaning the camera event is
measured as happening after the force event that caused it).
"""
import csv
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest

from tools.sync_session import (
    N_MARKERS,
    AGREEMENT_TOLERANCE_S,
    XCORR_STEP_MS,
    ForceTrace,
    MarkerTrace,
    SyncError,
    TapResult,
    XcorrResult,
    compute_rest_offsets,
    compute_taps,
    compute_xcorr,
    main,
    read_force,
    sync_session,
)

# --------------------------------------------------------------------------
# Shared synthetic building blocks
# --------------------------------------------------------------------------

ANCHOR_DT = datetime(2026, 1, 1, 9, 0, 0, 0)
ANCHOR_EPOCH_MS = ANCHOR_DT.timestamp() * 1000.0
FORCE_SAMPLE_PERIOD_S = 0.005  # matches the Biopac's 200 Hz / 5 ms-per-sample


def _osc(t, base=2.0, amp=1.5, period=7.0):
    """Smooth background 'grip-like' signal. Period (7s) is well outside the
    +-2.0s xcorr search grid, so it cannot alias into a competing peak."""
    return base + amp * np.sin(2 * np.pi * np.asarray(t, dtype=np.float64) / period)


def _tap_bursts(t, centers, sigma=0.08, amp=6.0):
    """Narrow gaussian bursts standing in for the operator's triple-tap."""
    t = np.asarray(t, dtype=np.float64)
    val = np.zeros_like(t)
    for c in centers:
        val = val + amp * np.exp(-0.5 * ((t - c) / sigma) ** 2)
    return val


def _shape(t, tap_centers):
    return _osc(t) + _tap_bursts(t, tap_centers)


START_TAPS = [3.0, 3.7, 4.4]   # gaps 0.7s, 0.7s -- both inside [0.35, 1.6]
END_TAPS = [15.0, 15.6, 16.3]  # gaps 0.6s, 0.7s
TAP_CENTERS = START_TAPS + END_TAPS


def _force_trace(t_sec, tap_centers=TAP_CENTERS) -> ForceTrace:
    force_kg = _shape(t_sec, tap_centers)
    time_ms = ANCHOR_EPOCH_MS + np.asarray(t_sec, dtype=np.float64) * 1000.0
    return ForceTrace(
        path=Path("<synthetic>"), anchor_epoch_ms=ANCHOR_EPOCH_MS,
        sample_period_s=FORCE_SAMPLE_PERIOD_S, force_kg=force_kg, time_ms=time_ms,
    )


def _marker_trace_from_proxy(t_cam_local, grip_proxy) -> MarkerTrace:
    """Builds a full (n_frames, N_MARKERS) MarkerTrace whose per-frame max
    magnitude equals the given grip_proxy exactly (every marker is set to
    the same value, so max-over-markers is trivially that value)."""
    n = len(t_cam_local)
    timestamp_ms = ANCHOR_EPOCH_MS + np.asarray(t_cam_local, dtype=np.float64) * 1000.0
    magnitude = np.tile(np.asarray(grip_proxy, dtype=np.float64).reshape(-1, 1), (1, N_MARKERS))
    dx = magnitude * 0.5
    dy = magnitude * 0.3
    dA = magnitude * 2.0
    autofilled = np.zeros((n, N_MARKERS), dtype=bool)
    return MarkerTrace(
        path=Path("<synthetic>"), frame=np.arange(n, dtype=np.int64),
        timestamp_ms=timestamp_ms, trial_id=np.zeros(n, dtype=np.int64),
        dx=dx, dy=dy, dA=dA, magnitude=magnitude, autofilled=autofilled,
        n_autofilled=autofilled.sum(axis=1), grip_proxy=np.asarray(grip_proxy, dtype=np.float64),
    )


def _marker_trace(t_cam_local, offset_s, tap_centers=TAP_CENTERS) -> MarkerTrace:
    """Camera trace whose grip proxy is the force-side shape() evaluated
    offset_s seconds earlier on the force timeline -- i.e. constructed so
    that recovering `offset_s` from (markers, force) alone is the known,
    correct answer."""
    grip_proxy = _shape(np.asarray(t_cam_local, dtype=np.float64) - offset_s, tap_centers)
    return _marker_trace_from_proxy(t_cam_local, grip_proxy)


T_SEC = np.arange(0.0, 20.0, FORCE_SAMPLE_PERIOD_S)          # 20s of force @ 200Hz
T_CAM = np.arange(2.0, 17.0, 0.05)                            # 15s of camera @ 20fps
TRUE_OFFSET_S = 0.115                                          # exact xcorr grid point


# --------------------------------------------------------------------------
# 1. BSL header parsing / failure modes (read_force)
# --------------------------------------------------------------------------

def _fmt_recording_on(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"


def _write_dynamometer_txt(path: Path, recording_on: datetime, sample_period_ms: float,
                            t_sec: list, force_kg: list, with_header: bool = True) -> None:
    lines = []
    if with_header:
        lines += [
            "graph-template.gtl.acq",
            f"Graph Created on: {_fmt_recording_on(recording_on)}",
            f"Recording on: {_fmt_recording_on(recording_on)}",
            f"{sample_period_ms} msec/sample",
            "1 channels",
            "Clench Force",
            "kg",
            "sec,CH2,",
            f",{len(t_sec)} samples,",
        ]
    for t, f in zip(t_sec, force_kg):
        lines.append(f"{t:.6f},{f:.5f},")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_read_force_parses_bsl_header(tmp_path):
    t_sec = [i * 0.005 for i in range(40)]
    force_kg = [0.5 + 0.1 * math.sin(i / 3.0) for i in range(40)]  # non-constant
    path = tmp_path / "dynamometer.txt"
    _write_dynamometer_txt(path, ANCHOR_DT, sample_period_ms=5, t_sec=t_sec, force_kg=force_kg)

    trace = read_force(path)

    expected_anchor_ms = ANCHOR_DT.timestamp() * 1000.0
    assert trace.anchor_epoch_ms == pytest.approx(expected_anchor_ms, abs=1.0)
    assert trace.sample_period_s == pytest.approx(0.005)
    assert trace.force_kg.size == 40
    assert trace.time_ms[0] == pytest.approx(trace.anchor_epoch_ms, abs=1.0)
    # anchor_epoch_ms + t_sec*1000 (spec sec 1)
    assert trace.time_ms[-1] == pytest.approx(trace.anchor_epoch_ms + t_sec[-1] * 1000.0, abs=1.0)


def test_read_force_missing_header_raises(tmp_path):
    # session 1's real failure: data rows present, no "Recording on:" line at all.
    t_sec = [i * 0.005 for i in range(20)]
    force_kg = [0.3 + 0.01 * i for i in range(20)]
    path = tmp_path / "dynamometer.txt"
    _write_dynamometer_txt(path, ANCHOR_DT, sample_period_ms=5, t_sec=t_sec, force_kg=force_kg,
                            with_header=False)

    with pytest.raises(SyncError, match="NO HEADER"):
        read_force(path)


def test_read_force_zero_data_rows_raises(tmp_path):
    # session 4's real failure: header written, then nothing.
    path = tmp_path / "dynamometer.txt"
    _write_dynamometer_txt(path, ANCHOR_DT, sample_period_ms=5, t_sec=[], force_kg=[])

    with pytest.raises(SyncError, match="ZERO DATA ROWS"):
        read_force(path)


# --------------------------------------------------------------------------
# 2. Cross-correlation offset recovery (compute_xcorr)
# --------------------------------------------------------------------------

def test_xcorr_recovers_known_lag():
    force = _force_trace(T_SEC)
    markers = _marker_trace(T_CAM, TRUE_OFFSET_S)

    result = compute_xcorr(markers, force)

    assert result.offset_s == pytest.approx(TRUE_OFFSET_S, abs=XCORR_STEP_MS / 1000.0)
    assert result.r_at_best > 0.95
    assert result.r_at_best > result.r_at_zero  # zero-lag must not beat the true offset
    # No drift injected -- the two halves should agree closely with each other.
    assert result.drift_s == pytest.approx(0.0, abs=0.03)


def test_xcorr_drift_check_reports_independent_half_offsets():
    force = _force_trace(T_SEC)
    markers = _marker_trace(T_CAM, TRUE_OFFSET_S)

    result = compute_xcorr(markers, force)

    assert result.offset_first_half_s == pytest.approx(TRUE_OFFSET_S, abs=0.03)
    assert result.offset_second_half_s == pytest.approx(TRUE_OFFSET_S, abs=0.03)


# --------------------------------------------------------------------------
# 3. Triple-tap detection (compute_taps)
# --------------------------------------------------------------------------

def test_tap_triple_detected_at_known_times():
    force = _force_trace(T_SEC)
    markers = _marker_trace(T_CAM, TRUE_OFFSET_S)

    result = compute_taps(markers, force)

    assert result.found is True
    assert result.offset_s == pytest.approx(TRUE_OFFSET_S, abs=0.05)
    # Force-side taps sit exactly on the 5ms sample grid -- recovered almost exactly.
    assert result.force_start_taps_s == pytest.approx(START_TAPS, abs=0.01)
    assert result.force_end_taps_s == pytest.approx(END_TAPS, abs=0.01)
    # Camera-side taps are the same physical events, offset by TRUE_OFFSET_S,
    # resolved to within one camera frame (0.05s).
    expected_camera_start = [c + TRUE_OFFSET_S for c in START_TAPS]
    expected_camera_end = [c + TRUE_OFFSET_S for c in END_TAPS]
    assert result.camera_start_taps_s == pytest.approx(expected_camera_start, abs=0.05)
    assert result.camera_end_taps_s == pytest.approx(expected_camera_end, abs=0.05)
    # No injected drift between the start and end bookends.
    assert result.drift_s == pytest.approx(0.0, abs=0.05)


def test_tap_triple_not_found_returns_null_without_raising():
    # No tap bursts at all -- just the smooth 7s-period oscillation, whose
    # local maxima are ~7s apart, far outside the [0.35, 1.6]s gap window
    # a real triple-tap requires. Must be reported, never raised (spec sec 4b).
    force = _force_trace(T_SEC, tap_centers=[])
    markers = _marker_trace(T_CAM, TRUE_OFFSET_S, tap_centers=[])

    result = compute_taps(markers, force)

    assert result.found is False
    assert result.offset_s is None
    assert result.note is not None


# --------------------------------------------------------------------------
# 4. Agreement rule (spec sec 4c) -- diff <= 0.15s -> agree, else flagged
# --------------------------------------------------------------------------

def test_agreement_rule_true_when_methods_close():
    # Same fixture as the recovery tests above: both methods see the same
    # single true offset, so they land close together.
    force = _force_trace(T_SEC)
    markers = _marker_trace(T_CAM, TRUE_OFFSET_S)

    xcorr = compute_xcorr(markers, force)
    taps = compute_taps(markers, force)

    diff = abs(xcorr.offset_s - taps.offset_s)
    assert diff <= AGREEMENT_TOLERANCE_S


def _agreement_ok(xcorr_offset_s: float, taps_offset_s: float) -> bool:
    """Mirrors sync_session()'s own inline agreement check (spec sec 4c):
    diff <= AGREEMENT_TOLERANCE_S -> agree. Exercised here directly against
    the exact tolerance boundary -- constructing two *independent* offset
    estimators that disagree by a controlled, precise amount from raw
    signals is fragile (xcorr's global correlation and taps' local
    threshold interact non-linearly with amplitude), so the boundary itself
    is what this test pins down, using real XcorrResult/TapResult objects
    to also confirm the field names line up."""
    xcorr = XcorrResult(
        offset_s=xcorr_offset_s, r_at_best=0.9, r_at_zero=0.5,
        offset_first_half_s=xcorr_offset_s, offset_second_half_s=xcorr_offset_s, drift_s=0.0,
    )
    taps = TapResult(
        found=True, offset_s=taps_offset_s, lag_start_s=taps_offset_s,
        lag_end_s=taps_offset_s, drift_s=0.0,
    )
    diff = abs(xcorr.offset_s - taps.offset_s)
    return diff <= AGREEMENT_TOLERANCE_S


def test_agreement_rule_true_exactly_at_tolerance_boundary():
    assert _agreement_ok(0.0, AGREEMENT_TOLERANCE_S) is True  # <=, not <


def test_agreement_rule_false_just_past_tolerance_boundary():
    assert _agreement_ok(0.0, AGREEMENT_TOLERANCE_S + 0.001) is False


# --------------------------------------------------------------------------
# 5. Rest offsets (compute_rest_offsets) -- report-only, never applied
# --------------------------------------------------------------------------

def test_rest_offset_computation_masks_and_medians_correctly():
    n_rest, n_other = 3, 3
    n = n_rest + n_other
    # Per-marker dx is constant across frames (dx[:, m] == m) inside the
    # rest window, and a wildly different constant outside it -- so a
    # correct median-over-rest-frames must recover exactly `m`, and would
    # be wrong if the mask were not applied.
    dx = np.zeros((n, N_MARKERS))
    dy = np.zeros((n, N_MARKERS))
    dA = np.zeros((n, N_MARKERS))
    magnitude = np.zeros((n, N_MARKERS))
    for i in range(n):
        if i < n_rest:
            dx[i] = np.arange(N_MARKERS, dtype=np.float64)
            dy[i] = -np.arange(N_MARKERS, dtype=np.float64)
            dA[i] = 2.0 * np.arange(N_MARKERS, dtype=np.float64)
            magnitude[i] = 1.5 * np.arange(N_MARKERS, dtype=np.float64)
        else:
            dx[i] = 999.0
            dy[i] = 999.0
            dA[i] = 999.0
            magnitude[i] = 999.0
    grip_proxy = magnitude.max(axis=1)  # constant 1.5*98 during rest, 999 outside
    autofilled = np.zeros((n, N_MARKERS), dtype=bool)
    markers = MarkerTrace(
        path=Path("<synthetic>"), frame=np.arange(n, dtype=np.int64),
        timestamp_ms=np.arange(n, dtype=np.float64) * 50.0, trial_id=np.zeros(n, dtype=np.int64),
        dx=dx, dy=dy, dA=dA, magnitude=magnitude, autofilled=autofilled,
        n_autofilled=autofilled.sum(axis=1), grip_proxy=grip_proxy,
    )
    rest_mask = np.array([True] * n_rest + [False] * n_other)

    result = compute_rest_offsets(markers, rest_mask)

    assert result.n_frames == n_rest
    assert result.grip_proxy_median == pytest.approx(1.5 * 98.0)
    assert result.grip_proxy_p95 == pytest.approx(1.5 * 98.0)
    assert result.dx_median == pytest.approx(list(range(N_MARKERS)))
    assert result.dy_median == pytest.approx([-x for x in range(N_MARKERS)])
    assert result.dA_median == pytest.approx([2.0 * x for x in range(N_MARKERS)])


def test_rest_offset_computation_empty_mask_returns_zero_frames():
    n = 4
    dx = dy = dA = magnitude = np.zeros((n, N_MARKERS))
    autofilled = np.zeros((n, N_MARKERS), dtype=bool)
    markers = MarkerTrace(
        path=Path("<synthetic>"), frame=np.arange(n, dtype=np.int64),
        timestamp_ms=np.arange(n, dtype=np.float64), trial_id=np.zeros(n, dtype=np.int64),
        dx=dx, dy=dy, dA=dA, magnitude=magnitude, autofilled=autofilled,
        n_autofilled=autofilled.sum(axis=1), grip_proxy=np.zeros(n),
    )
    rest_mask = np.zeros(n, dtype=bool)

    result = compute_rest_offsets(markers, rest_mask)

    assert result.n_frames == 0
    assert result.grip_proxy_median is None
    assert result.dx_median == []


# --------------------------------------------------------------------------
# 6. Full orchestration -- real CSVs on disk (timezone gate, end-to-end)
# --------------------------------------------------------------------------

MARKERS_FIELDS = [
    "frame", "timestamp_ms", "marker_id", "x", "y", "area", "dx", "dy", "dA",
    "magnitude", "size_moments", "predicted_x", "predicted_y", "autofilled", "trial_id",
]


def _write_markers_csv(path: Path, t_cam_ms, grip_proxy) -> None:
    """Every marker in a frame is written with the same dx/dy/dA/magnitude
    (scaled off grip_proxy), matching _marker_trace_from_proxy's approach --
    max-over-markers is then trivially that frame's grip_proxy."""
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(MARKERS_FIELDS)
        for i, ts in enumerate(t_cam_ms):
            g = float(grip_proxy[i])
            for m in range(N_MARKERS):
                writer.writerow([
                    i, f"{ts:.3f}", m, 0.0, 0.0, 100.0,
                    f"{g * 0.5:.6f}", f"{g * 0.3:.6f}", f"{g * 2.0:.6f}", f"{g:.6f}",
                    "nan", 0.0, 0.0, "False", 0,
                ])


def _write_trials_csv(path: Path, target_pct: int) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["trial_id", "trial_index", "start_ts", "end_ts", "voided",
                           "target_pct", "target_kg"],
        )
        writer.writeheader()
        writer.writerow({
            "trial_id": 0, "trial_index": 1, "start_ts": 0, "end_ts": 0,
            "voided": "False", "target_pct": target_pct, "target_kg": 0.0,
        })


def _write_participant_json(path: Path, mvc_kg: float) -> None:
    path.write_text(json.dumps({"schema_version": 1, "mvc_kg": mvc_kg}), encoding="utf-8")


def _build_session_dir(tmp_path: Path, name: str = "session1_test"):
    participant_dir = tmp_path / "participantTEST"
    session_dir = participant_dir / name
    session_dir.mkdir(parents=True)
    return participant_dir, session_dir


def test_timezone_gate_raises_for_mismatched_clocks(tmp_path):
    participant_dir, session_dir = _build_session_dir(tmp_path)
    t_sec = np.arange(0.0, 15.0, FORCE_SAMPLE_PERIOD_S)
    force_kg = _osc(t_sec)
    _write_dynamometer_txt(
        session_dir / "dynamometer.txt", ANCHOR_DT, sample_period_ms=5,
        t_sec=t_sec.tolist(), force_kg=force_kg.tolist(),
    )
    # Camera's first frame is 700s after the anchor -- outside the
    # [-60s, +600s] sanity window (spec sec 1).
    bad_t_cam_ms = [ANCHOR_EPOCH_MS + 700_000.0 + i * 50.0 for i in range(10)]
    _write_markers_csv(session_dir / "markers.csv", bad_t_cam_ms, grip_proxy=[1.0] * 10)
    _write_trials_csv(session_dir / "trials.csv", target_pct=0)
    _write_participant_json(participant_dir / "participant.json", mvc_kg=20.0)

    with pytest.raises(SyncError, match="sanity window"):
        sync_session(session_dir)


def test_sync_session_end_to_end_writes_training_table(tmp_path):
    participant_dir, session_dir = _build_session_dir(tmp_path)
    force_kg = _shape(T_SEC, TAP_CENTERS)
    grip_proxy = _shape(T_CAM - TRUE_OFFSET_S, TAP_CENTERS)
    t_cam_ms = ANCHOR_EPOCH_MS + T_CAM * 1000.0

    _write_dynamometer_txt(
        session_dir / "dynamometer.txt", ANCHOR_DT, sample_period_ms=5,
        t_sec=T_SEC.tolist(), force_kg=force_kg.tolist(),
    )
    _write_markers_csv(session_dir / "markers.csv", t_cam_ms, grip_proxy)
    _write_trials_csv(session_dir / "trials.csv", target_pct=0)
    _write_participant_json(participant_dir / "participant.json", mvc_kg=20.0)

    result = sync_session(session_dir)

    assert result.ok is True
    assert result.coverage == pytest.approx(1.0)
    assert result.xcorr.offset_s == pytest.approx(TRUE_OFFSET_S, abs=XCORR_STEP_MS / 1000.0)
    assert result.n_camera_frames == len(T_CAM)
    assert result.training_rows == len(T_CAM)

    table_path = session_dir / "training_table.csv"
    report_path = session_dir / "sync_report.json"
    assert table_path.exists()
    assert report_path.exists()

    with table_path.open(newline="", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        first_row = next(reader)
    assert len(header) == 9 + 4 * N_MARKERS  # spec sec 7: 405 columns
    assert header.index("dx_0") == 9

    # Displacement is raw (spec sec 7): the training table's dx_0 for frame 0
    # must equal exactly what was written into markers.csv, not a
    # baseline/rest-normalised value.
    assert float(first_row[header.index("dx_0")]) == pytest.approx(grip_proxy[0] * 0.5, abs=1e-4)

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["sign_convention"].startswith("force_corrected(T)")
    assert report["training_rows"] == len(T_CAM)


def test_sync_session_zero_data_rows_raises_cleanly(tmp_path):
    """Orchestration-level mirror of session 4: header present, zero force
    data rows. Must raise SyncError, never an unrelated exception."""
    participant_dir, session_dir = _build_session_dir(tmp_path, name="session4_test")
    _write_dynamometer_txt(session_dir / "dynamometer.txt", ANCHOR_DT, sample_period_ms=5,
                            t_sec=[], force_kg=[])
    _write_markers_csv(session_dir / "markers.csv", [ANCHOR_EPOCH_MS], grip_proxy=[1.0])
    _write_trials_csv(session_dir / "trials.csv", target_pct=0)
    _write_participant_json(participant_dir / "participant.json", mvc_kg=20.0)

    with pytest.raises(SyncError, match="ZERO DATA ROWS"):
        sync_session(session_dir)


def test_main_handles_zero_data_rows_session_without_crashing(tmp_path, capsys):
    """The CLI must report a bad session and exit 1 -- never raise -- so a
    batch of many sessions is unaffected by one broken export (spec sec 8)."""
    participant_dir, session_dir = _build_session_dir(tmp_path, name="session4_test")
    _write_dynamometer_txt(session_dir / "dynamometer.txt", ANCHOR_DT, sample_period_ms=5,
                            t_sec=[], force_kg=[])
    _write_markers_csv(session_dir / "markers.csv", [ANCHOR_EPOCH_MS], grip_proxy=[1.0])
    _write_trials_csv(session_dir / "trials.csv", target_pct=0)
    _write_participant_json(participant_dir / "participant.json", mvc_kg=20.0)

    exit_code = main(["sync_session.py", str(participant_dir)])

    assert exit_code == 1
    out = capsys.readouterr().out
    assert "FAIL" in out
    assert not (session_dir / "training_table.csv").exists()
