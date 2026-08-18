"""Put a session's Biopac force trace and camera marker trace on one timeline.

Two recordings of the same event, made by two independent programs that only
share a laptop clock:

  Biopac  -- dynamometer.txt, grip force in kg, 200 samples/sec
  GripVT  -- markers.csv, 99 marker positions per camera frame, ~16-27 fps

Their timestamps get you close (both machines share one OS clock) but not
exact -- neither clock sees the fixed hardware/software latency between "the
event happened" and "a timestamp got written". This module measures that
residual offset two independent ways and writes one wide training table with
a force label on every camera frame.

Method background: docs/design/sync/biopac-camera-time-sync.md
Implementation spec: docs/logs/20260806_133000_session_sync_plan.md

Sign convention (spec sec 4, pinned empirically against participant001
session 1 -- see the module-level constant SIGN_CONVENTION_NOTE below):

    force_corrected(T) = interp_force(T - offset)

for a camera frame at time T. A **positive** offset means the camera event
is measured (by the OS clock) as happening *after* the force event that
actually caused it -- i.e. GripVT's software+mechanical pipeline (frame
grab, USB transfer, tissue deformation reaching the sleeve) lags the
Biopac's force reading. To label a camera frame you therefore look
`offset` seconds *backward* into the force trace's own timeline.

Two independent offset estimates are computed and both are reported:

  xcorr -- Pearson correlation between the per-frame grip proxy (max
           baseline-relative marker displacement across non-autofilled
           markers, each referred to its own resting median -- MAX, not
           median, because a tap deforms only a subset of the 99
           markers, which a median discards) and the
           force trace, swept over a +-8.0s grid in 5ms steps. Primary:
           uses every frame (median 3,850) per estimate.
  taps  -- the operator's triple-tap bookend at the start and end of the
           recording, detected independently in both signals as a
           contiguous-above-threshold run inside a 25s search window.
           Secondary / cross-check: ~6 landmark points.

Neither estimate is silently preferred over the truth: both are always
computed and reported. `--corr-method` only selects which one feeds the
`*_corr` columns in the training table (default: xcorr, since it uses far
more data and needs no operator taps to have landed cleanly).

Usage:
    python tools/sync_session.py <session_dir | participant_dir> [--corr-method xcorr|taps]

Writes, per session, into the session directory:
    training_table.csv  -- one row per camera frame, 405 columns
    sync_report.json    -- anchor, coverage, both offsets, drift checks,
                            agreement verdict, rest-period offsets

Exit code is 0 only if every session synced successfully. A session that
cannot be synced (e.g. a Biopac export with zero data rows) is reported and
skipped -- it does not stop the rest of the batch, and it never writes a
partial/corrupt output pair.

Stdlib for parsing/IO (csv, json, pathlib, datetime, dataclasses, argparse,
statistics), matching tools/check_export.py. numpy for the numerical
search -- a pure-Python cross-correlation over ~3,200 offsets x ~3,850 frames
x 69 sessions would be needlessly slow, and numpy is already a hard
dependency (requirements.txt).
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np

if not __package__:
    # Allows `python tools/sync_session.py ...` (check_export.py's own
    # invocation style) in addition to `python -m tools.sync_session` and
    # `from tools import sync_session` -- without this, running the file
    # directly puts only tools/ on sys.path, and `tools.check_export`
    # cannot be found from inside its own package directory.
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.check_export import EXPORT_NAME, check_export, resolve_targets

N_MARKERS = 99
MARKERS_CSV_NAME = "markers.csv"
TRIALS_CSV_NAME = "trials.csv"
PARTICIPANT_JSON_NAME = "participant.json"
TRAINING_TABLE_NAME = "training_table.csv"
SYNC_REPORT_NAME = "sync_report.json"

# Cross-correlation search grid (spec sec 4a): the sweep half-width, at the
# Biopac's own 5ms sample period -- searching finer than the source data's
# resolution would just interpolate noise.
#
# Widened 2000 -> 8000 on 2026-08-14. The original +-2.0s was set from
# participants 001-004, whose offsets never exceed 0.905s. Participants
# 005-007 produced four sessions whose true offset lies outside +-2.0s;
# each one silently pinned to exactly the sweep edge, which reads as a
# plausible number rather than as a failure. Widening recovers all four and
# moves no previously-good session. Cross-checks, per session:
#   p005 s6  -1.715 -> +7.300 (r 0.20 -> 0.63); its independent tap
#            estimate is +7.295, so two methods agree to 5ms
#   p005 s8  +2.000 -> +2.265 (r 0.64 -> 0.76)
#   p006 s7  +2.000 -> +2.150 (r 0.85 -> 0.92)
#   p007 s3  +2.000 -> +2.695 (r 0.46 -> 0.90)
# In all four, the two drift halves agree to within 0.04s, so the new peak
# is a real alignment and not an alias of the trial cadence.
#
# 8000 stays far inside TZ_GATE_AFTER_MS, so it cannot mask a timezone fault.
XCORR_RANGE_MS = 8000
XCORR_STEP_MS = 5
XCORR_MIN_OVERLAP_FRAMES = 10

# Timezone sanity gate (spec sec 1). The Biopac export's "Recording on:"
# line is naive local wall-clock with no timezone field at all, so a
# machine/export-tool timezone mismatch would silently misalign every
# session rather than raise. This turns that into a loud, named failure.
# Verified correct on 2026-08-06: anchor 09:37:01.082 vs camera first frame
# 09:37:06.708, a 5.6s gap, matching "camera started 3-9s after the Biopac".
TZ_GATE_BEFORE_MS = 60_000
TZ_GATE_AFTER_MS = 600_000

# Triple-tap detection (spec sec 4b).
TAP_SEARCH_WINDOW_S = 25.0
TAP_THRESHOLD_FRAC = 0.35
#
# TAP_GAP_MIN_S lowered 0.35 -> 0.15 on 2026-08-14. Guillaume reported that
# some participants tap three VERY quick pulses. _find_triple rejects a triple
# unless BOTH gaps fall inside [TAP_GAP_MIN_S, TAP_GAP_MAX_S], so a fast
# tapper was being discarded rather than measured.
#
# Measured across all 69 sessions: the smallest genuine gap between the first
# three force peaks is 0.250s, and the 10th percentile is 0.395s. The old
# 0.35s floor therefore sat on top of the fast tail of a real distribution.
#
# Detection rate by floor (sessions with taps found in both signals):
#   floor   0.35   0.25   0.15   0.10   0.05
#   p002     8/10   9/10   9/10   9/10   9/10
#   p006     4/10   9/10   9/10   9/10   9/10
#   p007     5/10   9/10  10/10  10/10  10/10
# Nothing further is recovered below 0.15, which is the evidence that no
# genuine triple sits under it -- so 0.15 admits the real distribution with
# margin without inventing detections out of a single tap's ringing.
#
# 11 sessions gained a tap reading. 7 of the 11 agree with the wholly
# independent xcorr offset to within AGREEMENT_TOLERANCE_S, including two
# large-offset sessions (p006 s3: +1.690 xcorr vs +1.696 taps; p007 s3:
# +2.695 vs +2.671). The other 4 are false positives and are already caught
# by the agreement gate below. That trade is worth taking, because the taps
# are only ever a cross-check -- corr_method defaults to xcorr, so a bad tap
# reading never reaches the training table.
TAP_GAP_MIN_S = 0.15
TAP_GAP_MAX_S = 1.6

# xcorr vs taps agreement gate (spec sec 4c) -- ~2.4 camera frames at 16fps.
AGREEMENT_TOLERANCE_S = 0.15


class SyncError(Exception):
    """A session cannot be synced. Message is meant to be read by a human."""


# --------------------------------------------------------------------------
# Data model
# --------------------------------------------------------------------------

@dataclass
class ForceTrace:
    path: Path
    anchor_epoch_ms: float
    sample_period_s: float
    force_kg: np.ndarray
    time_ms: np.ndarray  # anchor_epoch_ms + t_sec*1000, ascending


@dataclass
class MarkerTrace:
    path: Path
    frame: np.ndarray
    timestamp_ms: np.ndarray
    trial_id: np.ndarray
    dx: np.ndarray  # (n_frames, N_MARKERS)
    dy: np.ndarray
    dA: np.ndarray
    magnitude: np.ndarray
    autofilled: np.ndarray  # (n_frames, N_MARKERS) bool
    n_autofilled: np.ndarray
    grip_proxy: np.ndarray  # max(magnitude) over non-autofilled markers
    # max_reref (docs/logs/20260818_174754_sync_proxy_sweep_plan.md sec 10):
    # as grip_proxy, but each marker's magnitude is first referenced to that
    # marker's own resting median before the per-frame max. Sharper xcorr
    # peak on all 69 sessions (median halfwidth 125ms vs 130ms for plain
    # max) and removes the effect of a drifting baseline (participant003
    # sessions 6/7/10). Set by sync_session() after rest frames are known --
    # None until then (e.g. still None inside read_markers(), and stays
    # None for a session with no target_pct==0 rest trial). compute_xcorr()
    # uses this field when set, falling back to grip_proxy otherwise.
    # compute_taps() always uses grip_proxy -- never this field -- because
    # TAP_GAP_MIN_S was tuned against plain-max peaks on 2026-08-14 and the
    # sweep never measured the tap detector against this signal.
    grip_proxy_reref: np.ndarray | None = None


@dataclass
class XcorrResult:
    offset_s: float
    r_at_best: float
    r_at_zero: float
    offset_first_half_s: float | None
    offset_second_half_s: float | None
    drift_s: float | None
    range_ms: int = XCORR_RANGE_MS
    at_sweep_edge: bool = False  # best offset sits on the edge -> range too narrow


@dataclass
class TapResult:
    found: bool
    offset_s: float | None
    lag_start_s: float | None
    lag_end_s: float | None
    drift_s: float | None
    camera_start_taps_s: list[float] | None = None
    force_start_taps_s: list[float] | None = None
    camera_end_taps_s: list[float] | None = None
    force_end_taps_s: list[float] | None = None
    note: str | None = None


@dataclass
class RestOffsets:
    """Grip-proxy and per-marker displacement over 0%-MVC trial windows.

    Report-only (spec sec 6) -- never subtracted from the written data.
    """
    n_frames: int
    grip_proxy_median: float | None = None
    grip_proxy_p95: float | None = None
    dx_median: list[float] = field(default_factory=list)
    dy_median: list[float] = field(default_factory=list)
    dA_median: list[float] = field(default_factory=list)
    magnitude_median: list[float] = field(default_factory=list)


@dataclass
class SessionSyncResult:
    session_dir: Path
    ok: bool
    error: str | None = None
    n_camera_frames: int | None = None
    n_force_samples: int | None = None
    coverage: float | None = None
    xcorr: XcorrResult | None = None
    taps: TapResult | None = None
    agreement_ok: bool | None = None
    corr_method: str = "xcorr"
    offset_used_s: float | None = None
    # True only when grip_proxy_reref could not be computed (no target_pct
    # == 0 rest trial in trials.csv) and compute_xcorr() fell back to the
    # plain max grip_proxy for this session.
    grip_proxy_reref_fallback: bool = False
    mvc_kg: float | None = None
    rest_force_kg: dict | None = None
    rest_offsets: RestOffsets | None = None
    training_table_path: Path | None = None
    training_rows: int | None = None


# --------------------------------------------------------------------------
# 1. Force parsing
# --------------------------------------------------------------------------

def _parse_force_samples(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Data rows are `t_sec,force_kg,` -- same two-float test check_export.py
    uses to tell a data row from header/metadata, just applied to every line
    instead of only the head."""
    t_list: list[float] = []
    force_list: list[float] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            parts = [p for p in line.split(",") if p.strip()]
            if len(parts) < 2:
                continue
            try:
                t_sec = float(parts[0])
                force_kg = float(parts[1])
            except ValueError:
                continue
            t_list.append(t_sec)
            force_list.append(force_kg)
    return np.asarray(t_list, dtype=np.float64), np.asarray(force_list, dtype=np.float64)


def read_force(path: Path) -> ForceTrace:
    """Parse a Biopac dynamometer.txt export into a force trace on the
    shared epoch-ms timeline.

    Validates with check_export.check_export() first (spec sec 1) -- this
    is what makes session 4 (header written, zero data rows) fail here,
    cleanly, instead of downstream as a confusing shape mismatch.
    """
    rep = check_export(path)
    if not rep.ok:
        raise SyncError(
            f"{path}: dynamometer export failed validation -- " + "; ".join(rep.errors)
        )
    if rep.recording_on is None or rep.sample_period_s is None:
        # Should already be covered by rep.ok above; kept as a named guard
        # so a future check_export.py change can't silently reintroduce a
        # crash here.
        raise SyncError(f"{path}: export passed validation but is missing anchor/sample period")

    try:
        anchor_dt = datetime.strptime(rep.recording_on, "%Y-%m-%d %H:%M:%S.%f")
    except ValueError as exc:
        raise SyncError(f"{path}: could not parse 'Recording on: {rep.recording_on}'") from exc
    anchor_epoch_ms = anchor_dt.timestamp() * 1000.0

    t_sec, force_kg = _parse_force_samples(path)
    if t_sec.size == 0:
        raise SyncError(f"{path}: no numeric data rows parsed")

    time_ms = anchor_epoch_ms + t_sec * 1000.0
    return ForceTrace(
        path=path,
        anchor_epoch_ms=anchor_epoch_ms,
        sample_period_s=rep.sample_period_s,
        force_kg=force_kg,
        time_ms=time_ms,
    )


# --------------------------------------------------------------------------
# 2. Marker parsing
# --------------------------------------------------------------------------

def read_markers(path: Path) -> MarkerTrace:
    """Parse markers.csv into per-frame arrays of 99 markers each.

    Groups rows by `frame` (contiguous blocks of N_MARKERS rows in the
    file) and scatters each row into its `marker_id` slot rather than
    trusting row order within the block, so a re-ordered or short frame
    raises instead of silently misaligning marker_id with data.

    `size_moments` is read by no one -- often the literal string "nan" --
    and is simply never touched. `autofilled` is the literal string
    "True"/"False", not a bool; compared as a string here, once.
    """
    frames: list[int] = []
    timestamps: list[float] = []
    trial_ids: list[int] = []
    dx_rows: list[list[float]] = []
    dy_rows: list[list[float]] = []
    dA_rows: list[list[float]] = []
    mag_rows: list[list[float]] = []
    autof_rows: list[list[bool]] = []

    def _flush(frame_id: str, rows: list[dict]) -> None:
        if not rows:
            return
        dx = [None] * N_MARKERS
        dy = [None] * N_MARKERS
        dA = [None] * N_MARKERS
        mag = [None] * N_MARKERS
        af = [None] * N_MARKERS
        for row in rows:
            mid = int(row["marker_id"])
            if not (0 <= mid < N_MARKERS):
                raise SyncError(f"{path}: frame {frame_id}: marker_id {mid} out of 0..{N_MARKERS - 1}")
            dx[mid] = float(row["dx"])
            dy[mid] = float(row["dy"])
            dA[mid] = float(row["dA"])
            mag[mid] = float(row["magnitude"])
            af[mid] = row["autofilled"] == "True"
        if any(v is None for v in dx):
            missing = [i for i, v in enumerate(dx) if v is None]
            raise SyncError(
                f"{path}: frame {frame_id}: missing marker_id(s) {missing[:5]}"
                + ("..." if len(missing) > 5 else "")
            )
        frames.append(int(frame_id))
        timestamps.append(float(rows[0]["timestamp_ms"]))
        trial_ids.append(int(rows[0]["trial_id"]))
        dx_rows.append(dx)
        dy_rows.append(dy)
        dA_rows.append(dA)
        mag_rows.append(mag)
        autof_rows.append(af)

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        current_frame: str | None = None
        current_rows: list[dict] = []
        for row in reader:
            if row["frame"] != current_frame:
                _flush(current_frame, current_rows)
                current_frame = row["frame"]
                current_rows = []
            current_rows.append(row)
        _flush(current_frame, current_rows)

    if not frames:
        raise SyncError(f"{path}: no frames parsed")

    dx = np.asarray(dx_rows, dtype=np.float64)
    dy = np.asarray(dy_rows, dtype=np.float64)
    dA = np.asarray(dA_rows, dtype=np.float64)
    mag = np.asarray(mag_rows, dtype=np.float64)
    autof = np.asarray(autof_rows, dtype=bool)
    n_autofilled = autof.sum(axis=1)

    # grip proxy = MAX magnitude over non-autofilled markers (spec sec 2,
    # methods doc sec "revised" point 2: median goes flat during taps
    # because a tap deforms only a subset of the 99 markers).
    masked = np.where(autof, -np.inf, mag)
    grip_proxy = masked.max(axis=1)
    all_autofilled = autof.all(axis=1)
    if np.any(all_autofilled):
        # No non-autofilled marker this frame (should not happen on real
        # data -- guarded rather than left to silently return -inf).
        grip_proxy = np.where(all_autofilled, mag.max(axis=1), grip_proxy)

    return MarkerTrace(
        path=path,
        frame=np.asarray(frames, dtype=np.int64),
        timestamp_ms=np.asarray(timestamps, dtype=np.float64),
        trial_id=np.asarray(trial_ids, dtype=np.int64),
        dx=dx, dy=dy, dA=dA, magnitude=mag,
        autofilled=autof, n_autofilled=n_autofilled, grip_proxy=grip_proxy,
    )


def read_trials(path: Path) -> dict[int, int]:
    """trial_id -> target_pct, skipping voided rows (spec: excluded everywhere).

    A frame whose trial_id maps to a voided trial, or to no trial at all,
    simply has no entry here -- callers treat a missing key as "no target".
    """
    target_pct_by_trial: dict[int, int] = {}
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if (row.get("voided") or "").strip() == "True":
                continue
            try:
                trial_id = int(row["trial_id"])
                target_pct = int(float(row["target_pct"]))
            except (KeyError, ValueError, TypeError):
                continue
            target_pct_by_trial[trial_id] = target_pct
    return target_pct_by_trial


def read_mvc_kg(session_dir: Path) -> float:
    """mvc_kg comes from participant.json in the *participant* directory --
    the parent of the session directory, not the session directory itself."""
    participant_path = session_dir.parent / PARTICIPANT_JSON_NAME
    if not participant_path.exists():
        raise SyncError(f"{participant_path}: not found (needed for mvc_kg)")
    with participant_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    try:
        return float(payload["mvc_kg"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SyncError(f"{participant_path}: no usable 'mvc_kg' field") from exc


# --------------------------------------------------------------------------
# 3. Coverage
# --------------------------------------------------------------------------

def compute_coverage(markers: MarkerTrace, force: ForceTrace) -> float:
    """Fraction of camera frames whose timestamp lies inside the force
    trace's own time span."""
    lo, hi = force.time_ms[0], force.time_ms[-1]
    inside = (markers.timestamp_ms >= lo) & (markers.timestamp_ms <= hi)
    return float(inside.mean())


# --------------------------------------------------------------------------
# 4a. Cross-correlation alignment
# --------------------------------------------------------------------------

def _pearson_r(a: np.ndarray, b: np.ndarray) -> float:
    if a.size < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _xcorr_search(
    camera_ts_ms: np.ndarray, grip_proxy: np.ndarray, force: ForceTrace,
) -> tuple[np.ndarray, np.ndarray]:
    """Sweep offsets in [-XCORR_RANGE_MS, +XCORR_RANGE_MS] at XCORR_STEP_MS
    resolution. Returns (offsets_ms, r_values) -- r_values[i] is nan where
    fewer than XCORR_MIN_OVERLAP_FRAMES camera frames land inside the
    force trace's span once shifted by that candidate offset.

    Convention: force_corrected(T) = interp_force(T - offset), so the
    query time for offset_ms is camera_ts_ms - offset_ms.
    """
    offsets_ms = np.arange(-XCORR_RANGE_MS, XCORR_RANGE_MS + 1, XCORR_STEP_MS)
    r_values = np.full(offsets_ms.shape, np.nan, dtype=np.float64)
    lo, hi = force.time_ms[0], force.time_ms[-1]
    for i, off_ms in enumerate(offsets_ms):
        t_query = camera_ts_ms - off_ms
        valid = (t_query >= lo) & (t_query <= hi)
        if valid.sum() < XCORR_MIN_OVERLAP_FRAMES:
            continue
        force_interp = np.interp(t_query[valid], force.time_ms, force.force_kg)
        r_values[i] = _pearson_r(force_interp, grip_proxy[valid])
    return offsets_ms, r_values


def _best_offset(offsets_ms: np.ndarray, r_values: np.ndarray) -> tuple[float, float] | None:
    if np.all(np.isnan(r_values)):
        return None
    best_idx = int(np.nanargmax(r_values))
    return float(offsets_ms[best_idx]) / 1000.0, float(r_values[best_idx])


def compute_xcorr(markers: MarkerTrace, force: ForceTrace) -> XcorrResult:
    # max_reref feeds this search when sync_session() has set it (plan
    # docs/logs/20260818_174754_sync_proxy_sweep_plan.md sec 10); a session
    # with no rest trial leaves grip_proxy_reref None and this falls back to
    # the plain max grip_proxy, same as before this change.
    proxy = markers.grip_proxy_reref if markers.grip_proxy_reref is not None else markers.grip_proxy
    offsets_ms, r_values = _xcorr_search(markers.timestamp_ms, proxy, force)
    best = _best_offset(offsets_ms, r_values)
    if best is None:
        raise SyncError(
            "cross-correlation search found no offset with enough overlapping frames "
            f"(need >= {XCORR_MIN_OVERLAP_FRAMES})"
        )
    offset_s, r_at_best = best
    zero_idx = int(np.argmin(np.abs(offsets_ms)))
    r_at_zero = float(r_values[zero_idx])

    # Drift check (spec sec 4a): independently re-run on each half of the
    # camera frames; report both and their difference.
    half = markers.timestamp_ms.size // 2
    off_first = off_second = None
    if half >= XCORR_MIN_OVERLAP_FRAMES:
        first = _best_offset(*_xcorr_search(markers.timestamp_ms[:half], proxy[:half], force))
        second = _best_offset(*_xcorr_search(markers.timestamp_ms[half:], proxy[half:], force))
        off_first = first[0] if first else None
        off_second = second[0] if second else None
    drift_s = (off_second - off_first) if (off_first is not None and off_second is not None) else None

    # A best offset that lands on the first or last grid point is not a
    # measurement -- the true peak is outside the window and the search was
    # truncated. Report it, so a too-narrow range can never pass silently again.
    at_edge = abs(abs(offset_s) - XCORR_RANGE_MS / 1000.0) < 1e-9

    return XcorrResult(
        offset_s=offset_s, r_at_best=r_at_best, r_at_zero=r_at_zero,
        offset_first_half_s=off_first, offset_second_half_s=off_second, drift_s=drift_s,
        range_ms=XCORR_RANGE_MS, at_sweep_edge=at_edge,
    )


# --------------------------------------------------------------------------
# 4b. Triple-tap alignment
# --------------------------------------------------------------------------

def _peaks_in_window(
    times_ms: np.ndarray, values: np.ndarray, window_start_ms: float, window_end_ms: float,
) -> list[tuple[float, float]]:
    """Contiguous above-threshold runs inside [window_start, window_end],
    each contributing its own max as one peak (spec sec 4b), time-ordered.

    threshold = base + 0.35*(peak-base); base = median in window,
    peak = max in window.
    """
    mask = (times_ms >= window_start_ms) & (times_ms <= window_end_ms)
    idx = np.nonzero(mask)[0]
    if idx.size == 0:
        return []
    t = times_ms[idx]
    v = values[idx]
    base = float(np.median(v))
    peak = float(np.max(v))
    threshold = base + TAP_THRESHOLD_FRAC * (peak - base)
    above = v > threshold

    peaks: list[tuple[float, float]] = []
    run_start: int | None = None
    for i in range(len(v) + 1):
        is_above = i < len(v) and above[i]
        if is_above and run_start is None:
            run_start = i
        elif not is_above and run_start is not None:
            run = slice(run_start, i)
            j = run_start + int(np.argmax(v[run]))
            peaks.append((float(t[j]), float(v[j])))
            run_start = None
    return peaks


def _find_triple(
    peaks: list[tuple[float, float]], *, earliest: bool,
) -> list[tuple[float, float]] | None:
    """3 consecutive peaks with both gaps in [TAP_GAP_MIN_S, TAP_GAP_MAX_S].

    `earliest=True` returns the first qualifying triple (used in the start
    window); `earliest=False` returns the last (used in the end window).
    """
    candidates = []
    for i in range(len(peaks) - 2):
        t0, t1, t2 = peaks[i][0], peaks[i + 1][0], peaks[i + 2][0]
        gap1_s = (t1 - t0) / 1000.0
        gap2_s = (t2 - t1) / 1000.0
        if TAP_GAP_MIN_S <= gap1_s <= TAP_GAP_MAX_S and TAP_GAP_MIN_S <= gap2_s <= TAP_GAP_MAX_S:
            candidates.append(peaks[i:i + 3])
    if not candidates:
        return None
    return candidates[0] if earliest else candidates[-1]


def compute_taps(markers: MarkerTrace, force: ForceTrace) -> TapResult:
    """Triple-tap bookend alignment (spec sec 4b), run independently at the
    start and end of the overlapping span. A missing triple in either
    signal at either end is recorded as `found=False` (or a partial
    result) and flagged -- never raised; the session is still usable via
    cross-correlation.
    """
    overlap_start = max(markers.timestamp_ms[0], force.time_ms[0])
    overlap_end = min(markers.timestamp_ms[-1], force.time_ms[-1])
    window_ms = TAP_SEARCH_WINDOW_S * 1000.0

    start_lo, start_hi = overlap_start, overlap_start + window_ms
    end_lo, end_hi = overlap_end - window_ms, overlap_end

    camera_start_peaks = _peaks_in_window(markers.timestamp_ms, markers.grip_proxy, start_lo, start_hi)
    force_start_peaks = _peaks_in_window(force.time_ms, force.force_kg, start_lo, start_hi)
    camera_end_peaks = _peaks_in_window(markers.timestamp_ms, markers.grip_proxy, end_lo, end_hi)
    force_end_peaks = _peaks_in_window(force.time_ms, force.force_kg, end_lo, end_hi)

    camera_start_triple = _find_triple(camera_start_peaks, earliest=True)
    force_start_triple = _find_triple(force_start_peaks, earliest=True)
    camera_end_triple = _find_triple(camera_end_peaks, earliest=False)
    force_end_triple = _find_triple(force_end_peaks, earliest=False)

    lag_start = lag_end = None
    if camera_start_triple and force_start_triple:
        lag_start = sum(c[0] - f[0] for c, f in zip(camera_start_triple, force_start_triple)) / 3.0 / 1000.0
    if camera_end_triple and force_end_triple:
        lag_end = sum(c[0] - f[0] for c, f in zip(camera_end_triple, force_end_triple)) / 3.0 / 1000.0

    def _to_relative_s(triple: list[tuple[float, float]] | None) -> list[float] | None:
        """Tap times reported relative to the Biopac anchor (t_sec on the
        force trace's own timeline) -- human-checkable against
        dynamometer.txt directly."""
        if triple is None:
            return None
        return [(t - force.anchor_epoch_ms) / 1000.0 for t, _v in triple]

    if lag_start is None and lag_end is None:
        return TapResult(
            found=False, offset_s=None, lag_start_s=None, lag_end_s=None, drift_s=None,
            note="no triple-tap burst found in both signals at either end",
        )

    if lag_start is not None and lag_end is not None:
        offset_s = (lag_start + lag_end) / 2.0
        drift_s = lag_end - lag_start
        note = None
    else:
        offset_s = lag_start if lag_start is not None else lag_end
        drift_s = None
        note = "only one end's triple-tap was found in both signals; no drift check possible"

    return TapResult(
        found=True, offset_s=offset_s, lag_start_s=lag_start, lag_end_s=lag_end, drift_s=drift_s,
        camera_start_taps_s=_to_relative_s(camera_start_triple),
        force_start_taps_s=_to_relative_s(force_start_triple),
        camera_end_taps_s=_to_relative_s(camera_end_triple),
        force_end_taps_s=_to_relative_s(force_end_triple),
        note=note,
    )


# --------------------------------------------------------------------------
# 5/6. Labels and rest offsets
# --------------------------------------------------------------------------

def _pct_mvc(force_kg: np.ndarray, mvc_kg: float) -> np.ndarray:
    with np.errstate(invalid="ignore"):
        pct = 100.0 * force_kg / mvc_kg
        return np.where(np.isnan(pct), np.nan, np.maximum(0.0, pct))


def compute_marker_rest_median(
    magnitude: np.ndarray, autofilled: np.ndarray, rest_mask: np.ndarray,
) -> tuple[np.ndarray, int] | None:
    """Per-marker median magnitude over the session's 0%-MVC rest frames.

    Camera-only by construction: `rest_mask` must come from trial_id joined
    to trials.csv's *scheduled* target_pct (this module's own sync_session(),
    at the `target_pct_by_trial.get(int(tid)) == 0` line), never from the
    force trace or from any xcorr/tap offset -- otherwise grip_proxy_reref
    would depend on the very offset it is meant to help find.

    Returns None if there are no rest frames at all (no target_pct==0 trial
    in trials.csv). A marker that is autofilled on every rest frame has no
    real rest magnitude to reference against; its median is left at 0.0, so
    grip_proxy_reref falls back to that marker's raw (unreferenced)
    magnitude rather than inventing a value -- the second element of the
    return is how many markers hit this case, for reporting.

    Ported verbatim from tools/sync_proxy_sweep.py's validated
    max_reref sweep (docs/logs/20260818_174754_sync_proxy_sweep_plan.md sec
    10); tools/sync_proxy_sweep.py now imports this rather than keeping its
    own copy, so the two cannot drift apart.
    """
    if rest_mask.sum() == 0:
        return None
    rest_mag = magnitude[rest_mask]
    rest_af = autofilled[rest_mask]
    masked_rest = np.where(rest_af, np.nan, rest_mag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        marker_median = np.nanmedian(masked_rest, axis=0)
    always_autofilled_at_rest = np.isnan(marker_median)
    marker_median = np.where(always_autofilled_at_rest, 0.0, marker_median)
    return marker_median, int(always_autofilled_at_rest.sum())


def compute_grip_proxy_reref(
    magnitude: np.ndarray, autofilled: np.ndarray, marker_rest_median: np.ndarray,
) -> np.ndarray:
    """max magnitude over non-autofilled markers, after each marker's
    magnitude is first referenced to its own resting median (the
    `max_reref` candidate of docs/logs/20260818_174754_sync_proxy_sweep_plan.md
    sec 2, ported verbatim from tools/sync_proxy_sweep.py's `_proxy_max_reref`,
    the validated implementation that won the pre-registered sweep)."""
    reref = magnitude - marker_rest_median[np.newaxis, :]
    masked = np.where(autofilled, -np.inf, reref)
    proxy = masked.max(axis=1)
    all_autofilled = autofilled.all(axis=1)
    if np.any(all_autofilled):
        proxy = np.where(all_autofilled, reref.max(axis=1), proxy)
    return proxy


def compute_rest_offsets(markers: MarkerTrace, rest_mask: np.ndarray) -> RestOffsets:
    n = int(rest_mask.sum())
    if n == 0:
        return RestOffsets(n_frames=0)
    proxy = markers.grip_proxy[rest_mask]
    return RestOffsets(
        n_frames=n,
        grip_proxy_median=float(np.median(proxy)),
        grip_proxy_p95=float(np.percentile(proxy, 95)),
        dx_median=np.median(markers.dx[rest_mask], axis=0).tolist(),
        dy_median=np.median(markers.dy[rest_mask], axis=0).tolist(),
        dA_median=np.median(markers.dA[rest_mask], axis=0).tolist(),
        magnitude_median=np.median(markers.magnitude[rest_mask], axis=0).tolist(),
    )


# --------------------------------------------------------------------------
# 7. Training table
# --------------------------------------------------------------------------

def _fmt(v: float) -> float | str:
    return "" if (v is None or (isinstance(v, float) and math.isnan(v))) else v


def write_training_table(
    path: Path,
    markers: MarkerTrace,
    target_pct_col: list[int | None],
    force_kg_anchor: np.ndarray,
    pct_mvc_anchor: np.ndarray,
    force_kg_corr: np.ndarray,
    pct_mvc_corr: np.ndarray,
) -> int:
    """One row per camera frame, 405 columns (spec sec 7). Displacement is
    written raw, never baseline-normalised beyond what markers.csv already
    is (dx/dy/dA are already baseline-relative per CLAUDE.md)."""
    fieldnames = (
        ["frame", "timestamp_ms", "trial_id", "target_pct", "n_autofilled",
         "force_kg_anchor", "pct_mvc_anchor", "force_kg_corr", "pct_mvc_corr"]
        + [f"dx_{i}" for i in range(N_MARKERS)]
        + [f"dy_{i}" for i in range(N_MARKERS)]
        + [f"dA_{i}" for i in range(N_MARKERS)]
        + [f"mag_{i}" for i in range(N_MARKERS)]
    )
    n = int(markers.frame.size)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(fieldnames)
        for i in range(n):
            row = [
                int(markers.frame[i]),
                float(markers.timestamp_ms[i]),
                int(markers.trial_id[i]),
                "" if target_pct_col[i] is None else target_pct_col[i],
                int(markers.n_autofilled[i]),
                _fmt(float(force_kg_anchor[i])),
                _fmt(float(pct_mvc_anchor[i])),
                _fmt(float(force_kg_corr[i])),
                _fmt(float(pct_mvc_corr[i])),
            ]
            row.extend(markers.dx[i].tolist())
            row.extend(markers.dy[i].tolist())
            row.extend(markers.dA[i].tolist())
            row.extend(markers.magnitude[i].tolist())
            writer.writerow(row)
    return n


# --------------------------------------------------------------------------
# Orchestration
# --------------------------------------------------------------------------

def sync_session(session_dir: Path, corr_method: str = "xcorr") -> SessionSyncResult:
    """Sync one sessionN_<ts> directory. Raises SyncError on anything that
    makes the session unsyncable (bad export, malformed CSVs, timezone
    mismatch) -- callers decide what "cleanly reported, batch continues"
    looks like."""
    dynamometer_path = session_dir / EXPORT_NAME
    markers_path = session_dir / MARKERS_CSV_NAME
    trials_path = session_dir / TRIALS_CSV_NAME

    if not markers_path.exists():
        raise SyncError(f"{markers_path}: not found")
    if not trials_path.exists():
        raise SyncError(f"{trials_path}: not found")

    force = read_force(dynamometer_path)
    markers = read_markers(markers_path)

    # Timezone sanity gate (spec sec 1) -- must run before anything else
    # that trusts the anchor lines the two clocks up.
    camera_first_ms = float(markers.timestamp_ms[0])
    lo = force.anchor_epoch_ms - TZ_GATE_BEFORE_MS
    hi = force.anchor_epoch_ms + TZ_GATE_AFTER_MS
    if not (lo <= camera_first_ms <= hi):
        gap_s = (camera_first_ms - force.anchor_epoch_ms) / 1000.0
        raise SyncError(
            f"{session_dir}: camera's first frame is {gap_s:+.1f}s from the Biopac anchor -- "
            f"outside the [-{TZ_GATE_BEFORE_MS / 1000:.0f}, +{TZ_GATE_AFTER_MS / 1000:.0f}]s "
            "sanity window. Likely cause: a timezone mismatch between the Biopac export and "
            "the camera clock (the export's 'Recording on:' line has no timezone field)."
        )

    coverage = compute_coverage(markers, force)

    # grip_proxy_reref (docs/logs/20260818_174754_sync_proxy_sweep_plan.md sec
    # 10): each marker's magnitude re-referenced to its own resting median
    # before the per-frame max, feeding compute_xcorr only -- compute_taps
    # keeps the plain grip_proxy (see MarkerTrace.grip_proxy_reref). The rest
    # mask is read from trials.csv here, once, and reused below for the
    # training-table target_pct column and for compute_rest_offsets, so it is
    # never re-derived from the force trace or from either offset estimate.
    target_pct_by_trial = read_trials(trials_path)
    rest_mask = np.asarray(
        [target_pct_by_trial.get(int(tid)) == 0 for tid in markers.trial_id]
    )
    rest_median_result = compute_marker_rest_median(markers.magnitude, markers.autofilled, rest_mask)
    grip_proxy_reref_fallback = rest_median_result is None
    if rest_median_result is None:
        print(
            f"WARNING: {session_dir.name}: no target_pct == 0 rest trial found in "
            "trials.csv -- grip_proxy_reref unavailable, xcorr falls back to the "
            "plain max grip_proxy for this session.",
            file=sys.stderr,
        )
    else:
        marker_rest_median, _n_always_af_at_rest = rest_median_result
        markers.grip_proxy_reref = compute_grip_proxy_reref(
            markers.magnitude, markers.autofilled, marker_rest_median
        )

    xcorr = compute_xcorr(markers, force)
    taps = compute_taps(markers, force)

    if taps.offset_s is not None:
        diff = abs(xcorr.offset_s - taps.offset_s)
        agreement_ok = diff <= AGREEMENT_TOLERANCE_S
        if not agreement_ok:
            print(
                f"WARNING: {session_dir.name}: xcorr offset ({xcorr.offset_s:+.3f}s) and "
                f"tap offset ({taps.offset_s:+.3f}s) disagree by {diff:.3f}s "
                f"(> {AGREEMENT_TOLERANCE_S}s) -- table still written.",
                file=sys.stderr,
            )
    else:
        agreement_ok = None

    if corr_method == "taps":
        if taps.offset_s is not None:
            offset_used = taps.offset_s
        else:
            print(
                f"WARNING: {session_dir.name}: --corr-method taps requested but no triple-tap "
                "was found; falling back to the cross-correlation offset.",
                file=sys.stderr,
            )
            offset_used = xcorr.offset_s
    else:
        offset_used = xcorr.offset_s

    mvc_kg = read_mvc_kg(session_dir)
    target_pct_col = [target_pct_by_trial.get(int(tid)) for tid in markers.trial_id]

    force_kg_anchor = np.interp(
        markers.timestamp_ms, force.time_ms, force.force_kg, left=np.nan, right=np.nan
    )
    force_kg_corr = np.interp(
        markers.timestamp_ms - offset_used * 1000.0, force.time_ms, force.force_kg,
        left=np.nan, right=np.nan,
    )
    pct_mvc_anchor = _pct_mvc(force_kg_anchor, mvc_kg)
    pct_mvc_corr = _pct_mvc(force_kg_corr, mvc_kg)

    rest_offsets = compute_rest_offsets(markers, rest_mask)

    rest_force_kg = None
    if rest_mask.any():
        rest_corr = force_kg_corr[rest_mask]
        rest_corr = rest_corr[~np.isnan(rest_corr)]
        if rest_corr.size:
            rest_force_kg = {
                "median": float(np.median(rest_corr)),
                "p95": float(np.percentile(rest_corr, 95)),
                "n_frames": int(rest_corr.size),
            }

    training_table_path = session_dir / TRAINING_TABLE_NAME
    training_rows = write_training_table(
        training_table_path, markers, target_pct_col,
        force_kg_anchor, pct_mvc_anchor, force_kg_corr, pct_mvc_corr,
    )

    result = SessionSyncResult(
        session_dir=session_dir, ok=True,
        n_camera_frames=int(markers.frame.size),
        n_force_samples=int(force.force_kg.size),
        coverage=coverage,
        xcorr=xcorr, taps=taps, agreement_ok=agreement_ok,
        corr_method=corr_method, offset_used_s=offset_used,
        grip_proxy_reref_fallback=grip_proxy_reref_fallback,
        mvc_kg=mvc_kg, rest_force_kg=rest_force_kg, rest_offsets=rest_offsets,
        training_table_path=training_table_path, training_rows=training_rows,
    )
    write_sync_report(session_dir / SYNC_REPORT_NAME, result, force)
    return result


def _sanitize(obj):
    """Recursively swap NaN/Path for JSON-safe equivalents, and round
    floats for a readable report (this is a diagnostic file, not a data
    channel -- no precision-sensitive consumer reads it)."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, float):
        return None if math.isnan(obj) else round(obj, 6)
    return obj


def write_sync_report(path: Path, result: SessionSyncResult, force: ForceTrace) -> None:
    payload = {
        "sign_convention": "force_corrected(T) = interp_force(T - offset); "
                            "positive offset means the camera event lags the force event",
        "anchor_epoch_ms": force.anchor_epoch_ms,
        **asdict(result),
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(_sanitize(payload), f, indent=2)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def print_result(result: SessionSyncResult) -> None:
    status = "PASS" if result.ok else "FAIL"
    print(f"\n[{status}] {result.session_dir}")
    if not result.ok:
        print(f"  ERROR {result.error}")
        return

    print(f"  coverage       : {result.coverage * 100:.1f}%  ({result.n_camera_frames} camera frames)")
    x = result.xcorr
    print(f"  offset xcorr   : {x.offset_s:+.3f}s  (r={x.r_at_best:.3f}, r@0={x.r_at_zero:.3f})")
    if x.at_sweep_edge:
        print(
            f"  WARNING range  : offset sits on the +-{x.range_ms / 1000:.1f}s sweep edge. "
            f"The true offset is outside the window. Re-run with a larger --xcorr-range-ms."
        )
    if x.drift_s is not None:
        print(
            f"                   drift check: first-half {x.offset_first_half_s:+.3f}s, "
            f"second-half {x.offset_second_half_s:+.3f}s, diff {x.drift_s:+.3f}s"
        )
    t = result.taps
    if t.found:
        drift = f", drift {t.drift_s:+.3f}s" if t.drift_s is not None else ""
        print(f"  offset taps    : {t.offset_s:+.3f}s  (start/end{drift})")
    else:
        print(f"  offset taps    : not found -- {t.note}")
    if result.agreement_ok is False:
        print(f"  WARNING agreement: xcorr/taps offsets disagree by more than {AGREEMENT_TOLERANCE_S}s")
    if result.grip_proxy_reref_fallback:
        print("  WARNING reref     : no 0%-MVC rest trial -- xcorr used plain max grip_proxy")
    print(f"  corr_method    : {result.corr_method}  (offset_used={result.offset_used_s:+.3f}s)")
    ro = result.rest_offsets
    print(
        f"  rest (0% MVC)  : grip proxy median={ro.grip_proxy_median}, p95={ro.grip_proxy_p95} px "
        f"(n={ro.n_frames} frames)"
    )
    print(f"  training table : {result.training_table_path}  ({result.training_rows} rows)")


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_session.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("path", type=Path, help="session dir, participant dir, or dynamometer.txt file")
    parser.add_argument(
        "--corr-method", choices=["xcorr", "taps"], default="xcorr",
        help="which offset estimate feeds the *_corr columns (default: xcorr)",
    )
    parser.add_argument(
        "--xcorr-range-ms", type=int, default=XCORR_RANGE_MS,
        help=(
            f"half-width of the cross-correlation offset sweep, in ms "
            f"(default: {XCORR_RANGE_MS}). Raise it if a session reports "
            f"WARNING range."
        ),
    )
    args = parser.parse_args(argv[1:])

    if args.xcorr_range_ms < XCORR_STEP_MS:
        print(f"--xcorr-range-ms must be at least {XCORR_STEP_MS}")
        return 2
    if args.xcorr_range_ms > TZ_GATE_AFTER_MS:
        print(f"--xcorr-range-ms must not exceed the timezone gate ({TZ_GATE_AFTER_MS} ms)")
        return 2
    # The sweep width is read from the module global by _xcorr_search and by
    # compute_xcorr's edge test, so set it once here rather than threading it
    # through five call sites.
    globals()["XCORR_RANGE_MS"] = args.xcorr_range_ms

    if not args.path.exists():
        print(f"path does not exist: {args.path}")
        return 2

    targets = resolve_targets(args.path)
    if not targets:
        print(f"no {EXPORT_NAME} found under {args.path}")
        return 2

    results: list[SessionSyncResult] = []
    for target in targets:
        session_dir = target.parent
        try:
            results.append(sync_session(session_dir, corr_method=args.corr_method))
        except SyncError as exc:
            results.append(SessionSyncResult(session_dir=session_dir, ok=False, error=str(exc)))
        except Exception as exc:  # a single bad session must never kill the batch
            results.append(SessionSyncResult(session_dir=session_dir, ok=False, error=f"unexpected error: {exc}"))

    for result in results:
        print_result(result)

    failed = [r for r in results if not r.ok]
    reref_fallback = [r for r in results if r.ok and r.grip_proxy_reref_fallback]
    print(f"\n{'=' * 60}")
    if len(results) > 1:
        print(f"{len(results) - len(failed)}/{len(results)} synced")
    if reref_fallback:
        print(f"{len(reref_fallback)} session(s) had no 0%-MVC rest trial -- xcorr used plain max grip_proxy:")
        for r in reref_fallback:
            print(f"  {r.session_dir}")
    if failed:
        print("FAILED:")
        for r in failed:
            print(f"  {r.session_dir}: {r.error}")
        return 1
    print("All sessions synced.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
