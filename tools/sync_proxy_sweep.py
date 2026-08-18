"""Sweep candidate camera-side alignment signals against the stored `max` proxy.

`tools/sync_session.py` aligns each session's force trace to its marker
trace by cross-correlating the force against `grip_proxy` -- the MAX
baseline-relative marker magnitude per frame, over non-autofilled markers.
Nobody ever measured whether `max` is the sharpest signal available; this
script measures that, read-only, without touching `sync_session.py` or any
file under `recordings/`.

Plan: docs/logs/20260818_174754_sync_proxy_sweep_plan.md (sections 2, 3, 4, 9
are binding). This script imports its search machinery from
`tools/sync_session.py` -- `_xcorr_search`, `_best_offset`, `read_force`,
`read_markers`, `read_trials` -- rather than reimplementing any of it, so a
result here is only ever as correct as the tool it is checking.

Six candidate camera-side proxies (plan sec 2), all computed over the same
loaded (mag, autofilled) arrays per session -- markers.csv is read once:

    max          max magnitude over non-autofilled markers (the CONTROL --
                 this is exactly sync_session.py's grip_proxy, and this
                 script's whole result is invalid if it does not reproduce
                 every stored sync_report.json offset)
    median       median magnitude over non-autofilled markers
    p95          95th percentile of magnitude over non-autofilled markers
    mean         mean magnitude over non-autofilled markers
    top10_mean   mean of the ten largest magnitudes in the frame
    max_reref    as `max`, after each marker's magnitude is first
                 referenced to that marker's own resting median (median
                 magnitude over the session's 0%-MVC rest frames, a purely
                 camera-side quantity -- trial_id -> trials.csv's *scheduled*
                 target_pct, never the force trace or any offset -- so this
                 cannot be circular with the offset it is meant to sharpen;
                 verified against tools/sync_session.py:823 before writing)

For each (session, proxy) this computes offset_s / r_at_best (via the
imported search), peak_halfwidth_ms and peak_prominence_z (sharpness), and
cross-checks against the session's independent triple-tap offset already
stored in its sync_report.json. Writes docs/results/sync_proxy_sweep.csv,
and prints a control PASS/FAIL check, a per-proxy summary, an outlier
verdict for six named large-offset sessions, and the plan's decision-rule
verdict.

Usage:
    python tools/sync_proxy_sweep.py                       # all 69 sessions
    python tools/sync_proxy_sweep.py --participant participant002
    python tools/sync_proxy_sweep.py --limit 5
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from tools.sync_session import (
    AGREEMENT_TOLERANCE_S,
    XCORR_MIN_OVERLAP_FRAMES,
    XCORR_RANGE_MS,
    XCORR_STEP_MS,
    ForceTrace,
    MarkerTrace,
    _best_offset,
    _xcorr_search,
    compute_grip_proxy_reref,
    compute_marker_rest_median,
    read_force,
    read_markers,
    read_trials,
)

DYNAMOMETER_NAME = "dynamometer.txt"
MARKERS_CSV_NAME = "markers.csv"
TRIALS_CSV_NAME = "trials.csv"
SYNC_REPORT_NAME = "sync_report.json"

DEFAULT_RECORDINGS_DIR = Path("recordings")
DEFAULT_OUT_CSV = Path("docs/results/sync_proxy_sweep.csv")

# "r >= r_max - PEAK_TOLERANCE" defines the peak region for peak_halfwidth_ms
# (plan sec 2). Fixed before the run, per plan sec 2's own instruction not to
# let the rule follow the results.
PEAK_TOLERANCE = 0.01

CSV_FIELDS = [
    "participant", "session", "proxy", "offset_s", "r_at_best",
    "peak_halfwidth_ms", "peak_prominence_z", "tap_offset_s",
    "tap_abs_diff_s", "at_sweep_edge",
]

# The six sessions Guillaume asked about by name (plan sec 3): does the large
# offset persist under every candidate, or collapse to ~0.1s under some?
OUTLIER_SESSIONS = [
    ("participant005", "session6", 7.300),
    ("participant007", "session3", 2.695),
    ("participant005", "session8", 2.265),
    ("participant006", "session7", 2.150),
    ("participant006", "session3", 1.690),
    ("participant006", "session4", 1.540),
]


# --------------------------------------------------------------------------
# Proxy registry -- name -> callable(mag, autofilled, ctx) -> (n_frames,)
# --------------------------------------------------------------------------
# `mag`/`autofilled` are the full (n_frames, N_MARKERS) arrays straight off
# MarkerTrace. `ctx` is a small per-session dict of precomputed extras a
# proxy may need beyond the raw arrays -- today just `marker_rest_median`,
# for max_reref -- every proxy takes it so the call site stays uniform even
# though five of the six ignore it.

def _guard_all_autofilled(proxy: np.ndarray, fallback: np.ndarray, autofilled: np.ndarray) -> np.ndarray:
    """Shared guard, mirroring read_markers()'s own grip_proxy guard: a
    frame with zero non-autofilled markers falls back to the unmasked
    reduction instead of leaking -inf/nan into the sweep (should not happen
    on real data -- guarded rather than left to silently poison a whole
    session's xcorr)."""
    all_autofilled = autofilled.all(axis=1)
    if np.any(all_autofilled):
        proxy = np.where(all_autofilled, fallback, proxy)
    return proxy


def _proxy_max(mag: np.ndarray, autofilled: np.ndarray, ctx: dict) -> np.ndarray:
    """Reproduces read_markers()'s grip_proxy computation exactly (same
    formula, same guard) -- this is the CONTROL and must equal
    sync_report.json's stored xcorr.offset_s for every session."""
    masked = np.where(autofilled, -np.inf, mag)
    proxy = masked.max(axis=1)
    return _guard_all_autofilled(proxy, mag.max(axis=1), autofilled)


def _proxy_median(mag: np.ndarray, autofilled: np.ndarray, ctx: dict) -> np.ndarray:
    masked = np.where(autofilled, np.nan, mag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proxy = np.nanmedian(masked, axis=1)
    return _guard_all_autofilled(proxy, np.median(mag, axis=1), autofilled)


def _proxy_p95(mag: np.ndarray, autofilled: np.ndarray, ctx: dict) -> np.ndarray:
    masked = np.where(autofilled, np.nan, mag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proxy = np.nanpercentile(masked, 95, axis=1)
    return _guard_all_autofilled(proxy, np.percentile(mag, 95, axis=1), autofilled)


def _proxy_mean(mag: np.ndarray, autofilled: np.ndarray, ctx: dict) -> np.ndarray:
    masked = np.where(autofilled, np.nan, mag)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proxy = np.nanmean(masked, axis=1)
    return _guard_all_autofilled(proxy, mag.mean(axis=1), autofilled)


def _proxy_top10_mean(mag: np.ndarray, autofilled: np.ndarray, ctx: dict) -> np.ndarray:
    """Mean of the ten largest non-autofilled magnitudes per frame. A frame
    with fewer than ten non-autofilled markers uses however many it has --
    the autofilled padding sorts to the tail and is dropped, never averaged
    in as -inf/0."""
    masked = np.where(autofilled, -np.inf, mag)
    sorted_desc = np.sort(masked, axis=1)[:, ::-1]
    top10 = sorted_desc[:, :10]
    top10 = np.where(np.isinf(top10), np.nan, top10)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        proxy = np.nanmean(top10, axis=1)
    fallback_sorted = np.sort(mag, axis=1)[:, ::-1]
    fallback = fallback_sorted[:, :10].mean(axis=1)
    return _guard_all_autofilled(proxy, fallback, autofilled)


# compute_marker_rest_median is imported from tools.sync_session (it is now
# the shared, single implementation -- see that module's docstring; ported
# there 2026-08-18 during the re-sync so the sweep and the production tool
# can never drift apart).


def _proxy_max_reref(mag: np.ndarray, autofilled: np.ndarray, ctx: dict) -> np.ndarray:
    marker_rest_median = ctx.get("marker_rest_median")
    if marker_rest_median is None:
        # No rest frames this session -- cannot re-reference. NaN every
        # frame rather than silently falling back to plain `max`, so a
        # session missing this proxy is visible in the CSV, not disguised
        # as a real (and misleadingly identical-to-max) result.
        return np.full(mag.shape[0], np.nan)
    return compute_grip_proxy_reref(mag, autofilled, marker_rest_median)


PROXIES = {
    "max": _proxy_max,
    "median": _proxy_median,
    "p95": _proxy_p95,
    "mean": _proxy_mean,
    "top10_mean": _proxy_top10_mean,
    "max_reref": _proxy_max_reref,
}


# --------------------------------------------------------------------------
# Sharpness metrics (plan sec 2)
# --------------------------------------------------------------------------

def peak_halfwidth_ms(r_values: np.ndarray, step_ms: float = XCORR_STEP_MS) -> float:
    """Total width, in ms, of the offset region where r >= r_max - PEAK_TOLERANCE.
    NaN entries (too little overlap at that candidate offset) are ignored,
    both for locating r_max and for the count."""
    valid = ~np.isnan(r_values)
    if not np.any(valid):
        return float("nan")
    r_max = float(np.nanmax(r_values))
    in_peak = valid & (r_values >= r_max - PEAK_TOLERANCE)
    return float(np.count_nonzero(in_peak) * step_ms)


def peak_prominence_z(r_values: np.ndarray) -> float:
    """(r_max - median(r)) / SD(r) over the whole swept range, ignoring NaNs."""
    valid_r = r_values[~np.isnan(r_values)]
    if valid_r.size < 2:
        return float("nan")
    sd = float(np.std(valid_r))
    if sd == 0:
        return float("nan")
    r_max = float(np.max(valid_r))
    med = float(np.median(valid_r))
    return (r_max - med) / sd


@dataclass
class ProxyMetrics:
    offset_s: float | None
    r_at_best: float | None
    peak_halfwidth_ms: float | None
    peak_prominence_z: float | None
    at_sweep_edge: bool | None


def _sweep_offset_and_metrics(
    timestamp_ms: np.ndarray, proxy_values: np.ndarray, force: ForceTrace,
) -> ProxyMetrics:
    offsets_ms, r_values = _xcorr_search(timestamp_ms, proxy_values, force)
    best = _best_offset(offsets_ms, r_values)
    if best is None:
        return ProxyMetrics(None, None, None, None, None)
    offset_s, r_at_best = best
    hw = peak_halfwidth_ms(r_values)
    prom = peak_prominence_z(r_values)
    at_edge = abs(abs(offset_s) - XCORR_RANGE_MS / 1000.0) < 1e-9
    return ProxyMetrics(offset_s, r_at_best, hw, prom, at_edge)


# --------------------------------------------------------------------------
# Session discovery / identification
# --------------------------------------------------------------------------

_PARTICIPANT_RE = re.compile(r"(participant\d+)")
_SESSION_RE = re.compile(r"(session\d+)")


def _participant_id(dirname: str) -> str:
    m = _PARTICIPANT_RE.match(dirname)
    return m.group(1) if m else dirname


def _session_id(dirname: str) -> str:
    m = _SESSION_RE.match(dirname)
    return m.group(1) if m else dirname


def _sort_key(path: Path) -> tuple[int, int]:
    p_match = re.search(r"participant(\d+)", path.parent.name)
    s_match = re.search(r"session(\d+)", path.name)
    return (
        int(p_match.group(1)) if p_match else 0,
        int(s_match.group(1)) if s_match else 0,
    )


def find_sessions(recordings_dir: Path, participant: str | None = None) -> list[Path]:
    session_dirs = sorted(
        (d for d in recordings_dir.glob("participant*/session*") if d.is_dir()),
        key=_sort_key,
    )
    if participant:
        session_dirs = [d for d in session_dirs if _participant_id(d.parent.name) == participant]
    return session_dirs


# --------------------------------------------------------------------------
# Per-session processing
# --------------------------------------------------------------------------

def process_session(session_dir: Path) -> tuple[list[dict], dict]:
    """Reads one session's markers.csv/dynamometer.txt/trials.csv/
    sync_report.json ONCE, runs the full sweep for all six proxies off the
    same loaded arrays, and returns (csv_rows, diagnostics).

    diagnostics carries the control-check inputs for this session
    (stored vs. computed `max` offset) and rest-mask bookkeeping for
    max_reref -- never re-derived later from the CSV.
    """
    report_path = session_dir / SYNC_REPORT_NAME
    dyn_path = session_dir / DYNAMOMETER_NAME
    markers_path = session_dir / MARKERS_CSV_NAME
    trials_path = session_dir / TRIALS_CSV_NAME

    if not report_path.exists() or not dyn_path.exists():
        return [], {"skipped": True, "reason": "missing sync_report.json or dynamometer.txt"}

    with report_path.open("r", encoding="utf-8") as f:
        report = json.load(f)
    stored_offset_s = (report.get("xcorr") or {}).get("offset_s")
    taps = report.get("taps") or {}
    tap_offset_s = taps.get("offset_s") if taps.get("found") else None

    force = read_force(dyn_path)
    markers = read_markers(markers_path)

    ctx: dict = {"marker_rest_median": None}
    rest_diag = {"n_rest_frames": 0, "n_always_autofilled_at_rest": 0, "trials_missing": False}
    if trials_path.exists():
        target_pct_by_trial = read_trials(trials_path)
        rest_mask = np.asarray(
            [target_pct_by_trial.get(int(tid)) == 0 for tid in markers.trial_id]
        )
        rest_result = compute_marker_rest_median(markers.magnitude, markers.autofilled, rest_mask)
        if rest_result is not None:
            marker_median, n_always_af = rest_result
            ctx["marker_rest_median"] = marker_median
            rest_diag["n_rest_frames"] = int(rest_mask.sum())
            rest_diag["n_always_autofilled_at_rest"] = n_always_af
    else:
        rest_diag["trials_missing"] = True

    participant_id = _participant_id(session_dir.parent.name)
    session_id = _session_id(session_dir.name)

    rows: list[dict] = []
    control_offset_s: float | None = None
    for proxy_name, proxy_fn in PROXIES.items():
        proxy_values = proxy_fn(markers.magnitude, markers.autofilled, ctx)
        metrics = _sweep_offset_and_metrics(markers.timestamp_ms, proxy_values, force)
        tap_abs_diff = (
            abs(metrics.offset_s - tap_offset_s)
            if (metrics.offset_s is not None and tap_offset_s is not None)
            else None
        )
        rows.append({
            "participant": participant_id,
            "session": session_id,
            "proxy": proxy_name,
            "offset_s": metrics.offset_s,
            "r_at_best": metrics.r_at_best,
            "peak_halfwidth_ms": metrics.peak_halfwidth_ms,
            "peak_prominence_z": metrics.peak_prominence_z,
            "tap_offset_s": tap_offset_s,
            "tap_abs_diff_s": tap_abs_diff,
            "at_sweep_edge": metrics.at_sweep_edge,
        })
        if proxy_name == "max":
            control_offset_s = metrics.offset_s

    control_pass = None
    if control_offset_s is not None and stored_offset_s is not None:
        control_pass = abs(control_offset_s - stored_offset_s) < 1e-6

    diagnostics = {
        "skipped": False,
        "stored_offset_s": stored_offset_s,
        "control_offset_s": control_offset_s,
        "control_pass": control_pass,
        "rest": rest_diag,
    }
    return rows, diagnostics


# --------------------------------------------------------------------------
# Output
# --------------------------------------------------------------------------

def _csv_fmt(v):
    if v is None:
        return ""
    if isinstance(v, float) and math.isnan(v):
        return ""
    return v


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_fmt(row[k]) for k in CSV_FIELDS})


def print_control_check(diagnostics_by_session: dict[Path, dict]) -> bool:
    print("\n" + "=" * 70)
    print("CONTROL CHECK -- max proxy vs stored sync_report.json xcorr.offset_s")
    print("=" * 70)
    n = len(diagnostics_by_session)
    failures = [
        (sd, diag["stored_offset_s"], diag["control_offset_s"])
        for sd, diag in diagnostics_by_session.items()
        if diag["control_pass"] is not True
    ]
    if failures:
        print(f"FAIL: {len(failures)}/{n} sessions' max proxy did NOT reproduce the stored offset.")
        print("This means the sweep script is wrong -- not that `max` is wrong. Investigate before trusting any other row.")
        for sd, stored, ours in failures:
            print(f"  {sd}: stored={stored}  computed={ours}")
        return False
    print(f"PASS: all {n} sessions' max proxy reproduced the stored offset exactly.")
    return True


def _median_iqr(values: list[float]) -> str:
    if not values:
        return "n/a"
    med = statistics.median(values)
    q1 = float(np.percentile(values, 25))
    q3 = float(np.percentile(values, 75))
    return f"{med:.3f} [{q1:.3f}, {q3:.3f}]"


def print_summary(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("PER-PROXY SUMMARY -- median [IQR] across usable sessions")
    print("=" * 70)
    by_proxy: dict[str, list[dict]] = {}
    for row in rows:
        by_proxy.setdefault(row["proxy"], []).append(row)

    print(f"{'proxy':<12} {'halfwidth_ms':<24} {'prominence_z':<22} {'r_at_best':<20} {'tap_abs_diff_s':<24} {'n_within_tol':<6}")
    for proxy_name in PROXIES:
        proxy_rows = by_proxy.get(proxy_name, [])
        hw = [r["peak_halfwidth_ms"] for r in proxy_rows
              if r["peak_halfwidth_ms"] is not None and not math.isnan(r["peak_halfwidth_ms"])]
        prom = [r["peak_prominence_z"] for r in proxy_rows
                if r["peak_prominence_z"] is not None and not math.isnan(r["peak_prominence_z"])]
        r_best = [r["r_at_best"] for r in proxy_rows if r["r_at_best"] is not None]
        tap_diff = [r["tap_abs_diff_s"] for r in proxy_rows if r["tap_abs_diff_s"] is not None]
        n_within_tol = sum(1 for d in tap_diff if d <= AGREEMENT_TOLERANCE_S)
        print(
            f"{proxy_name:<12} {_median_iqr(hw):<24} {_median_iqr(prom):<22} "
            f"{_median_iqr(r_best):<20} {_median_iqr(tap_diff):<24} {n_within_tol:<6}"
        )


def print_outlier_verdict(rows: list[dict]) -> None:
    print("\n" + "=" * 70)
    print("OUTLIER VERDICT -- offset_s (s) per proxy, six named large-offset sessions")
    print("=" * 70)
    by_key: dict[tuple[str, str], dict[str, float | None]] = {}
    for row in rows:
        by_key.setdefault((row["participant"], row["session"]), {})[row["proxy"]] = row["offset_s"]

    proxy_names = list(PROXIES.keys())
    header = f"{'session':<24}{'stored':>10}" + "".join(f"{p:>14}" for p in proxy_names)
    print(header)
    for participant, session, stored_offset in OUTLIER_SESSIONS:
        offsets = by_key.get((participant, session), {})
        line = f"{participant + '/' + session:<24}{stored_offset:>10.3f}"
        for p in proxy_names:
            v = offsets.get(p)
            line += f"{v:>14.3f}" if v is not None else f"{'n/a':>14}"
        print(line)
        if not offsets:
            print(f"  (no rows found for {participant}/{session} -- session missing or skipped)")


def print_decision(rows: list[dict]) -> list[str]:
    """Plan sec 2's decision rule, fixed before the numbers existed: a
    candidate replaces `max` only if BOTH (1) its median peak_halfwidth_ms
    is smaller than max's, and (2) its median tap_abs_diff_s is not worse
    than max's."""
    print("\n" + "=" * 70)
    print("DECISION RULE (plan sec 2) -- sharper peak AND not worse on tap agreement")
    print("=" * 70)
    by_proxy: dict[str, list[dict]] = {}
    for row in rows:
        by_proxy.setdefault(row["proxy"], []).append(row)

    def _median(values: list[float]) -> float:
        return statistics.median(values) if values else float("nan")

    def _hw_values(proxy_rows: list[dict]) -> list[float]:
        return [r["peak_halfwidth_ms"] for r in proxy_rows
                if r["peak_halfwidth_ms"] is not None and not math.isnan(r["peak_halfwidth_ms"])]

    def _tap_values(proxy_rows: list[dict]) -> list[float]:
        return [r["tap_abs_diff_s"] for r in proxy_rows if r["tap_abs_diff_s"] is not None]

    max_hw = _median(_hw_values(by_proxy.get("max", [])))
    max_tap = _median(_tap_values(by_proxy.get("max", [])))
    print(f"  max control: median halfwidth={max_hw:.3f}ms, median tap_abs_diff={max_tap:.3f}s")

    winners = []
    for proxy_name in PROXIES:
        if proxy_name == "max":
            continue
        proxy_rows = by_proxy.get(proxy_name, [])
        hw_med = _median(_hw_values(proxy_rows))
        tap_med = _median(_tap_values(proxy_rows))
        sharper = hw_med < max_hw if not math.isnan(hw_med) and not math.isnan(max_hw) else False
        not_worse = tap_med <= max_tap if not math.isnan(tap_med) and not math.isnan(max_tap) else False
        verdict = "PASSES" if (sharper and not_worse) else "fails"
        print(f"  {proxy_name:<12} halfwidth={hw_med:.3f}ms tap_abs_diff={tap_med:.3f}s -> {verdict}")
        if sharper and not_worse:
            winners.append(proxy_name)

    if winners:
        print(f"\nWinner(s) under the decision rule: {', '.join(winners)}")
    else:
        print("\nNo candidate satisfies the decision rule. Verdict: none.")
    return winners


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        prog="sync_proxy_sweep.py",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--recordings-dir", type=Path, default=DEFAULT_RECORDINGS_DIR,
        help=f"root containing participant*/session* dirs (default: {DEFAULT_RECORDINGS_DIR})",
    )
    parser.add_argument(
        "--out", type=Path, default=DEFAULT_OUT_CSV,
        help=f"CSV output path (default: {DEFAULT_OUT_CSV})",
    )
    parser.add_argument(
        "--participant", type=str, default=None,
        help="limit to one participant, e.g. participant002",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="process at most N sessions, after --participant filtering",
    )
    args = parser.parse_args(argv[1:])

    session_dirs = find_sessions(args.recordings_dir, participant=args.participant)
    if args.limit is not None:
        session_dirs = session_dirs[: args.limit]

    print(
        f"sweep params (imported from tools/sync_session.py): "
        f"range=+-{XCORR_RANGE_MS}ms step={XCORR_STEP_MS}ms min_overlap={XCORR_MIN_OVERLAP_FRAMES} frames"
    )
    print(f"{len(session_dirs)} session dir(s) found under {args.recordings_dir}")

    all_rows: list[dict] = []
    diagnostics_by_session: dict[Path, dict] = {}
    skipped: list[tuple[Path, str]] = []
    for session_dir in session_dirs:
        rows, diag = process_session(session_dir)
        if diag.get("skipped"):
            skipped.append((session_dir, diag.get("reason", "unknown")))
            continue
        all_rows.extend(rows)
        diagnostics_by_session[session_dir] = diag

    print(f"\n{len(diagnostics_by_session)} usable, {len(skipped)} skipped")
    for sd, reason in skipped:
        print(f"  SKIP {sd}: {reason}")

    n_reref_unavailable = sum(
        1 for diag in diagnostics_by_session.values()
        if diag["rest"]["n_rest_frames"] == 0
    )
    if n_reref_unavailable:
        print(f"\n{n_reref_unavailable} session(s) had no 0%-MVC rest frames -- max_reref is NaN for those.")
    n_always_af_markers = sum(
        diag["rest"]["n_always_autofilled_at_rest"] for diag in diagnostics_by_session.values()
    )
    if n_always_af_markers:
        print(
            f"{n_always_af_markers} (session, marker) pair(s) were autofilled on every rest frame -- "
            "those markers use rest_median=0.0 (unreferenced) in max_reref."
        )

    write_csv(args.out, all_rows)
    print(f"\nwrote {len(all_rows)} rows to {args.out}")

    control_ok = print_control_check(diagnostics_by_session)
    print_summary(all_rows)
    print_outlier_verdict(all_rows)
    print_decision(all_rows)

    return 0 if control_ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
