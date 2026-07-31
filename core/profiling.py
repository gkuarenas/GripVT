"""Opt-in stage timing for the camera poll loop.

This is a spike tool, not a feature: it exists to answer "how long does each
part of a frame take, and what happens if we change it" with numbers instead
of guesses -- see docs/logs/20260731_152249_framerate_spike_plan.md. Gated on
GRIPVT_PROFILE so a normal session pays nothing for it: stage() doesn't call
perf_counter() at all when PROFILE is off, and mark_period()/
maybe_report_every() are no-ops.
"""
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from statistics import mean, median
from typing import Iterator

PROFILE = os.environ.get("GRIPVT_PROFILE") == "1"

_samples: dict[str, list[float]] = defaultdict(list)
_last_period_ts: float | None = None
_poll_count = 0


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Times the wrapped block, appending elapsed ms under `name`. When
    PROFILE is off this yields immediately with no timing call at all, so it
    is safe to leave wrapped around hot-loop code permanently."""
    if not PROFILE:
        yield
        return
    start = time.perf_counter()
    try:
        yield
    finally:
        _samples[name].append((time.perf_counter() - start) * 1000.0)


def mark_period() -> None:
    """Records the wall-clock gap since the previous call under "period" --
    this, not any single stage, is the real achieved frame rate. The first
    call has no prior timestamp to diff against, so it records nothing."""
    global _last_period_ts
    if not PROFILE:
        return
    now = time.perf_counter()
    if _last_period_ts is not None:
        _samples["period"].append((now - _last_period_ts) * 1000.0)
    _last_period_ts = now


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, round(0.95 * (len(ordered) - 1)))]


def report(reset: bool = False) -> str:
    """One row per recorded stage (name, n, mean, median, p95 ms), sorted
    most-expensive-first, plus an implied-fps line derived from "period" --
    the single number this whole pass exists to move."""
    header = f"{'stage':<20}{'n':>6}{'mean':>10}{'median':>10}{'p95':>10}"
    rows = sorted(_samples.items(), key=lambda kv: mean(kv[1]), reverse=True)
    lines = [header] + [
        f"{name:<20}{len(vals):>6}{mean(vals):>10.2f}{median(vals):>10.2f}{_p95(vals):>10.2f}"
        for name, vals in rows
    ]
    if "period" in _samples:
        mean_period = mean(_samples["period"])
        if mean_period > 0:
            lines.append(f"implied fps: {1000.0 / mean_period:.2f}")
    text = "\n".join(lines)
    if reset:
        _samples.clear()
    return text


def maybe_report_every(n: int) -> None:
    """Prints report() to stdout every n calls. No-op when PROFILE is off."""
    global _poll_count
    if not PROFILE:
        return
    _poll_count += 1
    if _poll_count % n == 0:
        print(report())


def reset() -> None:
    """Clears all recorded samples (and the period baseline)."""
    global _last_period_ts
    _samples.clear()
    _last_period_ts = None
