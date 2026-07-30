"""Tests for output/mvc.py's parse_mvc_file (Phase 1 of
docs/logs/20260730_150048_mvc_file_load_plan.md). Pure tmp_path-based,
synthetic Biopac-shaped fixtures only -- no real participant data.
"""
import math

import pytest

from output.mvc import MvcParseError, THRESHOLD_KG, MIN_SEGMENT_SAMPLES, parse_mvc_file

SAMPLE_SPACING_MS = 5  # matches the observed Biopac export spacing

# Free-form header lines observed on the real export (device/graph info of
# unstable shape) -- none of these parse as two floats, so the header-agnostic
# parser (D1) must skip all of them regardless of order/count.
HEADER_LINES = [
    "data-settings.gtl",
    "Graph Created on: 07/30/2026",
    "Recording on: 07/30/2026",
    "5 msec/sample",
    "1 channels",
    "Clench Force",
    "kg",
    "milliSec,CH2,",
    ",3332 samples,",
]


def _baseline(i: int) -> float:
    """Slow sub-threshold drift, matching the observed ~-0.9..+0.05 kg band."""
    return -0.425 + 0.475 * math.sin(i / 37.0)


def _triangle_pulse(n_samples: int, peak: float) -> list[float]:
    """A triangular ramp 0 -> peak -> 0 over n_samples, apex exactly `peak`."""
    half = n_samples // 2
    up = [peak * (i / half) for i in range(half)]
    down = [peak * (1 - i / half) for i in range(half)]
    return up + down


def _build_rows(events: list, n_baseline_head: int = 50, n_baseline_gap: int = 50) -> list[float]:
    """events: list of ("baseline", n) or ("pulse", n_samples, peak) tuples,
    interleaved as given. Returns the flat kg series (baseline drift added
    under pulses too, negligible next to THRESHOLD_KG's 8x margin)."""
    kg_series: list[float] = []
    idx = 0
    for kind, *rest in events:
        if kind == "baseline":
            (n,) = rest
            for _ in range(n):
                kg_series.append(_baseline(idx))
                idx += 1
        elif kind == "pulse":
            # Pulses are kept as clean triangles (no baseline drift layered
            # on top) so each segment's peak is exactly the given value --
            # baseline drift is exercised separately, only during the
            # "baseline" (rest) stretches between clenches.
            n, peak = rest
            for v in _triangle_pulse(n, peak):
                kg_series.append(v)
                idx += 1
        else:
            raise ValueError(kind)
    return kg_series


def _write_fixture(tmp_path, kg_series: list[float], with_header: bool = True) -> str:
    path = tmp_path / "mvc.txt"
    lines = list(HEADER_LINES) if with_header else []
    for i, kg in enumerate(kg_series):
        ms = i * SAMPLE_SPACING_MS
        lines.append(f"{ms},{kg:.4f},")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return str(path)


# Pulse width chosen well above MIN_SEGMENT_SAMPLES once triangle apex
# clears THRESHOLD_KG -- with peaks ~40+ kg and threshold 5 kg, the vast
# majority of a 200-sample ramp sits above threshold.
REAL_PULSE_WIDTH = 200
REAL_PEAKS = [42.85, 43.81, 42.85]  # matches the doc's observed real trace


def test_three_clean_pulses_correct_mean(tmp_path):
    events = [("baseline", 50)]
    for peak in REAL_PEAKS:
        events.append(("pulse", REAL_PULSE_WIDTH, peak))
        events.append(("baseline", 50))
    kg_series = _build_rows(events)
    path = _write_fixture(tmp_path, kg_series)

    result = parse_mvc_file(path)

    assert result.mvc_kg == pytest.approx(sum(REAL_PEAKS) / 3, abs=1e-6)
    assert sorted(result.peaks_kg) == pytest.approx(sorted(REAL_PEAKS), abs=1e-6)
    assert result.source_path == path


def test_subthreshold_baseline_drift_ignored(tmp_path):
    # Baseline drift band tops out just under THRESHOLD_KG -- must never be
    # mistaken for a 4th segment.
    assert max(_baseline(i) for i in range(1000)) < THRESHOLD_KG

    events = [("baseline", 200)]
    for peak in REAL_PEAKS:
        events.append(("pulse", REAL_PULSE_WIDTH, peak))
        events.append(("baseline", 200))
    kg_series = _build_rows(events)
    path = _write_fixture(tmp_path, kg_series)

    result = parse_mvc_file(path)

    assert len(result.peaks_kg) == 3
    assert result.mvc_kg == pytest.approx(sum(REAL_PEAKS) / 3, abs=1e-6)


def test_fourth_spurious_pulse_excluded_by_top3(tmp_path):
    # A short but duration-qualifying spurious pulse (just above
    # MIN_SEGMENT_SAMPLES) with a much smaller peak than the 3 real clenches.
    spurious_width = MIN_SEGMENT_SAMPLES + 5
    spurious_peak = 10.0
    assert spurious_peak < min(REAL_PEAKS)

    events = [("baseline", 50)]
    for peak in REAL_PEAKS:
        events.append(("pulse", REAL_PULSE_WIDTH, peak))
        events.append(("baseline", 50))
    events.append(("pulse", spurious_width, spurious_peak))
    events.append(("baseline", 50))
    kg_series = _build_rows(events)
    path = _write_fixture(tmp_path, kg_series)

    result = parse_mvc_file(path)

    assert len(result.peaks_kg) == 3
    assert spurious_peak not in result.peaks_kg
    assert result.mvc_kg == pytest.approx(sum(REAL_PEAKS) / 3, abs=1e-6)


def test_fewer_than_three_segments_raises(tmp_path):
    events = [("baseline", 50)]
    for peak in REAL_PEAKS[:2]:
        events.append(("pulse", REAL_PULSE_WIDTH, peak))
        events.append(("baseline", 50))
    kg_series = _build_rows(events)
    path = _write_fixture(tmp_path, kg_series)

    with pytest.raises(MvcParseError):
        parse_mvc_file(path)


def test_header_only_no_numeric_rows_raises_parse_error(tmp_path):
    path = _write_fixture(tmp_path, kg_series=[], with_header=True)

    with pytest.raises(MvcParseError):
        parse_mvc_file(path)


def test_single_sample_glitch_excluded(tmp_path):
    # A lone above-threshold sample (1 row, under MIN_SEGMENT_SAMPLES) among
    # 3 real clenches must not be counted as a 4th segment.
    events = [("baseline", 50)]
    for peak in REAL_PEAKS:
        events.append(("pulse", REAL_PULSE_WIDTH, peak))
        events.append(("baseline", 50))
    kg_series = _build_rows(events)

    # Splice a single glitch sample well above THRESHOLD_KG into a baseline
    # stretch (index chosen to land inside the final "baseline" run).
    glitch_index = len(kg_series) - 25
    kg_series[glitch_index] = THRESHOLD_KG + 10.0

    path = _write_fixture(tmp_path, kg_series)

    result = parse_mvc_file(path)

    assert len(result.peaks_kg) == 3
    assert sorted(result.peaks_kg) == pytest.approx(sorted(REAL_PEAKS), abs=1e-6)
