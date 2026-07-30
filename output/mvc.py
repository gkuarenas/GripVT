"""Biopac MVC recording parser (docs/logs/20260730_150048_mvc_file_load_plan.md, Phase 1).

Parses a Biopac `mvc.txt` export (comma-delimited `milliSec,CH2` force trace,
free-form metadata header of unknown/variable shape) into the 3 clench peaks
and their mean, which becomes participant.mvc_kg.
"""
from dataclasses import dataclass

# 10x the largest observed baseline drift (<1 kg), 8x under the smallest
# observed peak (~42.85 kg) -- wide margin both directions (D2).
THRESHOLD_KG = 5.0

# 100 ms at the observed 5 ms sample spacing -- far shorter than any real
# clench, far longer than a single-sample glitch (D3).
MIN_SEGMENT_SAMPLES = 20


@dataclass
class MvcResult:
    peaks_kg: list[float]
    mvc_kg: float
    source_path: str


class MvcParseError(ValueError):
    pass


def parse_mvc_file(path: str) -> MvcResult:
    samples: list[tuple[float, float]] = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            tokens = line.split(",")
            if len(tokens) < 2:
                continue
            try:
                ms = float(tokens[0])
                kg = float(tokens[1])
            except ValueError:
                continue
            samples.append((ms, kg))

    if not samples:
        raise MvcParseError(f"no numeric data rows found in {path!r}")

    # Contiguous above-threshold segmentation (D2), min-duration filter (D3).
    segment_peaks: list[float] = []
    run: list[float] = []
    for _ms, kg in samples:
        if kg > THRESHOLD_KG:
            run.append(kg)
        else:
            if len(run) >= MIN_SEGMENT_SAMPLES:
                segment_peaks.append(max(run))
            run = []
    if len(run) >= MIN_SEGMENT_SAMPLES:
        segment_peaks.append(max(run))

    if len(segment_peaks) < 3:
        raise MvcParseError(
            f"expected at least 3 clench segments in {path!r}, found {len(segment_peaks)}"
        )

    # Top-3-by-peak selection (D5), but keep the returned list in file order
    # (the order the clenches actually occurred in), not peak-descending order.
    top3_indices = sorted(
        range(len(segment_peaks)), key=lambda i: segment_peaks[i], reverse=True
    )[:3]
    top3 = [segment_peaks[i] for i in sorted(top3_indices)]
    mvc_kg = sum(top3) / 3

    return MvcResult(peaks_kg=top3, mvc_kg=mvc_kg, source_path=path)
