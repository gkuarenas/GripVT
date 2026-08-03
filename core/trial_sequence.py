"""Balanced-multiset trial sequence generator for %MVC grip trials.

Implements the shuffle-and-reject rule in
docs/testing-protocol-files/mvc-trial-randomization-protocol.md: six reps of
each of the five %MVC levels per session, rejecting and reshuffling any
sequence that places a forbidden pair adjacent. Forbidden pairs are the
80/80 and 80/50 fatigue rule plus 0/0 -- two 0% (rest) reps back to back
waste a periodic baseline-drift check on the same instant; 50/50 stays
legal.

`TrialSequence.levels_pct` is 0-indexed but the display's trial number is
1-based: trial N's target is `sequence[N - 1]`, i.e. `levels_pct[0]` is trial
1's target, never trial 0's.
"""
from collections.abc import Sequence
from dataclasses import dataclass
import random

LEVELS_PCT = (0, 5, 20, 50, 80)
REPS_PER_LEVEL = 6
TRIAL_COUNT = 30
FORBIDDEN_ADJACENT = frozenset({(80, 80), (80, 50), (50, 80), (0, 0)})
MAX_SHUFFLE_ATTEMPTS = 10_000
GENERATOR_VERSION = "balanced_multiset_rejection_v2"
BAND_FRAC_OF_TARGET = 0.15
BAND_FLOOR_FRAC_OF_MVC = 0.01


class SequenceGenerationError(RuntimeError):
    """No valid shuffle was found within MAX_SHUFFLE_ATTEMPTS attempts."""


@dataclass(frozen=True)
class TrialSequence:
    seed: int
    levels_pct: tuple[int, ...]
    attempts: int

    def __post_init__(self) -> None:
        if len(self.levels_pct) != TRIAL_COUNT:
            raise ValueError(
                f"levels_pct must have {TRIAL_COUNT} entries, got {len(self.levels_pct)}"
            )


def is_valid(levels_pct: Sequence[int]) -> bool:
    return not any(
        (a, b) in FORBIDDEN_ADJACENT for a, b in zip(levels_pct, levels_pct[1:])
    )


def target_kg(pct: int, mvc_kg: float) -> float:
    return round(pct / 100 * mvc_kg, 3)


def accept_band_kg(pct: int, mvc_kg: float) -> tuple[float, float]:
    """The protocol's advisory acceptance band around a trial's target, in kg.

    Half-width is `max(15% of target, 1% of MVC)`. The floor governs the low
    targets -- 15% of a 5% target is only +-0.31kg at a 41.7kg MVC, tighter
    than deliberate control allows -- and the 15% rule governs everything from
    ~6.67% upward. The lower bound clamps at zero, which only bites the 0%
    target, where the band is the floor alone.

    Advisory only: nothing gates, scores, or persists on this. Computed from
    `target_kg()` rather than from `pct * mvc` directly, so the band always
    brackets the same rounded figure the UI displays.
    """
    target = target_kg(pct, mvc_kg)
    half = max(BAND_FRAC_OF_TARGET * target, BAND_FLOOR_FRAC_OF_MVC * mvc_kg)
    return round(max(0.0, target - half), 3), round(target + half, 3)


def generate_sequence(seed: int | None = None) -> TrialSequence:
    if seed is None:
        seed = random.SystemRandom().getrandbits(64)

    pool = [pct for pct in LEVELS_PCT for _ in range(REPS_PER_LEVEL)]
    rng = random.Random(seed)
    for attempt in range(1, MAX_SHUFFLE_ATTEMPTS + 1):
        rng.shuffle(pool)
        if is_valid(pool):
            return TrialSequence(seed=seed, levels_pct=tuple(pool), attempts=attempt)

    raise SequenceGenerationError(
        f"no valid sequence found in {MAX_SHUFFLE_ATTEMPTS} attempts (seed={seed})"
    )
