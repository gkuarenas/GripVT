from collections import Counter
import random

import pytest

import core.trial_sequence as trial_sequence
from core.trial_sequence import (
    LEVELS_PCT,
    REPS_PER_LEVEL,
    TRIAL_COUNT,
    SequenceGenerationError,
    TrialSequence,
    accept_band_kg,
    generate_sequence,
    is_valid,
    target_kg,
)


def test_generated_sequences_are_balanced_and_valid_over_many_seeds():
    expected = Counter({pct: REPS_PER_LEVEL for pct in LEVELS_PCT})
    saw_50_50 = False
    for seed in range(200):
        seq = generate_sequence(seed=seed)
        assert len(seq.levels_pct) == TRIAL_COUNT
        assert Counter(seq.levels_pct) == expected
        assert is_valid(seq.levels_pct)
        if (50, 50) in zip(seq.levels_pct, seq.levels_pct[1:]):
            saw_50_50 = True
    assert saw_50_50


def test_is_valid_rejects_80_80_and_80_50_adjacency():
    assert not is_valid([80, 80])
    assert not is_valid([80, 50])
    assert not is_valid([50, 80])


def test_is_valid_accepts_50_50_adjacency():
    assert is_valid([50, 50])


def test_is_valid_rejects_a_hand_built_violation():
    violating = [0] * 6 + [5] * 6 + [20] * 6 + [50] * 5 + [80] * 6 + [50]
    assert violating[-2:] == [80, 50]
    assert not is_valid(violating)


def test_same_seed_reproduces_identical_sequence():
    a = generate_sequence(seed=42)
    b = generate_sequence(seed=42)
    assert a.levels_pct == b.levels_pct
    assert a.attempts == b.attempts


def test_different_seeds_produce_different_sequences():
    a = generate_sequence(seed=1)
    b = generate_sequence(seed=2)
    assert a.levels_pct != b.levels_pct


def test_generate_sequence_draws_a_seed_when_none_given():
    a = generate_sequence()
    b = generate_sequence()
    assert isinstance(a.seed, int)
    assert a.seed != b.seed


def test_trial_sequence_rejects_wrong_length():
    with pytest.raises(ValueError):
        TrialSequence(seed=1, levels_pct=(0, 5, 20), attempts=1)


def test_target_kg_rounds_to_three_decimals():
    assert target_kg(50, 40.0) == 20.0
    assert target_kg(0, 40.0) == 0.0
    assert target_kg(5, 33.333) == round(5 / 100 * 33.333, 3)


def test_accept_band_matches_the_protocol_table_at_the_real_mvc():
    """The band table from the 2026-07-31 sync plan section 9, computed at
    participant001's real MVC. These are the numbers the operator reads off the
    UI, so they are pinned against the doc rather than recomputed from the
    formula (which would make the test tautological)."""
    mvc = 41.719
    assert accept_band_kg(0, mvc) == (0.0, 0.417)
    assert accept_band_kg(5, mvc) == (1.669, 2.503)
    assert accept_band_kg(20, mvc) == (7.092, 9.596)
    assert accept_band_kg(50, mvc) == (17.731, 23.989)
    assert accept_band_kg(80, mvc) == (28.369, 38.381)


def test_accept_band_lower_bound_never_goes_negative():
    """Only the 0% target can drive it negative -- its band is the 1%MVC floor
    alone, with no target to subtract it from."""
    lo, hi = accept_band_kg(0, 41.719)
    assert lo == 0.0
    assert hi == pytest.approx(0.01 * 41.719, abs=1e-3)


def test_accept_band_brackets_its_own_displayed_target():
    """A band that didn't contain its target would be incoherent on screen."""
    for mvc in (20.0, 41.719, 60.5):
        for pct in LEVELS_PCT:
            lo, hi = accept_band_kg(pct, mvc)
            assert lo <= target_kg(pct, mvc) <= hi


def test_floor_governs_below_the_crossover_and_the_15pct_rule_above_it():
    """15% of target overtakes the 1%MVC floor at pct = 100 * 0.01 / 0.15
    = 6.67%. Pinning the crossover stops a future MVC or constant change from
    silently flipping which rule governs a given target."""
    mvc = 41.719
    floor = 0.01 * mvc
    for pct in (1, 5, 6):
        lo, hi = accept_band_kg(pct, mvc)
        assert (hi - lo) / 2 == pytest.approx(floor, abs=1e-3)
    for pct in (7, 20, 50, 80):
        lo, hi = accept_band_kg(pct, mvc)
        assert (hi - lo) / 2 == pytest.approx(0.15 * target_kg(pct, mvc), abs=1e-3)


def test_accept_band_scales_with_mvc_not_with_a_hardcoded_kg():
    """Doubling MVC doubles both bounds -- catches any absolute-kg constant
    creeping into the formula."""
    lo1, hi1 = accept_band_kg(20, 30.0)
    lo2, hi2 = accept_band_kg(20, 60.0)
    assert (lo2, hi2) == pytest.approx((lo1 * 2, hi1 * 2), abs=1e-3)


def test_exhaustion_raises_sequence_generation_error(monkeypatch):
    monkeypatch.setattr(trial_sequence, "MAX_SHUFFLE_ATTEMPTS", 1)
    monkeypatch.setattr(random.Random, "shuffle", lambda self, x: None)
    with pytest.raises(SequenceGenerationError):
        generate_sequence(seed=1)


def test_empirical_acceptance_rate_matches_the_exact_probability(capsys):
    """D3's Poisson approximation estimates ~3% acceptance. The exact value,
    computed by DP over the multiset-with-forbidden-adjacency arrangements,
    is ~0.63% -- the Poisson model undercounts because forbidden pairs
    cluster (e.g. all six 80s landing consecutively kills many arrangements
    at once), which D3 itself flagged as "indicative, not exact." Either way
    the 10,000-attempt cap stays safe: even at 0.63%, exhausting it has
    probability ~(1-0.0063)**10000, i.e. effectively zero."""
    rng = random.Random(20260728)
    pool = [pct for pct in LEVELS_PCT for _ in range(REPS_PER_LEVEL)]
    trials = 20_000
    accepted = 0
    for _ in range(trials):
        rng.shuffle(pool)
        if is_valid(pool):
            accepted += 1
    rate = accepted / trials
    with capsys.disabled():
        print(
            f"\n  empirical acceptance rate: {rate:.4f} over {trials} shuffles "
            "(exact: 0.0063, D3's Poisson estimate: ~0.03)"
        )
    assert rate > 0.002
