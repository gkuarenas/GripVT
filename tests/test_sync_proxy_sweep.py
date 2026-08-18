"""Tests for tools/sync_proxy_sweep.py (docs/logs/20260818_174754_sync_proxy_sweep_plan.md).

Every fixture is synthetic, built inline with numpy -- this suite must pass
on a machine with no recordings/ directory at all. Proxy functions are
tested directly against small hand-built (mag, autofilled) arrays, not
against real markers.csv files, per the existing tests/test_sync_session.py
convention.
"""
import numpy as np
import pytest

from tools.sync_proxy_sweep import (
    PROXIES,
    _proxy_max,
    _proxy_max_reref,
    _proxy_mean,
    _proxy_median,
    _proxy_p95,
    _proxy_top10_mean,
    compute_marker_rest_median,
    peak_halfwidth_ms,
    peak_prominence_z,
)

# --------------------------------------------------------------------------
# 1. peak_halfwidth_ms -- known-width synthetic peak
# --------------------------------------------------------------------------

def test_peak_halfwidth_ms_known_plateau_width():
    # r_max = 1.0, threshold = r_max - 0.01 = 0.99. Exactly 3 grid points
    # (indices 2,3,4) sit at/above threshold -> halfwidth = 3 * step_ms.
    r_values = np.array([0.0, 0.5, 0.99, 1.0, 0.99, 0.5, 0.0])
    step_ms = 5.0

    result = peak_halfwidth_ms(r_values, step_ms=step_ms)

    assert result == pytest.approx(3 * step_ms)


def test_peak_halfwidth_ms_ignores_nan_entries():
    # Same plateau as above, but padded with NaN on both sides (as a real
    # sweep would have where too few frames overlap) -- must not count
    # towards the width, and must not break locating r_max.
    r_values = np.array([np.nan, np.nan, 0.0, 0.99, 1.0, 0.99, 0.0, np.nan])
    step_ms = 5.0

    result = peak_halfwidth_ms(r_values, step_ms=step_ms)

    assert result == pytest.approx(3 * step_ms)


def test_peak_halfwidth_ms_single_point_peak():
    r_values = np.array([0.0, 0.5, 0.9, 1.0, 0.9, 0.5, 0.0])
    step_ms = 5.0

    result = peak_halfwidth_ms(r_values, step_ms=step_ms)

    assert result == pytest.approx(1 * step_ms)


def test_peak_halfwidth_ms_all_nan_returns_nan():
    r_values = np.full(5, np.nan)

    result = peak_halfwidth_ms(r_values, step_ms=5.0)

    assert np.isnan(result)


# --------------------------------------------------------------------------
# 2. peak_prominence_z -- flat curve vs sharp-peaked curve
# --------------------------------------------------------------------------

def test_peak_prominence_z_sharp_peak_beats_flat_curve():
    flat = np.array([0.10, 0.11, 0.09, 0.10, 0.12, 0.10, 0.11, 0.09, 0.10, 0.10])
    sharp = np.array([0.10, 0.10, 0.10, 0.10, 5.00, 0.10, 0.10, 0.10, 0.10, 0.10])

    z_flat = peak_prominence_z(flat)
    z_sharp = peak_prominence_z(sharp)

    assert np.isfinite(z_flat)
    assert np.isfinite(z_sharp)
    assert z_sharp > z_flat


def test_peak_prominence_z_ignores_nan_entries():
    r_values = np.array([np.nan, 0.10, 0.10, 5.00, 0.10, 0.10, np.nan])
    r_values_no_nan = np.array([0.10, 0.10, 5.00, 0.10, 0.10])

    assert peak_prominence_z(r_values) == pytest.approx(peak_prominence_z(r_values_no_nan))


def test_peak_prominence_z_constant_curve_is_nan():
    # SD == 0 -- prominence is undefined, not zero.
    r_values = np.full(6, 0.42)

    assert np.isnan(peak_prominence_z(r_values))


# --------------------------------------------------------------------------
# 3. Proxy functions -- one value per frame, respect autofilled masking
# --------------------------------------------------------------------------

# 3 frames x 5 markers.
#   frame 0: marker 4 autofilled, [1,2,3,4] real         -> normal masking
#   frame 1: all markers autofilled                       -> guard fallback
#   frame 2: only marker 0 (=1.0) not autofilled           -> single survivor
MAG = np.array([
    [1.0, 2.0, 3.0, 4.0, 5.0],
    [10.0, 20.0, 30.0, 40.0, 50.0],
    [1.0, 1.0, 1.0, 1.0, 1.0],
])
AUTOFILLED = np.array([
    [False, False, False, False, True],
    [True, True, True, True, True],
    [False, True, True, True, True],
])
CTX: dict = {"marker_rest_median": None}


@pytest.mark.parametrize("proxy_fn,expected", [
    (_proxy_max, [4.0, 50.0, 1.0]),
    (_proxy_median, [2.5, 30.0, 1.0]),
    (_proxy_mean, [2.5, 30.0, 1.0]),
])
def test_proxy_masks_autofilled_and_falls_back_when_all_autofilled(proxy_fn, expected):
    result = proxy_fn(MAG, AUTOFILLED, CTX)

    assert result.shape == (3,)
    assert result == pytest.approx(expected)


def test_proxy_p95_masks_and_falls_back():
    result = _proxy_p95(MAG, AUTOFILLED, CTX)

    assert result.shape == (3,)
    # frame 0: p95 of [1,2,3,4]; frame 1 (guard): p95 of full row; frame 2: p95 of [1.0]
    assert result[0] == pytest.approx(np.percentile([1.0, 2.0, 3.0, 4.0], 95))
    assert result[1] == pytest.approx(np.percentile(MAG[1], 95))
    assert result[2] == pytest.approx(1.0)


def test_proxy_top10_mean_uses_fewer_than_ten_when_unavailable():
    result = _proxy_top10_mean(MAG, AUTOFILLED, CTX)

    assert result.shape == (3,)
    # Fewer than 10 markers exist at all here, so top10_mean degenerates to
    # plain mean-of-non-autofilled, same as _proxy_mean.
    assert result == pytest.approx([2.5, 30.0, 1.0])


def test_top10_mean_picks_only_the_ten_largest():
    mag = np.arange(20.0).reshape(1, 20)  # values 0..19, all real
    autofilled = np.zeros((1, 20), dtype=bool)

    result = _proxy_top10_mean(mag, autofilled, CTX)

    # Ten largest are 10..19 -> mean 14.5.
    assert result[0] == pytest.approx(14.5)


def test_all_registered_proxies_return_one_value_per_frame():
    ctx = {"marker_rest_median": np.zeros(5)}
    for name, fn in PROXIES.items():
        result = fn(MAG, AUTOFILLED, ctx)
        assert result.shape == (3,), f"{name} did not return one value per frame"


# --------------------------------------------------------------------------
# 4. max_reref -- rest-median re-referencing and its edge cases
# --------------------------------------------------------------------------

def test_compute_marker_rest_median_basic():
    # 4 frames, 3 markers. Frames 0,1 are "rest".
    mag = np.array([
        [10.0, 20.0, 30.0],
        [12.0, 18.0, 32.0],
        [100.0, 100.0, 100.0],
        [100.0, 100.0, 100.0],
    ])
    autofilled = np.zeros((4, 3), dtype=bool)
    rest_mask = np.array([True, True, False, False])

    result = compute_marker_rest_median(mag, autofilled, rest_mask)

    assert result is not None
    marker_median, n_always_af = result
    assert marker_median == pytest.approx([11.0, 19.0, 31.0])
    assert n_always_af == 0


def test_compute_marker_rest_median_no_rest_frames_returns_none():
    mag = np.zeros((4, 3))
    autofilled = np.zeros((4, 3), dtype=bool)
    rest_mask = np.zeros(4, dtype=bool)

    assert compute_marker_rest_median(mag, autofilled, rest_mask) is None


def test_compute_marker_rest_median_marker_always_autofilled_at_rest():
    # Marker 1 is autofilled on every rest frame -> its median falls back
    # to 0.0, and it is counted in n_always_af.
    mag = np.array([
        [10.0, 999.0, 30.0],
        [12.0, 999.0, 32.0],
    ])
    autofilled = np.array([
        [False, True, False],
        [False, True, False],
    ])
    rest_mask = np.array([True, True])

    marker_median, n_always_af = compute_marker_rest_median(mag, autofilled, rest_mask)

    assert marker_median == pytest.approx([11.0, 0.0, 31.0])
    assert n_always_af == 1


def test_proxy_max_reref_subtracts_rest_median_then_takes_max():
    mag = np.array([
        [10.0, 20.0, 30.0],   # rest frame
        [10.0, 20.0, 30.0],   # rest frame (rest median = [10,20,30])
        [15.0, 20.0, 60.0],   # grip frame: reref = [5, 0, 30] -> max 30
    ])
    autofilled = np.zeros((3, 3), dtype=bool)
    rest_mask = np.array([True, True, False])
    marker_median, _ = compute_marker_rest_median(mag, autofilled, rest_mask)
    ctx = {"marker_rest_median": marker_median}

    result = _proxy_max_reref(mag, autofilled, ctx)

    assert result[2] == pytest.approx(30.0)
    # Rest frames re-reference to exactly zero displacement from themselves.
    assert result[0] == pytest.approx(0.0)
    assert result[1] == pytest.approx(0.0)


def test_proxy_max_reref_is_nan_when_no_rest_frames_available():
    mag = np.array([[10.0, 20.0, 30.0], [15.0, 20.0, 60.0]])
    autofilled = np.zeros((2, 3), dtype=bool)
    ctx = {"marker_rest_median": None}

    result = _proxy_max_reref(mag, autofilled, ctx)

    assert np.all(np.isnan(result))


def test_proxy_max_reref_respects_all_autofilled_guard():
    # Frame 0: all autofilled -> guard falls back to reref.max() over the
    # full (unmasked) row, not -inf.
    mag = np.array([[10.0, 20.0, 30.0]])
    autofilled = np.array([[True, True, True]])
    ctx = {"marker_rest_median": np.array([5.0, 5.0, 5.0])}

    result = _proxy_max_reref(mag, autofilled, ctx)

    # reref = [5, 15, 25] -> guard fallback max = 25
    assert result[0] == pytest.approx(25.0)
