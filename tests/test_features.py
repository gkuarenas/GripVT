"""Tests for core/features.py -- the derived channels that survived notebook 01.

Expected values are hand-computed on small synthetic grids, not captured from a
previous run of the code under test. A test that asserts "whatever it did last time"
cannot catch a sign flip or an off-by-one in the edge fallback, which are exactly the
two failure modes here: both produce plausible arrays of the right shape.

Edge cases required by docs/logs/20260806_162640_feature_engineering_plan.md section 6:
first frame of a session, trial boundary, magnitude == 0, and the last row/column
one-sided fallback.
"""
import subprocess
import sys

import numpy as np
import pytest

from core.features import max_disp, strain, to_grid, velocity


# --- to_grid ----------------------------------------------------------------

def _shuffled_layout(n_rows, n_cols):
    """A layout where marker_id is deliberately NOT row * n_cols + col.

    grid_layout.json is the authority on marker positions, and the code must read it
    rather than assume the formula. Reversing the order makes any hidden assumption
    fail loudly instead of passing by coincidence.
    """
    return np.array(
        [(r, c) for r in range(n_rows) for c in range(n_cols)][::-1], dtype=int
    )


def test_to_grid_places_each_marker_at_its_layout_position():
    layout = _shuffled_layout(2, 3)          # marker 0 -> (1,2), marker 5 -> (0,0)
    values = np.array([[10.0, 11.0, 12.0, 13.0, 14.0, 15.0]])

    grid = to_grid(values, layout)

    assert grid.shape == (1, 2, 3)
    assert grid[0, 1, 2] == 10.0             # marker 0
    assert grid[0, 0, 0] == 15.0             # marker 5
    np.testing.assert_array_equal(grid[0], [[15.0, 14.0, 13.0], [12.0, 11.0, 10.0]])


def test_to_grid_keeps_frames_independent():
    layout = np.array([[0, 0], [0, 1]], dtype=int)
    values = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

    grid = to_grid(values, layout)

    assert grid.shape == (3, 1, 2)
    np.testing.assert_array_equal(grid[:, 0, 0], [1.0, 3.0, 5.0])


def test_to_grid_rejects_a_layout_that_leaves_a_hole():
    # Two markers claim (0, 0), so (0, 1) is never written. Silently returning NaN
    # here would misplace every downstream number.
    layout = np.array([[0, 0], [0, 0], [1, 1]], dtype=int)
    values = np.zeros((1, 3))

    with pytest.raises(ValueError, match='every grid cell'):
        to_grid(values, layout)


def test_to_grid_rejects_marker_count_mismatch():
    with pytest.raises(ValueError, match='markers'):
        to_grid(np.zeros((1, 4)), np.array([[0, 0], [0, 1]], dtype=int))


# --- max_disp ---------------------------------------------------------------

def test_max_disp_takes_the_largest_marker_per_frame():
    mag = np.array([
        [[1.0, 7.0], [3.0, 2.0]],            # frame 0 -> 7.0
        [[0.5, 0.25], [0.75, 0.1]],          # frame 1 -> 0.75
    ])

    np.testing.assert_allclose(max_disp(mag), [7.0, 0.75])


def test_max_disp_is_zero_when_no_marker_moved():
    # magnitude == 0 across the grid. Nothing divides by it now that sin/cos are cut,
    # so this must simply return 0.0 rather than warn or produce NaN.
    assert max_disp(np.zeros((2, 11, 9))).tolist() == [0.0, 0.0]


# --- strain -----------------------------------------------------------------

def test_strain_x_is_the_right_neighbour_minus_this_marker():
    # dx increases by 2 along each row, so every real column difference is +2.
    dx = np.array([[[0.0, 2.0, 4.0], [10.0, 12.0, 14.0]]])
    dy = np.zeros_like(dx)

    sx, _ = strain(dx, dy)

    assert sx.shape == dx.shape
    np.testing.assert_allclose(sx[0], [[2.0, 2.0, 2.0], [2.0, 2.0, 2.0]])


def test_strain_y_is_the_marker_below_minus_this_marker():
    # dy increases by 3 down each column.
    dy = np.array([[[0.0, 1.0], [3.0, 4.0], [6.0, 7.0]]])
    dx = np.zeros_like(dy)

    _, sy = strain(dx, dy)

    assert sy.shape == dy.shape
    np.testing.assert_allclose(sy[0], [[3.0, 3.0], [3.0, 3.0], [3.0, 3.0]])


def test_strain_sign_follows_the_neighbour_not_this_marker():
    # A single left-moving marker at column 0: the step from it to its neighbour is
    # positive. Getting this backwards would flip every strain map in the dataset.
    dx = np.array([[[-5.0, 0.0, 0.0]]])
    sx, _ = strain(dx, np.zeros_like(dx))

    assert sx[0, 0, 0] == pytest.approx(5.0)
    assert sx[0, 0, 1] == pytest.approx(0.0)


def test_strain_edge_fallback_copies_the_last_real_difference():
    # Non-uniform on purpose: the last real column step is 4 - 1 = 3, and the padded
    # final column must be that same 3, not 0 and not a wrapped value.
    dx = np.array([[[0.0, 1.0, 4.0]]])
    dy = np.array([[[0.0], [1.0], [5.0]]])

    sx, _ = strain(dx, np.zeros_like(dx))
    _, sy = strain(np.zeros_like(dy), dy)

    np.testing.assert_allclose(sx[0, 0], [1.0, 3.0, 3.0])
    assert sx[0, 0, -1] == sx[0, 0, -2]
    np.testing.assert_allclose(sy[0, :, 0], [1.0, 4.0, 4.0])
    assert sy[0, -1, 0] == sy[0, -2, 0]


def test_strain_of_a_rigid_shift_is_zero():
    # The whole sleeve translating by a constant is the between-mount baseline offset.
    # Strain must not see it at all -- that is why the channel exists.
    dx = np.full((1, 3, 3), 7.0)
    dy = np.full((1, 3, 3), -4.0)

    sx, sy = strain(dx, dy)

    np.testing.assert_allclose(sx, 0.0)
    np.testing.assert_allclose(sy, 0.0)


# --- velocity ---------------------------------------------------------------

def test_velocity_uses_real_elapsed_time():
    # 1 px over 0.5 s, then 3 px over 0.25 s. Capture is ~16 fps but the interval is
    # not constant, so a nominal frame rate would give the wrong answer here.
    dx = np.array([[[0.0]], [[1.0]], [[4.0]]])
    ts = np.array([1000.0, 1500.0, 1750.0])
    trial = np.array([0, 0, 0])

    vx, _ = velocity(dx, np.zeros_like(dx), ts, trial)

    assert np.isnan(vx[0, 0, 0])
    assert vx[1, 0, 0] == pytest.approx(2.0)      # 1 px / 0.5 s
    assert vx[2, 0, 0] == pytest.approx(12.0)     # 3 px / 0.25 s


def test_velocity_first_frame_of_the_session_is_nan_not_zero():
    dx = np.array([[[0.0]], [[5.0]]])
    vx, vy = velocity(dx, dx, np.array([0.0, 100.0]), np.array([0, 0]))

    assert np.isnan(vx[0]).all()
    assert np.isnan(vy[0]).all()
    assert not np.isnan(vx[1]).any()


def test_velocity_is_nan_across_a_trial_boundary():
    # Frame 2 starts a new trial. Its displacement jump spans the inter-trial pause,
    # so it measures the operator's key presses, not the participant's muscle.
    dx = np.array([[[0.0]], [[1.0]], [[9.0]], [[10.0]]])
    ts = np.array([0.0, 62.0, 124.0, 186.0])
    trial = np.array([0, 0, 1, 1])

    vx, vy = velocity(dx, dx, ts, trial)

    assert np.isnan(vx[2]).all()
    assert np.isnan(vy[2]).all()
    assert not np.isnan(vx[3]).any()


def test_velocity_boundary_masking_is_not_an_outlier_filter():
    # The pause makes dt LARGER, and dt is the denominator, so an unmasked boundary
    # velocity comes out low -- it would never be caught by an outlier rule. Here the
    # unmasked value would be a very ordinary 1.0 px/s.
    dx = np.array([[[0.0]], [[1.0]], [[2.0]]])
    ts = np.array([0.0, 62.0, 1062.0])           # 1 s pause before the new trial
    trial = np.array([0, 0, 1])

    vx, _ = velocity(dx, np.zeros_like(dx), ts, trial)

    assert np.isnan(vx[2]).all()


def test_velocity_handles_a_one_frame_trial():
    # session2 trial 13 is a single frame: the operator advanced twice in ~63 ms and
    # the trial never got a grip_start cue. Two boundaries land back to back, so both
    # frame 1 and frame 2 must be NaN and nothing may crash.
    dx = np.array([[[0.0]], [[1.0]], [[2.0]], [[3.0]]])
    ts = np.array([0.0, 62.0, 125.0, 187.0])
    trial = np.array([12, 13, 14, 14])

    vx, _ = velocity(dx, np.zeros_like(dx), ts, trial)

    assert np.isnan(vx[1]).all()
    assert np.isnan(vx[2]).all()
    assert not np.isnan(vx[3]).any()


def test_velocity_rejects_mismatched_metadata_length():
    dx = np.zeros((3, 1, 1))
    with pytest.raises(ValueError, match='frames'):
        velocity(dx, dx, np.array([0.0, 1.0]), np.array([0, 0, 0]))


def test_velocity_accepts_non_integer_trial_ids():
    # trial_id arrives from a CSV and may parse as float or object; comparison-based
    # boundary detection must not care which.
    dx = np.array([[[0.0]], [[1.0]], [[2.0]]])
    ts = np.array([0.0, 62.0, 124.0])

    vx, _ = velocity(dx, np.zeros_like(dx), ts, np.array([0.0, 0.0, 1.0]))

    assert np.isnan(vx[2]).all()


# --- runtime safety ---------------------------------------------------------

def test_module_imports_without_pandas():
    """core/ is imported by main.py on a collection machine, where pandas is absent.

    Run in a subprocess with pandas blocked at the import hook, so this catches a
    transitive import too -- checking sys.modules in-process would not, since the test
    session itself has already imported pandas.
    """
    code = (
        'import sys\n'
        'class Block:\n'
        '    def find_module(self, name, path=None):\n'
        '        if name == "pandas" or name.startswith("pandas."):\n'
        '            raise ImportError("pandas is blocked for this test")\n'
        'sys.meta_path.insert(0, Block())\n'
        'import core.features\n'
    )
    result = subprocess.run(
        [sys.executable, '-c', code], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
