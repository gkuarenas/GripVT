"""Derived per-marker feature channels for the grip-force dataset.

The channels that survived the screening in `notebooks/01_feature_validation.ipynb`:
`max_disp`, `strain_x`/`strain_y`, and `vx`/`vy`. `sin_theta`/`cos_theta` were cut --
they add no information over `dx`/`dy` and are numerically corrupt at rest, where
3-decimal storage makes the direction of a small displacement the angle of a
rounding error. See the verdict cell of notebook 01.

Everything here is pure arithmetic: arrays in, arrays out, no file I/O, no pandas,
no CSV column knowledge. That is deliberate on three counts.

- `core/` is imported by `main.py` on collection day, where pandas is not installed
  (`requirements.txt` splits runtime deps from analysis-only ones).
- These functions are idempotent -- same input, same output, safe to run a thousand
  times -- unlike `tools/sync_session.py`, which rewrites labels on data that can
  never be re-collected. The two must not be welded together.
  See `docs/logs/20260808_132517_feature_scope_revision.md`.
- Loading a session, reading `grid_layout.json`, and writing `features_table.csv`
  live in the notebook 02 assembly layer, not here.

Grid convention: every grid is `(n_frames, n_rows, n_cols)` = `(n, 11, 9)` for the
99-marker GripVT sleeve. Units are pixels throughout, never mm.
"""
import numpy as np


def to_grid(values, layout):
    """Scatter per-marker columns onto the sensor's physical row/col grid.

    `values` is (n_frames, n_markers) in marker_id order; `layout` is (n_markers, 2)
    of (row, col), also in marker_id order, so `layout[i]` is where marker i sits on
    the sleeve. Returns (n_frames, n_rows, n_cols).

    The scatter is what makes a 2D CNN and the strain differences meaningful:
    marker_id order is `row * n_cols + col` today, but `grid_layout.json` is the
    authority and a session may not match the formula. Read the layout, never assume.
    """
    values = np.asarray(values, dtype=float)
    layout = np.asarray(layout, dtype=int)

    if values.ndim != 2:
        raise ValueError(f'values must be (n_frames, n_markers), got {values.shape}')
    if layout.ndim != 2 or layout.shape[1] != 2:
        raise ValueError(f'layout must be (n_markers, 2), got {layout.shape}')
    if values.shape[1] != len(layout):
        raise ValueError(
            f'values has {values.shape[1]} markers but layout has {len(layout)}'
        )

    n_rows, n_cols = layout.max(axis=0) + 1

    out = np.full((len(values), n_rows, n_cols), np.nan)
    out[:, layout[:, 0], layout[:, 1]] = values

    # A hole means two markers claimed one cell, or the grid is not fully covered.
    # Either way the layout is wrong and every downstream number would be silently
    # misplaced -- fail here rather than propagate NaN.
    if np.isnan(out).any():
        raise ValueError('layout does not cover every grid cell exactly once')

    return out


def max_disp(mag_grid):
    """Largest marker displacement in each frame, in px. Returns (n_frames,).

    One column for the whole grid. r = 0.817 (session 1) / 0.644 (session 8) against
    `force_kg_corr`, matching `sync_report.json`'s `xcorr.r_at_best` exactly. Kept
    because trees cannot easily express a max over 99 separate features, and it costs
    a single column.
    """
    return np.asarray(mag_grid, dtype=float).max(axis=(1, 2))


def strain(dx_grid, dy_grid):
    """Neighbour-relative displacement. Returns (strain_x, strain_y), input shapes.

    `strain_x[r, c] = dx[r, c+1] - dx[r, c]` -- how much more the marker to the right
    moved than this one; `strain_y` does the same down a column. This is the only
    route by which grid adjacency reaches a model that sees markers as an unordered
    bag (Deep Sets), and it suppresses the rigid whole-sleeve shift that makes rest
    look different between mounts (it closes 45.5% of the session 1 vs 8 gap).

    The last column of `strain_x` and last row of `strain_y` have no neighbour, so the
    final real difference is COPIED into them. Every marker keeps a value and the grid
    keeps its shape, at the cost of 11% of `strain_x` cells (9% of `strain_y`) being
    duplicates rather than estimates. Settled in the plan; documented in the verdict.
    """
    dx_grid = np.asarray(dx_grid, dtype=float)
    dy_grid = np.asarray(dy_grid, dtype=float)

    sx = np.diff(dx_grid, axis=2)                             # axis 2 = columns
    sx = np.concatenate([sx, sx[:, :, -1:]], axis=2)

    sy = np.diff(dy_grid, axis=1)                             # axis 1 = rows
    sy = np.concatenate([sy, sy[:, -1:, :]], axis=1)

    return sx, sy


def velocity(dx_grid, dy_grid, timestamp_ms, trial_id):
    """Per-marker displacement rate in px/s. Returns (vx, vy), input shapes.

    NaN at the first frame of the session and at every trial boundary: there is no
    valid earlier frame to difference against, and 0 would be read by a model as
    "this marker did not move".

    Boundaries must be masked explicitly -- they do NOT show up as outliers. The
    inter-trial pause makes `dt` 5-8x larger, and `dt` is in the denominator, so a
    boundary velocity comes out lower than a normal one (session 1: median |vx| 0.70
    px/s at boundaries vs 1.06 elsewhere). Nothing would flag them for you.

    Uses real elapsed time rather than a nominal frame rate: capture runs at ~16 fps
    with a median 62 ms interval, but the interval is not constant.
    """
    dx_grid = np.asarray(dx_grid, dtype=float)
    dy_grid = np.asarray(dy_grid, dtype=float)
    timestamp_ms = np.asarray(timestamp_ms, dtype=float)
    trial_id = np.asarray(trial_id)

    n = len(dx_grid)
    if len(timestamp_ms) != n or len(trial_id) != n:
        raise ValueError(
            f'grids have {n} frames but timestamp_ms has {len(timestamp_ms)} '
            f'and trial_id has {len(trial_id)}'
        )

    dt = np.diff(timestamp_ms) / 1000.0
    vx = np.diff(dx_grid, axis=0) / dt[:, None, None]
    vy = np.diff(dy_grid, axis=0) / dt[:, None, None]

    # np.diff drops one frame -- prepend NaN so frame i lines up across channels.
    pad = np.full((1,) + dx_grid.shape[1:], np.nan)
    vx = np.concatenate([pad, vx], axis=0)
    vy = np.concatenate([pad, vy], axis=0)

    # A velocity spanning a trial boundary measures the pause, not the muscle.
    boundaries = np.flatnonzero(trial_id[1:] != trial_id[:-1]) + 1
    vx[boundaries] = np.nan
    vy[boundaries] = np.nan

    return vx, vy
