import time

import numpy as np
import pytest

from core.detector import default_params
from core.detector_search import (
    CORNER_LINES,
    GRID_SHAPE,
    MIN_ACCEPTABLE,
    POSE_KEYS,
    TARGET_MARKERS,
    AnchorError,
    geometry_from,
    grid_lines,
    pose_cost,
    region_counts,
    search,
)

COLS, ROWS = GRID_SHAPE
MARGIN = 60
PITCH_X = 50
PITCH_Y = 45
FRAME_W = MARGIN * 2 + PITCH_X * (COLS - 1)
FRAME_H = MARGIN * 2 + PITCH_Y * (ROWS - 1)


def grid_points(cols: int = COLS, rows: int = ROWS) -> list[tuple[float, float, float]]:
    return [
        (float(MARGIN + c * PITCH_X), float(MARGIN + r * PITCH_Y), 30.0)
        for r in range(rows)
        for c in range(cols)
    ]


def render(points, dim=(), shrink=(), radius=5, peak=210, background=30):
    """Grey frame with a Gaussian blob per point. `dim`/`shrink` hold indices
    rendered as the faint, smaller, elliptical blob an oblique edge marker
    produces -- the exact appearance that falls under thresh/min_area."""
    frame = np.full((FRAME_H, FRAME_W), background, dtype=np.float64)
    yy, xx = np.mgrid[0:FRAME_H, 0:FRAME_W]
    for i, (x, y, _size) in enumerate(points):
        amp = peak * (0.32 if i in dim else 1.0)
        rx = radius * (0.5 if i in shrink else 1.0)
        ry = radius * (0.85 if i in shrink else 1.0)
        frame += amp * np.exp(-(((xx - x) ** 2) / (2 * rx**2) + ((yy - y) ** 2) / (2 * ry**2)))
    return np.clip(frame, 0, 255).astype(np.uint8)


def base_params(thresh=90, min_area=34.0):
    return dict(default_params(), thresh=thresh, min_area=min_area, size_method="cc")


def bottom_right_index() -> int:
    return len(grid_points()) - 1


# ── grid line detection ────────────────────────────────────────────────────
def test_grid_lines_recovers_line_counts():
    points = grid_points()
    assert len(grid_lines(np.array([p[0] for p in points]))) == COLS
    assert len(grid_lines(np.array([p[1] for p in points]))) == ROWS


def test_grid_lines_survives_jitter():
    rng = np.random.default_rng(0)
    points = grid_points()
    xs = np.array([p[0] for p in points]) + rng.uniform(-5, 5, len(points))
    ys = np.array([p[1] for p in points]) + rng.uniform(-5, 5, len(points))
    assert len(grid_lines(xs)) == COLS
    assert len(grid_lines(ys)) == ROWS


def test_grid_lines_degenerate_inputs():
    assert grid_lines(np.array([])) == []
    assert grid_lines(np.array([7.0])) == [7.0]


# ── region counts, both orientations ───────────────────────────────────────
@pytest.mark.parametrize("transpose", [False, True])
def test_region_counts_is_orientation_agnostic(transpose):
    points = grid_points()
    if transpose:
        points = [(y, x, s) for x, y, s in points]
    geom = geometry_from(points)
    counts = region_counts(points, geom)

    assert sum(counts["bands"].values()) == TARGET_MARKERS
    assert all(v == CORNER_LINES**2 for v in counts["corners"].values()), counts["corners"]
    assert sorted((geom.x_lines, geom.y_lines)) == sorted(GRID_SHAPE)


def test_boundaries_clear_every_marker_by_a_third_of_a_pitch():
    points = grid_points()
    geom = geometry_from(points)
    xs = sorted({p[0] for p in points})
    ys = sorted({p[1] for p in points})
    for bound in geom.x_corner:
        assert min(abs(bound - v) for v in xs) >= PITCH_X / 3
    for bound in list(geom.y_corner) + list(geom.y_band):
        assert min(abs(bound - v) for v in ys) >= PITCH_Y / 3


def test_losing_the_bottom_right_marker_shows_in_that_cell():
    points = grid_points()
    geom = geometry_from(points)
    full = region_counts(points, geom)["corners"]
    without = region_counts(points[:-1], geom)["corners"]
    assert without["br"] == full["br"] - 1
    assert {k: without[k] for k in ("tl", "tr", "bl")} == {
        k: full[k] for k in ("tl", "tr", "bl")
    }


def test_geometry_from_rejects_too_few_lines():
    with pytest.raises(AnchorError):
        geometry_from(grid_points(cols=2, rows=2))


# ── cost function ──────────────────────────────────────────────────────────
def test_pose_cost_weights_drops_and_corners_above_false_positives():
    ref = {"tl": 9, "tr": 9, "bl": 9, "br": 9}
    perfect = pose_cost(99, 0, ref, ref)
    one_short = pose_cost(98, 0, dict(ref, br=8), ref)
    one_extra = pose_cost(100, 0, ref, ref)
    assert perfect == 0.0
    assert one_short > one_extra > perfect


# ── end-to-end search on synthetic three-pose data ─────────────────────────
def build_pose_frames(dim_indices):
    points = grid_points()
    return {
        "relaxed": render(points),
        "full_open": render(points, dim=dim_indices, shrink=dim_indices),
        "max_grip": render(points),
    }, points


def test_search_recovers_the_dim_corner_marker():
    corner = bottom_right_index()
    frames, points = build_pose_frames({corner})
    params = base_params()

    result = search(frames, params)

    assert result.grid_shape in (GRID_SHAPE, GRID_SHAPE[::-1])
    for key in POSE_KEYS:
        diag = result.poses[key]
        assert diag.n_detected == TARGET_MARKERS, (key, diag.n_detected)
        assert diag.oversized == 0, (key, diag.oversized)
        assert diag.corners["br"] == CORNER_LINES**2, (key, diag.corners)
    assert result.passed
    assert result.worst_count == TARGET_MARKERS


def test_seed_params_alone_would_have_dropped_that_marker():
    """Guards the test above from passing trivially: the seed must actually fail."""
    corner = bottom_right_index()
    frames, _points = build_pose_frames({corner})
    params = base_params()

    from core.detector_search import _evaluate, _prepare

    prepared = _prepare(frames["full_open"], params, None)
    dets, _oversized, _fb = _evaluate(prepared, params)
    assert len(dets) == TARGET_MARKERS - 1


def test_search_never_regresses_from_a_bad_seed():
    corner = bottom_right_index()
    frames, _points = build_pose_frames({corner})
    bad_seed = base_params(thresh=200, min_area=400.0)

    result = search(frames, bad_seed)

    assert result.worst_count >= MIN_ACCEPTABLE
    assert result.params["thresh"] != bad_seed["thresh"]


def test_search_prefers_the_conservative_tie():
    frames, _points = build_pose_frames(set())
    params = base_params(thresh=50)

    result = search(frames, params)

    assert all(d.cost == 0.0 for d in result.poses.values())
    # Every zero-cost candidate ties, so the tie-break must pick the most
    # conservative one in range, not the first one visited.
    assert (result.params["thresh"], result.params["min_area"]) == (60, 40.0)


def test_search_rejects_an_unusable_anchor():
    """Too few markers physically present -- no candidate can reach the floor."""
    frames = {k: render(grid_points()[:40]) for k in POSE_KEYS}
    with pytest.raises(AnchorError):
        search(frames, base_params())


def test_search_requires_all_three_poses():
    frames, _points = build_pose_frames(set())
    del frames["max_grip"]
    with pytest.raises(ValueError):
        search(frames, base_params())


def test_search_budget(capsys):
    frames, _points = build_pose_frames({bottom_right_index()})
    start = time.perf_counter()
    search(frames, base_params())
    elapsed = time.perf_counter() - start
    with capsys.disabled():
        print(f"\n  full 3-pose search: {elapsed:.2f}s on {FRAME_W}x{FRAME_H}")
    assert elapsed < 60
