import time

import numpy as np
import pytest

import core.detector_search as ds
from core.detector import default_params
from core.detector_search import (
    CORNER_LINES,
    GRID_SHAPE,
    MIN_ACCEPTABLE,
    POSE_KEYS,
    TARGET_MARKERS,
    AnchorError,
    _group_by_lines,
    clamp_manual_params,
    geometry_from,
    grid_lines,
    grid_positions,
    pose_cost,
    region_counts,
    rescore,
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


def curved_grid_points(bow_px=20.0, theta_span_deg=55.0, radius=250.0):
    """A cylindrical sensor's grid: rows bow in x, and row pitch compresses toward
    the top/bottom edges. bow_px=20 against a ~28px compressed edge pitch is the
    real geometry that made a global y-projection collapse 11 rows into 9."""
    thetas = np.radians(np.linspace(-theta_span_deg, theta_span_deg, ROWS))
    xs = np.linspace(0, PITCH_X * (COLS - 1), COLS)
    half = (xs[-1] - xs[0]) / 2
    return [
        (
            float(MARGIN + x),
            float(400 + radius * np.sin(th) + bow_px * ((x - half) / half) ** 2),
            30.0,
        )
        for th in thetas
        for x in xs
    ]


# ── region counts, both orientations and on a curved surface ───────────────
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


@pytest.mark.parametrize("bow", [0.0, 10.0, 20.0, 30.0])
def test_geometry_survives_a_bowed_compressed_grid(bow):
    """Regression: a global y-projection merged the two tightest row gaps here,
    reporting 9x9 for a 9x11 grid and aborting calibration on the real rig."""
    points = curved_grid_points(bow_px=bow)
    geom = geometry_from(points)

    assert (geom.x_lines, geom.y_lines) == (COLS, ROWS)
    counts = region_counts(points, geom)
    assert sum(counts["bands"].values()) == TARGET_MARKERS
    assert all(v == CORNER_LINES**2 for v in counts["corners"].values()), counts["corners"]


def test_bowed_grid_still_localises_a_lost_corner():
    points = curved_grid_points(bow_px=20.0)
    geom = geometry_from(points)
    full = region_counts(points, geom)["corners"]
    without = region_counts(points[:-1], geom)["corners"]
    assert without["br"] == full["br"] - 1, (full, without)


def test_global_projection_would_have_failed_on_the_bowed_grid():
    """Pins the diagnosis: clustering y globally really does lose two rows, so
    the per-line geometry is doing the work, not an incidental change."""
    points = curved_grid_points(bow_px=20.0)
    assert len(grid_lines(np.array([p[0] for p in points]))) == COLS
    assert len(grid_lines(np.array([p[1] for p in points]))) == 9


@pytest.mark.parametrize(
    "label,points", [("flat", grid_points()), ("curved", curved_grid_points(20.0))]
)
def test_boundaries_clear_every_marker_by_a_third_of_a_pitch(label, points):
    geom = geometry_from(points)
    _lines, groups = _group_by_lines(points, geom.axis)
    for i, group in enumerate(groups):
        vals = sorted(group)
        local_pitch = float(min(np.diff(vals)))
        for bound in (*geom.corner[i], *geom.band[i]):
            clearance = min(abs(bound - v) for v in vals)
            assert clearance >= local_pitch / 3, (label, i, bound, clearance, local_pitch)


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


# ── grid_positions: row/col resolution for marker_id ───────────────────────
def test_grid_positions_flat_bijection_and_corners():
    points = grid_points()
    geom = geometry_from(points)
    positions = grid_positions(points, geom)

    assert len(positions) == TARGET_MARKERS
    assert set(positions) == {(r, c) for r in range(ROWS) for c in range(COLS)}

    top_left = min(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    bottom_right = max(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    assert positions[top_left] == (0, 0)
    assert positions[bottom_right] == (ROWS - 1, COLS - 1)


def test_grid_positions_axis_1_branch():
    """Force geom.axis == 1 (lines run along y, i.e. a line IS a row) by
    breaking x's global clusterability while leaving y (the row grouping)
    untouched. Neither fixture reaches this branch unperturbed -- axis 0
    always validates first for them (see geometry_from's try-both-axes loop)
    -- so this merges two rows' x-offsets to push the x-line count to 10
    (outside GRID_SHAPE), forcing the fallback to axis 1. The per-row x order
    is preserved (the offset is constant within a row), so the resolved
    positions must match the unperturbed grid exactly.
    """
    points = grid_points()
    perturbed = [
        (x + min(i // COLS, ROWS - 2) * 10_000.0, y, s)
        for i, (x, y, s) in enumerate(points)
    ]
    geom = geometry_from(perturbed)
    assert geom.axis == 1

    positions = grid_positions(perturbed, geom)
    assert positions == grid_positions(points, geometry_from(points))


@pytest.mark.parametrize("bow", [20.0, 30.0])
def test_grid_positions_survives_a_bowed_grid_and_beats_the_naive_sort(bow):
    """Regression for the on-rig Mode A bug (core/tracker.py's old (y, x)
    sort): on a curved grid, per-row y bow means a flat sort scrambles
    within-row order. grid_positions must recover the true left-to-right
    order that the naive sort gets wrong."""
    points = curved_grid_points(bow_px=bow)
    geom = geometry_from(points)
    positions = grid_positions(points, geom)

    assert len(set(positions)) == TARGET_MARKERS
    assert set(positions) == {(r, c) for r in range(ROWS) for c in range(COLS)}

    naive_order = sorted(range(len(points)), key=lambda i: (points[i][1], points[i][0]))
    grid_order = sorted(range(len(points)), key=lambda i: positions[i])
    assert naive_order != grid_order

    true_top_left = min(range(COLS), key=lambda i: points[i][0])
    assert positions[true_top_left] == (0, 0)
    assert naive_order.index(true_top_left) != 0


def test_grid_positions_raises_on_dropped_marker():
    points = grid_points()[:-1]
    geom = geometry_from(points)
    with pytest.raises(AnchorError):
        grid_positions(points, geom)


def test_grid_positions_raises_on_duplicate_marker():
    points = grid_points() + [grid_points()[0]]
    geom = geometry_from(points)
    with pytest.raises(AnchorError):
        grid_positions(points, geom)


def test_grid_positions_raises_on_wrong_grid_shape():
    """An 11x9 mount (rotated ~90 degrees from the expected 9x11) must not be
    silently transposed into a plausible-looking layout (D8)."""
    points = grid_points(cols=11, rows=9)
    geom = geometry_from(points)
    with pytest.raises(AnchorError):
        grid_positions(points, geom)


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


# ── exactly-99 pass rule (Phase 3, D8/D9) ──────────────────────────────────
def test_search_fails_when_one_pose_overcounts():
    """The evidence for D8: the old `worst_count >= MIN_ACCEPTABLE` rule looks
    at the MINIMUM count across poses, so an over-detecting max_grip (100)
    alongside two clean poses (99) would still have worst_count == 99 and
    pass. The new all()-equal-target rule must catch it instead."""
    points = grid_points()
    extra = (20.0, 20.0, 30.0)  # margin corner, far from every real marker
    frames = {
        "relaxed": render(points),
        "full_open": render(points),
        "max_grip": render(points + [extra]),
    }
    params = base_params()

    result = search(frames, params)

    assert result.passed is False
    assert result.worst_count == TARGET_MARKERS
    assert result.worst_pose == "max_grip"


def test_search_passes_once_a_mask_hole_covers_the_extra_blob():
    """Ties Phase 2 (mask_holes) and Phase 3 (the exactly-99 gate) together:
    the same over-counting max_grip frame passes once the extra blob is
    masked out before detection reaches it."""
    points = grid_points()
    extra = (20.0, 20.0, 30.0)
    frames = {
        "relaxed": render(points),
        "full_open": render(points),
        "max_grip": render(points + [extra]),
    }
    params = base_params()
    mask = np.full((FRAME_H, FRAME_W), 255, dtype=np.uint8)
    mask[0:40, 0:40] = 0  # covers the extra blob only, no real marker sits here

    result = search(frames, params, mask)

    assert result.passed is True
    assert result.worst_count == TARGET_MARKERS


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


# ── Phase 4: manual thresh/min_area override backend ───────────────────────
def test_clamp_manual_params_boundaries():
    assert clamp_manual_params(10, 1.0) == (30, 4.0)      # both below range
    assert clamp_manual_params(45, 20.0) == (45, 20.0)    # both inside range
    assert clamp_manual_params(90, 100.0) == (60, 40.0)   # both above range
    assert clamp_manual_params(45.6, 20.3) == (46, 20.3)  # float thresh becomes an int
    assert isinstance(clamp_manual_params(45, 20)[0], int)
    assert isinstance(clamp_manual_params(45, 20)[1], float)


def _add_gaussian_blob(
    frame: np.ndarray, x: float, y: float, amp: float, radius: float, background: int = 30,
) -> np.ndarray:
    yy, xx = np.mgrid[0 : frame.shape[0], 0 : frame.shape[1]]
    bump = amp * np.exp(-(((xx - x) ** 2) / (2 * radius**2) + ((yy - y) ** 2) / (2 * radius**2)))
    return np.clip(frame.astype(np.float64) + bump, 0, 255).astype(np.uint8)


def test_rescore_recovers_from_an_overcount_the_automatic_sweep_cannot_reach(monkeypatch):
    """D15's own justification, proven end to end rather than argued from
    principle. Coarsen the sweep's own step so its candidates never include
    thresh=59 (still inside the real [30, 60] bound `clamp_manual_params`
    enforces, but only reachable at the manual step of 1), then show a
    manual rescore at that exact thresh clears an over-count no automatic
    candidate can reach."""
    monkeypatch.setattr(ds, "THRESH_RANGE", (30, 60, 7))
    monkeypatch.setattr(ds, "MIN_AREA_RANGE", (4, 40, 37))

    points = grid_points()
    relaxed = render(points)
    full_open = render(points)
    # amp=55 (peak 85 over background 30) at radius=2.5: empirically present
    # at every candidate the coarsened sweep tests (30, 37, 44, 51, 58) and
    # gone by thresh=59 -- verified against the real detect pipeline, not
    # hand-derived.
    max_grip = _add_gaussian_blob(render(points), 20.0, 20.0, amp=55.0, radius=2.5)
    frames = {"relaxed": relaxed, "full_open": full_open, "max_grip": max_grip}
    params = base_params(thresh=30, min_area=4.0)

    automatic = search(frames, params)
    assert automatic.passed is False
    assert automatic.worst_pose == "max_grip"
    assert automatic.poses["max_grip"].n_detected == TARGET_MARKERS + 1

    manual = rescore(
        frames, params, None, automatic.geometry, automatic.reference_corners,
        thresh=59, min_area=4.0, reference_bands=automatic.reference_bands,
    )
    assert manual.passed is True
    assert all(manual.poses[k].n_detected == TARGET_MARKERS for k in POSE_KEYS)


def test_rescore_reuses_the_given_geometry_without_rederiving(monkeypatch):
    """D14: a manual re-score must never re-derive geometry, even when the
    frames it is scoring differ from the frames the passed-in geometry came
    from -- proven with a spy rather than by reading the source."""
    frames_a, _points_a = build_pose_frames(set())
    frames_b, _points_b = build_pose_frames({bottom_right_index()})
    params = base_params()

    reference = search(frames_a, params)

    calls: list[int] = []
    original = ds.geometry_from

    def spy(*args, **kwargs):
        calls.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(ds, "geometry_from", spy)

    rescore(
        frames_b, params, None, reference.geometry, reference.reference_corners,
        thresh=44, min_area=20.0,
    )
    assert calls == []


def test_search_budget(capsys):
    frames, _points = build_pose_frames({bottom_right_index()})
    start = time.perf_counter()
    search(frames, base_params())
    elapsed = time.perf_counter() - start
    with capsys.disabled():
        print(f"\n  full 3-pose search: {elapsed:.2f}s on {FRAME_W}x{FRAME_H}")
    assert elapsed < 60
