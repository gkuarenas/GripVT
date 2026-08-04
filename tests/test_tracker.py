"""Headless test for Phase 3 of
docs/logs/20260731_152249_framerate_spike_plan.md (D4/D5): Tracker.
process_frame() accepts an optional pre-undistorted frame so _poll() doesn't
have to undistort the full frame twice (once for the display, once inside
process_frame()). Passing it must produce identical MarkerRecord field
values to letting process_frame() undistort internally -- same input, same
precomputed maps, same output, just computed once instead of twice.

Tracker.undistort and ROICrop.crop are monkeypatched to identity at the
class level (same idiom as test_session_csv_recording.py's fixtures), so
this test is about the optional-argument plumbing, not the fisheye math or
a real ROI's crop geometry.
"""
from dataclasses import astuple

import cv2
import numpy as np
import pytest

from core.calibration import CameraCalibration
from core.detector import default_params
from core.roi import ROICrop
from core.tracker import Tracker
from tests.test_detector_search import curved_grid_points

_BLOB_CENTERS = [(80, 60), (220, 60), (80, 180), (220, 180)]
_FRAME_SIZE = (240, 300, 3)  # (h, w, c) -- arbitrary, since undistort is identity here


def _synthetic_frame() -> np.ndarray:
    frame = np.full(_FRAME_SIZE, 30, dtype=np.uint8)
    for cx, cy in _BLOB_CENTERS:
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
    return frame


def _render_gaussian_blobs(
    points: list[tuple[float, float, float]], frame_h: int, frame_w: int,
    radius: float = 5.0, peak: float = 210.0, background: float = 30.0,
) -> np.ndarray:
    frame = np.full((frame_h, frame_w), background, dtype=np.float64)
    yy, xx = np.mgrid[0:frame_h, 0:frame_w]
    for x, y, _area in points:
        frame += peak * np.exp(-(((xx - x) ** 2 + (yy - y) ** 2) / (2 * radius**2)))
    return np.clip(frame, 0, 255).astype(np.uint8)


def _bowed_grid_frame(bow_px: float = 20.0) -> tuple[np.ndarray, list[tuple[float, float, float]]]:
    """A 9x11 blob frame on a curved (bowed) grid, so the naive (y, x) sort and
    real grid geometry disagree on marker 0 -- the on-rig Mode A shape (see
    docs/logs/20260804_161615_marker_grid_lookup_plan.md §1). Reuses
    test_detector_search.curved_grid_points's point layout; sized locally
    since that module's render() canvas is sized for the flat grid only and
    is too short for the curved grid's y-extent."""
    points = curved_grid_points(bow_px=bow_px)
    pad = 40
    frame_w = int(max(x for x, _y, _a in points) + pad)
    frame_h = int(max(y for _x, y, _a in points) + pad)
    gray = _render_gaussian_blobs(points, frame_h, frame_w)
    frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    return frame, points


@pytest.fixture
def bowed_grid_tracker(monkeypatch):
    """A real 9x11 baseline capture through capture_baseline() itself (not
    grid_positions() directly) -- exercises the Phase 2 wiring in
    core/tracker.py, including detection and clustering, not just the
    geometry math."""
    monkeypatch.setattr(Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(ROICrop, "crop", lambda self, frame: frame)

    frame, points = _bowed_grid_frame()
    cal = CameraCalibration()
    # "cc" for the same reason tracker_pair uses it: these are rendered
    # Gaussian blobs, and "cc" keeps this test about the marker_id wiring,
    # not detector tuning.
    params = dict(default_params(), thresh=90, min_area=34.0, size_method="cc")

    tracker = Tracker(cal, ROICrop())
    tracker.params = params

    count = tracker.capture_baseline([frame, frame, frame])
    return tracker, points, count


def test_capture_baseline_resolves_geometry_and_places_marker_zero_top_left(bowed_grid_tracker):
    tracker, points, count = bowed_grid_tracker

    assert count == 99
    assert set(tracker.kalman.states.keys()) == set(range(99))
    assert tracker.grid_layout_source == "geometry"
    assert tracker.grid_layout is not None
    assert tracker.grid_layout[0] == (0, 0)

    # points[0] is curved_grid_points' true top-left marker (row 0, the
    # smallest-theta row; col 0, the smallest-x point within it). marker_id 0
    # must land there, not on whichever marker the old naive (y, x) sort
    # picked closest to the bowed top row's centre.
    roi = ROICrop()
    expected_xy = roi.to_full_frame(points[0][0], points[0][1])
    state0 = tracker.kalman.states[0]
    assert state0.baseline_pos == pytest.approx(expected_xy, abs=4.0)


def test_capture_baseline_grid_layout_keys_match_kalman_states(bowed_grid_tracker):
    tracker, _points, _count = bowed_grid_tracker
    assert set(tracker.grid_layout.keys()) == set(tracker.kalman.states.keys())


def test_capture_baseline_initialises_states_in_ascending_id_order(bowed_grid_tracker):
    tracker, _points, _count = bowed_grid_tracker
    ids = list(tracker.kalman.states)
    assert ids == sorted(ids)


def test_capture_baseline_fallback_path_does_not_raise(tracker_pair):
    """The 4-blob tracker_pair fixture can't possibly resolve a 9x11 grid --
    capture_baseline() must still lock every state, via the naive sort, and
    must not raise (D5). If it had raised, tracker_pair itself would have
    failed already; this pins the specific fallback attributes."""
    tracker_a, tracker_b, _frame = tracker_pair
    for tracker in (tracker_a, tracker_b):
        assert set(tracker.kalman.states.keys()) == set(range(len(_BLOB_CENTERS)))
        assert list(tracker.kalman.states) == sorted(tracker.kalman.states)
        assert tracker.grid_layout is None
        assert tracker.grid_layout_source == "fallback_sort"
        assert tracker.grid_layout_reason is not None


@pytest.fixture
def tracker_pair(monkeypatch):
    """Two independently-constructed Trackers, each locking an identical
    baseline from the same synthetic frame -- detection here is
    deterministic (no RNG), so both trackers lock the same states, and
    process_frame() output can only differ between the two call styles
    below if the undistorted-argument plumbing itself is wrong."""
    monkeypatch.setattr(Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(ROICrop, "crop", lambda self, frame: frame)

    cal = CameraCalibration()
    frame = _synthetic_frame()
    # "cc" (connected-component centroid) rather than the default "gaussian"
    # fit -- these are solid disks, not the Gaussian-profile markers the
    # log-parabola fit expects, so "cc" keeps this test about the plumbing,
    # not detector tuning.
    params = dict(default_params(), size_method="cc")

    tracker_a = Tracker(cal, ROICrop())
    tracker_b = Tracker(cal, ROICrop())
    tracker_a.params = params
    tracker_b.params = params

    count_a = tracker_a.capture_baseline([frame, frame, frame])
    count_b = tracker_b.capture_baseline([frame, frame, frame])
    assert count_a == count_b == len(_BLOB_CENTERS)

    return tracker_a, tracker_b, frame


def test_precomputed_undistort_matches_internal(tracker_pair):
    tracker_a, tracker_b, frame = tracker_pair

    records_internal = tracker_a.process_frame(frame)

    undistorted = tracker_b.undistort(frame)
    records_precomputed = tracker_b.process_frame(frame, undistorted)

    assert len(records_internal) == len(records_precomputed) == len(_BLOB_CENTERS)
    for internal, precomputed in zip(records_internal, records_precomputed):
        # np.testing.assert_equal (unlike ==) treats NaN == NaN as a match,
        # which size_moments legitimately is here (size_method="cc" leaves
        # it unset) -- a real difference in any other field still fails.
        np.testing.assert_equal(astuple(internal), astuple(precomputed))

    # Both call styles must still populate last_undistorted -- overlay
    # drawing depends on it regardless of which path filled it (D4).
    assert tracker_a.last_undistorted is not None
    assert tracker_b.last_undistorted is not None
