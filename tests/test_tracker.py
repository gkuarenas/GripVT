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

_BLOB_CENTERS = [(80, 60), (220, 60), (80, 180), (220, 180)]
_FRAME_SIZE = (240, 300, 3)  # (h, w, c) -- arbitrary, since undistort is identity here


def _synthetic_frame() -> np.ndarray:
    frame = np.full(_FRAME_SIZE, 30, dtype=np.uint8)
    for cx, cy in _BLOB_CENTERS:
        cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1)
    return frame


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
