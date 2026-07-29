"""Headless test for the Session window's Capture Baseline state machine.

Drives SessionWindow's baseline state transitions directly, with no camera
and no polling loop. The test controls the camera itself rather than
assuming one is absent -- cv2.VideoCapture is monkeypatched inside
ui.session.camera_feed to a stub whose isOpened() is always False, so __init__
deterministically takes its _show_placeholder() branch and never schedules
_poll(), on any machine (including a rig with a real camera at
CAMERA_INDEX). Tracker.capture_baseline() is separately monkeypatched so the
marker count is controlled directly rather than depending on real detection
against synthetic frames.
"""
import tkinter as tk

import pytest

import ui.session.baseline as baseline_module
import ui.session.camera_feed as camera_feed_module
from output.participants import ParticipantRecord
from ui.session import SessionWindow


class _StubCapture:
    """Stands in for cv2.VideoCapture -- always closed, never blocks on
    hardware enumeration."""

    def isOpened(self) -> bool:
        return False

    def release(self) -> None:
        pass

    def set(self, *args, **kwargs) -> None:
        pass

    def read(self):
        return False, None


@pytest.fixture
def root(tk_root):
    # tk_root (tests/conftest.py) is session-scoped, not module-scoped -- a
    # second in-process tk.Tk() created after this module's own root had
    # already run and torn down intermittently raised TclError once
    # test_session_error_handling.py added a second Tk-based test module.
    # One process-wide root, with a fresh SessionWindow Toplevel per test,
    # avoids the destroy-then-create sequence that triggered it.
    return tk_root


@pytest.fixture
def window(root, monkeypatch):
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: _StubCapture())
    win = SessionWindow(root)
    assert win._cap is None, "stubbed VideoCapture.isOpened() should always be False"
    yield win
    win.release_camera()
    win.destroy()


def _finish_burst(win: SessionWindow, count: int) -> None:
    """Drive status none -> capturing -> pass/fail without a real camera."""
    win._tracker.capture_baseline = lambda frames: count
    win._on_capture_baseline_click()
    assert win._baseline["status"] == "capturing"
    for _ in range(baseline_module.BASELINE_FRAMES):
        win._capture_baseline_frame(object())


def _make_participant(tmp_path) -> ParticipantRecord:
    folder = tmp_path / "participant999_test"
    folder.mkdir()
    return ParticipantRecord(
        participant_id="participant999", folder=str(folder),
        created_at="2026-01-01T00:00:00", age=30, weight_kg=70.0, height_cm=170.0,
        forearm_circ_cm=25.0, mvc_kg=40.0, activity_level="active",
    )


def test_start_disabled_in_phase1(window):
    assert window._baseline["status"] == "none"
    assert window._start_btn.enabled is False


def test_start_enabled_after_pass(window):
    _finish_burst(window, 97)
    assert window._baseline["status"] == "pass"
    assert window._start_btn.enabled is True


def test_start_disabled_on_fail_not_overridden(window):
    _finish_burst(window, 60)
    assert window._baseline["status"] == "fail"
    assert window._baseline["overridden"] is False
    assert window._start_btn.enabled is False


def test_start_enabled_after_override(window):
    _finish_burst(window, 60)
    window._on_override_baseline_click()
    assert window._baseline["overridden"] is True
    assert window._start_btn.enabled is True


def test_override_flag_reaches_saved_payload(window, tmp_path, monkeypatch):
    saved = {}

    def fake_save_baseline(record, **kwargs):
        saved.update(kwargs)
        return "fake/path.json"

    monkeypatch.setattr(baseline_module.participants, "save_baseline", fake_save_baseline)
    window._participant = _make_participant(tmp_path)

    _finish_burst(window, 60)
    assert saved == {}  # not saved yet -- only on pass or override
    window._on_override_baseline_click()
    assert saved["overridden"] is True
    assert saved["passed"] is False
    assert saved["locked_count"] == 60


def test_recapture_disabled_after_start(window):
    _finish_burst(window, 97)
    assert window._strip_recapture_btn.enabled is True
    # Phase 5 gates Start behind a bookend-confirm modal -- _on_start() only
    # opens it now. _confirm_start() is the body that used to run directly,
    # so drive that here rather than opening (and closing) a real modal.
    window._confirm_start()
    assert window._recording_started is True
    assert window._strip_recapture_btn.enabled is False


def test_recapture_click_ignored_after_start(window):
    _finish_burst(window, 97)
    window._confirm_start()
    window._on_recapture_baseline_click()
    assert window._baseline["status"] == "pass"  # unchanged -- no re-lock mid-session
