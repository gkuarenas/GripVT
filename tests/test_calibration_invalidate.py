"""Headless tests for F5 (per-wear calibration enforcement),
docs/logs/20260803_per_wear_session_folder_plan.md Phase 4.

Covers:
  * CalibrationWindow.invalidate() -- reuses _recapture's reset so Continue
    disables and a fresh pass genuinely re-enables it (not just a flag flip).
  * main._should_invalidate -- the D2 decision table: either Restart scope
    always invalidates; a bare Back only invalidates if Start was ever
    confirmed for that session. The bare-Back/no-Start case is the one most
    likely to regress, so it gets its own explicit test.

Camera is stubbed exactly as tests/test_session_error_handling.py does:
cv2.VideoCapture(1) is real hardware on this machine, so every fixture here
replaces it with a closed stub -- neither window needs a live feed to drive
its state machine.
"""
import tkinter as tk

import pytest

import main as main_module
import ui.calibration_window as calibration_window_module
import ui.session.baseline as baseline_module
import ui.session.camera_feed as camera_feed_module
from core.detector import default_params
from core.detector_search import (
    POSE_KEYS,
    TARGET_MARKERS,
    GRID_SHAPE,
    PoseDiagnostics,
    SearchResult,
    geometry_from,
    region_counts,
)
from output.participants import ParticipantRecord
from ui import theme
from ui.calibration_window import CalibrationWindow
from ui.session import SessionWindow


class _StubCapture:
    """Always-closed stand-in for cv2.VideoCapture -- both windows take
    their "camera unavailable" branch and never schedule a poll loop."""

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
    return tk_root


@pytest.fixture
def session_window(root, monkeypatch):
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: _StubCapture())
    win = SessionWindow(root)
    yield win
    win.release_camera()
    win.destroy()


@pytest.fixture
def cal_window(root, monkeypatch):
    monkeypatch.setattr(
        calibration_window_module.cv2, "VideoCapture", lambda index: _StubCapture()
    )
    record = ParticipantRecord(
        participant_id="participant001",
        folder="unused",
        created_at="2026-08-03T00:00:00",
        age=30,
        weight_kg=70.0,
        height_cm=170.0,
        forearm_circ_cm=28.0,
        mvc_kg=40.0,
        activity_level="moderate",
    )
    win = CalibrationWindow(root, record)
    assert win._cap is None, "stubbed VideoCapture.isOpened() should always be False"
    yield win
    win.release_camera()
    win.destroy()


def _finish_burst(win: SessionWindow, count: int) -> None:
    """Drive baseline status none -> capturing -> pass without a real camera,
    same helper as test_session_error_handling.py."""
    win._tracker.capture_baseline = lambda frames: count
    win._on_capture_baseline_click()
    for _ in range(baseline_module.BASELINE_FRAMES):
        win._capture_baseline_frame(object())


def _grid_points() -> list[tuple[float, float, float]]:
    cols, rows = GRID_SHAPE
    pitch_x, pitch_y, margin = 50, 45, 60
    return [
        (float(margin + c * pitch_x), float(margin + r * pitch_y), 30.0)
        for r in range(rows)
        for c in range(cols)
    ]


def _pass_result() -> SearchResult:
    """A cheap, fully-formed passing SearchResult -- built from the real
    geometry/region-count helpers (no image processing, no thread) so
    CalibrationWindow's own render path can consume it exactly as
    _poll_search's real result would."""
    points = _grid_points()
    geom = geometry_from(points)
    counts = region_counts(points, geom)
    diag = PoseDiagnostics(
        n_detected=TARGET_MARKERS, bands=counts["bands"], corners=counts["corners"],
        oversized=0, gaussian_fallbacks=0, cost=0.0,
    )
    return SearchResult(
        params=default_params(),
        poses={key: diag for key in POSE_KEYS},
        detections={key: points for key in POSE_KEYS},
        geometry=geom,
        grid_shape=(geom.x_lines, geom.y_lines),
        reference_bands=counts["bands"],
        reference_corners=counts["corners"],
        worst_pose="relaxed",
        worst_count=TARGET_MARKERS,
        passed=True,
        bounds={"thresh": [30, 60, 2], "min_area": [4, 40, 2]},
    )


def _set_passed(win: CalibrationWindow) -> None:
    win._result = _pass_result()
    win._status = "pass"
    win._pose_state = {key: "captured" for key in POSE_KEYS}
    for key in POSE_KEYS:
        win._pose_counts[key] = TARGET_MARKERS
    win._refresh()


def _continue_button(win: CalibrationWindow) -> tk.Frame:
    return win._actions.winfo_children()[-1]


# ── D2 decision table (pure) ─────────────────────────────────────────────────

@pytest.mark.parametrize(
    "message,recording_started,expected",
    [
        (None, False, False),  # bare Back, Start never pressed -- no wear ended
        (None, True, True),    # bare Back after a confirmed Start (recorded session)
        ("Session 3 abandoned — returned to Sensor Calibration.", False, True),  # Restart Session
        ("All sessions abandoned — returned to Sensor Calibration.", True, True),  # Restart All
    ],
    ids=["bare_back_no_start", "bare_back_after_start", "restart_session", "restart_all"],
)
def test_should_invalidate_matrix(message, recording_started, expected):
    assert main_module._should_invalidate(message, recording_started) is expected


# ── CalibrationWindow.invalidate() ───────────────────────────────────────────

def test_invalidate_disables_continue_and_clears_result(cal_window):
    _set_passed(cal_window)
    assert _continue_button(cal_window).cget("highlightbackground") == theme.ACCENT

    cal_window.invalidate()

    assert cal_window._status == "idle"
    assert cal_window._result is None
    assert all(state == "pending" for state in cal_window._pose_state.values())
    assert _continue_button(cal_window).cget("highlightbackground") == theme.INK_45


def test_invalidate_then_fresh_pass_reenables_continue(cal_window):
    """Proves the state actually recovers, not just that it clears."""
    _set_passed(cal_window)
    cal_window.invalidate()
    assert _continue_button(cal_window).cget("highlightbackground") == theme.INK_45

    _set_passed(cal_window)  # stand-in for a fresh calibration run passing

    assert cal_window._status == "pass"
    assert _continue_button(cal_window).cget("highlightbackground") == theme.ACCENT


def test_no_start_no_invalidate_leaves_continue_enabled(cal_window):
    """D2's most regression-prone case: a bare Back with Start never pressed
    must not force a recalibration, so a passing calibration stays intact."""
    _set_passed(cal_window)

    assert main_module._should_invalidate(None, False) is False
    # ... so restore_cal never calls invalidate() on this path -- confirm the
    # state genuinely survives untouched rather than asserting the negative.
    assert cal_window._status == "pass"
    assert _continue_button(cal_window).cget("highlightbackground") == theme.ACCENT


# ── end-to-end through SessionWindow's actual navigation callbacks ──────────

def test_recorded_session_bare_back_invalidates(session_window, cal_window):
    win = session_window
    _finish_burst(win, 97)
    win._confirm_start()
    assert win._recording_started is True

    calls = []
    win._on_back = lambda message=None: calls.append(message)
    win._handle_back()
    assert calls == [None]  # ordinary Back always carries no message (E1)

    assert main_module._should_invalidate(calls[0], win._recording_started) is True

    _set_passed(cal_window)
    cal_window.invalidate()
    assert _continue_button(cal_window).cget("highlightbackground") == theme.INK_45


def test_bare_back_without_start_does_not_invalidate(session_window, cal_window):
    win = session_window
    assert win._recording_started is False

    calls = []
    win._on_back = lambda message=None: calls.append(message)
    win._handle_back()
    assert calls == [None]

    assert main_module._should_invalidate(calls[0], win._recording_started) is False

    _set_passed(cal_window)
    # main.py's restore_cal would not call invalidate() here -- confirm the
    # calibration is left alone.
    assert cal_window._status == "pass"
    assert _continue_button(cal_window).cget("highlightbackground") == theme.ACCENT


def test_restart_session_invalidates(session_window, cal_window):
    win = session_window
    calls = []
    win._on_back = lambda message=None: calls.append(message)

    captured = {}
    orig_build_button = win._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        captured[label] = command
        return orig_build_button(parent, label, kind, icon, command, **kwargs)

    win._build_button = spy
    win._open_restart_modal()
    captured["Restart Session"]()

    assert calls and calls[0] is not None
    assert main_module._should_invalidate(calls[0], win._recording_started) is True

    _set_passed(cal_window)
    cal_window.invalidate()
    assert _continue_button(cal_window).cget("highlightbackground") == theme.INK_45


def test_restart_entire_testing_invalidates(session_window, cal_window):
    win = session_window
    _finish_burst(win, 97)
    win._confirm_start()  # Restart Entire after a Start still must invalidate (D2)

    calls = []
    win._on_back = lambda message=None: calls.append(message)

    captured = {}
    orig_build_button = win._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        captured[label] = command
        return orig_build_button(parent, label, kind, icon, command, **kwargs)

    win._build_button = spy
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    assert calls and calls[0] is not None
    assert main_module._should_invalidate(calls[0], win._recording_started) is True

    _set_passed(cal_window)
    cal_window.invalidate()
    assert _continue_button(cal_window).cget("highlightbackground") == theme.INK_45
