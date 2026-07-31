"""Headless tests for the Session window's error-handling layer (Phases 4-8
of docs/logs/20260727_215358_session_error_handling_plan.md).

Reuses the cv2.VideoCapture monkeypatch idiom from test_session_baseline_gate
.py: on this machine cv2.VideoCapture(1) takes ~3.55s and returns
isOpened=True (a camera IS connected), so every fixture stubs it rather than
opening a real device.

Two camera stubs are used:
  * _StubCapture -- isOpened() always False. SessionWindow.__init__ then
    deterministically takes its _show_placeholder() branch and never
    schedules _poll(). Used for everything that doesn't need a live feed
    (transport matrix, modals, tiers, timer, trial/undo, navigation).
  * _StubCaptureConnected -- isOpened() always True, read() controlled by a
    `.fail` flag. Used for the camera-loss debounce and disconnect/reconnect
    tests, which need _poll()/_open_camera()/retry_camera() to actually run.
    Tracker.undistort and ROICrop.crop are monkeypatched to the identity
    function at the class level so even the automatic first _poll() inside
    __init__ never runs the real fisheye undistort + ROI crop against the
    tiny fake frame -- these tests are about _cap/_read_fail_streak/Kalman-
    state bookkeeping, not pixel correctness.
"""
import tkinter as tk

import numpy as np
import pytest

import ui.session.baseline as baseline_module
import ui.session.camera_feed as camera_feed_module
import ui.session.recording as recording_module
from core.detector_search import TARGET_MARKERS
from ui import theme
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


class _StubCaptureConnected:
    """Stands in for cv2.VideoCapture -- always opens; read() outcome is
    controlled via .fail so tests can drive a disconnect/reconnect cycle
    deterministically."""

    def __init__(self) -> None:
        self.fail = False
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def set(self, *args, **kwargs) -> None:
        pass

    def read(self):
        if self.fail:
            return False, None
        return True, self._frame


@pytest.fixture
def root(tk_root):
    # tk_root (tests/conftest.py) is session-scoped and shared with
    # test_session_baseline_gate.py -- a second module-scoped tk.Tk() in this
    # file intermittently raised TclError when both files ran in the same
    # process (~1/3 of runs, reproduced directly). One process-wide root
    # removes the destroy-then-create sequence between modules entirely.
    return tk_root


@pytest.fixture
def window(root, monkeypatch):
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: _StubCapture())
    win = SessionWindow(root)
    assert win._cap is None, "stubbed VideoCapture.isOpened() should always be False"
    yield win
    win.release_camera()
    win.destroy()


@pytest.fixture
def connected_window(root, monkeypatch):
    stub = _StubCaptureConnected()
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(camera_feed_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(camera_feed_module.ROICrop, "crop", lambda self, frame: frame)
    win = SessionWindow(root)
    assert win._cap is not None, "stubbed VideoCapture.isOpened() should always be True"
    yield win, stub
    win.release_camera()
    win.destroy()


def _finish_burst(win: SessionWindow, count: int) -> None:
    """Drive baseline status none -> capturing -> pass/fail without a real
    camera, same helper as test_session_baseline_gate.py."""
    win._tracker.capture_baseline = lambda frames: count
    win._on_capture_baseline_click()
    for _ in range(baseline_module.BASELINE_FRAMES):
        win._capture_baseline_frame(object())


# ── transport state matrix ──────────────────────────────────────────────────
# start_enabled = locked and not recording
# stop_enabled  = recording
# next_enabled  = locked and recording and trial < trial_max  -- usable
#                 throughout an active recording (E7 revision, rig-feedback
#                 correction), but not before Start (2026-07-28 correction,
#                 D9-addendum: self._trial at Start becomes persisted data)

def test_transport_unlocked_idle(window):
    assert window._locked() is False
    assert window._recording is False
    assert window._start_btn.enabled is False
    assert window._stop_btn.enabled is False
    assert window._next_btn.enabled is False


def test_transport_locked_idle(window):
    _finish_burst(window, 97)
    assert window._trial < window._trial_max
    assert window._start_btn.enabled is True
    assert window._stop_btn.enabled is False
    assert window._next_btn.enabled is False


def test_transport_locked_recording(window):
    _finish_burst(window, 97)
    window._confirm_start()
    assert window._start_btn.enabled is False
    assert window._stop_btn.enabled is True
    # E7 revision: Next Trial is recording-independent -- Start/Stop bracket
    # the whole session, not one trial, so it must stay enabled here.
    assert window._next_btn.enabled is True


def test_transport_locked_idle_trial_at_max(window):
    _finish_burst(window, 97)
    window._trial = window._trial_max
    window._refresh_transport()
    assert window._start_btn.enabled is True
    assert window._stop_btn.enabled is False
    assert window._next_btn.enabled is False


# ── baseline lock still gates Start (regression guard) ─────────────────────

def test_start_disabled_until_locked(window):
    assert window._locked() is False
    assert window._start_btn.enabled is False


def test_start_enabled_after_pass(window):
    _finish_burst(window, 97)
    assert window._baseline["status"] == "pass"
    assert window._start_btn.enabled is True


def test_start_enabled_after_fail_and_override(window):
    _finish_burst(window, 60)
    assert window._baseline["status"] == "fail"
    assert window._start_btn.enabled is False
    window._on_override_baseline_click()
    assert window._baseline["overridden"] is True
    assert window._start_btn.enabled is True


# ── D2 regression guard: Re-capture permanently disabled once recording has
# started. _recording toggles Start<->Stop; _recording_started latches
# forever -- the two must not be conflated (module docstring, line ~153). ──

def test_recapture_disabled_permanently_after_stop(window):
    _finish_burst(window, 97)
    assert window._strip_recapture_btn.enabled is True

    window._confirm_start()
    assert window._recording is True
    assert window._recording_started is True
    assert window._strip_recapture_btn.enabled is False

    window._confirm_stop()
    assert window._recording is False  # toggled off by Stop
    assert window._recording_started is True  # latched forever, NOT reset by Stop
    assert window._strip_recapture_btn.enabled is False

    window._on_recapture_baseline_click()
    assert window._baseline["status"] == "pass"  # click is a no-op, unchanged


# ── tracked-marker tiers (E3) ────────────────────────────────────────────────

def test_tier_calm_at_zero_untracked(window):
    window._set_tracked_badge(0)
    assert window._tracked_tier == "calm"
    assert window._tracked_label.cget("text") == f"{TARGET_MARKERS} / {TARGET_MARKERS} TRACKED"
    assert window._tracked_badge.cget("bg") == theme.INK
    assert window._tracked_blink_after is None


def test_tier_warning_at_one_untracked(window):
    window._set_tracked_badge(1)
    assert window._tracked_tier == "warning"
    assert window._tracked_label.cget("text") == f"{TARGET_MARKERS - 1} / {TARGET_MARKERS} TRACKED"
    assert window._tracked_badge.cget("bg") == theme.ACCENT_10
    assert window._tracked_blink_after is None


def test_tier_warning_at_five_untracked(window):
    window._set_tracked_badge(5)
    assert window._tracked_tier == "warning"
    assert window._tracked_label.cget("text") == f"{TARGET_MARKERS - 5} / {TARGET_MARKERS} TRACKED"
    assert window._tracked_blink_after is None


def test_tier_alarm_at_six_untracked(window):
    window._set_tracked_badge(6)
    assert window._tracked_tier == "alarm"
    assert window._tracked_label.cget("text") == f"{TARGET_MARKERS - 6} / {TARGET_MARKERS} TRACKED"
    assert window._tracked_badge.cget("bg") == theme.ACCENT
    assert window._tracked_blink_after is not None  # blink scheduled


def test_alarm_blink_cancelled_when_leaving_alarm_tier(window):
    window._set_tracked_badge(6)
    blink_after = window._tracked_blink_after
    assert blink_after is not None

    window._set_tracked_badge(0)
    assert window._tracked_tier == "calm"
    assert window._tracked_blink_after is None


# ── camera-loss debounce (E9) ────────────────────────────────────────────────

def test_camera_loss_debounced_below_threshold(connected_window):
    win, stub = connected_window
    stub.fail = True
    for _ in range(camera_feed_module.CAMERA_FAIL_POLLS - 1):
        win._poll()
    assert win._cap is not None
    assert win._read_fail_streak == camera_feed_module.CAMERA_FAIL_POLLS - 1
    assert win._disconnected_panel.winfo_manager() == ""


def test_camera_loss_flips_at_threshold(connected_window):
    win, stub = connected_window
    stub.fail = True
    for _ in range(camera_feed_module.CAMERA_FAIL_POLLS):
        win._poll()
    assert win._cap is None
    assert win._disconnected_panel.winfo_manager() == "place"


def test_single_failure_then_success_resets_streak(connected_window):
    win, stub = connected_window
    stub.fail = True
    win._poll()
    assert win._read_fail_streak == 1

    stub.fail = False
    win._poll()
    assert win._read_fail_streak == 0
    assert win._cap is not None  # never reached the disconnect threshold


# ── tracker survives disconnect -> reconnect (decisive: CLAUDE.md forbids
# re-initialising Kalman states mid-session) ────────────────────────────────

def test_tracker_survives_disconnect_reconnect(connected_window, monkeypatch):
    win, stub = connected_window
    # Keep the throttled tracked-marker indicator out of the picture -- this
    # test is about _tracker/kalman.states identity, not detection.
    monkeypatch.setattr(win, "_update_tracked_indicator", lambda frame, undistorted=None: None)

    win._tracker.baseline_set = True
    win._tracker.kalman.init_state(0, 10.0, 20.0, 5.0)
    win._tracker.kalman.init_state(1, 30.0, 40.0, 6.0)

    original_tracker = win._tracker
    original_states_dict = win._tracker.kalman.states
    original_state_objs = dict(original_states_dict)
    original_positions = {sid: s.baseline_pos for sid, s in original_states_dict.items()}

    def _must_not_recapture(*args, **kwargs):
        raise AssertionError("capture_baseline() must not be called on reconnect")

    monkeypatch.setattr(win._tracker, "capture_baseline", _must_not_recapture)

    # Disconnect.
    stub.fail = True
    for _ in range(camera_feed_module.CAMERA_FAIL_POLLS):
        win._poll()
    assert win._cap is None

    # Reconnect.
    stub.fail = False
    win.retry_camera()
    assert win._cap is not None

    assert win._tracker is original_tracker
    assert win._tracker.kalman.states is original_states_dict
    for sid, state_obj in original_state_objs.items():
        assert win._tracker.kalman.states[sid] is state_obj
    for sid, pos in original_positions.items():
        assert win._tracker.kalman.states[sid].baseline_pos == pos


# ── REC timer excludes the paused gap (E6) ──────────────────────────────────

def test_rec_timer_excludes_paused_gap(window, monkeypatch):
    clock = {"t": 0.0}
    monkeypatch.setattr(recording_module.time, "monotonic", lambda: clock["t"])

    window._confirm_start()  # segment_start anchored at t=0
    clock["t"] = 5.0
    window._pause_timer()  # accumulates 5s; freezes
    clock["t"] = 20.0  # 15s dead gap while paused -- must be excluded
    window._resume_timer()  # re-anchors at t=20
    clock["t"] = 25.0
    window._tick_timer()  # repaints the label

    assert window._rec_label.cget("text") == "REC 00:10"


# ── modal lifecycle (E5) ─────────────────────────────────────────────────────

def test_open_then_close_modal_leaves_no_grab(window):
    window._open_modal(lambda parent: tk.Label(parent, text="hi").pack())
    scrim, panel = window._modal_scrim, window._modal_panel
    assert scrim is not None and panel is not None

    window._close_modal()
    assert window._modal_scrim is None
    assert window._modal_panel is None
    assert not scrim.winfo_exists()
    assert not panel.winfo_exists()
    assert window.grab_current() is None


def test_opening_second_modal_closes_first(window):
    window._open_modal(lambda parent: tk.Label(parent, text="A").pack())
    first_scrim, first_panel = window._modal_scrim, window._modal_panel

    window._open_modal(lambda parent: tk.Label(parent, text="B").pack())
    assert not first_scrim.winfo_exists()
    assert not first_panel.winfo_exists()
    assert window._modal_panel is not None
    assert window._modal_panel is not first_panel

    window._close_modal()


def test_close_modal_safe_when_nothing_open(window):
    window._close_modal()
    window._close_modal()  # still safe, still a no-op
    assert window._modal_scrim is None
    assert window._modal_panel is None


def test_handle_back_closes_open_modal(window):
    window._open_modal(lambda parent: None)
    assert window._modal_panel is not None

    calls = []
    window._on_back = lambda message=None: calls.append(message)
    window._handle_back()

    assert window._modal_panel is None
    assert window._modal_scrim is None
    assert calls == [None]


def test_on_close_closes_open_modal(root, monkeypatch):
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: _StubCapture())
    win = SessionWindow(root)
    win._open_modal(lambda parent: None)
    assert win._modal_panel is not None

    win._on_close()  # closes the modal, releases the camera, destroys itself

    assert win._modal_panel is None
    assert win._modal_scrim is None


# ── modal reposition + drag-to-move (rig-feedback correction, 2026-07-27) ───
# A freshly-created overrideredirect Toplevel's geometry doesn't reliably
# "stick" on Windows until the next event-loop tick, so _open_modal now
# schedules a second _reposition_modal() via after_idle on top of the
# immediate call. Since there's no OS titlebar on an overrideredirect
# window, the panel also grew manual click-and-drag, bound directly to the
# panel Toplevel so it never intercepts clicks on the content built inside.

class _FakeEvent:
    """Minimal stand-in for a tkinter Event -- carries only the attributes
    _on_modal_drag_start/_on_modal_drag_motion read. Driving the real
    <ButtonPress-1>/<B1-Motion> sequence via event_generate is unreliable
    headlessly (no guarantee the synthetic pointer position lands on an
    overrideredirect window), so the bound handlers are called directly."""

    def __init__(self, x: int = 0, y: int = 0, x_root: int = 0, y_root: int = 0) -> None:
        self.x = x
        self.y = y
        self.x_root = x_root
        self.y_root = y_root


def test_open_modal_schedules_idle_reposition(window, monkeypatch):
    calls = []
    original = window._reposition_modal

    def spy():
        calls.append(1)
        original()

    monkeypatch.setattr(window, "_reposition_modal", spy)
    window._open_modal(lambda parent: None)
    calls_after_open = len(calls)
    assert calls_after_open >= 1  # the immediate call in _open_modal

    # update_idletasks() drains pending idle callbacks (after_idle included)
    # without needing a real running mainloop -- same idiom the rest of this
    # file relies on via _reposition_modal's own self.update_idletasks() call.
    window.update_idletasks()
    assert len(calls) > calls_after_open  # the after_idle callback also ran

    window._close_modal()


def test_modal_drag_moves_panel_but_not_scrim(window):
    window._open_modal(lambda parent: None)
    panel, scrim = window._modal_panel, window._modal_scrim
    window.update_idletasks()
    scrim_geometry_before = scrim.geometry()

    window._on_modal_drag_start(_FakeEvent(x=10, y=8))
    window._on_modal_drag_motion(_FakeEvent(x=10, y=8, x_root=300, y_root=250))
    window.update_idletasks()

    assert panel.winfo_rootx() == 300 - 10
    assert panel.winfo_rooty() == 250 - 8
    assert scrim.geometry() == scrim_geometry_before  # dragging never touches the scrim

    window._close_modal()


def test_modal_drag_start_records_pointer_offset(window):
    window._open_modal(lambda parent: None)

    window._on_modal_drag_start(_FakeEvent(x=42, y=17))

    assert window._modal_drag_offset == (42, 17)

    window._close_modal()


def test_close_modal_after_drag_is_clean(window):
    window._open_modal(lambda parent: None)
    window._on_modal_drag_start(_FakeEvent(x=5, y=5))
    window._on_modal_drag_motion(_FakeEvent(x=5, y=5, x_root=200, y_root=200))

    window._close_modal()

    assert window._modal_panel is None
    assert window._modal_scrim is None
    assert window.grab_current() is None

    # A stale after_idle(_reposition_modal) queued by the _open_modal call
    # above may still be pending -- firing it post-close must be a silent
    # no-op (guarded by the None checks at the top of _reposition_modal),
    # never an exception.
    window.update_idletasks()

    # A drag motion event delivered after close must be equally inert.
    window._on_modal_drag_motion(_FakeEvent(x=5, y=5, x_root=210, y_root=210))


# ── E1 navigation contract ───────────────────────────────────────────────────

def test_ordinary_back_calls_on_back_with_none(window):
    calls = []
    window._on_back = lambda message=None: calls.append(message)
    window._handle_back()
    assert calls == [None]


def test_restart_session_confirmation_calls_on_back_with_message(window):
    calls = []
    window._on_back = lambda message=None: calls.append(message)

    captured = {}
    orig_build_button = window._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        captured[label] = command
        return orig_build_button(parent, label, kind, icon, command, **kwargs)

    window._build_button = spy
    window._open_restart_modal()
    captured["Restart Session"]()

    assert len(calls) == 1
    assert calls[0] is not None
    assert isinstance(calls[0], str) and calls[0].strip() != ""


def test_restart_entire_testing_confirmation_calls_on_back_with_message(window):
    calls = []
    window._on_back = lambda message=None: calls.append(message)

    captured = {}
    orig_build_button = window._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        captured[label] = command
        return orig_build_button(parent, label, kind, icon, command, **kwargs)

    window._build_button = spy
    window._open_restart_modal()
    captured["Restart Entire Testing"]()

    assert len(calls) == 1
    assert calls[0] is not None
    assert isinstance(calls[0], str) and calls[0].strip() != ""


# ── trial counter + Undo (E7) ────────────────────────────────────────────────

def test_next_trial_advances_and_enables_undo(window):
    _finish_burst(window, 97)
    window._confirm_start()  # D9: Next Trial requires an active recording
    trial_before = window._trial
    assert window._undo_available is False
    assert window._undo_btn.enabled is False

    window._on_next_trial()

    assert window._trial == trial_before + 1
    assert window._undo_available is True
    assert window._undo_btn.enabled is True


def test_undo_steps_back_once_and_disables(window):
    _finish_burst(window, 97)
    # Not 0 -- Undo's floor is max(1, ...), so stepping back from trial 1
    # lands on 1, not 0 (trial 0 is only ever the pre-first-Start state).
    window._trial = 5
    trial_before = window._trial
    window._on_next_trial()

    window._on_undo_trial()

    assert window._trial == trial_before
    assert window._undo_available is False
    assert window._undo_btn.enabled is False


def test_undo_is_noop_when_unavailable(window):
    _finish_burst(window, 97)
    trial_before = window._trial
    assert window._undo_available is False

    window._on_undo_trial()

    assert window._trial == trial_before
    assert window._undo_available is False


def test_next_trial_disabled_and_noop_at_trial_max(window):
    _finish_burst(window, 97)
    window._trial = window._trial_max
    window._refresh_transport()
    assert window._next_btn.enabled is False

    window._on_next_trial()  # guarded no-op -- trial >= trial_max

    assert window._trial == window._trial_max


def test_undo_clamps_at_one(window):
    window._trial = 1
    window._undo_available = True

    window._on_undo_trial()

    assert window._trial == 1
    assert window._undo_available is False


# ── Next Trial stays usable throughout an active recording (E7 revision,
# rig-feedback correction) -- Start/Stop bracket the whole 30-trial session,
# not one trial. It is gated off before Start (2026-07-28, D9-addendum). ──

def test_next_trial_enabled_while_recording(window):
    _finish_burst(window, 97)
    window._confirm_start()
    assert window._recording is True
    assert window._next_btn.enabled is True


def test_next_trial_advances_while_recording(window):
    _finish_burst(window, 97)
    window._confirm_start()
    trial_before = window._trial

    window._on_next_trial()  # must not be a guarded no-op while recording

    assert window._trial == trial_before + 1
    assert window._undo_available is True


def test_next_trial_disabled_while_recording_at_trial_max(window):
    _finish_burst(window, 97)
    window._confirm_start()
    window._trial = window._trial_max
    window._refresh_transport()
    assert window._recording is True
    assert window._next_btn.enabled is False

    window._on_next_trial()  # guarded no-op -- trial >= trial_max, recording or not

    assert window._trial == window._trial_max


def test_next_trial_disabled_while_unlocked_regardless_of_recording(window):
    assert window._locked() is False
    assert window._next_btn.enabled is False

    window._on_next_trial()  # guarded no-op -- not locked

    assert window._trial == 0  # unchanged from __init__'s value
