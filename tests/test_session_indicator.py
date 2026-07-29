"""Headless tests for Phase 1 of
docs/logs/20260729_120040_session_indicator_plan.md -- SessionWindow's
_session_num / _session_num_at_open state (D2) and the single increment site
in _confirm_start (D4). No visible UI change lands yet (that's Phase 2+);
this file only proves the state.

Reuses the connected-camera stub idiom from test_trial_sequencing.py /
test_session_csv_recording.py.
"""
import tkinter as tk

import pytest

import ui.session.window as session_window_module
from ui.session import SessionWindow
from ui.session.progress import SESSION_COUNT

from tests.test_session_csv_recording import (
    MARKER_COUNT, _StubCaptureConnected, _install_fake_process_frame, _lock_baseline,
    _make_participant,
)


@pytest.fixture
def root(tk_root):
    return tk_root


@pytest.fixture
def make_window(root, monkeypatch):
    stub = _StubCaptureConnected()
    monkeypatch.setattr(session_window_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(session_window_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(session_window_module.ROICrop, "crop", lambda self, frame: frame)

    windows = []

    def build(session_num: int = 0) -> SessionWindow:
        win = SessionWindow(root, session_num=session_num)
        assert win._cap is not None, "stubbed VideoCapture.isOpened() should always be True"
        windows.append(win)
        return win

    yield build
    for win in windows:
        win.release_camera()
        win.destroy()


def test_fresh_window_at_zero_reports_matching_live_and_snapshot(make_window):
    win = make_window(session_num=0)
    assert win._session_num == 0
    assert win._session_num_at_open == 0


def test_confirmed_start_increments_live_num_but_not_snapshot(make_window, tmp_path):
    win = make_window(session_num=0)
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()

    assert win._session_num == 1
    assert win._session_num_at_open == 0


def test_window_seeded_above_zero_without_start_leaves_both_fields_unchanged(make_window):
    win = make_window(session_num=4)
    assert win._session_num == 4
    assert win._session_num_at_open == 4


# ── Phase 2: live SESSION block (_build_session_block / _refresh_session_block) ──


def test_fresh_window_session_block_shows_zero(make_window):
    win = make_window(session_num=0)
    assert win._session_value_label.cget("text") == "0 / 10"
    assert float(win._session_fill.place_info()["relwidth"]) == 0 / SESSION_COUNT


def test_confirmed_start_refreshes_session_block_to_one(make_window, tmp_path):
    win = make_window(session_num=0)
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()

    assert win._session_value_label.cget("text") == "1 / 10"
    assert float(win._session_fill.place_info()["relwidth"]) == 1 / SESSION_COUNT


def test_second_start_after_stop_refreshes_session_block_to_two(make_window, tmp_path):
    win = make_window(session_num=0)
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    win._poll()
    win._confirm_stop()

    win._confirm_start()

    assert win._session_value_label.cget("text") == "2 / 10"
    assert float(win._session_fill.place_info()["relwidth"]) == 2 / SESSION_COUNT


# ── Phase 3: Restart gating + dynamic modal/toast copy ──────────────────────


def _open_restart_modal_capturing_commands(win, monkeypatch):
    """D7/D6 verification needs to fire confirm_session/confirm_all, but
    they're closures local to _open_restart_modal, not SessionWindow methods
    -- capture the callables _build_button receives for those two labels
    instead of simulating a real pointer-position click (on_release checks
    winfo_pointerxy(), which a headless test can't fake reliably)."""
    captured: dict[str, object] = {}
    original = session_window_module.SessionWindow._build_button

    def patched(self, parent, label, kind, icon, command, *args, **kwargs):
        if label in ("Restart Session", "Restart Entire Testing"):
            captured[label] = command
        return original(self, parent, label, kind, icon, command, *args, **kwargs)

    monkeypatch.setattr(session_window_module.SessionWindow, "_build_button", patched)
    win._open_restart_modal()
    monkeypatch.setattr(session_window_module.SessionWindow, "_build_button", original)
    return captured


def test_restart_btn_disabled_at_construction_session_num_zero(make_window):
    win = make_window(session_num=0)
    assert win._restart_btn.enabled is False


def test_restart_btn_enabled_at_construction_session_num_one_and_above(make_window):
    win = make_window(session_num=1)
    assert win._restart_btn.enabled is True

    win2 = make_window(session_num=4)
    assert win2._restart_btn.enabled is True


def test_restart_btn_flips_enabled_after_in_window_start_increment(make_window, tmp_path):
    win = make_window(session_num=0)
    assert win._restart_btn.enabled is False
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()

    assert win._session_num == 1
    assert win._restart_btn.enabled is True


def test_kept_sessions_phrase_at_one_claims_no_priors(make_window):
    win = make_window(session_num=1)
    phrase = win._kept_sessions_phrase()
    assert "1-0" not in phrase
    assert phrase == "No prior sessions yet."


def test_kept_sessions_phrase_above_one_claims_priors_kept(make_window):
    win = make_window(session_num=3)
    assert win._kept_sessions_phrase() == "Sessions 1-2 are kept."


def test_confirm_session_rolls_session_num_back_to_snapshot(make_window, monkeypatch):
    win = make_window(session_num=3)
    assert win._session_num_at_open == 3
    win._session_num = 5  # simulate an in-window Start beyond the open snapshot

    commands = _open_restart_modal_capturing_commands(win, monkeypatch)
    commands["Restart Session"]()

    assert win._session_num == win._session_num_at_open == 3


def test_confirm_all_resets_session_num_to_zero(make_window, monkeypatch):
    win = make_window(session_num=5)

    commands = _open_restart_modal_capturing_commands(win, monkeypatch)
    commands["Restart Entire Testing"]()

    assert win._session_num == 0


def _label_texts(widget) -> list[str]:
    texts = []
    if isinstance(widget, tk.Label):
        texts.append(widget.cget("text"))
    for child in widget.winfo_children():
        texts.extend(_label_texts(child))
    return texts


def test_restart_modal_row_text_uses_live_numbers_not_old_literals(make_window, monkeypatch):
    win = make_window(session_num=3)
    win._open_restart_modal()
    assert win._modal_panel is not None
    texts = " ".join(_label_texts(win._modal_panel))

    assert "Abandons session 3" in texts
    assert "Sessions 1-2 are kept" in texts
    assert f"session 3 of {SESSION_COUNT}" in texts
    assert "next session recorded will be session 1 of 10" in texts
    assert "resets to" not in texts  # old row2 phrasing, replaced (Q1)


def test_restart_toasts_use_live_numbers_not_old_literals(make_window, monkeypatch):
    win = make_window(session_num=3)
    toasts: list[str] = []
    monkeypatch.setattr(win, "_on_back", lambda message=None: toasts.append(message))

    commands = _open_restart_modal_capturing_commands(win, monkeypatch)
    commands["Restart Session"]()

    assert toasts[-1] == "Session 3 abandoned — returned to Sensor Calibration. Sessions 1-2 are kept."

    win2 = make_window(session_num=3)
    toasts2: list[str] = []
    monkeypatch.setattr(win2, "_on_back", lambda message=None: toasts2.append(message))
    commands2 = _open_restart_modal_capturing_commands(win2, monkeypatch)
    commands2["Restart Entire Testing"]()

    assert toasts2[-1] == (
        "All sessions abandoned — returned to Sensor Calibration. "
        f"The next session recorded will be session 1 of {SESSION_COUNT}."
    )
