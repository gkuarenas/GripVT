"""Headless tests for the post-session unmount/rest modal:
docs/logs/20260803_post_session_unmount_rest_modal_plan.md.

Covers `RecordingMixin._confirm_stop`'s new D1 trigger check and
`ModalMixin._open_post_session_modal` / `_build_unmount_rest_content` /
`_build_visit_complete_content` / `_tick_rest_timer` / `_continue_from_
post_session`.

Reuses test_calibration_buffering.py's `session_window` fixture idiom: stub
`cv2.VideoCapture`, stub `Tracker.undistort` / `ROICrop.crop`, then construct
a real `SessionWindow(root, ...)` -- no camera, no GUI event loop needed for
`_confirm_start`/`_confirm_stop` themselves.
"""
import numpy as np
import pytest

import ui.session.camera_feed as camera_feed_module
from output.participants import ParticipantRecord
from ui.session import SessionWindow
from ui.session.modals import REST_DURATION_S
from ui.session.progress import SESSION_COUNT


# ── fixtures/helpers ─────────────────────────────────────────────────────────

class _StubCaptureConnected:
    def __init__(self) -> None:
        self._frame = np.zeros((4, 4, 3), dtype=np.uint8)

    def isOpened(self) -> bool:
        return True

    def release(self) -> None:
        pass

    def set(self, *args, **kwargs) -> None:
        pass

    def read(self):
        return True, self._frame


@pytest.fixture
def root(tk_root):
    return tk_root


@pytest.fixture(autouse=True)
def _stub_camera(monkeypatch):
    stub = _StubCaptureConnected()
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(camera_feed_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(camera_feed_module.ROICrop, "crop", lambda self, frame: frame)


def _make_participant(tmp_path, name="participant999_test") -> ParticipantRecord:
    folder = tmp_path / name
    folder.mkdir()
    return ParticipantRecord(
        participant_id="participant999", folder=str(folder),
        created_at="2026-01-01T00:00:00", age=30, weight_kg=70.0, height_cm=170.0,
        forearm_circ_cm=25.0, mvc_kg=40.0, activity_level="active",
    )


def _make_window(root, tmp_path, **kwargs) -> SessionWindow:
    win = SessionWindow(root, **kwargs)
    win._participant = _make_participant(tmp_path)
    return win


def _spy_open_post_session_modal(win, monkeypatch):
    calls = []
    orig = win._open_post_session_modal

    def spy():
        calls.append(True)
        orig()

    monkeypatch.setattr(win, "_open_post_session_modal", spy)
    return calls


# ── (a): modal opens iff _trial == _trial_max at Stop ──────────────────────

def test_modal_opens_only_when_all_30_trials_ran(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path)
    calls = _spy_open_post_session_modal(win, monkeypatch)

    win._confirm_start()
    assert win._trial == 1
    win._trial = 15  # early/manual stop -- well short of 30
    win._confirm_stop()

    assert calls == []

    win.release_camera()
    win.destroy()


def test_modal_opens_at_30_of_30(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path)
    calls = _spy_open_post_session_modal(win, monkeypatch)

    win._confirm_start()
    win._trial = win._trial_max
    win._confirm_stop()

    assert calls == [True]

    win.release_camera()
    win.destroy()


# ── (b): session N < 10 renders unmount/rest, N == 10 renders visit-complete

def test_session_below_10_shows_unmount_rest_branch(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path, session_num=2)  # -> 3 after _confirm_start
    win._confirm_start()
    win._trial = win._trial_max

    built = []
    monkeypatch.setattr(
        win, "_build_unmount_rest_content", lambda parent: built.append("unmount_rest"),
    )
    monkeypatch.setattr(
        win, "_build_visit_complete_content", lambda parent: built.append("visit_complete"),
    )

    win._confirm_stop()

    assert win._session_num == 3
    assert built == ["unmount_rest"]

    win._close_modal()
    win.release_camera()
    win.destroy()


def test_session_10_shows_visit_complete_branch(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path, session_num=SESSION_COUNT - 1)  # -> 10
    win._confirm_start()
    win._trial = win._trial_max

    built = []
    monkeypatch.setattr(
        win, "_build_unmount_rest_content", lambda parent: built.append("unmount_rest"),
    )
    monkeypatch.setattr(
        win, "_build_visit_complete_content", lambda parent: built.append("visit_complete"),
    )

    win._confirm_stop()

    assert win._session_num == SESSION_COUNT
    assert built == ["visit_complete"]

    win._close_modal()
    win.release_camera()
    win.destroy()


# ── (c): Continue calls release_camera() then on_back(None) ────────────────

def test_continue_releases_camera_then_calls_on_back(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path)
    win._confirm_start()
    win._trial = win._trial_max
    win._confirm_stop()  # opens the real modal (unmount/rest branch, session 1)

    order = []
    monkeypatch.setattr(win, "release_camera", lambda: order.append("release_camera"))
    monkeypatch.setattr(win, "_on_back", lambda message=None: order.append(("on_back", message)))

    win._continue_from_post_session()

    assert order == ["release_camera", ("on_back", None)]
    # The countdown must be cancelled, not left dangling.
    assert win._rest_after_id is None

    win.destroy()


def test_continue_handles_visit_complete_branch_with_no_timer(root, tmp_path, monkeypatch):
    """D8 guard: the visit-complete branch never starts _rest_after_id, so
    cancelling it in the Continue handler must be a safe no-op."""
    win = _make_window(root, tmp_path, session_num=SESSION_COUNT - 1)
    win._confirm_start()
    win._trial = win._trial_max
    win._confirm_stop()  # opens visit-complete branch, no timer started

    assert win._rest_after_id is None

    order = []
    monkeypatch.setattr(win, "release_camera", lambda: order.append("release_camera"))
    monkeypatch.setattr(win, "_on_back", lambda message=None: order.append(("on_back", message)))

    win._continue_from_post_session()  # must not raise

    assert order == ["release_camera", ("on_back", None)]

    win.destroy()


# ── (d): countdown ticks from 03:00, clamps at 00:00, never disables Continue

def test_countdown_ticks_down_and_clamps_without_disabling_continue(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path)

    captured_buttons = {}
    orig_build_button = win._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        btn = orig_build_button(parent, label, kind, icon, command, **kwargs)
        captured_buttons[label] = btn
        return btn

    monkeypatch.setattr(win, "_build_button", spy)

    win._confirm_start()
    win._trial = win._trial_max
    win._confirm_stop()  # unmount/rest branch (session 1 of 10)

    assert win._rest_remaining_s == REST_DURATION_S
    assert win._rest_label.cget("text") == "03:00"

    continue_btn = captured_buttons["Continue to Calibration"]
    assert continue_btn.enabled is True

    # Advance the countdown well past zero -- it must clamp, not go negative,
    # and must never touch the button's enabled state.
    for _ in range(REST_DURATION_S + 10):
        win._tick_rest_timer()

    assert win._rest_remaining_s == 0
    assert win._rest_label.cget("text") == "00:00"
    assert continue_btn.enabled is True

    win._cancel_rest_timer()
    win._close_modal()
    win.release_camera()
    win.destroy()


def test_countdown_ticks_once_per_call(root, tmp_path, monkeypatch):
    win = _make_window(root, tmp_path)
    win._confirm_start()
    win._trial = win._trial_max
    win._confirm_stop()

    assert win._rest_label.cget("text") == "03:00"
    win._tick_rest_timer()
    assert win._rest_label.cget("text") == "02:59"
    win._tick_rest_timer()
    assert win._rest_label.cget("text") == "02:58"

    win._cancel_rest_timer()
    win._close_modal()
    win.release_camera()
    win.destroy()


# ── (e): fires identically on a Resume-superseded session (D2) ─────────────

def test_modal_fires_on_resume_superseded_session(root, tmp_path, monkeypatch):
    """D2: no branch on _pending_supersede_session -- a session that started
    by superseding a trailing incomplete one must still get the modal at
    30/30, same as a fresh session."""
    folder = tmp_path / "participant001_20260803110000"
    folder.mkdir()
    import json
    from output import participants

    (folder / "participant.json").write_text(json.dumps({
        "schema_version": participants.SCHEMA_VERSION,
        "participant_id": "participant001",
        "created_at": "2026-08-03T11:00:00",
        "age": 22, "weight_kg": 70.0, "height_cm": 175.0,
        "forearm_circ_cm": 26.0, "mvc_kg": 41.719, "activity_level": "moderate",
    }))
    trailing_dir = folder / "session2_20260803110200"
    trailing_dir.mkdir()

    win = SessionWindow(
        root, session_num=1, pending_supersede_session="session2_20260803110200",
    )
    win._participant = ParticipantRecord(
        participant_id="participant001", folder=str(folder),
        created_at="2026-08-03T11:00:00", age=22, weight_kg=70.0,
        height_cm=175.0, forearm_circ_cm=26.0, mvc_kg=41.719,
        activity_level="moderate",
    )
    calls = _spy_open_post_session_modal(win, monkeypatch)

    win._confirm_start()
    assert win._pending_supersede_session is None  # consumed
    win._trial = win._trial_max
    win._confirm_stop()

    assert calls == [True]

    win._close_modal()
    win.release_camera()
    win.destroy()
