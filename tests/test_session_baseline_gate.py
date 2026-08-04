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
import json
import tkinter as tk
from pathlib import Path

import cv2
import numpy as np
import pytest

import ui.session.baseline as baseline_module
import ui.session.camera_feed as camera_feed_module
from output import participants
from output.participants import ParticipantRecord
from ui.session import SessionWindow

_CLIP_FRAME_SIZE = (48, 64, 3)  # (h, w, c) -- small, for fast test encodes

# The `window` fixture monkeypatches cv2.VideoCapture on the shared cv2
# module (camera_feed.py's `cv2` is the same module object), which would
# also stub out re-reads of a written clip within the same test. Grab the
# real class before any test can patch it.
_REAL_VIDEO_CAPTURE = cv2.VideoCapture


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


def _finish_burst_with_video(win: SessionWindow, count: int) -> None:
    """Like _finish_burst, but feeds real frames so the resulting baseline
    clip is a genuine, re-readable video."""
    win._tracker.capture_baseline = lambda frames: count
    win._on_capture_baseline_click()
    assert win._baseline["status"] == "capturing"
    for _ in range(baseline_module.BASELINE_FRAMES):
        frame = np.random.randint(0, 256, _CLIP_FRAME_SIZE, dtype=np.uint8)
        win._capture_baseline_frame(frame)


def _finish_burst_with_grid(
    win: SessionWindow, count: int, *,
    layout: dict[int, tuple[int, int]] | None = None,
    source: str = "geometry",
    reason: str | None = None,
) -> None:
    """Like _finish_burst_with_video, but also sets the tracker's
    grid_layout* attributes the way the real Tracker.capture_baseline() does
    (see core/tracker.py) -- so the Phase 3 grid-layout buffering path has
    real data to read instead of the Tracker.__init__ defaults. Real numpy
    frames (not _finish_burst's bare object()) because these tests drive all
    the way through _confirm_start(), which flushes save_baseline() and
    therefore needs frames with a real .shape for the video clip write."""
    def fake_capture_baseline(frames):
        win._tracker.grid_layout = layout
        win._tracker.grid_layout_source = source
        win._tracker.grid_layout_reason = reason
        return count

    win._tracker.capture_baseline = fake_capture_baseline
    win._on_capture_baseline_click()
    assert win._baseline["status"] == "capturing"
    for _ in range(baseline_module.BASELINE_FRAMES):
        frame = np.random.randint(0, 256, _CLIP_FRAME_SIZE, dtype=np.uint8)
        win._capture_baseline_frame(frame)


def _count_video_frames(path: Path) -> int:
    cap = _REAL_VIDEO_CAPTURE(str(path))
    n = 0
    for _ in range(baseline_module.BASELINE_FRAMES + 5):  # bounded: no hang on a stuck backend
        ok, _ = cap.read()
        if not ok:
            break
        n += 1
    cap.release()
    return n


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


def test_override_flag_reaches_saved_payload(window, tmp_path):
    """F6b (D3): _save_baseline_record no longer writes -- it buffers the
    save_baseline() kwargs into window._pending_baseline, which
    _confirm_start later flushes. Nothing reaches disk from the gate alone."""
    window._participant = _make_participant(tmp_path)

    _finish_burst(window, 60)
    assert window._pending_baseline is None  # not buffered yet -- only on pass or override
    window._on_override_baseline_click()
    saved = window._pending_baseline
    assert saved is not None
    assert saved["overridden"] is True
    assert saved["passed"] is False
    assert saved["locked_count"] == 60
    assert list(Path(window._participant.folder).glob("baseline_*")) == []


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


def test_pass_path_writes_matching_baseline_clip(window, tmp_path):
    """F6b: nothing lands on disk until Start flushes the buffered baseline
    -- so it must appear in the SESSION dir, not the participant root."""
    window._participant = _make_participant(tmp_path)
    _finish_burst_with_video(window, 97)

    root_before = Path(window._participant.folder)
    assert list(root_before.glob("baseline_*")) == []

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)

    jsons = list(session_dir.glob("baseline_*.json"))
    clips = list(session_dir.glob("baseline_*.mp4"))
    assert len(jsons) == 1
    assert len(clips) == 1
    assert jsons[0].stem == clips[0].stem
    assert _count_video_frames(clips[0]) == baseline_module.BASELINE_FRAMES
    assert list(root_before.glob("baseline_*")) == []
    window._confirm_stop()


def test_override_path_writes_matching_baseline_clip(window, tmp_path):
    window._participant = _make_participant(tmp_path)
    _finish_burst_with_video(window, 60)
    assert window._baseline["status"] == "fail"
    window._on_override_baseline_click()

    root_before = Path(window._participant.folder)
    assert list(root_before.glob("baseline_*")) == []

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)

    jsons = list(session_dir.glob("baseline_*.json"))
    clips = list(session_dir.glob("baseline_*.mp4"))
    assert len(jsons) == 1
    assert len(clips) == 1
    assert jsons[0].stem == clips[0].stem
    assert _count_video_frames(clips[0]) == baseline_module.BASELINE_FRAMES
    assert list(root_before.glob("baseline_*")) == []
    window._confirm_stop()


def test_fail_recapture_pass_writes_exactly_one_clip(window, tmp_path):
    """The subtle F6b requirement: a fail -> recapture -> fail -> recapture
    -> pass sequence must flush EXACTLY ONE baseline JSON and ONE clip at
    Start, and its content must be the WINNING (97-count) attempt's, not
    either discarded one."""
    window._participant = _make_participant(tmp_path)

    _finish_burst_with_video(window, 50)  # fail, discarded
    assert window._baseline["status"] == "fail"
    assert window._pending_baseline is None  # a bare fail is never buffered
    window._on_recapture_baseline_click()

    _finish_burst_with_video(window, 55)  # fail again, discarded
    assert window._baseline["status"] == "fail"
    assert window._pending_baseline is None
    window._on_recapture_baseline_click()

    _finish_burst_with_video(window, 97)  # winning attempt
    assert window._pending_baseline["locked_count"] == 97

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)

    jsons = list(session_dir.glob("baseline_*.json"))
    clips = list(session_dir.glob("baseline_*.mp4"))
    assert len(jsons) == 1
    assert len(clips) == 1
    assert jsons[0].stem == clips[0].stem

    payload = json.loads(jsons[0].read_text())
    assert payload["locked_count"] == 97  # the winning attempt, not 50 or 55
    assert _count_video_frames(clips[0]) == baseline_module.BASELINE_FRAMES
    window._confirm_stop()


def test_grid_layout_not_written_before_start(window, tmp_path):
    """Phase 3 (marker-grid-lookup plan): grid_layout.json follows the same
    F6b buffer-then-flush pattern as baseline_*.json -- nothing lands on disk
    from the gate alone."""
    window._participant = _make_participant(tmp_path)
    layout = {0: (0, 0), 1: (0, 1), 8: (0, 8), 98: (10, 8)}
    _finish_burst_with_grid(window, 99, layout=layout, source="geometry")
    assert window._pending_grid_layout is not None

    root_before = Path(window._participant.folder)
    assert list(root_before.glob("grid_layout.json")) == []

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)
    files = list(session_dir.glob("grid_layout.json"))
    assert len(files) == 1
    window._confirm_stop()


def test_fail_recapture_pass_writes_one_grid_layout_matching_winner(window, tmp_path):
    """Mirrors test_fail_recapture_pass_writes_exactly_one_clip: a
    fail -> recapture -> pass sequence must flush exactly one grid_layout.json,
    and its content must be the WINNING attempt's, not the discarded one."""
    window._participant = _make_participant(tmp_path)

    _finish_burst_with_grid(window, 50, layout=None, source="fallback_sort", reason="too few markers")
    assert window._baseline["status"] == "fail"
    assert window._pending_grid_layout is None  # a bare fail is never buffered
    window._on_recapture_baseline_click()

    winning_layout = {0: (0, 0), 1: (0, 1), 9: (1, 0)}
    _finish_burst_with_grid(window, 97, layout=winning_layout, source="geometry")
    assert window._pending_grid_layout["locked_count"] == 97

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)
    files = list(session_dir.glob("grid_layout.json"))
    assert len(files) == 1

    payload = json.loads(files[0].read_text())
    assert payload["locked_count"] == 97  # the winning attempt, not 50
    assert payload["source"] == "geometry"
    window._confirm_stop()


def test_fallback_capture_writes_grid_layout_with_null_layout(window, tmp_path):
    """D6: a capture whose geometry never resolves still writes the file --
    a missing file is ambiguous, an explicit fallback_sort record is not."""
    window._participant = _make_participant(tmp_path)
    _finish_burst_with_grid(
        window, 99, layout=None, source="fallback_sort", reason="mount rotated 90deg",
    )
    assert window._baseline["status"] == "pass"

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)
    payload = json.loads((session_dir / "grid_layout.json").read_text())
    assert payload["source"] == "fallback_sort"
    assert payload["layout"] is None
    assert payload["reason"] == "mount rotated 90deg"
    window._confirm_stop()


def test_grid_layout_round_trip_matches_tracker_dict(window, tmp_path):
    """Reading `layout` back and int()-ing its (necessarily string) keys must
    reproduce the tracker's in-memory grid_layout dict exactly."""
    window._participant = _make_participant(tmp_path)
    layout = {0: (0, 0), 1: (0, 1), 9: (1, 0), 98: (10, 8)}
    _finish_burst_with_grid(window, 99, layout=layout, source="geometry")

    window._confirm_start()
    session_dir = Path(window._writer.session_dir)
    payload = json.loads((session_dir / "grid_layout.json").read_text())

    round_tripped = {
        int(k): (v["row"], v["col"]) for k, v in payload["layout"].items()
    }
    assert round_tripped == layout
    window._confirm_stop()


def test_grid_layout_never_written_to_participant_root(window, tmp_path):
    window._participant = _make_participant(tmp_path)
    layout = {0: (0, 0)}
    _finish_burst_with_grid(window, 99, layout=layout, source="geometry")

    window._confirm_start()
    assert list(Path(window._participant.folder).glob("grid_layout.json")) == []
    window._confirm_stop()


def test_save_baseline_writes_clip_under_json_ts(tmp_path):
    record = _make_participant(tmp_path)
    frames = [
        np.random.randint(0, 256, _CLIP_FRAME_SIZE, dtype=np.uint8) for _ in range(15)
    ]

    json_path = Path(
        participants.save_baseline(
            record,
            dest_dir=record.folder,
            locked_count=97,
            passed=True,
            overridden=False,
            corners=None,
            bands=None,
            params={"thresh": 40, "min_area": 5},
            frames=frames,
        )
    )
    clip_path = json_path.with_suffix(".mp4")

    assert json_path.exists()
    assert clip_path.exists()
    assert json_path.stem == clip_path.stem
    assert _count_video_frames(clip_path) == 15


def test_save_baseline_without_frames_writes_no_clip(tmp_path):
    record = _make_participant(tmp_path)
    json_path = Path(
        participants.save_baseline(
            record,
            dest_dir=record.folder,
            locked_count=97,
            passed=True,
            overridden=False,
            corners=None,
            bands=None,
            params={"thresh": 40, "min_area": 5},
        )
    )
    assert json_path.exists()
    assert list(json_path.parent.glob("baseline_*.mp4")) == []
