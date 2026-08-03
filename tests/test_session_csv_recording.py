"""Headless tests for Phase 3 of
docs/logs/20260728_121200_session_csv_recording_plan.md -- the recording
lifecycle wired into SessionWindow (_confirm_start/_confirm_stop/
_finalize_recording) that turns Tracker.process_frame() output into
markers.csv/trials.csv/events.json/overlay.png under a real participant
folder.

Reuses the connected-camera stub idiom from test_session_error_handling.py:
Tracker.undistort/ROICrop.crop are monkeypatched to identity at the class
level so _poll()'s display path never runs the real fisheye undistort
against a tiny fake frame. Tracker.process_frame is replaced on the instance
(same idiom as capture_baseline in test_session_baseline_gate.py) with a
canned MarkerRecord list, so this file is about the CSV-writing plumbing,
not detection correctness.
"""
import csv
import json
import os
import time

import cv2
import numpy as np
import pytest

import ui.session.baseline as baseline_module
import ui.session.camera_feed as camera_feed_module
import ui.session.modals as modals_module
import ui.session.recording as recording_module
from core.tracker import MarkerRecord
from output.participants import ParticipantRecord
from output.video import VIDEO_EXT
from output.writer import make_session_dir
from ui.session import SessionWindow

MARKER_COUNT = 5

# camera_feed_module.cv2 is the same module object as cv2 itself -- fixtures
# below monkeypatch cv2.VideoCapture globally for the fake-camera duration,
# which would also stub out a test's own re-read of a just-written video
# file. Grab the real class before any monkeypatching happens.
_REAL_VIDEO_CAPTURE = cv2.VideoCapture


class _StubCaptureConnected:
    """Always-open camera stub returning one fixed tiny frame -- read()
    never fails, so _poll() takes the live path every call."""

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


class _StubCaptureFailable(_StubCaptureConnected):
    """Same as _StubCaptureConnected, plus a .fail toggle -- mirrors
    test_session_error_handling.py's _StubCaptureConnected, so Phase 5's
    camera-loss tests can drive a disconnect/reconnect cycle deterministically."""

    def __init__(self) -> None:
        super().__init__()
        self.fail = False

    def read(self):
        if self.fail:
            return False, None
        return True, self._frame


@pytest.fixture
def root(tk_root):
    # Shared session-scoped Tk root -- see tests/conftest.py.
    return tk_root


@pytest.fixture
def connected_window(root, monkeypatch):
    stub = _StubCaptureConnected()
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(camera_feed_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(camera_feed_module.ROICrop, "crop", lambda self, frame: frame)
    win = SessionWindow(root)
    assert win._cap is not None, "stubbed VideoCapture.isOpened() should always be True"
    yield win
    win.release_camera()
    win.destroy()


@pytest.fixture
def failable_window(root, monkeypatch):
    stub = _StubCaptureFailable()
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(camera_feed_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(camera_feed_module.ROICrop, "crop", lambda self, frame: frame)
    win = SessionWindow(root)
    assert win._cap is not None, "stubbed VideoCapture.isOpened() should always be True"
    yield win, stub
    win.release_camera()
    win.destroy()


class _StubCaptureFullSize(_StubCaptureConnected):
    """Same as _StubCaptureConnected, but the frame matches
    self._cal.image_size (1024x768) rather than the 4x4 stub -- required for
    video-log tests, since VideoLogWriter opened at that size writes an
    undecodable (0-byte) file if the frames it receives are a different
    size (verified empirically against this machine's MSMF backend)."""

    def __init__(self) -> None:
        super().__init__()
        self._frame = np.zeros((768, 1024, 3), dtype=np.uint8)


@pytest.fixture
def video_window(root, monkeypatch):
    stub = _StubCaptureFullSize()
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(camera_feed_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(camera_feed_module.ROICrop, "crop", lambda self, frame: frame)
    win = SessionWindow(root)
    assert win._cap is not None, "stubbed VideoCapture.isOpened() should always be True"
    yield win
    win.release_camera()
    win.destroy()


def _make_participant(tmp_path) -> ParticipantRecord:
    folder = tmp_path / "participant999_test"
    folder.mkdir()
    return ParticipantRecord(
        participant_id="participant999", folder=str(folder),
        created_at="2026-01-01T00:00:00", age=30, weight_kg=70.0, height_cm=170.0,
        forearm_circ_cm=25.0, mvc_kg=40.0, activity_level="active",
    )


def _lock_baseline(win: SessionWindow, count: int = 97) -> None:
    """Drive status none -> capturing -> pass without a real camera (same
    helper as test_session_baseline_gate.py), then set baseline_set directly
    since the monkeypatched capture_baseline() -- unlike the real one -- does
    not lock any Kalman states or flip that flag itself."""
    win._tracker.capture_baseline = lambda frames: count
    win._on_capture_baseline_click()
    for _ in range(baseline_module.BASELINE_FRAMES):
        frame = np.random.randint(0, 256, (48, 64, 3), dtype=np.uint8)
        win._capture_baseline_frame(frame)
    assert win._baseline["status"] == "pass"
    win._tracker.baseline_set = True


def _install_fake_process_frame(win: SessionWindow, marker_count: int) -> None:
    """Replaces Tracker.process_frame on the instance with a canned,
    deterministic MarkerRecord list -- this suite is about the CSV-writing
    plumbing downstream of process_frame(), not real detection."""

    def fake_process_frame(raw_frame, undistorted=None):
        win._tracker.frame_index += 1
        win._tracker._last_undistorted = np.zeros((8, 8, 3), dtype=np.uint8)
        return [
            MarkerRecord(
                marker_id=i, x=float(i), y=float(i), area=10.0,
                dx=0.0, dy=0.0, dA=0.0, magnitude=0.0,
                predicted_x=float(i), predicted_y=float(i),
                autofilled=False, size_moments=10.0,
            )
            for i in range(marker_count)
        ]

    win._tracker.process_frame = fake_process_frame


def test_start_stop_writes_expected_files(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    assert win._writer is not None
    session_dir = win._writer.session_dir

    n_polls = 5
    for _ in range(n_polls):
        win._poll()

    win._confirm_stop()

    markers_csv = os.path.join(session_dir, "markers.csv")
    trials_csv = os.path.join(session_dir, "trials.csv")
    events_json = os.path.join(session_dir, "events.json")
    overlay_png = os.path.join(session_dir, "overlay.png")
    for path in (markers_csv, trials_csv, events_json, overlay_png):
        assert os.path.exists(path), path

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert list(rows[0].keys()) == [
        "frame", "timestamp_ms", "marker_id", "x", "y", "area", "dx", "dy", "dA",
        "magnitude", "size_moments", "predicted_x", "predicted_y", "autofilled",
        "trial_id",
    ]
    assert len(rows) == n_polls * MARKER_COUNT
    timestamps = [float(r["timestamp_ms"]) for r in rows]
    assert timestamps == sorted(timestamps)
    assert all(r["trial_id"] == "0" for r in rows)

    with open(trials_csv, newline="") as f:
        trial_rows = list(csv.DictReader(f))
    assert len(trial_rows) == 1
    assert trial_rows[0]["trial_id"] == "0"
    assert trial_rows[0]["start_ts"] != ""
    assert trial_rows[0]["end_ts"] != ""
    assert trial_rows[0]["voided"] == "False"

    with open(events_json) as f:
        envelope = json.load(f)
    assert envelope["schema_version"] == 1
    assert envelope["participant_id"] == "participant999"
    assert "started_at" in envelope
    assert isinstance(envelope["events"], list)

    # Finalized: all three handles nulled together (D8's single source of
    # truth for "is a writer open").
    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None


def test_second_session_in_same_participant_folder_numbers_up(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    first_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()

    win._confirm_start()
    second_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()

    assert os.path.basename(first_dir).startswith("session1_")
    assert os.path.basename(second_dir).startswith("session2_")


def test_buffer_flushed_before_max_frames(connected_window, tmp_path, monkeypatch):
    """D7 safety valve: force-commits early and logs buffer_flushed rather
    than growing the in-RAM buffer without bound."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    monkeypatch.setattr(recording_module, "MAX_BUFFER_FRAMES", 3)

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(4):
        win._poll()

    assert win._writer.pending_count == 1  # 4th frame buffered; 3rd forced a commit

    markers_csv = os.path.join(session_dir, "markers.csv")
    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * MARKER_COUNT  # committed early, before Stop

    win._confirm_stop()

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 4 * MARKER_COUNT  # remaining frame committed at Stop

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    flush_events = [e for e in envelope["events"] if e["type"] == "buffer_flushed"]
    assert len(flush_events) == 1
    assert flush_events[0]["frame_count"] == 3


def test_next_trial_then_undo_tombstones_the_right_segment(connected_window, tmp_path):
    """Phase 4: Start -> N polls (segment 0) -> Next Trial -> M polls
    (segment 1) -> Undo -> K polls (segment 1, redone) -> Next Trial ->
    polls (segment 2) -> Stop. Three segments total, exactly one voided
    (segment 0, the one Next Trial closed right before Undo fired), and
    markers.csv's row count must never decrease -- Undo discards only the
    uncommitted buffer of the in-flight redo, never already-written rows."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    markers_csv = os.path.join(session_dir, "markers.csv")

    def row_count() -> int:
        with open(markers_csv, newline="") as f:
            return len(list(csv.DictReader(f)))

    n_polls, m_polls, k_polls, p_polls = 3, 2, 4, 2
    for _ in range(n_polls):
        win._poll()
    assert row_count() == 0  # still buffered, not yet committed

    win._on_next_trial()
    assert win._segment_id == 1
    assert row_count() == n_polls * MARKER_COUNT  # committed on the boundary

    for _ in range(m_polls):
        win._poll()
    count_before_undo = row_count()

    win._on_undo_trial()
    assert win._segment_id == 1  # never decremented (D2)
    assert row_count() == count_before_undo  # discard drops only the buffer

    for _ in range(k_polls):
        win._poll()

    win._on_next_trial()
    assert win._segment_id == 2

    for _ in range(p_polls):
        win._poll()

    win._confirm_stop()

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    # segment 0 (n_polls, voided but never deleted) + segment 1 redone
    # (k_polls, the discarded m_polls buffer never made it to disk) +
    # segment 2 (p_polls).
    expected = (n_polls + k_polls + p_polls) * MARKER_COUNT
    assert len(rows) == expected
    trial_ids = [r["trial_id"] for r in rows]
    assert trial_ids.count("0") == n_polls * MARKER_COUNT
    assert trial_ids.count("1") == k_polls * MARKER_COUNT
    assert trial_ids.count("2") == p_polls * MARKER_COUNT

    with open(os.path.join(session_dir, "trials.csv"), newline="") as f:
        trial_rows = {r["trial_id"]: r for r in csv.DictReader(f)}
    assert set(trial_rows.keys()) == {"0", "1", "2"}
    assert trial_rows["0"]["voided"] == "True"
    assert trial_rows["1"]["voided"] == "False"
    assert trial_rows["2"]["voided"] == "False"
    # No phantom completed row: segment 1's surviving row (post-Undo) must
    # carry the same start/end pairing as any normally-closed segment, and
    # must not be a leftover from the discarded first opening.
    for tid in ("0", "1", "2"):
        assert trial_rows[tid]["start_ts"] != ""
        assert trial_rows[tid]["end_ts"] != ""


def test_confirm_start_without_participant_is_noop(connected_window):
    win = connected_window
    win._participant = None
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None
    assert win._recording is True  # existing UI state still flips normally

    win._poll()  # must not crash with no writer open
    win._confirm_stop()  # _finalize_recording no-ops on self._writer is None
    assert win._writer is None


# ── Phase 5: camera-loss gaps and tracking failures ─────────────────────────

def test_camera_loss_during_recording_closes_paired_event(failable_window, tmp_path):
    """A disconnect followed by a retry_camera() recovery appends exactly one
    camera_loss event with both start_ts and end_ts, and markers.csv has no
    rows for the disconnected interval (the failed-read polls never reach
    _update_tracked_indicator -- see _poll())."""
    win, stub = failable_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir

    n_before = 3
    for _ in range(n_before):
        win._poll()

    stub.fail = True
    for _ in range(camera_feed_module.CAMERA_FAIL_POLLS):
        win._poll()
    assert win._cap is None
    assert win._open_camera_loss_event is not None
    assert win._open_camera_loss_event["start_ts"] > 0
    assert "end_ts" not in win._open_camera_loss_event

    time.sleep(0.02)  # ensure the outage has a measurable, real duration
    stub.fail = False
    win.retry_camera()  # retry_camera() itself drives one _poll() (1860)
    assert win._cap is not None
    assert win._open_camera_loss_event is None

    n_after = 2
    for _ in range(n_after):
        win._poll()

    win._confirm_stop()

    markers_csv = os.path.join(session_dir, "markers.csv")
    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == (n_before + 1 + n_after) * MARKER_COUNT

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    loss_events = [e for e in envelope["events"] if e["type"] == "camera_loss"]
    assert len(loss_events) == 1
    assert loss_events[0]["end_ts"] > loss_events[0]["start_ts"]

    # Strict interior check: the first post-reconnect frame is legitimately
    # timestamped at/after end_ts (retry_camera() sets end_ts, then drives a
    # _poll() in the same call), so the boundary itself is not part of the
    # gap. Round both sides to markers.csv's own precision (CSVWriter rounds
    # timestamp_ms to 3 decimals; events.json does not) so a same-instant
    # boundary frame can't look "interior" from rounding noise alone.
    timestamps = [float(r["timestamp_ms"]) for r in rows]
    gap_start = round(loss_events[0]["start_ts"], 3)
    gap_end = round(loss_events[0]["end_ts"], 3)
    assert not any(gap_start < ts < gap_end for ts in timestamps)


def test_camera_loss_never_retried_closed_at_finalize(failable_window, tmp_path):
    """A disconnect that is never retried before Stop must still leave a
    closed (end_ts populated) camera_loss event in the final events.json --
    _finalize_recording's dangling-event close, not retry_camera()."""
    win, stub = failable_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir

    win._poll()
    stub.fail = True
    for _ in range(camera_feed_module.CAMERA_FAIL_POLLS):
        win._poll()
    assert win._open_camera_loss_event is not None

    win._confirm_stop()  # never retried -- Stop must still close the event
    assert win._open_camera_loss_event is None

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    loss_events = [e for e in envelope["events"] if e["type"] == "camera_loss"]
    assert len(loss_events) == 1
    assert "end_ts" in loss_events[0]
    assert loss_events[0]["end_ts"] >= loss_events[0]["start_ts"]


def test_tracking_failure_during_recording_finalizes(connected_window, tmp_path):
    """D8: a process_frame() exception while recording must not leave the
    REC badge counting over a dead writer -- one tracking_failed event, then
    the writer/trial_log/event_log are finalized (nulled) rather than
    silently continuing. Also the Phase 6 follow-up: the same UI teardown
    _confirm_stop performs (self._recording, badges, paused banner, timer,
    transport) must happen too, so the operator doesn't see a live-looking
    REC badge over an already-finalized session."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._poll()  # one good frame first, so the session isn't empty
    assert win._recording is True

    def raising_process_frame(raw_frame, undistorted=None):
        raise RuntimeError("boom")

    win._tracker.process_frame = raising_process_frame
    win._poll()  # triggers the except-block finalize path

    assert win._tracking_disabled is True
    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    failures = [e for e in envelope["events"] if e["type"] == "tracking_failed"]
    assert len(failures) == 1
    assert failures[0]["error"] == "boom"

    # Phase 6 follow-up: mirrors what _confirm_stop leaves behind.
    assert win._recording is False
    assert win._rec_badge.winfo_manager() == ""
    assert win._rec_paused_badge.winfo_manager() == ""
    assert win._paused_banner.winfo_manager() == ""
    assert win._rec_after_id is None


# ── Phase 6: exit paths and the abort record ─────────────────────────────────

def test_handle_back_finalizes_recording_mid_session(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(3):
        win._poll()

    win._handle_back()

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None

    markers_csv = os.path.join(session_dir, "markers.csv")
    trials_csv = os.path.join(session_dir, "trials.csv")
    events_json = os.path.join(session_dir, "events.json")
    overlay_png = os.path.join(session_dir, "overlay.png")
    for path in (markers_csv, trials_csv, events_json, overlay_png):
        assert os.path.exists(path), path

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * MARKER_COUNT

    with open(trials_csv, newline="") as f:
        trial_rows = list(csv.DictReader(f))
    assert len(trial_rows) == 1
    assert trial_rows[0]["end_ts"] != ""

    with open(events_json) as f:
        json.load(f)  # parses cleanly -- not truncated


def test_on_close_finalizes_recording_mid_session(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(3):
        win._poll()

    win._on_close()

    markers_csv = os.path.join(session_dir, "markers.csv")
    trials_csv = os.path.join(session_dir, "trials.csv")
    events_json = os.path.join(session_dir, "events.json")
    overlay_png = os.path.join(session_dir, "overlay.png")
    for path in (markers_csv, trials_csv, events_json, overlay_png):
        assert os.path.exists(path), path

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 3 * MARKER_COUNT

    with open(events_json) as f:
        json.load(f)  # parses cleanly -- not truncated

    # _on_close() calls self.destroy() -- no post-close attribute assertions
    # against win, mirroring test_on_close_closes_open_modal's approach in
    # test_session_error_handling.py.


def _capture_restart_buttons(win: SessionWindow) -> dict:
    """Same spy idiom as test_session_error_handling.py's restart-modal
    tests: intercept _build_button() to grab the confirm callbacks by label,
    since confirm_session/confirm_all are closures with no other seam."""
    captured: dict = {}
    orig_build_button = win._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        captured[label] = command
        return orig_build_button(parent, label, kind, icon, command, **kwargs)

    win._build_button = spy
    return captured


def _moved_dir(win, session_dir: str) -> str:
    """F7: Restart moves the just-finalized session dir into aborted/, under
    its original (unrenamed) name -- see move_sessions_to_aborted."""
    return os.path.join(win._participant.folder, "aborted", os.path.basename(session_dir))


def test_restart_session_aborts_open_segment_keeps_committed_trial(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir

    n_committed = 3
    for _ in range(n_committed):
        win._poll()
    win._on_next_trial()  # commits segment 0 to disk

    n_open = 2
    for _ in range(n_open):
        win._poll()  # buffered, not yet committed -- segment 1 in progress

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Session"]()

    moved_dir = _moved_dir(win, session_dir)
    assert not os.path.isdir(session_dir)  # moved, not left in place
    assert os.path.isdir(moved_dir)

    with open(os.path.join(moved_dir, "markers.csv"), newline="") as f:
        rows = list(csv.DictReader(f))
    # segment 0's n_committed rows survive; segment 1's n_open buffered rows
    # were discarded, never reaching disk (D6).
    assert len(rows) == n_committed * MARKER_COUNT

    with open(os.path.join(moved_dir, "trials.csv"), newline="") as f:
        trial_rows = {r["trial_id"]: r for r in csv.DictReader(f)}
    assert trial_rows["0"]["voided"] == "False"
    assert trial_rows["0"]["end_ts"] != ""
    # Rig-found 2026-07-28 (participant008): the discarded in-progress
    # segment must be voided too, or trials.csv shows a "completed, valid"
    # trial with zero markers.csv rows behind it -- indistinguishable from
    # real (empty) data.
    assert trial_rows["1"]["voided"] == "True"
    assert trial_rows["1"]["end_ts"] != ""

    with open(os.path.join(moved_dir, "events.json")) as f:
        envelope = json.load(f)
    aborts = [e for e in envelope["events"] if e["type"] == "session_aborted"]
    assert len(aborts) == 1
    assert aborts[0]["scope"] == "session"

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None


def test_restart_all_aborts_open_segment_with_scope_all(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(2):
        win._poll()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    moved_dir = _moved_dir(win, session_dir)
    assert not os.path.isdir(session_dir)  # moved, not left in place

    with open(os.path.join(moved_dir, "events.json")) as f:
        envelope = json.load(f)
    aborts = [e for e in envelope["events"] if e["type"] == "session_aborted"]
    assert len(aborts) == 1
    assert aborts[0]["scope"] == "all"

    with open(os.path.join(moved_dir, "trials.csv"), newline="") as f:
        trial_rows = {r["trial_id"]: r for r in csv.DictReader(f)}
    assert trial_rows["0"]["voided"] == "True"

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None


def test_restart_before_start_records_no_session_data(connected_window, tmp_path):
    """D6: a Restart pressed before any confirmed Start has no event_log, so
    no session_aborted event and no session dir are produced.

    Renamed from ..._is_noop 2026-08-03: it is no longer a full no-op. F4's
    participant-level `aborted.json` is written unconditionally, precisely
    because this path has no event_log to write into. What stays true, and is
    what this test guards, is that no *session* artifacts appear."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    assert win._event_log is None
    assert win._writer is None

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Session"]()  # must not raise

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None
    # No session dir was ever created (D5: lazy at first confirmed Start).
    assert not any(
        name.startswith("session") for name in os.listdir(win._participant.folder)
    )


def test_close_after_committed_trial_still_finalizes(connected_window, tmp_path):
    """Rig-found bug (2026-07-28, participant007 session5): closing the
    window after at least one Next-Trial commit is a different code path
    from closing an untouched first segment -- test_handle_back/on_close_
    finalizes_recording_mid_session above only ever exercise a single,
    never-committed segment. This reproduces the real sequence: commit one
    segment via Next Trial, then close mid-second-segment."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(3):
        win._poll()
    win._on_next_trial()
    for _ in range(3):
        win._poll()

    win._handle_back()

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None

    markers_csv = os.path.join(session_dir, "markers.csv")
    trials_csv = os.path.join(session_dir, "trials.csv")
    events_json = os.path.join(session_dir, "events.json")
    overlay_png = os.path.join(session_dir, "overlay.png")
    for path in (markers_csv, trials_csv, events_json, overlay_png):
        assert os.path.exists(path), path

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 6 * MARKER_COUNT
    trial_ids = sorted(set(r["trial_id"] for r in rows))
    assert trial_ids == ["0", "1"]

    with open(trials_csv, newline="") as f:
        trial_rows = list(csv.DictReader(f))
    assert len(trial_rows) == 2
    assert all(r["end_ts"] != "" for r in trial_rows), "both segments must be closed"


def test_finalize_survives_overlay_failure(connected_window, tmp_path, monkeypatch):
    """The actual guarantee behind the fix above: a failure while drawing/
    writing overlay.png (a best-effort visualization) must never prevent
    trials.csv/events.json from being flushed or the writer from closing --
    those are the data-integrity-critical files. Forces draw_overlay to
    raise and asserts finalize still completes correctly."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    def _boom(*args, **kwargs):
        raise ValueError("simulated overlay failure")

    monkeypatch.setattr(recording_module, "draw_overlay", _boom)

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(3):
        win._poll()

    win._handle_back()  # must not raise, despite draw_overlay blowing up

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None

    trials_csv = os.path.join(session_dir, "trials.csv")
    events_json = os.path.join(session_dir, "events.json")
    markers_csv = os.path.join(session_dir, "markers.csv")
    for path in (markers_csv, trials_csv, events_json):
        assert os.path.exists(path), path

    with open(trials_csv, newline="") as f:
        trial_rows = list(csv.DictReader(f))
    assert trial_rows[0]["end_ts"] != ""

    with open(events_json) as f:
        json.load(f)  # parses cleanly


# ── Phase 4: session video (docs/logs/20260730_110247_raw_video_log_plan.md) ─

def test_confirm_start_creates_session_video(video_window, tmp_path):
    win = video_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir

    assert win._video is not None
    video_path = os.path.join(session_dir, f"raw_video{VIDEO_EXT}")
    assert win._video.path == video_path
    assert os.path.exists(video_path)

    win._confirm_stop()


def test_finalize_closes_video_and_rereads_expected_frame_count(video_window, tmp_path):
    win = video_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    video_path = os.path.join(session_dir, f"raw_video{VIDEO_EXT}")

    n_polls = 5
    for _ in range(n_polls):
        win._poll()

    win._confirm_stop()

    assert win._video is None  # closed and nulled by _finalize_recording

    cap = _REAL_VIDEO_CAPTURE(video_path)
    decoded = 0
    for _ in range(n_polls + 5):  # bounded: a stuck backend must not hang the suite
        ok, _frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    assert decoded == n_polls

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    video_events = [e for e in envelope["events"] if e["type"] == "video_log"]
    assert len(video_events) == 1
    assert video_events[0]["path"] == f"raw_video{VIDEO_EXT}"
    assert video_events[0]["frame_count"] == n_polls


def test_video_log_event_carries_real_duration_and_achieved_fps(video_window, tmp_path):
    """docs/logs/20260730_163901_video_fps_labeling_plan.md, Phase 3 -- the
    video_log event must carry the true per-frame timing, not just frame_count,
    since VIDEO_FPS is a nominal write-cadence, not the achieved rate."""
    win = video_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir

    n_polls = 3
    for _ in range(n_polls):
        win._poll()
        time.sleep(0.02)  # ensure a measurable, real duration between frames

    win._confirm_stop()

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    event = [e for e in envelope["events"] if e["type"] == "video_log"][0]

    assert event["real_duration_ms"] > 0
    assert event["achieved_fps"] == pytest.approx(
        n_polls / (event["real_duration_ms"] / 1000), rel=1e-6
    )


def test_video_log_open_failure_degrades_to_no_video(connected_window, tmp_path, monkeypatch):
    """D10: a VideoLogWriter that raises on construction must never block the
    session -- markers.csv/trials.csv/events.json still get written normally,
    and a video_log_failed event records the failure."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    def _boom(*args, **kwargs):
        raise RuntimeError("no codec available")

    monkeypatch.setattr(recording_module, "VideoLogWriter", _boom)

    win._confirm_start()
    session_dir = win._writer.session_dir
    assert win._video is None

    win._poll()
    win._confirm_stop()

    markers_csv = os.path.join(session_dir, "markers.csv")
    trials_csv = os.path.join(session_dir, "trials.csv")
    events_json = os.path.join(session_dir, "events.json")
    for path in (markers_csv, trials_csv, events_json):
        assert os.path.exists(path), path

    with open(events_json) as f:
        envelope = json.load(f)
    failures = [e for e in envelope["events"] if e["type"] == "video_log_failed"]
    assert len(failures) == 1
    assert failures[0]["error"] == "no codec available"
    assert win._video is None


def test_video_survives_overlay_failure(video_window, tmp_path, monkeypatch):
    """Mirrors test_finalize_survives_overlay_failure: a draw_overlay failure
    during _finalize_recording must not leave the video writer unclosed --
    an unfinalized H.264 container may not decode at all (D8)."""
    win = video_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    def _boom(*args, **kwargs):
        raise ValueError("simulated overlay failure")

    monkeypatch.setattr(recording_module, "draw_overlay", _boom)

    win._confirm_start()
    session_dir = win._writer.session_dir
    video_path = os.path.join(session_dir, f"raw_video{VIDEO_EXT}")

    n_polls = 3
    for _ in range(n_polls):
        win._poll()

    win._handle_back()  # must not raise, despite draw_overlay blowing up

    assert win._video is None

    cap = _REAL_VIDEO_CAPTURE(video_path)
    decoded = 0
    for _ in range(n_polls + 5):  # bounded: a stuck backend must not hang the suite
        ok, _frame = cap.read()
        if not ok:
            break
        decoded += 1
    cap.release()
    assert decoded == n_polls  # closed cleanly, not left as a corrupt container


# ── F2: Grip Start marker (2026-08-03) ──────────────────────────────────────
# Marks where rest ends and grip begins inside a segment. trials.csv cannot
# supply this: trials are contiguous (end_ts[n] == start_ts[n+1]), so rest
# lives inside a segment and is otherwise invisible.

def _grip_events(session_dir: str) -> list[dict]:
    with open(os.path.join(session_dir, "events.json")) as f:
        return [e for e in json.load(f)["events"] if e["type"] == "grip_start"]


def test_grip_start_writes_one_well_formed_event(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._poll()
    win._on_grip_start()
    win._poll()
    win._confirm_stop()

    events = _grip_events(session_dir)
    assert len(events) == 1
    assert events[0]["trial_id"] == 0
    assert events[0]["trial_index"] == 1
    assert isinstance(events[0]["ts"], float)


def test_grip_start_timestamp_falls_inside_its_own_trial_window(connected_window, tmp_path):
    """The event is only meaningful if it shares markers.csv's clock -- a
    different time source would put it outside the segment it splits."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._poll()
    win._on_grip_start()
    win._poll()
    win._confirm_stop()

    event = _grip_events(session_dir)[0]
    with open(os.path.join(session_dir, "trials.csv"), newline="") as f:
        trial = next(r for r in csv.DictReader(f) if int(r["trial_id"]) == event["trial_id"])
    # All three timestamps are quantized to 3dp (1us). This test compresses
    # into microseconds an interval that is seconds long in real use, so the
    # bounds are compared at that quantum rather than exactly -- a tie at the
    # closing boundary is a test artifact, not a clock mismatch.
    quantum = 0.001
    assert float(trial["start_ts"]) - quantum <= event["ts"] <= float(trial["end_ts"]) + quantum


def test_second_grip_press_in_the_same_trial_is_ignored(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._on_grip_start()
    win._on_grip_start()
    win._on_grip_start()
    win._confirm_stop()

    assert len(_grip_events(session_dir)) == 1


def test_next_trial_re_arms_grip_start(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._on_grip_start()
    win._on_next_trial()
    assert win._grip_marked is False
    win._on_grip_start()
    win._confirm_stop()

    events = _grip_events(session_dir)
    assert [(e["trial_id"], e["trial_index"]) for e in events] == [(0, 1), (1, 2)]


def test_undo_re_arms_grip_start_and_drops_the_discarded_segments_mark(
    connected_window, tmp_path,
):
    """Undo voids the trial it just completed (segment_id - 1) and re-opens
    the SAME segment_id for the redo, discarding that segment's buffered
    marker rows. The grip mark describes those discarded frames, so it goes
    with them -- otherwise the redo's mark and the stale one share a trial_id
    and nothing downstream can tell them apart.

    This corrects D6 of the plan, which assumed segment_id is never reused.
    Undo reuses it; only Next Trial advances it."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._on_grip_start()          # marks segment 0
    win._on_next_trial()          # commits segment 0, opens segment 1
    win._on_grip_start()          # marks segment 1
    win._on_undo_trial()          # voids segment 0, re-opens segment 1
    assert win._grip_marked is False
    win._on_grip_start()          # marks the re-opened segment 1
    win._confirm_stop()

    events = _grip_events(session_dir)
    # One mark per surviving segment -- the stale segment-1 mark is gone, not
    # duplicated alongside the redo's.
    assert [e["trial_id"] for e in events] == [0, 1]
    assert len([e for e in events if e["trial_id"] == 1]) == 1

    with open(os.path.join(session_dir, "trials.csv"), newline="") as f:
        voided = {int(r["trial_id"]): r["voided"] == "True" for r in csv.DictReader(f)}
    assert voided[0] is True      # the mark survives on the voided trial it describes
    assert voided[1] is False


def test_undo_without_a_grip_mark_leaves_other_events_untouched(
    connected_window, tmp_path,
):
    """drop() must be surgical -- it filters grip_start for one segment, not
    the envelope."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._on_grip_start()          # segment 0
    win._on_next_trial()
    win._on_undo_trial()          # segment 1 had no mark; segment 0's must stay
    win._confirm_stop()

    assert [e["trial_id"] for e in _grip_events(session_dir)] == [0]
    with open(os.path.join(session_dir, "events.json")) as f:
        assert any(e["type"] == "video_log" for e in json.load(f)["events"])


def test_session_finalizes_normally_with_no_grip_start_events(connected_window, tmp_path):
    """D4: a missed press must never block Next Trial, Stop, or finalization."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._poll()
    win._on_next_trial()
    win._poll()
    win._confirm_stop()

    assert _grip_events(session_dir) == []
    for name in ("markers.csv", "trials.csv", "events.json", "overlay.png"):
        assert os.path.exists(os.path.join(session_dir, name)), name


def test_grip_start_before_recording_writes_nothing(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._on_grip_start()          # no recording open yet
    assert win._grip_marked is False
    assert win._event_log is None


# ── F4: participant-level abandonment record (2026-08-03) ───────────────────
# docs/logs/20260803_session_abandonment_record_plan.md. The in-session
# session_aborted event only exists while a recording is open; aborted.json
# covers the common case of Restart pressed between sessions.

def _aborted_json(win) -> dict:
    # F7 (docs/logs/20260803_per_wear_session_folder_plan.md, D5): aborted.json
    # moved from the participant root into <participant>/aborted/.
    with open(os.path.join(win._participant.folder, "aborted", "aborted.json")) as f:
        return json.load(f)


def test_restart_between_sessions_still_records_the_abandonment(
    connected_window, tmp_path,
):
    """The gap F4 closes: no recording is open, so there is no event_log and
    the old code recorded nothing anywhere."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    win._poll()
    win._confirm_stop()                      # session on disk, event_log closed
    assert win._event_log is None

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    aborts = _aborted_json(win)["aborts"]
    assert len(aborts) == 1
    assert aborts[0]["scope"] == "all"
    assert len(aborts[0]["abandoned_sessions"]) == 1


def test_restart_while_recording_writes_both_records(connected_window, tmp_path):
    """aborted.json and the in-session session_aborted event are
    complementary, not duplicates -- one names the abandoned directories, the
    other captures the in-progress segment's abort.

    F7: the session dir is finalized, recorded in aborted.json, then MOVED
    into aborted/ by the same Restart -- so events.json is read back from its
    post-move location, not the original session_dir path.
    """
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._poll()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Session"]()

    assert not os.path.isdir(session_dir)  # moved, not left in place
    moved_dir = os.path.join(win._participant.folder, "aborted", os.path.basename(session_dir))
    with open(os.path.join(moved_dir, "events.json")) as f:
        in_session = [e for e in json.load(f)["events"] if e["type"] == "session_aborted"]
    assert len(in_session) == 1
    assert _aborted_json(win)["aborts"][0]["scope"] == "session"


def test_restart_deletes_no_session_directory(connected_window, tmp_path):
    """D1/D8 at the integration level: the recovery path must never destroy
    collected data. F7 relocates the abandoned session into aborted/ rather
    than leaving it in place, but every byte must survive the move
    byte-identical -- this is the invariant the whole design turns on."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    first_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()
    rows_before = open(os.path.join(first_dir, "markers.csv")).read()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    moved_dir = _moved_dir(win, first_dir)
    assert not os.path.isdir(first_dir)  # gone from its original location...
    assert os.path.isdir(moved_dir)  # ...because it was moved, not deleted
    assert open(os.path.join(moved_dir, "markers.csv")).read() == rows_before


def test_a_failed_abort_record_does_not_block_restart(
    connected_window, tmp_path, monkeypatch,
):
    """D8: Restart is the recovery path. Losing the record is bad; being stuck
    in a broken session with a participant waiting is worse."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    backs: list = []
    win._on_back = lambda message=None: backs.append(message)

    monkeypatch.setattr(
        modals_module, "record_abort",
        lambda *a, **k: (_ for _ in ()).throw(OSError("disk full")),
    )

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()      # must not raise

    assert len(backs) == 1                    # navigation still happened


# ── F7: move abandoned sessions on Restart (2026-08-03) ─────────────────────
# docs/logs/20260803_per_wear_session_folder_plan.md, D7/D9, Phase 5. Restart
# already recorded which sessions were abandoned in aborted.json (F4); this
# closes the loop by relocating those same directories into aborted/, which
# is what makes _next_session_number (and the SESSION x/10 indicator) agree
# with what is actually on disk after a restart.

def _write_participant_root_files(win) -> None:
    """F7 D4: the participant root should hold exactly participant.json and
    mvc.txt alongside session/aborted dirs. _make_participant only creates the
    bare folder, so tests asserting the exact root listing write these two
    stand-ins themselves."""
    folder = win._participant.folder
    with open(os.path.join(folder, "participant.json"), "w") as f:
        json.dump({"participant_id": win._participant.participant_id}, f)
    with open(os.path.join(folder, "mvc.txt"), "w") as f:
        f.write("40.0\n")


def test_restart_all_moves_every_session_and_renumbers_from_one(
    connected_window, tmp_path,
):
    """D7: Restart Entire Testing moves ALL session dirs into aborted/, so
    the next make_session_dir() call -- exactly what the next SessionWindow
    performs at Start -- goes back to session1_*, matching the reset SESSION
    x/10 indicator."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    first_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()

    win._confirm_start()
    second_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    assert not os.path.isdir(first_dir)
    assert not os.path.isdir(second_dir)
    assert os.path.isdir(_moved_dir(win, first_dir))
    assert os.path.isdir(_moved_dir(win, second_dir))

    next_dir = make_session_dir(win._participant.folder)
    assert os.path.basename(next_dir).startswith("session1_")


def test_restart_session_moves_only_current_and_reuses_same_number(
    connected_window, tmp_path,
):
    """D7: Restart Session moves only the current session, leaving earlier
    ones in place, and the next make_session_dir() call reuses the SAME
    number -- exactly what the shipped modal's prose already promises
    ("the next attempt is still session N of 10")."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    first_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()

    win._confirm_start()  # session 2, in progress -- never stopped
    second_dir = win._writer.session_dir
    win._poll()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Session"]()

    assert os.path.isdir(first_dir)  # earlier session untouched
    assert not os.path.isdir(second_dir)  # only the current one moved
    assert os.path.isdir(_moved_dir(win, second_dir))

    next_dir = make_session_dir(win._participant.folder)
    assert os.path.basename(next_dir).startswith("session2_")  # same number, reused


def test_moved_session_markers_csv_is_byte_identical(connected_window, tmp_path):
    """D8: moving is not deleting -- the moved markers.csv must be exactly
    the bytes it was before the move, not a copy that could subtly differ."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir
    for _ in range(4):
        win._poll()
    win._confirm_stop()

    markers_path = os.path.join(session_dir, "markers.csv")
    with open(markers_path, "rb") as f:
        bytes_before = f.read()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    moved_markers_path = os.path.join(_moved_dir(win, session_dir), "markers.csv")
    with open(moved_markers_path, "rb") as f:
        bytes_after = f.read()
    assert bytes_after == bytes_before


def test_participant_root_holds_only_expected_entries_after_restart_all(
    connected_window, tmp_path,
):
    """D4: after Restart Entire Testing moves every session away, the
    participant root holds exactly participant.json, mvc.txt, and aborted/ --
    no stray sessionN_<ts>/ dirs left behind."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None
    _write_participant_root_files(win)

    win._confirm_start()
    win._poll()
    win._confirm_stop()

    win._confirm_start()
    win._poll()
    win._confirm_stop()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    root_entries = set(os.listdir(win._participant.folder))
    assert root_entries == {"participant.json", "mvc.txt", "aborted"}


def test_participant_root_keeps_surviving_session_after_restart_session(
    connected_window, tmp_path,
):
    """D4: Restart Session only moves the current session, so a prior
    session dir legitimately survives in the root alongside aborted/."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None
    _write_participant_root_files(win)

    win._confirm_start()
    win._poll()
    win._confirm_stop()

    win._confirm_start()
    win._poll()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Session"]()

    root_entries = set(os.listdir(win._participant.folder))
    session_entries = {e for e in root_entries if e.startswith("session")}
    assert len(session_entries) == 1  # session 1 survives, session 2 moved
    assert root_entries == {"participant.json", "mvc.txt", "aborted"} | session_entries


def test_aborted_json_named_sessions_agree_with_actually_moved_dirs(
    connected_window, tmp_path,
):
    """The core F7 guarantee: aborted.json naming a directory that wasn't
    actually moved (or vice versa) would be worse than either failing alone
    -- the two records must always agree exactly."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    win._poll()
    win._confirm_stop()

    win._confirm_start()
    win._poll()
    win._confirm_stop()

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()

    named = set(_aborted_json(win)["aborts"][-1]["abandoned_sessions"])
    aborted_dir = os.path.join(win._participant.folder, "aborted")
    actually_moved = {
        name for name in os.listdir(aborted_dir)
        if name.startswith("session") and os.path.isdir(os.path.join(aborted_dir, name))
    }
    assert named == actually_moved
    assert len(named) == 2  # both sessions were named and moved


def test_move_failure_does_not_block_restart(connected_window, tmp_path, monkeypatch):
    """D9: same reasoning as the aborted.json write -- a failure to move must
    not leave the operator stuck in a broken session with a participant
    waiting. Restart still navigates away even if the move raises."""
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    backs: list = []
    win._on_back = lambda message=None: backs.append(message)

    win._confirm_start()
    session_dir = win._writer.session_dir
    win._poll()
    win._confirm_stop()

    monkeypatch.setattr(
        modals_module, "move_sessions_to_aborted",
        lambda *a, **k: (_ for _ in ()).throw(OSError("cross-device move failed")),
    )

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Entire Testing"]()  # must not raise

    assert len(backs) == 1  # navigation still happened
    # aborted.json was still written -- only the move failed.
    assert len(_aborted_json(win)["aborts"]) == 1
    # The session dir is left exactly where it was -- a failed move must not
    # half-move or otherwise corrupt it.
    assert os.path.isdir(session_dir)
    assert os.path.exists(os.path.join(session_dir, "markers.csv"))
