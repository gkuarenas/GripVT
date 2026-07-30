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
import ui.session.recording as recording_module
from core.tracker import MarkerRecord
from output.participants import ParticipantRecord
from output.video import VIDEO_EXT
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

    def fake_process_frame(raw_frame):
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

    def raising_process_frame(raw_frame):
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


def test_restart_session_aborts_open_segment_keeps_committed_trial(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    win._on_back = lambda message=None: None

    win._confirm_start()
    session_dir = win._writer.session_dir
    markers_csv = os.path.join(session_dir, "markers.csv")

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

    with open(markers_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    # segment 0's n_committed rows survive; segment 1's n_open buffered rows
    # were discarded, never reaching disk (D6).
    assert len(rows) == n_committed * MARKER_COUNT

    with open(os.path.join(session_dir, "trials.csv"), newline="") as f:
        trial_rows = {r["trial_id"]: r for r in csv.DictReader(f)}
    assert trial_rows["0"]["voided"] == "False"
    assert trial_rows["0"]["end_ts"] != ""
    # Rig-found 2026-07-28 (participant008): the discarded in-progress
    # segment must be voided too, or trials.csv shows a "completed, valid"
    # trial with zero markers.csv rows behind it -- indistinguishable from
    # real (empty) data.
    assert trial_rows["1"]["voided"] == "True"
    assert trial_rows["1"]["end_ts"] != ""

    with open(os.path.join(session_dir, "events.json")) as f:
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

    with open(os.path.join(session_dir, "events.json")) as f:
        envelope = json.load(f)
    aborts = [e for e in envelope["events"] if e["type"] == "session_aborted"]
    assert len(aborts) == 1
    assert aborts[0]["scope"] == "all"

    with open(os.path.join(session_dir, "trials.csv"), newline="") as f:
        trial_rows = {r["trial_id"]: r for r in csv.DictReader(f)}
    assert trial_rows["0"]["voided"] == "True"

    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None


def test_restart_before_start_is_noop(connected_window, tmp_path):
    """D6: a Restart pressed before any confirmed Start has no event_log --
    nothing was ever recorded, so there is nothing to abort or persist."""
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
