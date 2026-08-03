"""Headless tests for the resume-supersede plan (2026-08-03):
docs/logs/20260803_203149_resume_session_complete_count_plan.md.

Covers `SessionWindow._confirm_start`'s D11/D12/D14 block -- the first
confirmed Start after a resume supersedes a trailing incomplete session
(moved into `aborted/`, audit-logged via `record_abort`, its session number
reused by the fresh recording) rather than orphaning it while a new,
higher-numbered session is created.

Reuses test_calibration_buffering.py's `session_window` fixture idiom: stub
`cv2.VideoCapture`, stub `Tracker.undistort` / `ROICrop.crop`, then construct
a real `SessionWindow(root, ...)` -- no camera, no GUI event loop needed for
`_confirm_start` itself.
"""
import csv
import json
import os

import numpy as np
import pytest

import ui.session.camera_feed as camera_feed_module
from core.trial_sequence import TRIAL_COUNT
from output import participants
from output.participants import ParticipantRecord
from output.writer import _TRIAL_COLUMNS
from ui.session import SessionWindow


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


def _write_trials_csv(session_dir, *, valid=30, voided=0, open_rows=0):
    """Same byte-shape as tests/test_participants.py's helper -- a real
    _TRIAL_COLUMNS header via csv.DictWriter, Python True/False in `voided`."""
    session_dir.mkdir(parents=True, exist_ok=True)
    csv_path = session_dir / "trials.csv"
    rows = []
    trial_id = 1
    for _ in range(valid):
        rows.append({
            "trial_id": trial_id, "trial_index": trial_id, "start_ts": 0.0,
            "end_ts": 1.0, "voided": False, "target_pct": 50, "target_kg": 5.0,
        })
        trial_id += 1
    for _ in range(voided):
        rows.append({
            "trial_id": trial_id, "trial_index": trial_id, "start_ts": 0.0,
            "end_ts": 1.0, "voided": True, "target_pct": 50, "target_kg": 5.0,
        })
        trial_id += 1
    for _ in range(open_rows):
        rows.append({
            "trial_id": trial_id, "trial_index": trial_id, "start_ts": 0.0,
            "end_ts": "", "voided": False, "target_pct": 50, "target_kg": 5.0,
        })
        trial_id += 1
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TRIAL_COLUMNS)
        w.writeheader()
        for row in rows:
            w.writerow(row)


def _make_participant_folder(tmp_path, *, complete=1, trailing_valid=None):
    """Fabricates <tmp_path>/participant001_<ts>/ with `complete` fully
    recorded sessions and, if `trailing_valid` is given, one more trailing
    session with that many valid rows (< TRIAL_COUNT, i.e. incomplete)."""
    folder = tmp_path / "participant001_20260803110000"
    folder.mkdir(parents=True)
    (folder / "participant.json").write_text(json.dumps({
        "schema_version": participants.SCHEMA_VERSION,
        "participant_id": "participant001",
        "created_at": "2026-08-03T11:00:00",
        "age": 22, "weight_kg": 70.0, "height_cm": 175.0,
        "forearm_circ_cm": 26.0, "mvc_kg": 41.719, "activity_level": "moderate",
    }))
    n = 0
    for n in range(1, complete + 1):
        _write_trials_csv(
            folder / f"session{n}_2026080311{n:02d}00", valid=TRIAL_COUNT,
        )
    if trailing_valid is not None:
        trailing_n = complete + 1
        _write_trials_csv(
            folder / f"session{trailing_n}_2026080311{trailing_n:02d}00",
            valid=trailing_valid,
        )
    return folder


def _record(folder) -> ParticipantRecord:
    return ParticipantRecord(
        participant_id="participant001", folder=str(folder),
        created_at="2026-08-03T11:00:00", age=22, weight_kg=70.0,
        height_cm=175.0, forearm_circ_cm=26.0, mvc_kg=41.719,
        activity_level="moderate",
    )


# ── (a)/(b): Start supersedes the trailing incomplete session ──────────────

def test_confirm_start_supersedes_trailing_incomplete_session(root, tmp_path):
    folder = _make_participant_folder(tmp_path, complete=1, trailing_valid=27)
    trailing_name = "session2_20260803110200"

    # Item 11's headless end-to-end check: derive session_num/
    # pending_supersede_session the same way main.py::open_calibration does
    # (D8), not by hand, so this test exercises the real computation too.
    assert participants.count_complete_sessions(str(folder)) == 1
    assert participants.trailing_incomplete_session(str(folder)) == trailing_name

    win = SessionWindow(
        root,
        session_num=participants.count_complete_sessions(str(folder)),
        pending_supersede_session=participants.trailing_incomplete_session(str(folder)),
    )
    win._participant = _record(folder)

    win._confirm_start()

    aborted_dir = folder / "aborted"
    assert (aborted_dir / trailing_name).is_dir()
    assert not (folder / trailing_name).exists()

    session_dir_name = os.path.basename(win._writer.session_dir)
    assert session_dir_name.startswith("session2_")
    assert session_dir_name != trailing_name  # different <ts>, number reused

    aborted_json = json.loads((aborted_dir / "aborted.json").read_text())
    assert len(aborted_json["aborts"]) == 1
    entry = aborted_json["aborts"][0]
    assert entry["scope"] == "session"
    assert entry["reason"] == "resume_supersede"
    assert entry["note"] == ""
    assert entry["session_num_at_abort"] == win._session_num_at_open == 1
    assert entry["abandoned_sessions"] == [trailing_name]

    win._confirm_stop()
    win.release_camera()
    win.destroy()


# ── (c): consumed exactly once -- a second Start in the same window instance
#         does not touch aborted/ again ─────────────────────────────────────

def test_pending_supersede_consumed_exactly_once(root, tmp_path):
    folder = _make_participant_folder(tmp_path, complete=1, trailing_valid=27)
    trailing_name = "session2_20260803110200"

    win = SessionWindow(
        root, session_num=1, pending_supersede_session=trailing_name,
    )
    win._participant = _record(folder)

    win._confirm_start()
    assert win._pending_supersede_session is None
    win._confirm_stop()

    # Simulate Restart Session / a second confirmed Start in the same window
    # instance -- must create session3_<ts> without touching aborted/ again.
    win._confirm_start()
    session_dir_name = os.path.basename(win._writer.session_dir)
    assert session_dir_name.startswith("session3_")

    aborted_json = json.loads((folder / "aborted" / "aborted.json").read_text())
    assert len(aborted_json["aborts"]) == 1  # still just the one entry

    win._confirm_stop()
    win.release_camera()
    win.destroy()


# ── (d): the New Participant path (pending_supersede_session=None) writes
#         no aborted/ at all ────────────────────────────────────────────────

def test_no_pending_supersede_writes_no_aborted_dir(root, tmp_path):
    folder = tmp_path / "participant002_20260803120000"
    folder.mkdir(parents=True)
    (folder / "participant.json").write_text(json.dumps({
        "schema_version": participants.SCHEMA_VERSION,
        "participant_id": "participant002",
        "created_at": "2026-08-03T12:00:00",
        "age": 22, "weight_kg": 70.0, "height_cm": 175.0,
        "forearm_circ_cm": 26.0, "mvc_kg": 41.719, "activity_level": "moderate",
    }))

    win = SessionWindow(root, session_num=0, pending_supersede_session=None)
    win._participant = ParticipantRecord(
        participant_id="participant002", folder=str(folder),
        created_at="2026-08-03T12:00:00", age=22, weight_kg=70.0,
        height_cm=175.0, forearm_circ_cm=26.0, mvc_kg=41.719,
        activity_level="moderate",
    )

    win._confirm_start()

    assert not (folder / "aborted").exists()
    session_dir_name = os.path.basename(win._writer.session_dir)
    assert session_dir_name.startswith("session1_")

    win._confirm_stop()
    win.release_camera()
    win.destroy()


# ── (e): a failed move_sessions_to_aborted must not block the recording ────

def test_failed_move_does_not_block_the_recording(root, tmp_path, monkeypatch):
    folder = _make_participant_folder(tmp_path, complete=1, trailing_valid=27)
    trailing_name = "session2_20260803110200"

    win = SessionWindow(
        root, session_num=1, pending_supersede_session=trailing_name,
    )
    win._participant = _record(folder)

    import ui.session.recording as recording_module

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(recording_module.participants, "move_sessions_to_aborted", _boom)

    win._confirm_start()  # must not raise

    assert win._writer is not None
    session_dir = win._writer.session_dir
    assert os.path.exists(os.path.join(session_dir, "markers.csv"))
    # D14's documented consequence: the move failed, so the trailing dir is
    # still on disk, unmoved -- make_session_dir allocated N+1 instead of
    # reusing N.
    assert (folder / trailing_name).exists()

    win._confirm_stop()
    win.release_camera()
    win.destroy()
