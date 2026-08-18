"""Headless tests for Phase 2 (F6a) of
docs/logs/20260803_per_wear_session_folder_plan.md -- calibration is buffered
in memory at CalibrationWindow.Continue and only reaches disk once a session
directory exists, flushed by SessionWindow._confirm_start.

Closes the coverage gap flagged at the end of Phase 1: prior to this file,
ui/calibration_window.py:631 (now :639, `_continue`) called
`participants.save_calibration(...)` with no `dest_dir`, which raises
TypeError the instant it runs -- and nothing in the suite exercised that call
path, so nothing failed. These tests exercise the real call sites: `_continue`
building the bundle, `main.py`'s closures threading it through, and
`_confirm_start` flushing it into the session directory.
"""
import csv
import json
import os

import numpy as np
import pytest

import ui.session.camera_feed as camera_feed_module
from core.detector_search import Geometry, PoseDiagnostics, SearchResult
from core.trial_sequence import TRIAL_COUNT
from output import participants
from output.participants import ParticipantRecord
from output.writer import _TRIAL_COLUMNS
from ui.calibration_window import CalibrationWindow
from ui.session import SessionWindow

import main as main_module


# ── shared fixtures/helpers (same idiom as test_participants.py /
#    test_session_csv_recording.py) ─────────────────────────────────────────

def _geometry():
    return Geometry(
        axis=0, lines=(10.0, 20.0), corner=((0.0, 0.0),), band=((0.0, 0.0),),
        n_lines=9, n_per_line=11,
    )


def _search_result(passed=True):
    pose = PoseDiagnostics(
        n_detected=99, bands={"top": 9, "bottom": 9},
        corners={"tl": 3, "tr": 3, "bl": 3, "br": 3},
        oversized=0, gaussian_fallbacks=0, cost=0.0,
    )
    return SearchResult(
        params={"thresh": 40, "min_area": 10},
        poses={"relaxed": pose, "full_open": pose, "max_grip": pose},
        detections={"relaxed": [], "full_open": [], "max_grip": []},
        geometry=_geometry(), grid_shape=(9, 11),
        reference_bands={"top": 9, "bottom": 9},
        reference_corners={"tl": 3, "tr": 3, "bl": 3, "br": 3},
        worst_pose="relaxed", worst_count=99, passed=passed,
        bounds={"thresh": [30, 60], "min_area": [4, 40]},
    )


def _pose_frames():
    return {
        "relaxed": np.zeros((8, 8), dtype=np.uint8),
        "full_open": np.zeros((8, 8), dtype=np.uint8),
        "max_grip": np.zeros((8, 8), dtype=np.uint8),
    }


def _make_participant(tmp_path) -> ParticipantRecord:
    folder = tmp_path / "participant999_test"
    folder.mkdir()
    return ParticipantRecord(
        participant_id="participant999", folder=str(folder),
        created_at="2026-01-01T00:00:00", age=30, weight_kg=70.0, height_cm=170.0,
        forearm_circ_cm=25.0, mvc_kg=40.0, activity_level="active",
    )


def _all_calibration_artifacts(root: str) -> list[str]:
    """Every calibration_*.json / pose_*.png anywhere under `root`."""
    found = []
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if name.startswith("calibration_") or name.startswith("pose_"):
                found.append(os.path.join(dirpath, name))
    return found


@pytest.fixture
def root(tk_root):
    return tk_root


@pytest.fixture
def cal_window(root):
    """CalibrationWindow with no real camera -- CAMERA_INDEX (1) is not open
    in the test environment, so _open_camera() takes its `_show_unavailable`
    branch and never schedules an `after()` poll loop, same as the real
    "camera unplugged" state. The record's folder is never touched: _continue
    no longer writes anything (that is the whole point of F6a)."""
    win = CalibrationWindow(root, _make_participant_dummy())
    yield win
    win.release_camera()
    win.destroy()


def _make_participant_dummy() -> ParticipantRecord:
    return ParticipantRecord(
        participant_id="participant999", folder="unused",
        created_at="2026-01-01T00:00:00", age=30, weight_kg=70.0, height_cm=170.0,
        forearm_circ_cm=25.0, mvc_kg=40.0, activity_level="active",
    )


def _drive_to_pass(win: CalibrationWindow) -> None:
    """Sets the post-search state _continue requires, without running the
    real threaded search() -- mirrors the plan's guidance to test the
    _continue call site itself, not detector_search."""
    win._pose_frames = _pose_frames()
    win._pose_dets = {"relaxed": [], "full_open": [], "max_grip": []}
    win._result = _search_result(passed=True)
    win._status = "pass"


# ── 1/3: calibration completing writes nothing, anywhere ───────────────────

def test_continue_writes_no_calibration_artifacts_anywhere(cal_window, tmp_path):
    win = cal_window
    _drive_to_pass(win)

    captured = {}
    win._on_continue = lambda result, pending: captured.update(
        result=result, pending=pending
    )
    win._continue()

    assert captured  # on_continue was actually called
    assert _all_calibration_artifacts(str(tmp_path)) == []


def test_continue_bundle_carries_everything_save_calibration_needs(cal_window):
    """The bundle's keys/shape must match what save_calibration's signature
    expects (output/participants.py) -- this is the coverage gap Phase 1
    flagged: nothing previously asserted this contract at all."""
    win = cal_window
    _drive_to_pass(win)

    captured = {}
    win._on_continue = lambda result, pending: captured.update(
        result=result, pending=pending
    )
    win._continue()

    pending = captured["pending"]
    assert set(pending.keys()) == {
        "result", "pose_frames", "overridden", "roi", "calibration_image_size",
        "manual_override", "auto_params",
    }
    assert pending["result"] is captured["result"]
    assert set(pending["pose_frames"].keys()) == {"relaxed", "full_open", "max_grip"}
    assert pending["overridden"] is False  # result.passed was True
    assert pending["calibration_image_size"] == list(win._cal.image_size)
    assert pending["manual_override"] is False  # _drive_to_pass never ran a rescore
    assert pending["auto_params"] is None


# ── 2/3: Start flushes into the SESSION dir; participant root stays clean ──

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
def session_window(root, monkeypatch):
    stub = _StubCaptureConnected()
    monkeypatch.setattr(camera_feed_module.cv2, "VideoCapture", lambda index: stub)
    monkeypatch.setattr(camera_feed_module.Tracker, "undistort", lambda self, frame: frame)
    monkeypatch.setattr(camera_feed_module.ROICrop, "crop", lambda self, frame: frame)
    win = SessionWindow(root)
    assert win._cap is not None
    yield win
    win.release_camera()
    win.destroy()


def _pending_bundle() -> dict:
    return {
        "result": _search_result(passed=True),
        "pose_frames": _pose_frames(),
        "overridden": False,
        "roi": None,
        "calibration_image_size": [1024, 768],
        "manual_override": False,
        "auto_params": None,
    }


def test_start_writes_calibration_into_the_session_dir_only(session_window, tmp_path):
    win = session_window
    win._participant = _make_participant(tmp_path)
    win._pending_calibration = _pending_bundle()

    win._confirm_start()
    session_dir = win._writer.session_dir

    session_cals = [n for n in os.listdir(session_dir) if n.startswith("calibration_")]
    session_poses = [n for n in os.listdir(session_dir) if n.startswith("pose_")]
    assert len(session_cals) == 1
    assert len(session_poses) == 3

    root_entries = os.listdir(win._participant.folder)
    root_cals = [n for n in root_entries if n.startswith("calibration_")]
    root_poses = [n for n in root_entries if n.startswith("pose_")]
    assert root_cals == []
    assert root_poses == []

    win._confirm_stop()


def test_start_without_pending_calibration_does_not_crash(session_window, tmp_path):
    """No CalibrationWindow bundle at all (e.g. a headless/test fixture, or a
    future caller that never went through Calibration) must degrade cleanly --
    same D5/D8 idiom as the self._participant is None guard right above it."""
    win = session_window
    win._participant = _make_participant(tmp_path)
    win._pending_calibration = None

    win._confirm_start()  # must not raise
    session_dir = win._writer.session_dir
    assert [n for n in os.listdir(session_dir) if n.startswith("calibration_")] == []
    win._confirm_stop()


def test_calibration_write_failure_does_not_block_start(session_window, tmp_path, monkeypatch):
    """D5/D8-style guard (per the task): Start is the critical path -- a
    broken calibration write must not prevent the 30-trial recording from
    opening."""
    win = session_window
    win._participant = _make_participant(tmp_path)
    win._pending_calibration = _pending_bundle()

    import ui.session.recording as recording_module

    def _boom(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(recording_module.participants, "save_calibration", _boom)

    win._confirm_start()  # must not raise
    assert win._writer is not None  # the session itself still opened
    session_dir = win._writer.session_dir
    assert os.path.exists(os.path.join(session_dir, "markers.csv"))
    win._confirm_stop()


# ── 4: Restart before Start still leaves no session folder (regression) ────

def _capture_restart_buttons(win: SessionWindow) -> dict:
    captured: dict = {}
    orig_build_button = win._build_button

    def spy(parent, label, kind, icon, command, **kwargs):
        captured[label] = command
        return orig_build_button(parent, label, kind, icon, command, **kwargs)

    win._build_button = spy
    return captured


def test_restart_before_start_still_leaves_no_session_folder(session_window, tmp_path):
    """Guards the guarantee D3 depends on: buffering calibration in memory
    must not change test_restart_before_start_records_no_session_data's
    outcome. A pending calibration sitting unwritten in RAM must vanish with
    nothing on disk when Restart fires before Start."""
    win = session_window
    win._participant = _make_participant(tmp_path)
    win._pending_calibration = _pending_bundle()
    win._on_back = lambda message=None: None

    captured = _capture_restart_buttons(win)
    win._open_restart_modal()
    captured["Restart Session"]()  # must not raise

    assert win._writer is None
    assert not any(
        name.startswith("session") for name in os.listdir(win._participant.folder)
    )
    assert _all_calibration_artifacts(win._participant.folder) == []


# ── 5: the bundle survives the real CalibrationWindow -> main.open_session
#       -> SessionWindow handoff -- assert on the actual wiring, not a mock
#       of it. Every class main.py constructs is faked (records its own
#       constructor kwargs/callbacks), but the closures under test
#       (open_new_participant/open_calibration/open_session, all defined
#       inside main.main()) are the real, unmodified ones.  ─────────────────

class _FakeWindow:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)
        if "session_num" in kwargs:
            self._session_num = kwargs["session_num"]
        if "on_back" in kwargs:
            self.on_back = kwargs["on_back"]
        type(self).instances.append(self)

    def withdraw(self):
        pass

    def deiconify(self):
        pass

    def protocol(self, *a, **k):
        pass

    def release_camera(self):
        pass

    def retry_camera(self):
        pass

    def show_toast(self, message=None):
        pass

    def destroy(self):
        pass

    def mainloop(self):
        pass


class _FakeLanding(_FakeWindow):
    instances: list = []

    def __init__(self, on_new_participant=None, on_resume_session=None):
        super().__init__(
            on_new_participant=on_new_participant, on_resume_session=on_resume_session,
        )


class _FakeNewParticipant(_FakeWindow):
    instances: list = []


class _FakeCalibration(_FakeWindow):
    instances: list = []


class _FakeResumeSession(_FakeWindow):
    instances: list = []


class _FakeSession(_FakeWindow):
    instances: list = []
    _session_num = 0
    _recording_started = False


def test_bundle_survives_the_real_main_py_handoff_wiring(tmp_path, monkeypatch):
    for cls in (_FakeLanding, _FakeNewParticipant, _FakeCalibration, _FakeSession):
        cls.instances = []

    monkeypatch.setattr(main_module, "LandingWindow", _FakeLanding)
    monkeypatch.setattr(main_module, "NewParticipantWindow", _FakeNewParticipant)
    monkeypatch.setattr(main_module, "CalibrationWindow", _FakeCalibration)
    monkeypatch.setattr(main_module, "SessionWindow", _FakeSession)

    main_module.main()  # app.mainloop() is a no-op on the fake -- returns immediately

    landing = _FakeLanding.instances[-1]
    landing.on_new_participant()  # simulate "New Participant" clicked

    np_win = _FakeNewParticipant.instances[-1]
    record = _make_participant(tmp_path)
    np_win.on_save(record)  # simulate Save -> open_calibration

    cal_win = _FakeCalibration.instances[-1]
    assert cal_win.args[1] is record  # open_calibration passed the real record through

    result = _search_result(passed=True)
    pending = {
        "result": result, "pose_frames": _pose_frames(), "overridden": False,
        "roi": None, "calibration_image_size": [1024, 768],
        "manual_override": False, "auto_params": None,
    }
    cal_win.on_continue(result, pending)  # simulate Continue -> open_session

    session_win = _FakeSession.instances[-1]
    assert session_win.pending_calibration is pending
    assert session_win.participant is record
    assert session_win.calibration_result is result


# ── 6: resume-supersede plan (2026-08-03) D8/D9 -- the Resume Session path
#       through the same real main.py closures, asserting open_calibration
#       computes count_complete_sessions/trailing_incomplete_session and
#       open_session threads them into SessionWindow as
#       session_num/pending_supersede_session. ────────────────────────────

def _write_trials_csv(session_dir, *, valid=30):
    session_dir.mkdir(parents=True, exist_ok=True)
    with open(session_dir / "trials.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=_TRIAL_COLUMNS)
        w.writeheader()
        for trial_id in range(1, valid + 1):
            w.writerow({
                "trial_id": trial_id, "trial_index": trial_id, "start_ts": 0.0,
                "end_ts": 1.0, "voided": False, "target_pct": 50, "target_kg": 5.0,
            })


def test_resume_flow_threads_complete_count_and_pending_supersede(tmp_path, monkeypatch):
    for cls in (_FakeLanding, _FakeNewParticipant, _FakeCalibration, _FakeResumeSession, _FakeSession):
        cls.instances = []

    monkeypatch.setattr(main_module, "LandingWindow", _FakeLanding)
    monkeypatch.setattr(main_module, "NewParticipantWindow", _FakeNewParticipant)
    monkeypatch.setattr(main_module, "CalibrationWindow", _FakeCalibration)
    monkeypatch.setattr(main_module, "ResumeSessionWindow", _FakeResumeSession)
    monkeypatch.setattr(main_module, "SessionWindow", _FakeSession)

    # Fabricate a participant folder with one complete session and one
    # trailing incomplete session, mirroring the resume-supersede plan's
    # end-to-end scenario.
    folder = tmp_path / "participant001_20260803110000"
    folder.mkdir()
    (folder / "participant.json").write_text(json.dumps({
        "schema_version": participants.SCHEMA_VERSION,
        "participant_id": "participant001",
        "created_at": "2026-08-03T11:00:00",
        "age": 22, "weight_kg": 70.0, "height_cm": 175.0,
        "forearm_circ_cm": 26.0, "mvc_kg": 41.719, "activity_level": "moderate",
    }))
    _write_trials_csv(folder / "session1_20260803110100", valid=TRIAL_COUNT)
    _write_trials_csv(folder / "session2_20260803110200", valid=27)
    record = ParticipantRecord(
        participant_id="participant001", folder=str(folder),
        created_at="2026-08-03T11:00:00", age=22, weight_kg=70.0,
        height_cm=175.0, forearm_circ_cm=26.0, mvc_kg=41.719,
        activity_level="moderate",
    )

    main_module.main()

    landing = _FakeLanding.instances[-1]
    landing.on_resume_session()  # simulate "Resume Session" clicked

    resume_win = _FakeResumeSession.instances[-1]
    resume_win.on_continue(record)  # simulate Continue -> open_calibration

    cal_win = _FakeCalibration.instances[-1]
    assert cal_win.args[1] is record

    result = _search_result(passed=True)
    pending = {
        "result": result, "pose_frames": _pose_frames(), "overridden": False,
        "roi": None, "calibration_image_size": [1024, 768],
        "manual_override": False, "auto_params": None,
    }
    cal_win.on_continue(result, pending)  # simulate Continue -> open_session

    session_win = _FakeSession.instances[-1]
    assert session_win.session_num == 1
    assert session_win.pending_supersede_session == "session2_20260803110200"
