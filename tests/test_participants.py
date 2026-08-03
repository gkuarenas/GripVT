"""Headless tests for output.participants.create_participant.

No existing test file covered create_participant before this one (confirmed
via grep -r create_participant tests/ -- no matches). Covers the pre-existing
folder/JSON/key-file behavior plus Phase 2 of
docs/logs/20260730_150048_mvc_file_load_plan.md: the mvc_source_path -> copied
mvc.txt provenance step, and D10's "a failed copy aborts the whole call, no
participant.json is left behind" guarantee.

RECORDINGS_DIR and KEY_PATH are monkeypatched to tmp_path locations (same
module-level-constant-patching idiom as test_session_baseline_gate.py /
test_session_csv_recording.py's camera_feed_module.cv2 patching), so this
suite never touches the real gitignored recordings/ folder or the real
external participant key file.
"""
import csv
import json
import os
from dataclasses import asdict

import numpy as np
import pytest

from core.detector_search import Geometry, PoseDiagnostics, SearchResult
from core.trial_sequence import TRIAL_COUNT
from output import participants
from output.writer import _TRIAL_COLUMNS


@pytest.fixture
def isolated_dirs(tmp_path, monkeypatch):
    recordings_dir = tmp_path / "recordings"
    key_path = tmp_path / "keys" / "participant_key.json"
    monkeypatch.setattr(participants, "RECORDINGS_DIR", str(recordings_dir))
    monkeypatch.setattr(participants, "KEY_PATH", key_path)
    return recordings_dir, key_path


@pytest.fixture
def mvc_source(tmp_path):
    path = tmp_path / "source_mvc.txt"
    path.write_bytes(b"milliSec,CH2,\n0,0.01,\n5,0.02,\n10,42.85,\n")
    return path


def _make_participant(mvc_source_path):
    return participants.create_participant(
        name="Test Participant",
        age=30,
        weight_kg=70.0,
        height_cm=170.0,
        forearm_circ_cm=25.0,
        mvc_kg=42.85,
        activity_level="active",
        mvc_source_path=str(mvc_source_path),
    )


def test_create_participant_writes_folder_json_and_mvc_copy(isolated_dirs, mvc_source):
    recordings_dir, key_path = isolated_dirs

    record = _make_participant(mvc_source)

    folder = recordings_dir / os.path.basename(record.folder)
    assert folder.is_dir()
    assert record.participant_id == "participant001"

    participant_json = folder / "participant.json"
    assert participant_json.exists()
    payload = json.loads(participant_json.read_text())
    assert payload["schema_version"] == participants.SCHEMA_VERSION
    assert payload["participant_id"] == "participant001"
    assert payload["age"] == 30
    assert payload["weight_kg"] == 70.0
    assert payload["height_cm"] == 170.0
    assert payload["forearm_circ_cm"] == 25.0
    assert payload["mvc_kg"] == 42.85
    assert payload["activity_level"] == "active"
    assert "folder" not in payload  # implicit in the folder itself, not data
    assert "name" not in json.dumps(payload)  # de-identified at the source

    mvc_copy = folder / "mvc.txt"
    assert mvc_copy.exists()
    assert mvc_copy.read_bytes() == mvc_source.read_bytes()

    assert key_path.exists()
    key_data = json.loads(key_path.read_text())
    assert key_data["participant001"]["name"] == "Test Participant"


def test_create_participant_missing_mvc_source_raises_and_leaves_no_json(isolated_dirs, tmp_path):
    recordings_dir, _key_path = isolated_dirs
    missing_source = tmp_path / "does_not_exist.txt"
    assert not missing_source.exists()

    with pytest.raises((OSError, FileNotFoundError)):
        _make_participant(missing_source)

    # The folder create_participant attempted to populate must not contain a
    # participant.json -- D10: a failed copy aborts the whole call rather
    # than leaving a record with no backing source file.
    if recordings_dir.exists():
        for folder in recordings_dir.iterdir():
            assert not (folder / "participant.json").exists()


# ── F4: session abandonment record (2026-08-03) ─────────────────────────────
# docs/logs/20260803_session_abandonment_record_plan.md. Restart previously
# wrote session_aborted only when a recording was open, so restarting between
# sessions left no trace of which folders were abandoned.

def _record(folder) -> participants.ParticipantRecord:
    return participants.ParticipantRecord(
        participant_id="participant003", folder=str(folder),
        created_at="2026-08-03T11:32:05", age=22, weight_kg=70.0,
        height_cm=175.0, forearm_circ_cm=26.0, mvc_kg=41.719,
        activity_level="moderate",
    )


def _make_sessions(folder, n):
    folder.mkdir(parents=True, exist_ok=True)
    for i in range(1, n + 1):
        (folder / f"session{i}_2026080311{i:02d}00").mkdir()


def _write_trials_csv(session_dir, *, valid=30, voided=0, open_rows=0):
    """Write a `trials.csv` shaped exactly like `TrialLog.flush()`'s output
    (resume-supersede plan, item 4): real `_TRIAL_COLUMNS` header via
    `csv.DictWriter`, with Python `True`/`False` in `voided` so the file is
    byte-shaped the same way a real recording produces it (`str()`-ed by
    `csv.DictWriter` into the literal strings `"True"`/`"False"`).

    Writes `valid` closed+non-voided rows, then `voided` closed+voided rows,
    then `open_rows` non-voided rows with an empty `end_ts` (still open).
    """
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


def _make_complete_sessions(folder, n):
    """`_make_sessions` plus a `TRIAL_COUNT`-valid-row `trials.csv` in each,
    so every session dir made is *complete* per `_session_is_complete`."""
    _make_sessions(folder, n)
    for i in range(1, n + 1):
        session_dir = folder / f"session{i}_2026080311{i:02d}00"
        _write_trials_csv(session_dir, valid=TRIAL_COUNT)


def test_session_dirs_are_ordered_numerically_not_lexically(tmp_path):
    """session10 must not sort between session1 and session2."""
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 11)
    (folder / "participant.json").write_text("{}")
    names = participants.session_dirs(str(folder))
    assert [int(n.split("_")[0][7:]) for n in names] == list(range(1, 12))


def test_session_dirs_on_a_folder_with_no_sessions_is_empty(tmp_path):
    folder = tmp_path / "participant003_x"
    folder.mkdir()
    assert participants.session_dirs(str(folder)) == []


def test_session_dirs_on_a_missing_folder_is_empty_not_an_error(tmp_path):
    assert participants.session_dirs(str(tmp_path / "nope")) == []


def test_record_abort_all_lists_every_session_dir(tmp_path):
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 3)
    path = participants.record_abort(
        _record(folder), scope="all", reason="sensor slip",
        note="strap loosened", session_num_at_abort=3,
    )
    data = json.loads(open(path).read())
    assert data["participant_id"] == "participant003"
    assert len(data["aborts"]) == 1
    entry = data["aborts"][0]
    assert entry["scope"] == "all"
    assert entry["reason"] == "sensor slip"
    assert entry["note"] == "strap loosened"
    assert entry["session_num_at_abort"] == 3
    assert len(entry["abandoned_sessions"]) == 3


def test_record_abort_session_lists_only_the_current_one(tmp_path):
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 3)
    path = participants.record_abort(
        _record(folder), scope="session", reason="bad baseline", note="",
        session_num_at_abort=3,
    )
    entry = json.loads(open(path).read())["aborts"][0]
    assert entry["abandoned_sessions"] == ["session3_20260803110300"]


def test_second_abort_appends_rather_than_truncating(tmp_path):
    """A visit can be restarted more than once; each is its own fact (D6)."""
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 2)
    rec = _record(folder)
    participants.record_abort(
        rec, scope="session", reason="first", note="", session_num_at_abort=1,
    )
    path = participants.record_abort(
        rec, scope="all", reason="second", note="", session_num_at_abort=2,
    )
    aborts = json.loads(open(path).read())["aborts"]
    assert [a["reason"] for a in aborts] == ["first", "second"]


def test_record_abort_deletes_nothing(tmp_path):
    """D1: the whole point -- abandoned sessions stay exactly where they are."""
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 3)
    (folder / "session2_20260803110200" / "markers.csv").write_text("frame\n1\n")
    before = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))

    participants.record_abort(
        _record(folder), scope="all", reason="x", note="", session_num_at_abort=3,
    )

    after = sorted(p.relative_to(folder).as_posix() for p in folder.rglob("*"))
    assert set(before).issubset(set(after))          # nothing removed
    # D5: aborted.json now lives in aborted/, not the participant root.
    assert set(after) - set(before) == {"aborted", "aborted/aborted.json"}
    assert (folder / "session2_20260803110200" / "markers.csv").read_text() == "frame\n1\n"


def test_aborted_json_carries_no_participant_name(tmp_path):
    """URERB Form 6A / CLAUDE.md: names live only in the external key file."""
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 1)
    path = participants.record_abort(
        _record(folder), scope="all", reason="participant withdrew",
        note="asked to stop", session_num_at_abort=1,
    )
    raw = open(path).read().lower()
    assert "guillaume" not in raw
    assert "name" not in json.loads(raw)["aborts"][0]


# ── Phase 1 (2026-08-03 per-wear-session-folders plan): dest_dir / move API ──
# docs/logs/20260803_per_wear_session_folder_plan.md. save_calibration and
# save_baseline now take a keyword-only dest_dir (no default) so a session's
# artifacts can land in its own sessionN_<ts>/ folder rather than the
# participant root (D4); move_sessions_to_aborted relocates abandoned wears
# into aborted/ without ever deleting (D8/D9).

def _geometry():
    return Geometry(
        axis=0,
        lines=(10.0, 20.0),
        corner=((0.0, 0.0),),
        band=((0.0, 0.0),),
        n_lines=9,
        n_per_line=11,
    )


def _search_result(passed=True):
    pose = PoseDiagnostics(
        n_detected=99,
        bands={"top": 9, "bottom": 9},
        corners={"tl": 3, "tr": 3, "bl": 3, "br": 3},
        oversized=0,
        gaussian_fallbacks=0,
        cost=0.0,
    )
    return SearchResult(
        params={"thresh": 40, "min_area": 10},
        poses={"relaxed": pose, "full_open": pose, "max_grip": pose},
        detections={"relaxed": [], "full_open": [], "max_grip": []},
        geometry=_geometry(),
        grid_shape=(9, 11),
        reference_bands={"top": 9, "bottom": 9},
        reference_corners={"tl": 3, "tr": 3, "bl": 3, "br": 3},
        worst_pose="relaxed",
        worst_count=99,
        passed=passed,
        bounds={"thresh": [30, 60], "min_area": [4, 40]},
    )


def _pose_frames():
    return {
        "relaxed": np.zeros((8, 8), dtype=np.uint8),
        "full_open": np.zeros((8, 8), dtype=np.uint8),
        "max_grip": np.zeros((8, 8), dtype=np.uint8),
    }


def test_save_calibration_honours_dest_dir(tmp_path):
    folder = tmp_path / "participant003_x"
    folder.mkdir()
    session_dir = folder / "session1_20260803110100"
    session_dir.mkdir()

    path = participants.save_calibration(
        _record(folder), _search_result(), _pose_frames(), dest_dir=str(session_dir),
    )

    assert os.path.dirname(path) == str(session_dir)
    assert len(list(session_dir.glob("calibration_*.json"))) == 1
    assert len(list(session_dir.glob("pose_*.png"))) == 3
    # The participant root gets neither -- dest_dir is the only place written to.
    assert list(folder.glob("calibration_*.json")) == []
    assert list(folder.glob("pose_*.png")) == []


def test_save_baseline_honours_dest_dir(tmp_path):
    folder = tmp_path / "participant003_x"
    folder.mkdir()
    session_dir = folder / "session1_20260803110100"
    session_dir.mkdir()

    path = participants.save_baseline(
        _record(folder),
        dest_dir=str(session_dir),
        locked_count=99,
        passed=True,
        overridden=False,
        corners={"tl": 3},
        bands={"top": 9},
        params={"thresh": 40, "min_area": 10},
    )

    assert os.path.dirname(path) == str(session_dir)
    assert len(list(session_dir.glob("baseline_*.json"))) == 1
    assert list(folder.glob("baseline_*.json")) == []


def test_save_calibration_failure_lands_in_aborted(tmp_path):
    folder = tmp_path / "participant003_x"
    folder.mkdir()

    path = participants.save_calibration_failure(
        _record(folder),
        _pose_frames(),
        detections={"relaxed": [(1.0, 2.0, 3.0)]},
        message="anchor pose unusable",
    )

    aborted_dir = folder / "aborted"
    assert os.path.dirname(path) == str(aborted_dir)
    assert len(list(aborted_dir.glob("calibration_failed_*.json"))) == 1
    assert len(list(aborted_dir.glob("failed_*.png"))) == 3
    assert list(folder.glob("calibration_failed_*.json")) == []
    assert list(folder.glob("failed_*.png")) == []


def test_move_sessions_to_aborted_is_byte_identical_and_cleans_root(tmp_path):
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 3)
    (folder / "participant.json").write_text("{}")
    (folder / "mvc.txt").write_text("milliSec,CH2,\n")
    payload = b"frame,x,y\n1,10.5,20.25\n2,11.0,20.75\n"
    (folder / "session2_20260803110200" / "markers.csv").write_bytes(payload)

    moved = participants.move_sessions_to_aborted(
        _record(folder),
        ["session1_20260803110100", "session2_20260803110200"],
    )

    assert moved == ["session1_20260803110100", "session2_20260803110200"]
    aborted_dir = folder / "aborted"
    assert (aborted_dir / "session2_20260803110200" / "markers.csv").read_bytes() == payload
    assert not (folder / "session1_20260803110100").exists()
    assert not (folder / "session2_20260803110200").exists()

    remaining = {p.name for p in folder.iterdir()}
    assert remaining == {"participant.json", "mvc.txt", "session3_20260803110300", "aborted"}


def test_move_sessions_to_aborted_skips_already_gone_name(tmp_path):
    """D9: a name that no longer exists is skipped rather than raised."""
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 1)

    moved = participants.move_sessions_to_aborted(
        _record(folder), ["session9_20260803110900"],
    )

    assert moved == []
    assert (folder / "session1_20260803110100").exists()


def test_move_sessions_to_aborted_returns_only_names_actually_moved(tmp_path):
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 2)

    moved = participants.move_sessions_to_aborted(
        _record(folder),
        ["session1_20260803110100", "session_missing", "session2_20260803110200"],
    )

    assert moved == ["session1_20260803110100", "session2_20260803110200"]


# ── Resume Session picker (2026-08-03 resume_session_window plan): D1-D9 ─────
# docs/logs/20260803_191346_resume_session_window_plan.md. list_resumable_
# participants/load_participant back the Resume Session picker; the scan
# never opens participant.json (D4), never consults aborted.json (D8), and
# sorts most-recent-session-first (D9).

def _write_participant_json(folder, participant_id="participant001", **overrides):
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": participants.SCHEMA_VERSION,
        "participant_id": participant_id,
        "created_at": "2026-08-03T11:32:05",
        "age": 22,
        "weight_kg": 70.0,
        "height_cm": 175.0,
        "forearm_circ_cm": 26.0,
        "mvc_kg": 41.719,
        "activity_level": "moderate",
    }
    payload.update(overrides)
    (folder / "participant.json").write_text(json.dumps(payload))
    return payload


def test_list_resumable_participants_excludes_folder_without_participant_json(isolated_dirs):
    recordings_dir, _key_path = isolated_dirs
    folder = recordings_dir / "participant001_20260803110000"
    folder.mkdir(parents=True)  # no participant.json inside

    result = participants.list_resumable_participants(10)

    assert result == []


def test_list_resumable_participants_excludes_folder_at_max_sessions(isolated_dirs):
    recordings_dir, _key_path = isolated_dirs
    folder = recordings_dir / "participant001_20260803110000"
    _write_participant_json(folder)
    # D6: the cap now gates on *complete* sessions, so bare dirs (as the old
    # fixture used) would stay resumable -- use complete ones to actually
    # exercise the at-cap exclusion.
    _make_complete_sessions(folder, 3)

    result = participants.list_resumable_participants(3)

    assert result == []
    # One session short of the cap, the same folder is resumable.
    result_below_cap = participants.list_resumable_participants(4)
    assert [r.participant_id for r in result_below_cap] == ["participant001"]


def test_list_resumable_participants_includes_zero_session_folder(isolated_dirs):
    recordings_dir, _key_path = isolated_dirs
    folder = recordings_dir / "participant001_20260803110000"
    _write_participant_json(folder)

    result = participants.list_resumable_participants(10)

    assert len(result) == 1
    assert result[0].complete_session_count == 0
    assert result[0].last_session_at is None


def test_list_resumable_participants_orders_most_recent_session_first(isolated_dirs):
    recordings_dir, _key_path = isolated_dirs
    # _last_session_at parses the "%Y%m%d_%H%M%S" suffix (D7) -- the
    # timestamp needs its own internal underscore, matching writer.py's
    # make_session_dir (session{n}_{ts} with ts = "%Y%m%d_%H%M%S").
    older = recordings_dir / "participant001_20260803_110000"
    older.mkdir(parents=True)
    _write_participant_json(older, participant_id="participant001")
    (older / "session1_20260803_110100").mkdir()  # 11:01:00

    newer = recordings_dir / "participant002_20260803_120000"
    newer.mkdir(parents=True)
    _write_participant_json(newer, participant_id="participant002")
    (newer / "session1_20260803_150000").mkdir()  # 15:00:00, well after `older`

    result = participants.list_resumable_participants(10)

    assert [r.participant_id for r in result] == ["participant002", "participant001"]
    assert result[0].last_session_at > result[1].last_session_at


def test_list_resumable_participants_skips_unmatched_entry_without_raising(isolated_dirs):
    recordings_dir, _key_path = isolated_dirs
    folder = recordings_dir / "participant001_20260803110000"
    _write_participant_json(folder)
    # A stray file sitting alongside real participant folders -- not a
    # participantNNN_<ts> directory at all -- must be skipped, not raise.
    recordings_dir.mkdir(parents=True, exist_ok=True)
    (recordings_dir / "readme.txt").write_text("not a participant folder")

    result = participants.list_resumable_participants(10)

    assert [r.participant_id for r in result] == ["participant001"]


def test_load_participant_round_trips_every_field(isolated_dirs, mvc_source):
    recordings_dir, _key_path = isolated_dirs
    record = _make_participant(mvc_source)
    folder = str(recordings_dir / os.path.basename(record.folder))

    loaded = participants.load_participant(folder)

    assert loaded.folder == folder
    assert loaded.participant_id == record.participant_id
    assert loaded.created_at == record.created_at
    assert loaded.age == record.age
    assert loaded.weight_kg == record.weight_kg
    assert loaded.height_cm == record.height_cm
    assert loaded.forearm_circ_cm == record.forearm_circ_cm
    assert loaded.mvc_kg == record.mvc_kg
    assert loaded.activity_level == record.activity_level


def test_load_participant_ignores_unknown_extra_key(isolated_dirs):
    recordings_dir, _key_path = isolated_dirs
    folder = recordings_dir / "participant001_20260803110000"
    _write_participant_json(folder, some_future_field="unexpected")

    loaded = participants.load_participant(str(folder))

    assert loaded.participant_id == "participant001"
    assert not hasattr(loaded, "some_future_field")


def test_resumable_participants_and_load_participant_never_carry_a_name(isolated_dirs, mvc_source):
    recordings_dir, key_path = isolated_dirs
    record = _make_participant(mvc_source)  # writes the name only to KEY_PATH
    folder = str(recordings_dir / os.path.basename(record.folder))

    resumables = participants.list_resumable_participants(10)
    loaded = participants.load_participant(folder)

    resumables_dump = json.dumps(
        [{**asdict(r), "last_session_at": str(r.last_session_at)} for r in resumables]
    )
    assert "Test Participant" not in resumables_dump
    assert "Test Participant" not in json.dumps(asdict(loaded))
    # KEY_PATH itself is untouched by either read path -- it still holds the name.
    assert key_path.exists()
    assert json.loads(key_path.read_text())["participant001"]["name"] == "Test Participant"


# ── Resume-supersede plan (2026-08-03): count_complete_sessions /
# trailing_incomplete_session, D1-D4. See
# docs/logs/20260803_203149_resume_session_complete_count_plan.md.

def test_count_complete_sessions_counts_a_fully_recorded_session(tmp_path):
    folder = tmp_path / "participant003_x"
    session_dir = folder / "session1_20260803110100"
    _write_trials_csv(session_dir, valid=TRIAL_COUNT)

    assert participants.count_complete_sessions(str(folder)) == 1


def test_count_complete_sessions_no_trials_csv_counts_zero_not_an_error(tmp_path):
    folder = tmp_path / "participant003_x"
    _make_sessions(folder, 1)  # bare dir, no trials.csv at all

    assert participants.count_complete_sessions(str(folder)) == 0


def test_count_complete_sessions_27_valid_rows_counts_zero(tmp_path):
    folder = tmp_path / "participant003_x"
    session_dir = folder / "session1_20260803110100"
    _write_trials_csv(session_dir, valid=27)

    assert participants.count_complete_sessions(str(folder)) == 0


def test_count_complete_sessions_voided_rows_dont_count_toward_30(tmp_path):
    folder = tmp_path / "participant003_x"
    session_dir = folder / "session1_20260803110100"
    _write_trials_csv(session_dir, valid=27, voided=3)

    assert participants.count_complete_sessions(str(folder)) == 0


def test_count_complete_sessions_open_rows_dont_count_toward_30(tmp_path):
    folder = tmp_path / "participant003_x"
    session_dir = folder / "session1_20260803110100"
    _write_trials_csv(session_dir, valid=28, open_rows=2)

    assert participants.count_complete_sessions(str(folder)) == 0


def test_count_and_trailing_incomplete_mixed_folder(tmp_path):
    folder = tmp_path / "participant003_x"
    _write_trials_csv(folder / "session1_20260803110100", valid=TRIAL_COUNT)
    _write_trials_csv(folder / "session2_20260803110200", valid=27)

    assert participants.count_complete_sessions(str(folder)) == 1
    assert participants.trailing_incomplete_session(str(folder)) == "session2_20260803110200"


def test_trailing_incomplete_session_is_none_when_every_session_is_complete(tmp_path):
    folder = tmp_path / "participant003_x"
    _make_complete_sessions(folder, 2)

    assert participants.trailing_incomplete_session(str(folder)) is None


def test_zero_sessions_folder_counts_and_trailing_are_both_empty(tmp_path):
    folder = tmp_path / "participant003_x"
    folder.mkdir()

    assert participants.count_complete_sessions(str(folder)) == 0
    assert participants.trailing_incomplete_session(str(folder)) is None


def test_corrupt_trials_csv_counts_zero_not_an_error(tmp_path):
    folder = tmp_path / "participant003_x"
    session_dir = folder / "session1_20260803110100"
    session_dir.mkdir(parents=True)
    (session_dir / "trials.csv").write_bytes(b"\x00\x01\xffnot,a,real,csv\xfe\xfd")

    assert participants.count_complete_sessions(str(folder)) == 0

    # Header-only (no data rows) is likewise 0, not an error.
    (session_dir / "trials.csv").write_text(",".join(_TRIAL_COLUMNS) + "\n")
    assert participants.count_complete_sessions(str(folder)) == 0


def test_missing_folder_counts_and_trailing_are_both_empty(tmp_path):
    missing = tmp_path / "does_not_exist"

    assert participants.count_complete_sessions(str(missing)) == 0
    assert participants.trailing_incomplete_session(str(missing)) is None
