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
import json
import os

import pytest

from output import participants


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
    assert set(after) - set(before) == {"aborted.json"}   # only the record added
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
