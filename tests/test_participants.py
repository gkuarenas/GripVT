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
