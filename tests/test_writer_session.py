"""Headless tests for output/writer.py's Phase 1 and Phase 2 changes
(session-dir numbering, plain filenames, trial_id tagging, trials.csv,
events.json, and the wall-clock helper) -- see
docs/logs/20260728_121200_session_csv_recording_plan.md. Pure tmp_path-based:
no Tk, no camera, no mocking beyond constructing fake MarkerRecord instances
directly (and, for the wall-clock test, patching time.monotonic()).
"""
import csv
import json
import os
from unittest.mock import patch

from core.tracker import MarkerRecord
from core.trial_sequence import FORBIDDEN_ADJACENT, LEVELS_PCT, REPS_PER_LEVEL, generate_sequence
from output.writer import (
    COLUMNS,
    CSVWriter,
    EventLog,
    SEQUENCE_SCHEMA_VERSION,
    TrialLog,
    anchor_wall_clock,
    make_session_dir,
    now_ms,
    save_sequence,
)


def make_record(marker_id: int) -> MarkerRecord:
    return MarkerRecord(
        marker_id=marker_id,
        x=10.0 + marker_id,
        y=20.0 + marker_id,
        area=30.0,
        dx=1.0,
        dy=2.0,
        dA=0.5,
        magnitude=2.236,
        predicted_x=10.5 + marker_id,
        predicted_y=20.5 + marker_id,
        autofilled=False,
        size_moments=29.5,
    )


def read_rows(csv_path: str) -> list[dict]:
    with open(csv_path, newline="") as f:
        return list(csv.DictReader(f))


# ── COLUMNS / header shape ──────────────────────────────────────────────────


def test_columns_has_fifteen_entries_trial_id_last():
    assert len(COLUMNS) == 15
    assert COLUMNS[-1] == "trial_id"
    assert COLUMNS[:14] == [
        "frame", "timestamp_ms", "marker_id",
        "x", "y", "area", "dx", "dy", "dA",
        "magnitude",
        "size_moments",
        "predicted_x", "predicted_y", "autofilled",
    ]


def test_csv_header_matches_columns(tmp_path):
    session_dir = tmp_path / "session1_20260728_120000"
    writer = CSVWriter(str(session_dir))
    writer.close()

    with open(writer.csv_path, newline="") as f:
        header = next(csv.reader(f))
    assert header == COLUMNS


# ── plain filenames, no embedded timestamp ──────────────────────────────────


def test_filenames_are_plain_no_timestamp_suffix(tmp_path):
    session_dir = tmp_path / "session1_20260728_120000"
    writer = CSVWriter(str(session_dir))
    writer.close()

    assert os.path.basename(writer.csv_path) == "markers.csv"
    assert os.path.basename(writer.png_path) == "overlay.png"


# ── buffer_frame / write_window / trial_id tagging ──────────────────────────


def test_buffer_and_write_tags_trial_id(tmp_path):
    writer = CSVWriter(str(tmp_path / "session1_20260728_120000"))
    records_a = [make_record(0), make_record(1)]
    records_b = [make_record(0), make_record(1)]

    writer.buffer_frame(records_a, frame_idx=10, timestamp_ms=100.0, trial_id=0)
    writer.buffer_frame(records_b, frame_idx=11, timestamp_ms=133.0, trial_id=1)
    writer.write_window()
    writer.close()

    rows = read_rows(writer.csv_path)
    assert len(rows) == 4
    assert [r["trial_id"] for r in rows] == ["0", "0", "1", "1"]
    assert [r["frame"] for r in rows] == ["10", "10", "11", "11"]


def test_buffer_frame_default_trial_id_is_zero(tmp_path):
    writer = CSVWriter(str(tmp_path / "session1_20260728_120000"))
    writer.buffer_frame([make_record(0)], frame_idx=1, timestamp_ms=0.0)
    writer.write_window()
    writer.close()

    rows = read_rows(writer.csv_path)
    assert rows[0]["trial_id"] == "0"


def test_write_window_clears_pending_and_flushes(tmp_path):
    writer = CSVWriter(str(tmp_path / "session1_20260728_120000"))
    writer.buffer_frame([make_record(0)], frame_idx=1, timestamp_ms=0.0, trial_id=0)
    writer.write_window()

    # write_window() flushes (D7) -- the row must be on disk even though the
    # writer's file handle is still open (not yet closed).
    rows = read_rows(writer.csv_path)
    assert len(rows) == 1
    assert writer._pending == []
    writer.close()


def test_discard_window_drops_pending_rows(tmp_path):
    writer = CSVWriter(str(tmp_path / "session1_20260728_120000"))
    writer.buffer_frame([make_record(0)], frame_idx=1, timestamp_ms=0.0, trial_id=0)
    writer.discard_window()
    writer.write_window()
    writer.close()

    rows = read_rows(writer.csv_path)
    assert rows == []


def test_row_count_equals_frames_times_marker_count(tmp_path):
    writer = CSVWriter(str(tmp_path / "session1_20260728_120000"))
    for frame_idx in range(5):
        records = [make_record(m) for m in range(3)]
        writer.buffer_frame(records, frame_idx=frame_idx, timestamp_ms=frame_idx * 33.0, trial_id=0)
    writer.write_window()
    writer.close()

    rows = read_rows(writer.csv_path)
    assert len(rows) == 5 * 3


# ── make_session_dir numbering ──────────────────────────────────────────────


def test_make_session_dir_first_session(tmp_path):
    participant_folder = tmp_path / "participant001_20260728_100000"
    participant_folder.mkdir()

    session_dir = make_session_dir(str(participant_folder))

    assert os.path.basename(session_dir).startswith("session1_")
    assert os.path.isdir(session_dir)


def test_make_session_dir_scans_existing_and_increments(tmp_path):
    participant_folder = tmp_path / "participant001_20260728_100000"
    participant_folder.mkdir()
    (participant_folder / "session1_20260728_100100").mkdir()
    (participant_folder / "session2_20260728_100200").mkdir()

    session_dir = make_session_dir(str(participant_folder))

    assert os.path.basename(session_dir).startswith("session3_")


def test_make_session_dir_ignores_unrelated_names(tmp_path):
    participant_folder = tmp_path / "participant001_20260728_100000"
    participant_folder.mkdir()
    (participant_folder / "session1_20260728_100100").mkdir()
    (participant_folder / "participant.json").write_text("{}")
    (participant_folder / "pose_front_20260728_100100.png").write_text("")

    session_dir = make_session_dir(str(participant_folder))

    assert os.path.basename(session_dir).startswith("session2_")


def test_make_session_dir_missing_participant_folder_starts_at_one(tmp_path):
    participant_folder = tmp_path / "does_not_exist_yet"

    session_dir = make_session_dir(str(participant_folder))

    assert os.path.basename(session_dir).startswith("session1_")
    assert os.path.isdir(session_dir)


# ── TrialLog ─────────────────────────────────────────────────────────────────


def test_trial_log_open_close_flush_round_trips(tmp_path):
    log = TrialLog(str(tmp_path / "session1_20260728_120000"))
    log.open_trial(trial_id=0, trial_index=1, start_ts=1000.0)
    log.close_trial(trial_id=0, end_ts=2000.0)
    log.flush()

    rows = read_rows(log.csv_path)
    assert len(rows) == 1
    row = rows[0]
    assert set(row.keys()) == {
        "trial_id", "trial_index", "start_ts", "end_ts", "voided",
        "target_pct", "target_kg",
    }
    assert row["trial_id"] == "0"
    assert row["trial_index"] == "1"
    assert row["start_ts"] == "1000.0"
    assert row["end_ts"] == "2000.0"
    assert row["voided"] == "False"


def test_trial_log_open_trial_defaults_target_columns_to_empty_string(tmp_path):
    log = TrialLog(str(tmp_path / "session1_20260728_120000"))
    log.open_trial(trial_id=0, trial_index=1, start_ts=0.0)
    log.flush()

    rows = read_rows(log.csv_path)
    assert rows[0]["target_pct"] == ""
    assert rows[0]["target_kg"] == ""


def test_trial_log_open_trial_stores_target_pct_and_kg(tmp_path):
    log = TrialLog(str(tmp_path / "session1_20260728_120000"))
    log.open_trial(trial_id=0, trial_index=1, start_ts=0.0, target_pct=20, target_kg=8.0)
    log.flush()

    rows = read_rows(log.csv_path)
    assert rows[0]["target_pct"] == "20"
    assert rows[0]["target_kg"] == "8.0"


def test_trial_log_open_trial_target_pct_zero_is_not_treated_as_missing(tmp_path):
    log = TrialLog(str(tmp_path / "session1_20260728_120000"))
    log.open_trial(trial_id=0, trial_index=1, start_ts=0.0, target_pct=0, target_kg=0.0)
    log.flush()

    rows = read_rows(log.csv_path)
    assert rows[0]["target_pct"] == "0"
    assert rows[0]["target_kg"] == "0.0"


def test_trial_log_void_only_affects_voided_trial(tmp_path):
    log = TrialLog(str(tmp_path / "session1_20260728_120000"))
    log.open_trial(trial_id=0, trial_index=1, start_ts=0.0)
    log.close_trial(trial_id=0, end_ts=100.0)
    log.open_trial(trial_id=1, trial_index=2, start_ts=100.0)
    log.close_trial(trial_id=1, end_ts=200.0)
    log.void_trial(trial_id=0)
    log.flush()

    rows = {r["trial_id"]: r for r in read_rows(log.csv_path)}
    assert rows["0"]["voided"] == "True"
    assert rows["1"]["voided"] == "False"
    # Untouched trial's other fields are unaffected by voiding its sibling.
    assert rows["1"]["trial_index"] == "2"
    assert rows["1"]["start_ts"] == "100.0"
    assert rows["1"]["end_ts"] == "200.0"


def test_trial_log_flush_rewrites_not_appends(tmp_path):
    log = TrialLog(str(tmp_path / "session1_20260728_120000"))
    log.open_trial(trial_id=0, trial_index=1, start_ts=0.0)
    log.flush()
    log.flush()
    log.flush()

    rows = read_rows(log.csv_path)
    assert len(rows) == 1

    log.close_trial(trial_id=0, end_ts=50.0)
    for _ in range(5):
        log.flush()

    rows = read_rows(log.csv_path)
    assert len(rows) == 1
    assert rows[0]["end_ts"] == "50.0"


# ── EventLog ─────────────────────────────────────────────────────────────────


def test_event_log_envelope_shape_and_round_trip(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    log = EventLog(
        session_dir,
        participant_id="participant001",
        started_at="2026-07-28T12:00:00",
        frame_index_at_start=42,
    )
    log.append({"type": "camera_loss", "start_ts": 100.0, "end_ts": 200.0})
    log.append({"type": "rate_fallback", "target_hz": 15, "ts": 300.0})
    log.flush()

    with open(log.json_path) as f:
        envelope = json.load(f)

    assert envelope["schema_version"] == 1
    assert envelope["participant_id"] == "participant001"
    assert envelope["session_dir"] == session_dir
    assert envelope["started_at"] == "2026-07-28T12:00:00"
    assert envelope["frame_index_at_start"] == 42
    assert envelope["events"] == [
        {"type": "camera_loss", "start_ts": 100.0, "end_ts": 200.0},
        {"type": "rate_fallback", "target_hz": 15, "ts": 300.0},
    ]


def test_event_log_no_participant_name_in_envelope(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    log = EventLog(
        session_dir,
        participant_id="participant001",
        started_at="2026-07-28T12:00:00",
        frame_index_at_start=0,
    )
    log.flush()

    with open(log.json_path) as f:
        contents = f.read()
    assert "participant_id" in contents
    # No free-text name field anywhere in the de-identified envelope.
    assert "name" not in json.loads(contents)


def test_event_log_flush_is_atomic_no_tmp_file_left_behind(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    log = EventLog(
        session_dir,
        participant_id="participant001",
        started_at="2026-07-28T12:00:00",
        frame_index_at_start=0,
    )
    log.append({"type": "session_aborted", "scope": "session", "ts": 10.0})
    log.flush()

    assert os.path.exists(log.json_path)
    assert not os.path.exists(log.json_path + ".tmp")


# ── wall-clock helper ────────────────────────────────────────────────────────


def test_now_ms_monotonically_non_decreasing_with_real_clock():
    t0_wall_ms, t0_monotonic = anchor_wall_clock()

    readings = [now_ms(t0_wall_ms, t0_monotonic) for _ in range(5)]

    assert readings == sorted(readings)
    assert all(r >= t0_wall_ms for r in readings)


def test_now_ms_anchoring_formula_with_mocked_monotonic():
    t0_wall_ms = 1_700_000_000_000.0
    t0_monotonic = 500.0

    with patch("output.writer.time.monotonic", return_value=502.5):
        result = now_ms(t0_wall_ms, t0_monotonic)

    assert result == 1_700_000_000_000.0 + 2500.0


def test_anchor_wall_clock_returns_matched_pair():
    with patch("output.writer.time.time", return_value=1_700_000_000.0), \
         patch("output.writer.time.monotonic", return_value=123.456):
        t0_wall_ms, t0_monotonic = anchor_wall_clock()

    assert t0_wall_ms == 1_700_000_000_000.0
    assert t0_monotonic == 123.456


# ── save_sequence ────────────────────────────────────────────────────────────


def test_save_sequence_writes_sequence_json_in_session_dir(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=40.0)

    assert path == os.path.join(session_dir, "sequence.json")
    assert os.path.exists(path)


def test_save_sequence_envelope_has_all_schema_keys(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=40.0)
    with open(path) as f:
        payload = json.load(f)

    assert set(payload.keys()) == {
        "schema_version", "participant_id", "session_dir", "generated_at",
        "generator", "seed", "attempts", "mvc_kg", "levels_pct",
        "reps_per_level", "forbidden_adjacent_pairs", "sequence_pct", "sequence_kg",
    }
    assert payload["schema_version"] == SEQUENCE_SCHEMA_VERSION
    assert payload["participant_id"] == "participant007"
    assert payload["session_dir"] == session_dir
    assert payload["generator"] == "balanced_multiset_rejection_v2"
    assert payload["seed"] == seq.seed
    assert payload["attempts"] == seq.attempts
    assert payload["levels_pct"] == list(LEVELS_PCT)
    assert payload["reps_per_level"] == REPS_PER_LEVEL
    assert sorted(tuple(p) for p in payload["forbidden_adjacent_pairs"]) == sorted(
        {tuple(sorted(p, reverse=True)) for p in FORBIDDEN_ADJACENT}
    )
    assert [0, 0] in payload["forbidden_adjacent_pairs"]


def test_save_sequence_has_thirty_pct_and_kg_entries(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=40.0)
    with open(path) as f:
        payload = json.load(f)

    assert len(payload["sequence_pct"]) == 30
    assert len(payload["sequence_kg"]) == 30
    assert payload["sequence_pct"] == list(seq.levels_pct)


def test_save_sequence_kg_matches_pct_over_mvc_conversion(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)
    mvc = 33.333

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=mvc)
    with open(path) as f:
        payload = json.load(f)

    assert payload["mvc_kg"] == mvc
    for pct, kg in zip(payload["sequence_pct"], payload["sequence_kg"]):
        assert kg == round(pct / 100 * mvc, 3)


def test_save_sequence_with_no_mvc_writes_null_shape(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=None)
    with open(path) as f:
        payload = json.load(f)

    assert payload["mvc_kg"] is None
    assert payload["sequence_kg"] == [None] * 30
    assert len(payload["sequence_kg"]) == 30


def test_save_sequence_is_atomic_no_tmp_file_left_behind(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=40.0)

    assert os.path.exists(path)
    assert not os.path.exists(path + ".tmp")


def test_save_sequence_no_participant_name_in_payload(tmp_path):
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=1)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=40.0)
    with open(path) as f:
        contents = f.read()

    assert "name" not in json.loads(contents)


def test_save_sequence_seed_round_trips_through_generate_sequence(tmp_path):
    """Success criterion 7, tested at the persistence boundary: replaying the
    logged seed through generate_sequence() must reproduce the persisted
    sequence_pct byte-for-byte, including the rejected shuffle attempts the
    seed also determines."""
    session_dir = str(tmp_path / "session1_20260728_120000")
    os.makedirs(session_dir)
    seq = generate_sequence(seed=987654321)

    path = save_sequence(session_dir, participant_id="participant007", sequence=seq, mvc_kg=40.0)
    with open(path) as f:
        payload = json.load(f)

    replayed = generate_sequence(seed=payload["seed"])
    assert list(replayed.levels_pct) == payload["sequence_pct"]
    assert replayed.attempts == payload["attempts"]
