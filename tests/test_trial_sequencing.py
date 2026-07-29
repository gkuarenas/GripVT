"""Headless tests for Phases 3-5 of
docs/logs/20260728_160955_trial_sequencing_target_force_plan.md -- SessionWindow's
_sequence state, the two pure lookup helpers (_target_pct_for_trial /
_target_kg_for_trial), the live TARGET FORCE block (_refresh_target_block), and
(Phase 5) generating/locking/persisting the sequence at a confirmed Start.
No visible UI change lands until later phases; this file only proves the state,
the D2 indexing mapping (sequence[trial - 1]), and the Phase 4 rendering rules.

Reuses the connected-camera stub idiom from test_session_csv_recording.py.
"""
from collections import Counter
import csv
import json
import os

import pytest

import ui.session.camera_feed as camera_feed_module
import ui.session.recording as recording_module
import ui.theme as theme
from core.trial_sequence import SequenceGenerationError, TRIAL_COUNT, generate_sequence, is_valid
from output.participants import ParticipantRecord
from ui.session import SessionWindow

from tests.test_session_csv_recording import (
    MARKER_COUNT, _StubCaptureConnected, _install_fake_process_frame, _lock_baseline,
    _make_participant,
)


@pytest.fixture
def root(tk_root):
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


def test_fresh_window_has_no_sequence(connected_window):
    win = connected_window
    assert win._trial == 0
    assert win._trial_max == 30
    assert win._trial_max == TRIAL_COUNT
    assert win._sequence is None
    assert win._target_pct_for_trial() is None
    assert win._target_kg_for_trial() is None


@pytest.mark.parametrize("trial", [1, 15, 30])
def test_helpers_index_sequence_at_trial_minus_one(connected_window, tmp_path, trial):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    win._sequence = generate_sequence(seed=12345)
    win._trial = trial

    expected_pct = win._sequence.levels_pct[trial - 1]
    assert win._target_pct_for_trial() == expected_pct

    expected_kg = round(expected_pct / 100 * win._participant.mvc_kg, 3)
    assert win._target_kg_for_trial() == expected_kg


def test_target_kg_none_when_no_participant_even_with_sequence(connected_window):
    win = connected_window
    win._participant = None
    win._sequence = generate_sequence(seed=999)
    win._trial = 1

    assert win._target_pct_for_trial() == win._sequence.levels_pct[0]
    assert win._target_kg_for_trial() is None


@pytest.mark.parametrize("trial", [1, 30])
def test_refresh_target_block_live_values(connected_window, tmp_path, trial):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    win._sequence = generate_sequence(seed=12345)
    win._trial = trial

    win._refresh_target_block()

    pct = win._sequence.levels_pct[trial - 1]
    kg = round(pct / 100 * win._participant.mvc_kg, 3)
    assert win._target_num_label.cget("text") == f"{kg:.1f}"
    assert win._target_sub_label.cget("text") == f"{pct}% MVC"
    assert win._target_num_label.cget("fg") == theme.INK
    assert win._target_sub_label.cget("fg") == theme.INK


def test_refresh_target_block_at_zero_pct_trial(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    win._sequence = generate_sequence(seed=12345)
    zero_index = win._sequence.levels_pct.index(0)
    win._trial = zero_index + 1

    win._refresh_target_block()

    assert win._target_num_label.cget("text") == "0.0"
    assert win._target_sub_label.cget("text") == "0% MVC"
    assert win._target_num_label.cget("fg") == theme.INK
    assert win._target_sub_label.cget("fg") == theme.INK


def test_refresh_target_block_placeholder_before_start(connected_window):
    win = connected_window
    assert win._sequence is None

    win._refresh_target_block()

    assert win._target_num_label.cget("text") == "—"
    assert win._target_sub_label.cget("text") == "AWAITING START"
    assert win._target_num_label.cget("fg") == theme.INK_55
    assert win._target_sub_label.cget("fg") == theme.INK_55


def test_refresh_target_block_kg_placeholder_without_participant(connected_window):
    win = connected_window
    win._participant = None
    win._sequence = generate_sequence(seed=999)
    win._trial = 1

    win._refresh_target_block()

    pct = win._sequence.levels_pct[0]
    assert win._target_num_label.cget("text") == "—"
    assert win._target_sub_label.cget("text") == f"{pct}% MVC"
    assert win._target_num_label.cget("fg") == theme.INK
    assert win._target_sub_label.cget("fg") == theme.INK


# ── Phase 5: generate, lock, and persist the sequence at Start ─────────────


def _make_zero_mvc_participant(tmp_path) -> ParticipantRecord:
    folder = tmp_path / "participant998_test"
    folder.mkdir()
    return ParticipantRecord(
        participant_id="participant998", folder=str(folder),
        created_at="2026-01-01T00:00:00", age=30, weight_kg=70.0, height_cm=170.0,
        forearm_circ_cm=25.0, mvc_kg=0.0, activity_level="active",
    )


def test_confirm_start_writes_balanced_valid_sequence_json(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()

    session_dir = win._writer.session_dir
    with open(os.path.join(session_dir, "sequence.json")) as f:
        payload = json.load(f)

    assert len(payload["sequence_pct"]) == TRIAL_COUNT
    assert Counter(payload["sequence_pct"]) == Counter({0: 6, 5: 6, 20: 6, 50: 6, 80: 6})
    assert is_valid(payload["sequence_pct"])
    assert payload["participant_id"] == "participant999"
    assert payload["mvc_kg"] == 40.0


def test_second_start_after_stop_writes_different_sequence_and_resets_display(
    connected_window, tmp_path,
):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    first_dir = win._writer.session_dir
    first_sequence = win._sequence
    win._poll()
    win._confirm_stop()

    win._confirm_start()
    second_dir = win._writer.session_dir

    assert second_dir != first_dir
    with open(os.path.join(second_dir, "sequence.json")) as f:
        second_payload = json.load(f)
    assert second_payload["seed"] != first_sequence.seed
    assert tuple(second_payload["sequence_pct"]) != first_sequence.levels_pct

    assert win._trial == 1
    assert win._trial_value_label.cget("text") == "1 / 30"


def test_target_force_block_at_start_equals_sequence_first_entry(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()

    pct = win._sequence.levels_pct[0]
    kg = round(pct / 100 * win._participant.mvc_kg, 3)
    assert win._target_num_label.cget("text") == f"{kg:.1f}"
    assert win._target_sub_label.cget("text") == f"{pct}% MVC"


def test_sequence_generation_error_aborts_start_before_any_mutation(
    connected_window, tmp_path, monkeypatch,
):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    def _raise(seed=None):
        raise SequenceGenerationError("no valid sequence found")

    monkeypatch.setattr(recording_module, "generate_sequence", _raise)
    before = set(os.listdir(win._participant.folder))

    win._confirm_start()

    assert win._recording is False
    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None
    assert win._sequence is None
    assert set(os.listdir(win._participant.folder)) == before
    assert win._toast_label.cget("text") == "Could not generate a trial sequence — Start again"


def test_zero_mvc_blocks_start_with_toast(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_zero_mvc_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)
    before = set(os.listdir(win._participant.folder))

    win._confirm_start()

    assert win._recording is False
    assert win._writer is None
    assert win._trial_log is None
    assert win._event_log is None
    assert win._trial == 0
    assert win._sequence is None
    assert set(os.listdir(win._participant.folder)) == before
    assert win._toast_label.cget("text") == (
        "Participant MVC is 0 kg — re-enter MVC before recording"
    )


# ── Phase 6: Next Trial / Undo <-> sequence-index correctness ──────────────


def _target_label_text(win):
    return win._target_num_label.cget("text"), win._target_sub_label.cget("text")


def _expected_label_text(win):
    pct = win._sequence.levels_pct[win._trial - 1]
    kg = round(pct / 100 * win._participant.mvc_kg, 3)
    return f"{kg:.1f}", f"{pct}% MVC"


def test_full_walk_target_matches_sequence_at_every_trial(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    assert win._trial == 1
    assert _target_label_text(win) == _expected_label_text(win)

    while win._trial < TRIAL_COUNT:
        win._on_next_trial()
        assert _target_label_text(win) == _expected_label_text(win)

    assert win._trial == TRIAL_COUNT


def test_undo_restores_identical_target_label_and_trial_index(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    for _ in range(4):
        win._on_next_trial()
    assert win._trial == 5
    before = _target_label_text(win)
    voided_segment_id = win._segment_id

    win._on_next_trial()
    assert win._trial == 6

    win._on_undo_trial()
    assert win._trial == 5
    assert _target_label_text(win) == before

    win._trial_log.flush()
    with open(os.path.join(win._writer.session_dir, "trials.csv"), newline="") as f:
        rows = {r["trial_id"]: r for r in csv.DictReader(f)}
    voided_row = rows[str(voided_segment_id)]
    redo_row = rows[str(voided_segment_id + 1)]
    assert voided_row["voided"] == "True"
    assert redo_row["voided"] == "False"
    assert voided_row["trial_index"] == redo_row["trial_index"] == "5"


def test_interleaved_next_undo_walk_never_desyncs(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    assert _target_label_text(win) == _expected_label_text(win)

    steps = ["next", "next", "undo", "next", "undo", "undo", "next"]
    for step in steps:
        if step == "next":
            win._on_next_trial()
        else:
            win._on_undo_trial()
        assert _target_label_text(win) == _expected_label_text(win)


# ── Phase 7: trials.csv target_pct/target_kg columns ────────────────────────


def test_trials_csv_target_columns_match_sequence_including_voided_rows(
    connected_window, tmp_path,
):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    _install_fake_process_frame(win, MARKER_COUNT)

    win._confirm_start()
    win._on_next_trial()
    win._on_next_trial()
    win._on_undo_trial()
    win._on_next_trial()

    win._trial_log.flush()
    with open(os.path.join(win._writer.session_dir, "trials.csv"), newline="") as f:
        rows = list(csv.DictReader(f))

    assert len(rows) > 0
    for row in rows:
        trial_index = int(row["trial_index"])
        expected_pct = win._sequence.levels_pct[trial_index - 1]
        expected_kg = round(expected_pct / 100 * win._participant.mvc_kg, 3)
        assert int(row["target_pct"]) == expected_pct
        assert float(row["target_kg"]) == expected_kg


def test_next_trial_before_start_is_noop_even_when_locked(connected_window, tmp_path):
    win = connected_window
    win._participant = _make_participant(tmp_path)
    _lock_baseline(win)
    assert win._locked() is True
    assert win._recording is False
    assert win._trial == 0
    assert win._sequence is None

    win._on_next_trial()

    assert win._trial == 0
    assert win._sequence is None
