"""Participant persistence for the GripVT human-data app.

Enforces the URERB Form 6A data-management rule: the de-identified record
(coded ID + anthropometrics) lands in ``recordings/participantNNN_<ts>/``,
while the participant's name is written only to an external key file, so the
recordings tree can be backed up without leaking identities.
"""
import csv
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from core.detector_search import MIN_ACCEPTABLE, TARGET_MARKERS, SearchResult
from core.trial_sequence import TRIAL_COUNT
from output.video import VIDEO_EXT, VideoLogWriter

RECORDINGS_DIR = "recordings"
KEY_PATH = Path(r"C:\Users\Guillaume\Documents\DLSU\THESIS\GripVT_keys\participant_key.json")
SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 2
BASELINE_SCHEMA_VERSION = 2
ABORTED_SCHEMA_VERSION = 1
GRID_LAYOUT_SCHEMA_VERSION = 1

_FOLDER_RE = re.compile(r"participant(\d+)_")
_SESSION_RE = re.compile(r"session(\d+)_")


@dataclass(frozen=True)
class ParticipantRecord:
    participant_id: str
    folder: str
    created_at: str
    age: int
    weight_kg: float
    height_cm: float
    forearm_circ_cm: float
    mvc_kg: float
    activity_level: str


@dataclass(frozen=True)
class ResumableParticipant:
    """One row of the Resume Session picker (D3 of the resume-session plan).

    Deliberately thin -- no anthropometrics, no name -- so the scan behind
    `list_resumable_participants` never has to open `participant.json` (D4).

    `complete_session_count` (D5 of the resume-supersede plan, renamed from
    `session_count`) is the count of *fully recorded* (30/30 trial) sessions
    on disk, from `count_complete_sessions` -- not a raw dir count, since a
    partially recorded trailing session still leaves real work to resume.
    """
    participant_id: str
    folder: str
    complete_session_count: int
    last_session_at: datetime | None


def next_participant_number() -> int:
    if not os.path.isdir(RECORDINGS_DIR):
        return 1
    numbers = [
        int(m.group(1))
        for name in os.listdir(RECORDINGS_DIR)
        if (m := _FOLDER_RE.match(name))
    ]
    return max(numbers, default=0) + 1


def create_participant(
    name: str,
    age: int,
    weight_kg: float,
    height_cm: float,
    forearm_circ_cm: float,
    mvc_kg: float,
    activity_level: str,
    mvc_source_path: str,
) -> ParticipantRecord:
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    created_at = now.isoformat(timespec="seconds")
    pid = f"participant{next_participant_number():03d}"

    folder = os.path.join(RECORDINGS_DIR, f"{pid}_{ts}")
    os.makedirs(folder, exist_ok=True)

    # mvc.txt is the primary provenance record behind mvc_kg -- a failed copy
    # (missing/unreadable source) must abort here, before participant.json is
    # ever written, rather than degrade silently (D10 of
    # docs/logs/20260730_150048_mvc_file_load_plan.md).
    shutil.copy2(mvc_source_path, os.path.join(folder, "mvc.txt"))

    record = ParticipantRecord(
        participant_id=pid,
        folder=folder,
        created_at=created_at,
        age=age,
        weight_kg=weight_kg,
        height_cm=height_cm,
        forearm_circ_cm=forearm_circ_cm,
        mvc_kg=mvc_kg,
        activity_level=activity_level,
    )

    payload = {"schema_version": SCHEMA_VERSION, **asdict(record)}
    payload.pop("folder")  # path is implicit in the folder itself, not data
    with open(os.path.join(folder, "participant.json"), "w") as f:
        json.dump(payload, f, indent=2)

    _append_key(pid, name, created_at)
    return record


def save_calibration(
    record: ParticipantRecord,
    result: SearchResult,
    pose_frames: dict[str, np.ndarray],
    *,
    dest_dir: str,
    overridden: bool = False,
    roi: dict | None = None,
    calibration_image_size: list[int] | None = None,
    manual_override: bool = False,
    auto_params: dict | None = None,
) -> str:
    """Persist one wear's calibration into `dest_dir`, de-identified.

    `dest_dir` is keyword-only with no default (per
    docs/logs/20260803_per_wear_session_folder_plan.md Phase 1): every caller
    must state where it writes, so a future caller cannot silently re-litter
    the participant root. Callers pass the session directory once F6a buffers
    calibration until Start (Phase 2); until then callers pass `record.folder`.

    The three median pose frames go alongside the JSON so the search can be
    re-run offline without the rig -- the per-pose diagnostics are the study's
    calibration-evaluation record, so they have to stay reproducible.

    `manual_override`/`auto_params` record Phase 4's post-hoc detector
    override (D17): `manual_override` is always written, but `auto_params`
    and `final_params` are written only when it is true (D17) -- a plain
    automatic pass has nothing to contrast `params` against, so those two
    keys would just restate `params` for every ordinary calibration. `params`
    itself is left alone either way, so old readers keep working.
    """
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "participant_id": record.participant_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "passed": result.passed,
        "pass_rule": "every_pose_equals_target",
        "overridden": overridden,
        "manual_override": manual_override,
        "target_markers": TARGET_MARKERS,
        "min_acceptable": MIN_ACCEPTABLE,
        "params": result.params,
        "search_bounds": result.bounds,
        "grid_shape": {"x_lines": result.grid_shape[0], "y_lines": result.grid_shape[1]},
        "geometry": asdict(result.geometry),
        "reference_bands": result.reference_bands,
        "reference_corners": result.reference_corners,
        "poses": {key: asdict(diag) for key, diag in result.poses.items()},
        "worst_pose": result.worst_pose,
        "worst_count": result.worst_count,
        "roi": roi,
        "calibration_image_size": calibration_image_size,
    }
    if manual_override:
        payload["auto_params"] = auto_params
        payload["final_params"] = result.params

    path = os.path.join(dest_dir, f"calibration_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    for key, frame in pose_frames.items():
        cv2.imwrite(os.path.join(dest_dir, f"pose_{key}_{ts}.png"), frame)
    return path


def save_baseline(
    record: ParticipantRecord,
    *,
    dest_dir: str,
    locked_count: int,
    passed: bool,
    overridden: bool,
    corners: dict[str, int] | None,
    bands: dict[str, int] | None,
    params: dict,
    frames: list[np.ndarray] | None = None,
) -> str:
    """Persist one session's Capture Baseline lock into `dest_dir`.

    `dest_dir` is keyword-only with no default (per
    docs/logs/20260803_per_wear_session_folder_plan.md Phase 1): every caller
    must state where it writes. Once F6b buffers the baseline until Start
    (Phase 3), callers pass the session directory; until then callers pass
    `record.folder`.

    Written once, when the baseline locks -- on a pass, or on an operator
    override of a below-threshold capture (`overridden=True`, so the record is
    auditable).

    `frames`, if non-empty, is the winning burst's raw frames -- written as
    `baseline_<ts>.<VIDEO_EXT>` under the same `ts` as the JSON (D4 of
    docs/logs/20260730_110247_raw_video_log_plan.md), so the clip and its
    record can never drift apart in time.
    """
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": BASELINE_SCHEMA_VERSION,
        "participant_id": record.participant_id,
        "captured_at": now.isoformat(timespec="seconds"),
        "locked_count": locked_count,
        "target": TARGET_MARKERS,
        "min_acceptable": MIN_ACCEPTABLE,
        "passed": passed,
        "pass_rule": "locked_count_equals_target",
        "overridden": overridden,
        "corners": corners,
        "bands": bands,
        "params_thresh": params.get("thresh"),
        "params_min_area": params.get("min_area"),
    }
    path = os.path.join(dest_dir, f"baseline_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    if frames:
        h, w = frames[0].shape[:2]
        clip_path = os.path.join(dest_dir, f"baseline_{ts}{VIDEO_EXT}")
        writer = VideoLogWriter(clip_path, size=(w, h))
        for frame in frames:
            writer.write(frame)
        writer.close()

    return path


def save_grid_layout(
    record: ParticipantRecord,
    *,
    dest_dir: str,
    layout: dict[int, tuple[int, int]] | None,
    source: str,
    reason: str | None,
    grid_shape: dict[str, int],
    locked_count: int,
    baseline_file: str,
) -> str:
    """Persist one session's marker_id -> (row, col) lookup into `dest_dir`.

    `dest_dir` is keyword-only with no default -- same rule
    `save_baseline`/`save_calibration` already establish (per
    docs/logs/20260803_per_wear_session_folder_plan.md Phase 1): every caller
    must state where it writes, so a future caller cannot silently re-litter
    the participant root.

    Filename is the literal, untimestamped `grid_layout.json` (D3 of
    docs/logs/20260804_161615_marker_grid_lookup_plan.md) -- only the winning
    baseline attempt is ever flushed into a session dir (F6b buffering, see
    ui/session/baseline.py), so there is exactly one per session and a
    timestamp suffix buys nothing. `baseline_file` pairs this record with its
    `baseline_<ts>.json` unambiguously.

    `layout` is `None` whenever `source == "fallback_sort"` (D6): geometry
    could not be resolved for this capture, so no row/col mapping exists for
    it -- callers must check `source` before trusting `layout`. When present,
    keys are written as strings (JSON object keys are always strings), so
    readers must do `int(k)` on `layout`'s keys to recover marker_id ints.

    De-identified per URERB Form 6A, same as `save_baseline`: coded
    participant_id only, no name, no free text beyond the geometry error
    message carried in `reason`.
    """
    payload = {
        "schema_version": GRID_LAYOUT_SCHEMA_VERSION,
        "participant_id": record.participant_id,
        "captured_at": datetime.now().isoformat(timespec="seconds"),
        "baseline_file": baseline_file,
        "source": source,
        "reason": reason,
        "locked_count": locked_count,
        "grid_shape": grid_shape,
        "id_formula": "marker_id = row * n_cols + col",
        "layout": (
            None if layout is None
            else {str(mid): {"row": row, "col": col} for mid, (row, col) in layout.items()}
        ),
    }
    path = os.path.join(dest_dir, "grid_layout.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


def session_dirs(folder: str) -> list[str]:
    """Every `sessionN_<ts>` directory in a participant folder, ordered by N.

    Scanned from disk rather than read off the in-memory session counter,
    which tracks the operator's progress through the visit and cannot answer
    "what is actually on disk".

    Only the participant root is scanned, so sessions that F7 moved into
    `aborted/` are correctly invisible here -- the same reason
    `writer._next_session_number` stops counting them, which is what lets
    folder numbering agree with the SESSION x/10 indicator again (D7).
    The old display-vs-folder divergence, documented as deliberate on
    2026-07-29, was caused solely by abandoned folders squatting in the
    numbering space; F7 relocates them, so it no longer arises.
    """
    if not os.path.isdir(folder):
        return []
    named = [
        (int(m.group(1)), name)
        for name in os.listdir(folder)
        if (m := _SESSION_RE.match(name)) and os.path.isdir(os.path.join(folder, name))
    ]
    return [name for _, name in sorted(named)]


def _session_is_complete(session_dir: str) -> bool:
    """A `sessionN_<ts>` dir is complete iff `trials.csv` has exactly
    `TRIAL_COUNT` (30) valid rows (D1 of the resume-supersede plan).

    A row is valid iff it is non-voided AND closed. `voided` is compared as
    a literal string, not truth-tested and not compared against a Python
    bool: `TrialLog.flush()` writes Python `True`/`False` through
    `csv.DictWriter`, which `str()`s them, so the file on disk holds the
    literal strings `"True"`/`"False"` -- truth-testing the raw string or
    comparing to the Python bool reads every row as voided (or as
    non-voided), silently and with no exception. `closed` means `end_ts` is
    non-empty. Columns are read by name via `csv.DictReader` against
    `output/writer.py::_TRIAL_COLUMNS` (the schema's source of truth) --
    never by index -- so an unknown/missing column degrades via `.get`
    rather than raising.

    A missing `trials.csv` reads as 0 valid rows, i.e. incomplete, not an
    error -- this is deliberate: a clean early Stop and a true crash (no
    `trials.csv` at all, since `TrialLog.flush()`/`EventLog.flush()` only
    run inside `_finalize_recording()`) are the same fact for the protocol,
    so both are treated identically here.

    Never raises: any exception (bad path, corrupt/binary CSV, encoding
    error) degrades to `False`.
    """
    try:
        csv_path = os.path.join(session_dir, "trials.csv")
        if not os.path.isfile(csv_path):
            return False
        valid = 0
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                not_voided = (row.get("voided") or "").strip() not in ("True", "true", "1")
                closed = (row.get("end_ts") or "").strip() != ""
                if not_voided and closed:
                    valid += 1
        return valid == TRIAL_COUNT
    except Exception:
        return False


def count_complete_sessions(folder: str) -> int:
    """Count of `sessionN_<ts>` dirs under `folder` that are complete (D1).

    Scans every session dir returned by `session_dirs` (cheap: at most
    `SESSION_COUNT` dirs, at most `TRIAL_COUNT` rows each) even though only
    the trailing (highest-N) dir can ever actually be incomplete --
    `writer._next_session_number` assigns strictly increasing N from
    existing dirs, and the resume-supersede numbering reuse guarantees at
    most one incomplete trailing dir ever coexists with complete
    predecessors. No caller may rely on a mid-list dir being incomplete.

    Never raises: any exception degrades to `0`.
    """
    try:
        return sum(
            1 for name in session_dirs(folder)
            if _session_is_complete(os.path.join(folder, name))
        )
    except Exception:
        return 0


def trailing_incomplete_session(folder: str) -> str | None:
    """Bare folder name of the trailing incomplete session, if any (D4).

    Returns the highest-N `sessionN_<ts>` dir's bare name (e.g.
    `"session2_20260803_120000"`, no path prefix) when that dir is not
    complete per `_session_is_complete`, else `None` -- including when
    `folder` has no session dirs at all. Only the trailing dir is ever
    considered, per the same invariant `count_complete_sessions` documents.

    This exists so callers (namely `main.py`) never re-derive "which dir to
    supersede" themselves, and so the bare-name convention that
    `move_sessions_to_aborted` requires is enforced in exactly one place.

    Never raises: any exception degrades to `None`.
    """
    try:
        names = session_dirs(folder)
        if not names:
            return None
        last = names[-1]
        return last if not _session_is_complete(os.path.join(folder, last)) else None
    except Exception:
        return None


def load_participant(folder: str) -> ParticipantRecord:
    """Read `<folder>/participant.json` back into a `ParticipantRecord` (D6).

    Called only when the operator commits to a folder (Continue on the
    Resume Session picker) -- the picker's list scan itself never opens this
    file (see `list_resumable_participants`'s docstring for why). Fields are
    named explicitly rather than `**payload`, so `schema_version` and any
    future extra key in the JSON can never reach the constructor. Raises
    (OSError / ValueError / KeyError / TypeError) on a missing file, bad
    JSON, or a missing key -- the caller is expected to catch this, since a
    non-atomic write (`create_participant`) is exactly what can leave a
    half-written file behind after a crash.
    """
    with open(os.path.join(folder, "participant.json")) as f:
        payload = json.load(f)
    return ParticipantRecord(
        participant_id=payload["participant_id"],
        folder=folder,
        created_at=payload["created_at"],
        age=payload["age"],
        weight_kg=payload["weight_kg"],
        height_cm=payload["height_cm"],
        forearm_circ_cm=payload["forearm_circ_cm"],
        mvc_kg=payload["mvc_kg"],
        activity_level=payload["activity_level"],
    )


def _last_session_at(folder: str) -> datetime | None:
    """Timestamp of the most recent `sessionN_<ts>` dir under `folder` (D7).

    Read from the folder name itself, not `os.stat` and not `aborted.json`
    (which is only written by the in-app Restart/Abort flow, never by a
    crash, so it would under-report resumability -- D8). Unparseable name or
    zero sessions both fall back to `None`.
    """
    sessions = session_dirs(folder)
    if not sessions:
        return None
    _, _, ts = sessions[-1].partition("_")
    try:
        return datetime.strptime(ts, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def list_resumable_participants(max_sessions: int) -> list[ResumableParticipant]:
    """Every `participantNNN_*` folder under RECORDINGS_DIR with room for another
    session (D2, D4).

    `max_sessions` is a required positional argument -- this module must not
    import `SESSION_COUNT` from `ui.session.progress`, or `output/` would
    depend on `ui/`. Both callers (the landing window's gate and the picker
    itself) pass `ui.session.progress.SESSION_COUNT`.

    The predicate is **not** byte-identical to the one
    `_has_resumable_participant` used to inline anymore (resume-supersede
    plan D6): a directory under RECORDINGS_DIR, `participant.json` exists in
    it (existence only -- never opened/parsed here), and it has fewer than
    `max_sessions` *complete* sessions on disk (`count_complete_sessions`),
    not fewer than `max_sessions` raw session dirs. A participant with a
    partially recorded trailing session must stay resumable -- that trailing
    session still has real work left, and the resume-supersede numbering
    reuse (see `output/participants.py` module docs referenced from the
    plan) means the same session number is still available to redo it.
    `participant.json` is parsed lazily, exactly once, only when the
    operator commits to a row via `load_participant` -- not during this
    scan -- so a folder with a corrupt or half-written `participant.json`
    still shows up as resumable rather than silently vanishing from the
    list.

    Never raises: returns `[]` if RECORDINGS_DIR itself can't be scanned, and
    skips (rather than aborts on) any individual folder that raises while
    being inspected, matching the old `_has_resumable_participant` contract
    that the landing window depends on to open at all.
    """
    try:
        if not os.path.isdir(RECORDINGS_DIR):
            return []
        items: list[ResumableParticipant] = []
        for name in os.listdir(RECORDINGS_DIR):
            try:
                folder = os.path.join(RECORDINGS_DIR, name)
                if not os.path.isdir(folder):
                    continue
                if not os.path.isfile(os.path.join(folder, "participant.json")):
                    continue
                if count_complete_sessions(folder) >= max_sessions:
                    continue
                items.append(ResumableParticipant(
                    participant_id=name.split("_")[0],
                    folder=folder,
                    complete_session_count=count_complete_sessions(folder),
                    last_session_at=_last_session_at(folder),
                ))
            except Exception:
                continue
        items.sort(
            key=lambda item: item.last_session_at or datetime.min, reverse=True,
        )
        return items
    except Exception:
        return []


def _aborted_dir(folder: str) -> str:
    """`<participant>/aborted/`, creating it if absent (D5/D6).

    Abandoned wears -- failed calibrations, moved session dirs, and the
    `aborted.json` record explaining them -- all live here, keeping the
    participant root down to exactly `participant.json`, `mvc.txt`,
    `sessionN_<ts>/` dirs, and this one directory (D4).
    """
    path = os.path.join(folder, "aborted")
    os.makedirs(path, exist_ok=True)
    return path


def move_sessions_to_aborted(record: ParticipantRecord, session_names: list[str]) -> list[str]:
    """Relocate named `sessionN_<ts>` dirs into `<participant>/aborted/` (D8, D9).

    Used by Restart (Phase 5) to move exactly the sessions it just recorded in
    `aborted.json`. `os.replace` on a whole directory is atomic on the same
    volume and never copies, so a move cannot half-succeed.

    Never deletes anything -- a name that no longer exists under the
    participant folder is skipped rather than raised, since Restart is a
    recovery path and must not be broken by a second attempt to move a name
    that was already relocated (D9). Session dirs keep their original
    timestamped names, so repeated restarts cannot collide inside `aborted/`.

    Returns the names actually moved, in the order given.
    """
    dest_dir = _aborted_dir(record.folder)
    moved = []
    for name in session_names:
        src = os.path.join(record.folder, name)
        if not os.path.isdir(src):
            continue
        dst = os.path.join(dest_dir, name)
        os.replace(src, dst)
        moved.append(name)
    return moved


def record_abort(
    record: ParticipantRecord,
    *,
    scope: str,
    reason: str,
    note: str,
    session_num_at_abort: int,
) -> str:
    """Record that the operator walked away from one session, or a whole visit.

    Written to `aborted.json` in `<participant>/aborted/` (D5),
    **unconditionally** -- the in-session `session_aborted` event in
    `events.json` only exists when a recording was open at the moment Restart
    was pressed, which is not the common case. Pressing Restart between
    sessions previously left no trace at all, making an abandoned visit
    indistinguishable from a real multi-session one after the fact.

    Nothing is deleted. The abandoned session directories stay exactly where
    they are (Phase 5 moves them into `aborted/` too, but that is a separate
    call); this file is what lets analysis (and a later, deliberate purge
    tool) tell them apart from real data.

    De-identified per URERB Form 6A: coded ID, directory names, timestamps,
    the reason chip, and the operator's note only -- never a participant name.
    """
    sessions = session_dirs(record.folder)
    if scope == "session":
        sessions = sessions[-1:]

    path = os.path.join(_aborted_dir(record.folder), "aborted.json")
    payload = {"schema_version": ABORTED_SCHEMA_VERSION,
               "participant_id": record.participant_id,
               "aborts": []}
    if os.path.exists(path):
        # Append, never truncate -- a visit can be restarted more than once
        # and each abandonment is its own fact (D6).
        with open(path) as f:
            payload = json.load(f)
        payload.setdefault("aborts", [])

    payload["aborts"].append({
        "scope": scope,
        "reason": reason,
        "note": note,
        "at": datetime.now().isoformat(timespec="seconds"),
        "session_num_at_abort": session_num_at_abort,
        "abandoned_sessions": sessions,
    })

    # Same atomic idiom as _append_key and EventLog.flush: a crash mid-write
    # cannot leave a half-written aborted.json.
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)
    return path


def save_calibration_failure(
    record: ParticipantRecord,
    pose_frames: dict[str, np.ndarray],
    detections: dict[str, list],
    message: str,
) -> str:
    """Dump the frames and detected centroids when calibration cannot even anchor.

    Written into `<participant>/aborted/` (D6): a calibration that never
    converged is an abandoned wear attempt, and it is Gap 3 evidence --
    sensitivity to mounting is precisely what a failure records. Without this
    the operator sees only a toast and the geometry that defeated the search
    is gone with the frames, so the failure cannot be reproduced off-rig.
    """
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "participant_id": record.participant_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "failed": True,
        "message": message,
        "detections": {
            key: [[round(x, 2), round(y, 2), round(size, 2)] for x, y, size in dets]
            for key, dets in detections.items()
        },
    }
    dest_dir = _aborted_dir(record.folder)
    path = os.path.join(dest_dir, f"calibration_failed_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    for key, frame in pose_frames.items():
        cv2.imwrite(os.path.join(dest_dir, f"failed_{key}_{ts}.png"), frame)
    return path


def _append_key(pid: str, name: str, created_at: str) -> None:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if KEY_PATH.exists():
        data = json.loads(KEY_PATH.read_text())
    data[pid] = {"name": name, "created_at": created_at}

    tmp = KEY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, KEY_PATH)
