"""Participant persistence for the GripVT human-data app.

Enforces the URERB Form 6A data-management rule: the de-identified record
(coded ID + anthropometrics) lands in ``recordings/participantNNN_<ts>/``,
while the participant's name is written only to an external key file, so the
recordings tree can be backed up without leaking identities.
"""
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
from output.video import VIDEO_EXT, VideoLogWriter

RECORDINGS_DIR = "recordings"
KEY_PATH = Path(r"C:\Users\Guillaume\Documents\DLSU\THESIS\GripVT_keys\participant_key.json")
SCHEMA_VERSION = 1
CALIBRATION_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1
ABORTED_SCHEMA_VERSION = 1

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
    overridden: bool = False,
    roi: dict | None = None,
    calibration_image_size: list[int] | None = None,
) -> str:
    """Persist one wear's calibration into the participant's folder, de-identified.

    The three median pose frames go alongside the JSON so the search can be
    re-run offline without the rig -- the per-pose diagnostics are the study's
    calibration-evaluation record, so they have to stay reproducible.
    """
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    payload = {
        "schema_version": CALIBRATION_SCHEMA_VERSION,
        "participant_id": record.participant_id,
        "timestamp": now.isoformat(timespec="seconds"),
        "passed": result.passed,
        "overridden": overridden,
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

    path = os.path.join(record.folder, f"calibration_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    for key, frame in pose_frames.items():
        cv2.imwrite(os.path.join(record.folder, f"pose_{key}_{ts}.png"), frame)
    return path


def save_baseline(
    record: ParticipantRecord,
    *,
    locked_count: int,
    passed: bool,
    overridden: bool,
    corners: dict[str, int] | None,
    bands: dict[str, int] | None,
    params: dict,
    frames: list[np.ndarray] | None = None,
) -> str:
    """Persist one session's Capture Baseline lock into the participant's folder.

    Written once, when the baseline locks -- on a pass, or on an operator
    override of a below-threshold capture (`overridden=True`, so the record is
    auditable). Lands beside `calibration_<ts>.json`, not in a session dir --
    a baseline is captured once per session, before trial 1 exists.

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
        "overridden": overridden,
        "corners": corners,
        "bands": bands,
        "params_thresh": params.get("thresh"),
        "params_min_area": params.get("min_area"),
    }
    path = os.path.join(record.folder, f"baseline_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)

    if frames:
        h, w = frames[0].shape[:2]
        clip_path = os.path.join(record.folder, f"baseline_{ts}{VIDEO_EXT}")
        writer = VideoLogWriter(clip_path, size=(w, h))
        for frame in frames:
            writer.write(frame)
        writer.close()

    return path


def session_dirs(folder: str) -> list[str]:
    """Every `sessionN_<ts>` directory in a participant folder, ordered by N.

    Scanned from disk rather than read off the in-memory session counter: the
    counter deliberately diverges from folder numbering after a Restart (see
    the SESSION x/10 indicator notes in the session-recording checklist), so it
    cannot answer "what is actually on disk".
    """
    if not os.path.isdir(folder):
        return []
    named = [
        (int(m.group(1)), name)
        for name in os.listdir(folder)
        if (m := _SESSION_RE.match(name)) and os.path.isdir(os.path.join(folder, name))
    ]
    return [name for _, name in sorted(named)]


def record_abort(
    record: ParticipantRecord,
    *,
    scope: str,
    reason: str,
    note: str,
    session_num_at_abort: int,
) -> str:
    """Record that the operator walked away from one session, or a whole visit.

    Written to `aborted.json` in the participant folder, **unconditionally** --
    the in-session `session_aborted` event in `events.json` only exists when a
    recording was open at the moment Restart was pressed, which is not the
    common case. Pressing Restart between sessions previously left no trace at
    all, making an abandoned visit indistinguishable from a real multi-session
    one after the fact.

    Nothing is deleted. The abandoned session directories stay exactly where
    they are; this file is what lets analysis (and a later, deliberate purge
    tool) tell them apart from real data.

    De-identified per URERB Form 6A: coded ID, directory names, timestamps,
    the reason chip, and the operator's note only -- never a participant name.
    """
    sessions = session_dirs(record.folder)
    if scope == "session":
        sessions = sessions[-1:]

    path = os.path.join(record.folder, "aborted.json")
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

    Without this the operator sees only a toast and the geometry that defeated the
    search is gone with the frames, so the failure cannot be reproduced off-rig.
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
    path = os.path.join(record.folder, f"calibration_failed_{ts}.json")
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    for key, frame in pose_frames.items():
        cv2.imwrite(os.path.join(record.folder, f"failed_{key}_{ts}.png"), frame)
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
