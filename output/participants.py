"""Participant persistence for the GripVT human-data app.

Enforces the URERB Form 6A data-management rule: the de-identified record
(coded ID + anthropometrics) lands in ``recordings/participantNNN_<ts>/``,
while the participant's name is written only to an external key file, so the
recordings tree can be backed up without leaking identities.
"""
import json
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

RECORDINGS_DIR = "recordings"
KEY_PATH = Path(r"C:\Users\Guillaume\Documents\DLSU\THESIS\GripVT_keys\participant_key.json")
SCHEMA_VERSION = 1

_FOLDER_RE = re.compile(r"participant(\d+)_")


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
) -> ParticipantRecord:
    now = datetime.now()
    ts = now.strftime("%Y%m%d_%H%M%S")
    created_at = now.isoformat(timespec="seconds")
    pid = f"participant{next_participant_number():03d}"

    folder = os.path.join(RECORDINGS_DIR, f"{pid}_{ts}")
    os.makedirs(folder, exist_ok=True)

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


def _append_key(pid: str, name: str, created_at: str) -> None:
    KEY_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {}
    if KEY_PATH.exists():
        data = json.loads(KEY_PATH.read_text())
    data[pid] = {"name": name, "created_at": created_at}

    tmp = KEY_PATH.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=2))
    os.replace(tmp, KEY_PATH)
