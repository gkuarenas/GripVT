import csv
import json
import os
import re
import time
from datetime import datetime

from core.tracker import MarkerRecord


COLUMNS = [
    "frame", "timestamp_ms", "marker_id",
    "x", "y", "area", "dx", "dy", "dA",
    "magnitude",
    "size_moments",
    "predicted_x", "predicted_y", "autofilled",
    "trial_id",
]

# Mirrors output/participants.py's _FOLDER_RE / next_participant_number()
# idiom one level down: sessionN_<ts> dirs inside a participant folder,
# instead of participantNNN_<ts> dirs inside recordings/.
_SESSION_FOLDER_RE = re.compile(r"session(\d+)_")


def _next_session_number(participant_folder: str) -> int:
    if not os.path.isdir(participant_folder):
        return 1
    numbers = [
        int(m.group(1))
        for name in os.listdir(participant_folder)
        if (m := _SESSION_FOLDER_RE.match(name))
    ]
    return max(numbers, default=0) + 1


def make_session_dir(participant_folder: str) -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    session_num = _next_session_number(participant_folder)
    path = os.path.join(participant_folder, f"session{session_num}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path


class CSVWriter:
    def __init__(self, session_dir: str) -> None:
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.csv_path = os.path.join(session_dir, "markers.csv")
        self.png_path = os.path.join(session_dir, "overlay.png")
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS)
        self._w.writeheader()
        self._pending: list[tuple[list[MarkerRecord], int, float, int]] = []

    def buffer_frame(
        self, records: list[MarkerRecord], frame_idx: int, timestamp_ms: float, trial_id: int = 0
    ) -> None:
        self._pending.append((records, frame_idx, timestamp_ms, trial_id))

    @property
    def pending_count(self) -> int:
        """Frames buffered since the last write_window()/discard_window() --
        lets a caller (SessionWindow's D7 safety valve) force an early commit
        without CSVWriter owning any policy about when that should happen."""
        return len(self._pending)

    def write_window(self) -> None:
        for records, frame_idx, ms, trial_id in self._pending:
            for r in records:
                self._w.writerow({
                    "frame": frame_idx,
                    "timestamp_ms": round(ms, 3),
                    "marker_id": r.marker_id,
                    "x": round(r.x, 3),
                    "y": round(r.y, 3),
                    "area": round(r.area, 3),
                    "dx": round(r.dx, 3),
                    "dy": round(r.dy, 3),
                    "dA": round(r.dA, 3),
                    "magnitude": round(r.magnitude, 3),
                    "size_moments": round(r.size_moments, 3),
                    "predicted_x": round(r.predicted_x, 3),
                    "predicted_y": round(r.predicted_y, 3),
                    "autofilled": r.autofilled,
                    "trial_id": trial_id,
                })
        self._pending = []
        self._f.flush()

    def discard_window(self) -> None:
        self._pending = []

    def close(self) -> None:
        self._f.close()


_TRIAL_COLUMNS = ["trial_id", "trial_index", "start_ts", "end_ts", "voided"]


class TrialLog:
    """One row per trial segment (D2), rewritten in full on each flush().

    Small and bounded (<=30 rows for this study), so it is not an
    append-only stream like CSVWriter -- flush() overwrites trials.csv from
    the in-memory table each time, which is also what keeps void_trial()'s
    tombstone (D6) simple: mutate the row in place, rewrite.
    """

    def __init__(self, session_dir: str) -> None:
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.csv_path = os.path.join(session_dir, "trials.csv")
        self._trials: dict[int, dict] = {}

    def open_trial(self, trial_id: int, trial_index: int, start_ts: float) -> None:
        self._trials[trial_id] = {
            "trial_id": trial_id,
            "trial_index": trial_index,
            "start_ts": round(start_ts, 3),
            "end_ts": "",
            "voided": False,
        }

    def close_trial(self, trial_id: int, end_ts: float) -> None:
        self._trials[trial_id]["end_ts"] = round(end_ts, 3)

    def void_trial(self, trial_id: int) -> None:
        self._trials[trial_id]["voided"] = True

    def flush(self) -> None:
        with open(self.csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=_TRIAL_COLUMNS)
            w.writeheader()
            for row in self._trials.values():
                w.writerow(row)


class EventLog:
    """Session-level event log, written as one JSON envelope (D3/D6/D8).

    append() is generic on purpose -- this phase does not emit any events
    itself, later phases will (camera_loss, tracking_failed,
    session_aborted, rate_fallback), and the envelope should not need to
    change when a new event type is added.
    """

    def __init__(
        self,
        session_dir: str,
        participant_id: str,
        started_at: str,
        frame_index_at_start: int,
    ) -> None:
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        self.json_path = os.path.join(session_dir, "events.json")
        self._envelope = {
            "schema_version": 1,
            "participant_id": participant_id,
            "session_dir": session_dir,
            "started_at": started_at,
            "frame_index_at_start": frame_index_at_start,
            "events": [],
        }

    def append(self, event: dict) -> None:
        self._envelope["events"].append(event)

    def flush(self) -> None:
        # Same atomic-write idiom as output/participants.py's _append_key:
        # write to a .tmp file, then os.replace() so a crash mid-write
        # cannot leave events.json half-written.
        tmp_path = self.json_path + ".tmp"
        with open(tmp_path, "w") as f:
            json.dump(self._envelope, f, indent=2)
        os.replace(tmp_path, self.json_path)


def anchor_wall_clock() -> tuple[float, float]:
    """Capture (t0_wall_ms, t0_monotonic) together, for now_ms() (D1).

    t0_wall_ms is epoch milliseconds (subject to NTP steps); t0_monotonic is
    the matching time.monotonic() reading. now_ms() combines them so a
    session's timestamps stay monotonic even if the wall clock steps mid-
    session, while still being wall-clock-aligned for offline correlation
    (e.g. against Biopac).
    """
    return time.time() * 1000, time.monotonic()


def now_ms(t0_wall_ms: float, t0_monotonic: float) -> float:
    return t0_wall_ms + (time.monotonic() - t0_monotonic) * 1000
