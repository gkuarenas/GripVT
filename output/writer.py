import csv
import os
from datetime import datetime

from core.tracker import MarkerRecord


COLUMNS = [
    "frame", "timestamp_ms", "marker_id",
    "x", "y", "area", "dx", "dy", "dA",
    "magnitude",
    "size_moments",
    "predicted_x", "predicted_y", "autofilled",
]


def make_session_dir() -> str:
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join("output", "sessions", ts)
    os.makedirs(path, exist_ok=True)
    return path


class CSVWriter:
    def __init__(self, session_dir: str) -> None:
        os.makedirs(session_dir, exist_ok=True)
        self.session_dir = session_dir
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(session_dir, f"markers_{ts}.csv")
        self.png_path = os.path.join(session_dir, f"overlay_{ts}.png")
        self._f = open(self.csv_path, "w", newline="")
        self._w = csv.DictWriter(self._f, fieldnames=COLUMNS)
        self._w.writeheader()
        self._pending: list[tuple[list[MarkerRecord], int, float]] = []

    def buffer_frame(self, records: list[MarkerRecord], frame_idx: int, timestamp_ms: float) -> None:
        self._pending.append((records, frame_idx, timestamp_ms))

    def write_window(self) -> None:
        for records, frame_idx, ms in self._pending:
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
                })
        self._pending = []

    def discard_window(self) -> None:
        self._pending = []

    def close(self) -> None:
        self._f.close()
