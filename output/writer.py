import csv
import os
from datetime import datetime

import cv2
import numpy as np

from core.tracker import MarkerRecord


COLUMNS = [
    "frame", "timestamp_ms", "marker_id",
    "x", "y", "area", "dx", "dy", "dA",
    "magnitude", "dx_mm", "dy_mm", "delta_z_mm", "magnitude_mm",
    "predicted_x", "predicted_y", "autofilled",
    "rep", "force_n", "indenter_z_mm", "window_type",
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

    def write_window(self, rep: int, force_n: float, window_type: str, indenter_z_mm: float = 0.0) -> None:
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
                    "dx_mm": round(r.dx_mm, 4),
                    "dy_mm": round(r.dy_mm, 4),
                    "delta_z_mm": round(r.delta_z_mm, 4),
                    "magnitude_mm": round(r.magnitude_mm, 4),
                    "predicted_x": round(r.predicted_x, 3),
                    "predicted_y": round(r.predicted_y, 3),
                    "autofilled": r.autofilled,
                    "rep": rep,
                    "force_n": round(force_n, 3),
                    "indenter_z_mm": round(indenter_z_mm, 3),
                    "window_type": window_type,
                })
        self._pending = []

    def discard_window(self) -> None:
        self._pending = []

    def close(self) -> None:
        self._f.close()


class VideoWriter:
    def __init__(self, session_dir: str, rep: int, win_type: str,
                 fps: float, frame_size: tuple[int, int]) -> None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._path = os.path.join(session_dir, f"recording_rep{rep}_{win_type}_{ts}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(self._path, fourcc, fps, frame_size)

    def write_frame(self, frame: np.ndarray) -> None:
        self._writer.write(frame)

    def close(self) -> str:
        self._writer.release()
        return self._path

    def discard(self) -> None:
        self._writer.release()
        if os.path.exists(self._path):
            os.remove(self._path)
