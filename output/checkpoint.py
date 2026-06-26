import json
import os


class CheckpointManagerV4:
    """Atomic JSON checkpoint for the two-phase v4 protocol (calibration +
    sensitivity/repeatability collection). Mirrors CheckpointManager's
    tmp -> os.replace pattern but tracks both phases independently so either
    can be resumed on its own."""

    FILENAME = "checkpoint_v4.json"
    TMPFILE  = "checkpoint_v4.tmp"

    def save(
        self,
        session_dir: str,
        session_ts: str,
        blend_id: str,
        sample_n: int,
        phase: str,
        z_thresh_map_path: str,
        completed_calibration_bins: list[int],
        completed_collection_reps: dict[str, list[int]],
        csv_path: str,
        summary_csv_path: str,
        z_target_mm: float = float("nan"),
    ) -> None:
        payload = {
            "session_ts": session_ts,
            "blend_id": blend_id,
            "sample_n": sample_n,
            "phase": phase,
            "z_thresh_map_path": z_thresh_map_path,
            "completed_calibration_bins": completed_calibration_bins,
            "completed_collection_reps": completed_collection_reps,
            "csv_path": csv_path,
            "summary_csv_path": summary_csv_path,
            "z_target_mm": z_target_mm,
        }
        tmp = os.path.join(session_dir, self.TMPFILE)
        os.makedirs(session_dir, exist_ok=True)
        with open(tmp, "w") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp, os.path.join(session_dir, self.FILENAME))

    def load(self, session_dir: str) -> dict | None:
        path = os.path.join(session_dir, self.FILENAME)
        try:
            with open(path, "r") as f:
                data = json.load(f)
            data["session_dir"] = session_dir
            return data
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            return None

    def scan_for_resume(self, sessions_root: str = "output/sessions") -> dict | None:
        if not os.path.isdir(sessions_root):
            return None
        candidates = []
        for entry in os.listdir(sessions_root):
            entry_path = os.path.join(sessions_root, entry)
            if not os.path.isdir(entry_path):
                continue
            if entry.endswith("_sensitivity"):
                # Legacy layout: sessions_root/<timestamp>_sensitivity/
                cp = self.load(entry_path)
                if cp is not None and cp.get("phase") != "complete":
                    candidates.append(cp)
            else:
                # New layout: sessions_root/<blend_id>/<timestamp>_sensitivity/
                for sub in os.listdir(entry_path):
                    if not sub.endswith("_sensitivity"):
                        continue
                    sub_path = os.path.join(entry_path, sub)
                    if not os.path.isdir(sub_path):
                        continue
                    cp = self.load(sub_path)
                    if cp is not None and cp.get("phase") != "complete":
                        candidates.append(cp)
        if not candidates:
            return None
        return max(candidates, key=lambda c: c.get("session_ts", ""))
