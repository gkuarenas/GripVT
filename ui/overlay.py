import cv2
import numpy as np

_MAX_VIS_PX = 50.0

from core.tracker import MarkerRecord


def draw_overlay(
    frame: np.ndarray,
    records: list[MarkerRecord],
    session_active: bool,
    frame_index: int = 0,
) -> np.ndarray:
    out = frame.copy()

    for r in records:
        cx, cy = int(r.x), int(r.y)
        bx = int(r.x - r.dx)
        by = int(r.y - r.dy)

        if session_active:
            mag = float(np.hypot(r.dx, r.dy))
            if mag >= 1.0:
                scale = min(1.0, _MAX_VIS_PX / mag)
                ex = int(bx + r.dx * scale)
                ey = int(by + r.dy * scale)
                tip_len = min(0.4, 10.0 / max(mag * scale, 1.0))
                mag_mm = r.magnitude_mm
                if mag_mm < 1.5:
                    color = (255, 80, 0)    # blue   — small   (< 1.5 mm)
                elif mag_mm < 4.0:
                    color = (0, 140, 255)   # orange — medium  (1.5–4.0 mm)
                else:
                    color = (0, 0, 255)     # red    — large   (>= 4.0 mm)
                cv2.arrowedLine(out, (bx, by), (ex, ey), color, 2, tipLength=tip_len)
            dot_color = (0, 200, 255) if r.autofilled else (200, 200, 200)
            cv2.circle(out, (cx, cy), 3, dot_color, 2)
        else:
            color = (0, 200, 0) if not r.autofilled else (0, 200, 255)
            cv2.circle(out, (cx, cy), 6, color, 2)

        cv2.putText(
            out,
            str(r.marker_id),
            (cx + 4, cy - 4),
            cv2.FONT_HERSHEY_PLAIN,
            0.7,
            (0, 255, 255),
            1,
        )

    state_str = "RECORDING" if session_active else "IDLE"
    hud = f"frame {frame_index}  markers {len(records)}  [{state_str}]"
    cv2.putText(out, hud, (8, 20), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 255, 200), 1)

    return out
