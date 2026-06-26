from dataclasses import dataclass, field
import numpy as np


@dataclass
class KalmanState:
    id: int
    x: np.ndarray          # shape (6,): [x, y, dx, dy, ddx, ddy]
    P: np.ndarray          # shape (6, 6) covariance
    baseline_pos: tuple[float, float]
    baseline_area: float
    autofilled: bool = False


def _build_F(dt: float = 1.0) -> np.ndarray:
    return np.array([
        [1, 0, dt, 0,  0.5 * dt * dt, 0            ],
        [0, 1, 0,  dt, 0,             0.5 * dt * dt],
        [0, 0, 1,  0,  dt,            0            ],
        [0, 0, 0,  1,  0,             dt           ],
        [0, 0, 0,  0,  1,             0            ],
        [0, 0, 0,  0,  0,             1            ],
    ], dtype=float)


class KalmanManager:
    def __init__(self) -> None:
        self.states: dict[int, KalmanState] = {}
        self.F = _build_F(dt=1.0)
        self.H = np.array([[1, 0, 0, 0, 0, 0],
                           [0, 1, 0, 0, 0, 0]], dtype=float)
        self.Q = np.eye(6) * 0.1
        self.R = np.eye(2) * 5.0

    def init_state(self, marker_id: int, x: float, y: float, area: float) -> None:
        state_vec = np.zeros(6, dtype=float)
        state_vec[0] = x
        state_vec[1] = y
        self.states[marker_id] = KalmanState(
            id=marker_id,
            x=state_vec,
            P=np.eye(6) * 100.0,
            baseline_pos=(x, y),
            baseline_area=area,
        )

    def predict_all(self) -> dict[int, np.ndarray]:
        priors: dict[int, np.ndarray] = {}
        for sid, s in self.states.items():
            s.x = self.F @ s.x
            s.P = self.F @ s.P @ self.F.T + self.Q
            priors[sid] = s.x[:2].copy()
        return priors

    def correct(self, marker_id: int, z: np.ndarray) -> None:
        s = self.states[marker_id]
        innov = z - self.H @ s.x
        S = self.H @ s.P @ self.H.T + self.R
        K = s.P @ self.H.T @ np.linalg.inv(S)
        s.x = s.x + K @ innov
        s.P = (np.eye(6) - K @ self.H) @ s.P
        s.autofilled = False

    def mark_autofilled(self, marker_id: int) -> None:
        s = self.states[marker_id]
        s.autofilled = True
        s.x[2:] = 0.0  # prevent kinematic drift while no detection
