import numpy as np
from scipy.optimize import linear_sum_assignment


def assign(
    priors: dict[int, np.ndarray],
    detections: list[tuple[float, float, float]],
    gate_px: float = 100.0,
) -> tuple[dict[int, int], list[int]]:
    if not priors:
        return {}, []
    if not detections:
        return {}, list(priors.keys())

    ids = list(priors.keys())
    prior_xy = np.array([priors[i] for i in ids], dtype=float)        # (N, 2)
    det_xy = np.array([(d[0], d[1]) for d in detections], dtype=float) # (M, 2)

    diff = prior_xy[:, None, :] - det_xy[None, :, :]   # (N, M, 2)
    cost = np.linalg.norm(diff, axis=2)                 # (N, M)

    cost_gated = np.where(cost <= gate_px, cost, 1e6)

    row_ind, col_ind = linear_sum_assignment(cost_gated)

    matches: dict[int, int] = {}
    unmatched: list[int] = []
    assigned_rows: set[int] = set(row_ind.tolist())

    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if cost[r, c] <= gate_px:
            matches[ids[r]] = c
        else:
            unmatched.append(ids[r])

    for i, sid in enumerate(ids):
        if i not in assigned_rows:
            unmatched.append(sid)

    return matches, unmatched
