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
    n = len(ids)
    m = len(detections)
    prior_xy = np.array([priors[i] for i in ids], dtype=float)        # (N, 2)
    det_xy = np.array([(d[0], d[1]) for d in detections], dtype=float) # (M, 2)

    diff = prior_xy[:, None, :] - det_xy[None, :, :]   # (N, M, 2)
    cost = np.linalg.norm(diff, axis=2)                 # (N, M)

    # scipy.linear_sum_assignment on an NxM matrix is forced into a complete
    # bijection over min(N, M) pairs -- whenever N and M happen to line up
    # (e.g. one state's detection is missing but a stray extra detection
    # exists elsewhere), it has no way to leave either side unmatched, so it
    # minimizes TOTAL cost by reshuffling otherwise-correct neighbouring
    # pairs to absorb the mismatch (e.g. state A gives up its own ~0-cost
    # detection to a neighbour so state B, whose real detection is missing,
    # isn't left unassigned) instead of just rejecting the one true outlier.
    # Augmenting the matrix with an explicit "leave unmatched" option per
    # state/detection gives the solver a way out that doesn't require
    # touching a correct pair. The reject price must stay well under
    # gate_px/2: rejecting a genuinely-orphaned state+detection pair (2
    # rejects) must always cost less than reassigning even a single
    # near-gate-boundary "steal" (up to ~gate_px), or the same chain-swap
    # bug reappears with the reject option unused.
    reject_cost = 0.4 * gate_px
    out_of_gate = gate_px * 1000.0 + 1.0  # must dominate any real/reject cost
    real_block = np.where(cost <= gate_px, cost, out_of_gate)
    reject_states = np.full((n, n), out_of_gate)
    np.fill_diagonal(reject_states, reject_cost)
    reject_detections = np.full((m, m), out_of_gate)
    np.fill_diagonal(reject_detections, reject_cost)
    filler = np.zeros((m, n))

    augmented = np.block([
        [real_block, reject_states],
        [reject_detections, filler],
    ])

    row_ind, col_ind = linear_sum_assignment(augmented)

    matches: dict[int, int] = {}
    unmatched: list[int] = []
    for r, c in zip(row_ind.tolist(), col_ind.tolist()):
        if r >= n:
            continue  # a detection-reject row matched to a state-reject column
        if c < m:
            matches[ids[r]] = c
        else:
            unmatched.append(ids[r])

    return matches, unmatched
