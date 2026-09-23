"""Numba kernels for Simple Online Alignment (SOA).

The accumulated cost matrix is stored as a ring buffer of rows: row ``i`` of
the full matrix lives in ``D[i % D.shape[0]]``. The buffer only needs
``max(row_steps) + 1`` rows, so memory does not grow with the query length.
"""

# library imports
import numpy as np
from numba import njit


@njit(cache=True)
def soa_row_update(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    dn: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
    normalize: bool,
) -> int:
    """Compute row ``i`` of the accumulated cost matrix and return the best column.

    The caller must reset ``D[i % D.shape[0]]`` to ``inf`` before calling.

    Args:
        i: Current query frame index (row of the full matrix).
        costs: Local cost vector for frame ``i``. Shape (ref_length,)
        D: Ring buffer of accumulated cost rows. Shape (n_rows, ref_length)
        dn: Row (query) step sizes. Shape (n_steps,)
        dm: Column (reference) step sizes. Shape (n_steps,)
        dw: Step weights. Shape (n_steps,)
        normalize: If True, choose predecessors and the best column by
            accumulated cost divided by path length ``i + j + 2``.

    Returns:
        Reference column index with the lowest (normalized) accumulated cost.
    """
    n_rows, ref_length = D.shape
    n_steps = dn.shape[0]
    cur = i % n_rows

    # ring-buffer row of each step's predecessor, or -1 if it is before the first row
    prev_rows = np.empty(n_steps, dtype=np.int64)
    for k in range(n_steps):
        prev_i = i - dn[k]
        prev_rows[k] = prev_i % n_rows if prev_i >= 0 else -1

    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step_score = np.inf
        best_step = -1

        for k in range(n_steps):
            prev_j = j - dm[k]
            if prev_rows[k] < 0 or prev_j < 0:
                continue

            cur_cost = D[prev_rows[k], prev_j] + costs[j] * dw[k]
            if normalize:
                score = cur_cost / (i + 1 + j + 1)  # normalize by path length
            else:
                score = cur_cost

            if score < best_step_score:
                best_step_score = score
                best_step_cost = cur_cost
                best_step = k

        if best_step != -1:
            D[cur, j] = best_step_cost

            if best_step_score < best_cost:
                best_cost = best_step_score
                best_j = j

    return best_j
