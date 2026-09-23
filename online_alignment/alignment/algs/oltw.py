"""Numba kernels for banded Online Time Warping (OLTW), per Dixon (2005).

The accumulated cost matrix is indexed ``D[t, j]`` with ``t`` the reference
frame and ``j`` the query frame. It is stored in a 2D ring buffer: cell
``(t, j)`` lives at ``D[t % D.shape[0], j % D.shape[1]]``. Callers reset a
ring row (column) to ``inf`` before filling a new reference row (query
column), so stale cells read as ``inf``, exactly like cells outside the band.
"""

# library imports
import numpy as np
from numba import njit

# path transition indices into window_steps
BOTH = 0
ROW = 1
COLUMN = 2


@njit(cache=True)
def _fill_cell(D: np.ndarray, t: int, j: int, cost: float, steps: np.ndarray, weights: np.ndarray):
    """Set ``D[t, j]`` to the cheapest predecessor plus the weighted local cost.

    Matches the recurrence of ``librosa.sequence.dtw``.
    """
    n_rows, n_cols = D.shape
    best = np.inf
    for k in range(steps.shape[0]):
        prev_t = t - steps[k, 0]
        prev_j = j - steps[k, 1]
        if prev_t < 0 or prev_j < 0:
            continue
        cur = D[prev_t % n_rows, prev_j % n_cols] + weights[k] * cost
        if cur < best:
            best = cur
    D[t % n_rows, j % n_cols] = best


@njit(cache=True)
def oltw_fill_row(
    D: np.ndarray, t: int, j_start: int, costs: np.ndarray, steps: np.ndarray, weights: np.ndarray
):
    """Fill reference row ``t`` for query columns ``j_start .. j_start + len(costs) - 1``."""
    for k in range(costs.shape[0]):
        _fill_cell(D, t, j_start + k, costs[k], steps, weights)


@njit(cache=True)
def oltw_fill_column(
    D: np.ndarray, j: int, t_start: int, costs: np.ndarray, steps: np.ndarray, weights: np.ndarray
):
    """Fill query column ``j`` for reference rows ``t_start .. t_start + len(costs) - 1``."""
    for k in range(costs.shape[0]):
        _fill_cell(D, t_start + k, j, costs[k], steps, weights)


@njit(cache=True)
def _min_cost_indices(D: np.ndarray, t: int, j: int, c: int):
    """Location of the lowest path-length-normalized cost in the current row and column.

    Searches ``D[t, j-c+1 .. j]`` and ``D[t-c+1 .. t, j]`` (the whole row and
    column when ``c < 0``). The row wins ties. Returns ``(x, y)``.
    """
    n_rows, n_cols = D.shape
    row_start = max(0, j - c + 1) if c >= 0 else 0
    col_start = max(0, t - c + 1) if c >= 0 else 0

    row_min = np.inf
    row_idx = row_start
    for k in range(row_start, j + 1):
        v = D[t % n_rows, k % n_cols] / (t + k + 1)
        if v < row_min:
            row_min = v
            row_idx = k

    col_min = np.inf
    col_idx = col_start
    for k in range(col_start, t + 1):
        v = D[k % n_rows, j % n_cols] / (k + j + 1)
        if v < col_min:
            col_min = v
            col_idx = k

    if row_min <= col_min:
        return t, row_idx
    return col_idx, j


@njit(cache=True)
def oltw_get_inc(
    D: np.ndarray,
    t: int,
    j: int,
    c: int,
    run_count: int,
    prev: int,
    max_run_count: int,
) -> int:
    """Choose the next path transition: BOTH, ROW, or COLUMN.

    Args:
        D: Ring buffer of accumulated costs.
        t: Current reference frame.
        j: Current query frame.
        c: Band width, or a negative number for no band.
        run_count: Number of consecutive times ``prev`` was taken.
        prev: Last non-BOTH transition, or -1 if none yet.
        max_run_count: Maximum consecutive ROW or COLUMN transitions.
    """
    if c >= 0 and t < c:
        return BOTH
    if run_count >= max_run_count:
        return COLUMN if prev == ROW else ROW
    x, y = _min_cost_indices(D, t, j, c)
    if x < t:
        return COLUMN
    if y < j:
        return ROW
    return BOTH
