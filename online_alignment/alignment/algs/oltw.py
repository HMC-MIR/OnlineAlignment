"""Numba kernels for banded Online Time Warping (OLTW), per Dixon (2005).

The accumulated cost matrix is indexed ``D[t, j]`` with ``t`` the reference
frame and ``j`` the query frame. It is stored in a 2D ring buffer: cell
``(t, j)`` lives at ``D[t % D.shape[0], j % D.shape[1]]``. A ring row (column)
is reset to ``inf`` before a new reference row (query column) is filled, so
stale cells read as ``inf``, exactly like cells outside the band.

Ring buffers are always larger than any step, so an index that steps below
zero wraps with a single addition.
"""

# library imports
import numpy as np
from numba import njit

# path transition indices into window_steps
BOTH = 0
ROW = 1
COLUMN = 2

# layout of the state array passed to oltw_advance_cosine
STATE_T = 0
STATE_J = 1
STATE_RUN_COUNT = 2
STATE_PREV = 3
STATE_PATH_LEN = 4


# ---------------------------------------------------------------------------
# Accumulated cost updates
# ---------------------------------------------------------------------------


@njit(cache=True)
def _best_predecessor(D, t, j, rt, jc, cost, steps, weights):
    """Cheapest predecessor of cell (t, j), at ring position (rt, jc), plus the weighted cost.

    Matches the recurrence of ``librosa.sequence.dtw``.
    """
    n_rows, n_cols = D.shape
    best = np.inf
    for k in range(steps.shape[0]):
        dt = steps[k, 0]
        dj = steps[k, 1]
        if t < dt or j < dj:
            continue
        pr = rt - dt
        if pr < 0:
            pr += n_rows
        pc = jc - dj
        if pc < 0:
            pc += n_cols
        cur = D[pr, pc] + weights[k] * cost
        if cur < best:
            best = cur
    return best


@njit(cache=True)
def oltw_fill_row(
    D: np.ndarray, t: int, j_start: int, costs: np.ndarray, steps: np.ndarray, weights: np.ndarray
):
    """Fill reference row ``t`` for query columns ``j_start .. j_start + len(costs) - 1``."""
    n_rows, n_cols = D.shape
    rt = t % n_rows
    jc = j_start % n_cols
    for k in range(costs.shape[0]):
        D[rt, jc] = _best_predecessor(D, t, j_start + k, rt, jc, costs[k], steps, weights)
        jc += 1
        if jc == n_cols:
            jc = 0


@njit(cache=True)
def oltw_fill_column(
    D: np.ndarray, j: int, t_start: int, costs: np.ndarray, steps: np.ndarray, weights: np.ndarray
):
    """Fill query column ``j`` for reference rows ``t_start .. t_start + len(costs) - 1``."""
    n_rows, n_cols = D.shape
    rt = t_start % n_rows
    jc = j % n_cols
    for k in range(costs.shape[0]):
        D[rt, jc] = _best_predecessor(D, t_start + k, j, rt, jc, costs[k], steps, weights)
        rt += 1
        if rt == n_rows:
            rt = 0


# ---------------------------------------------------------------------------
# Path transitions
# ---------------------------------------------------------------------------


@njit(cache=True)
def _min_cost_indices(D: np.ndarray, t: int, j: int, c: int):
    """Location of the lowest path-length-normalized cost in the current row and column.

    Searches ``D[t, j-c+1 .. j]`` and ``D[t-c+1 .. t, j]`` (the whole row and
    column when ``c < 0``). The row wins ties. Returns ``(x, y)``.
    """
    n_rows, n_cols = D.shape
    row_start = max(0, j - c + 1) if c >= 0 else 0
    col_start = max(0, t - c + 1) if c >= 0 else 0
    rt = t % n_rows
    jc = j % n_cols

    row_min = np.inf
    row_idx = row_start
    kc = row_start % n_cols
    for k in range(row_start, j + 1):
        v = D[rt, kc] / (t + k + 1)
        if v < row_min:
            row_min = v
            row_idx = k
        kc += 1
        if kc == n_cols:
            kc = 0

    col_min = np.inf
    col_idx = col_start
    kr = col_start % n_rows
    for k in range(col_start, t + 1):
        v = D[kr, jc] / (k + j + 1)
        if v < col_min:
            col_min = v
            col_idx = k
        kr += 1
        if kr == n_rows:
            kr = 0

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


# ---------------------------------------------------------------------------
# Fused cosine path walk
# ---------------------------------------------------------------------------


@njit(cache=True)
def _cosine_cost(reference: np.ndarray, query: np.ndarray, t: int, jc: int) -> float:
    """Cosine distance of prenormalized columns; same arithmetic as cosine_mat2mat_prenormalized."""
    dot = 0.0
    for k in range(reference.shape[0]):
        dot += reference[k, t] * query[k, jc]
    return 1.0 - dot


@njit(cache=True)
def _fill_row_cosine(D, reference, query, t, j_lo, j_hi, steps, weights):
    """Fill reference row ``t`` for query columns ``j_lo .. j_hi`` with cosine costs."""
    n_rows, n_cols = D.shape
    rt = t % n_rows
    jc = j_lo % n_cols
    for j in range(j_lo, j_hi + 1):
        cost = _cosine_cost(reference, query, t, jc)
        D[rt, jc] = _best_predecessor(D, t, j, rt, jc, cost, steps, weights)
        jc += 1
        if jc == n_cols:
            jc = 0


@njit(cache=True)
def _fill_column_cosine(D, reference, query, j, t_lo, t_hi, steps, weights):
    """Fill query column ``j`` for reference rows ``t_lo .. t_hi`` with cosine costs."""
    n_rows, n_cols = D.shape
    rt = t_lo % n_rows
    jc = j % n_cols
    for t in range(t_lo, t_hi + 1):
        cost = _cosine_cost(reference, query, t, jc)
        D[rt, jc] = _best_predecessor(D, t, j, rt, jc, cost, steps, weights)
        rt += 1
        if rt == n_rows:
            rt = 0


@njit(cache=True)
def oltw_advance_cosine(
    D: np.ndarray,
    reference: np.ndarray,
    query: np.ndarray,
    state: np.ndarray,
    path: np.ndarray,
    window_steps: np.ndarray,
    steps: np.ndarray,
    weights: np.ndarray,
    c: int,
    max_run_count: int,
    last_frame: int,
    final: bool,
):
    """Take OLTW path transitions until the path needs a query frame not yet fed.

    Same algorithm as the generic Python loop in ``OLTW._advance``, with cosine
    costs computed inline.

    Args:
        D: Ring buffer of accumulated costs. Shape (n_rows, n_cols)
        reference: Prenormalized reference features. Shape (n_features, ref_length)
        query: Ring buffer of prenormalized query frames. Shape (n_features, n_cols)
        state: [t, j, run_count, prev, path_len], updated in place.
        path: Path buffer, rows [query, reference]. Must have room for every
            remaining step: ``(ref_length - 1 - t) + (last_frame - j)`` more points.
        window_steps: Transitions in [BOTH, ROW, COLUMN] order. Shape (3, 2)
        steps: DTW steps. Shape (n_steps, 2)
        weights: DTW step weights. Shape (n_steps,)
        c: Band width, or a negative number for no band.
        max_run_count: Maximum consecutive ROW or COLUMN transitions.
        last_frame: Index of the newest query frame fed.
        final: True once the query has ended; transitions past the last frame
            are then clamped to it.
    """
    n_rows, n_cols = D.shape
    ref_length = reference.shape[1]
    t = state[STATE_T]
    j = state[STATE_J]
    run_count = state[STATE_RUN_COUNT]
    prev = state[STATE_PREV]
    path_len = state[STATE_PATH_LEN]

    while t < ref_length - 1 and j < last_frame:
        inc = oltw_get_inc(D, t, j, c, run_count, prev, max_run_count)
        dt = window_steps[inc, 0]
        dj = window_steps[inc, 1]
        if not final and j + dj > last_frame:
            break  # wait for more query frames

        t_new = min(t + dt, ref_length - 1)
        j_new = min(j + dj, last_frame)

        # new reference rows over the band of query frames
        j_lo = 0 if c < 0 else max(0, j - c + 1)
        for tt in range(t + 1, t_new + 1):
            if n_rows < ref_length:
                D[tt % n_rows, :] = np.inf
            _fill_row_cosine(D, reference, query, tt, j_lo, j, steps, weights)

        # new query columns over the band of reference frames
        t_lo = 0 if c < 0 else max(0, t_new - c + 1)
        for jj in range(j + 1, j_new + 1):
            if c >= 0:
                D[:, jj % n_cols] = np.inf
            _fill_column_cosine(D, reference, query, jj, t_lo, t_new, steps, weights)

        # update run count
        if inc == prev:
            run_count += 1
        else:
            run_count = 1
        if inc != BOTH:
            prev = inc

        t = t_new
        j = j_new
        path[0, path_len] = j
        path[1, path_len] = t
        path_len += 1

    state[STATE_T] = t
    state[STATE_J] = j
    state[STATE_RUN_COUNT] = run_count
    state[STATE_PREV] = prev
    state[STATE_PATH_LEN] = path_len
