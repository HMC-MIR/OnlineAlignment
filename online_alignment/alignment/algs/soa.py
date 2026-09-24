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


@njit(cache=True)
def soa_row_update_flexible(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    dn: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
) -> int:
    """Compute row ``i`` for a flexible start, where paths may begin at any reference frame.

    ``S`` holds the reference frame where each cell's best path began (-1 if the
    cell is unreachable). Predecessors and the best column are compared by
    accumulated cost divided by the path length ``i + j - start``. The caller must
    reset ``D[i % D.shape[0]]`` to ``inf`` and ``S[i % S.shape[0]]`` to -1 first.

    Args:
        i: Current query frame index (row of the full matrix).
        costs: Local cost vector for frame ``i``. Shape (ref_length,)
        D: Ring buffer of accumulated cost rows. Shape (n_rows, ref_length)
        S: Ring buffer of path start frames, same shape as ``D``.
        dn: Row (query) step sizes. Shape (n_steps,)
        dm: Column (reference) step sizes. Shape (n_steps,)
        dw: Step weights. Shape (n_steps,)

    Returns:
        Reference column index with the lowest normalized accumulated cost.
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
        best_step_score = np.inf
        best_step = -1

        for k in range(n_steps):
            prev_j = j - dm[k]
            if prev_rows[k] < 0 or prev_j < 0:
                continue
            start = S[prev_rows[k], prev_j]
            if start < 0:
                continue

            score = (D[prev_rows[k], prev_j] + costs[j] * dw[k]) / (i + j - start)
            if score < best_step_score:
                best_step_score = score
                best_step = k

        if best_step != -1:
            pr = prev_rows[best_step]
            pj = j - dm[best_step]
            D[cur, j] = D[pr, pj] + costs[j] * dw[best_step]
            S[cur, j] = S[pr, pj]

            if best_step_score < best_cost:
                best_cost = best_step_score
                best_j = j

    return best_j


# ---------------------------------------------------------------------------
# Unrolled kernels for three-step patterns (the SOA default)
# ---------------------------------------------------------------------------
#
# Same results as the general kernels above, but faster:
# - every column of the new row is written, so the row needs no reset to inf;
# - with a fixed start, all candidate steps into a cell share the path length
#   i + j + 2, so the cheapest candidate is chosen by raw cost and only the
#   winner is divided. The costs are float32 and the division is float64, so
#   dividing cannot merge or reorder two different candidates: the choice is
#   the same as comparing normalized scores;
# - the three steps are unrolled, with their predecessor rows looked up once;
# - with a flexible start, normalized costs v / l (float32 cost v, integer path
#   length l) are compared by cross-multiplying, v_a * l_b < v_b * l_a, with no
#   division. In float64 these products are exact (24 + 30 bits < 53), and two
#   different quotients of this form differ by far more than float64 rounding,
#   so the decisions are the same as comparing the divided values.


@njit(cache=True)
def soa_row_update3(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    rows: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
    normalize: bool,
) -> int:
    """Fixed-start row update for exactly three steps. Same result as ``soa_row_update``.

    Args:
        i: Current query frame index (row of the full matrix).
        costs: Local cost vector for frame ``i``. Shape (ref_length,)
        D: Ring buffer of accumulated cost rows. Shape (n_rows, ref_length)
        rows: Ring-buffer row of each step's predecessor, or -1. Shape (3,)
        dm: Column (reference) step sizes. Shape (3,)
        dw: Step weights. Shape (3,)
        normalize: If True, pick the best column by cost over path length.

    Returns:
        Reference column index with the lowest (normalized) accumulated cost.
    """
    n_rows, ref_length = D.shape
    cur = D[i % n_rows]
    r0, r1, r2 = rows[0], rows[1], rows[2]
    m0, m1, m2 = dm[0], dm[1], dm[2]
    w0, w1, w2 = dw[0], dw[1], dw[2]

    best_j = 0
    best_cost = np.inf

    # columns (or early frames) where some predecessor falls outside the matrix
    n_checked = ref_length
    if r0 >= 0 and r1 >= 0 and r2 >= 0:
        n_checked = min(max(m0, m1, m2), ref_length)
    for j in range(n_checked):
        c = costs[j]
        best = np.inf
        if r0 >= 0 and j >= m0:
            v = D[r0, j - m0] + c * w0
            if v < best:
                best = v
        if r1 >= 0 and j >= m1:
            v = D[r1, j - m1] + c * w1
            if v < best:
                best = v
        if r2 >= 0 and j >= m2:
            v = D[r2, j - m2] + c * w2
            if v < best:
                best = v
        cur[j] = best
        if best < np.inf:
            score = best / (i + 1 + j + 1) if normalize else best
            if score < best_cost:
                best_cost = score
                best_j = j
    if n_checked == ref_length:
        return best_j

    # every predecessor is inside the matrix: no bounds checks
    P0, P1, P2 = D[r0], D[r1], D[r2]
    if normalize:
        for j in range(n_checked, ref_length):
            c = costs[j]
            best = P0[j - m0] + c * w0
            v = P1[j - m1] + c * w1
            if v < best:
                best = v
            v = P2[j - m2] + c * w2
            if v < best:
                best = v
            cur[j] = best
            if best < np.inf:
                score = best / (i + 1 + j + 1)
                if score < best_cost:
                    best_cost = score
                    best_j = j
    else:
        for j in range(n_checked, ref_length):
            c = costs[j]
            best = P0[j - m0] + c * w0
            v = P1[j - m1] + c * w1
            if v < best:
                best = v
            v = P2[j - m2] + c * w2
            if v < best:
                best = v
            cur[j] = best
            if best < best_cost:
                best_cost = best
                best_j = j
    return best_j


@njit(cache=True)
def soa_row_update_flexible3(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    rows: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
) -> int:
    """Flexible-start row update for exactly three steps. Same result as
    ``soa_row_update_flexible``; see it for the arguments (``rows`` as in
    ``soa_row_update3``)."""
    n_rows, ref_length = D.shape
    cur_row = i % n_rows
    Dc, Sc = D[cur_row], S[cur_row]
    r0, r1, r2 = rows[0], rows[1], rows[2]
    m0, m1, m2 = dm[0], dm[1], dm[2]
    w0, w1, w2 = dw[0], dw[1], dw[2]

    best_j = 0
    best_cost = np.inf  # normalized: best_cost / best_len
    best_len = 1

    n_checked = ref_length
    if r0 >= 0 and r1 >= 0 and r2 >= 0:
        n_checked = min(max(m0, m1, m2), ref_length)
    for j in range(n_checked):
        c = costs[j]
        best_d = np.inf  # best candidate so far: best_d / best_l
        best_l = 1
        best_s = -1
        if r0 >= 0 and j >= m0:
            s = S[r0, j - m0]
            if s >= 0:
                v = D[r0, j - m0] + c * w0
                length = i + j - s
                if v * best_l < best_d * length:
                    best_d, best_l, best_s = v, length, s
        if r1 >= 0 and j >= m1:
            s = S[r1, j - m1]
            if s >= 0:
                v = D[r1, j - m1] + c * w1
                length = i + j - s
                if v * best_l < best_d * length:
                    best_d, best_l, best_s = v, length, s
        if r2 >= 0 and j >= m2:
            s = S[r2, j - m2]
            if s >= 0:
                v = D[r2, j - m2] + c * w2
                length = i + j - s
                if v * best_l < best_d * length:
                    best_d, best_l, best_s = v, length, s
        Dc[j] = best_d
        Sc[j] = best_s
        if best_d * best_len < best_cost * best_l:
            best_cost = best_d
            best_len = best_l
            best_j = j
    if n_checked == ref_length:
        return best_j

    # every predecessor is inside the matrix: no bounds checks
    P0, P1, P2 = D[r0], D[r1], D[r2]
    Q0, Q1, Q2 = S[r0], S[r1], S[r2]
    for j in range(n_checked, ref_length):
        c = costs[j]
        best_d = np.inf  # best candidate so far: best_d / best_l
        best_l = 1
        best_s = -1
        s = Q0[j - m0]
        if s >= 0:
            v = P0[j - m0] + c * w0
            length = i + j - s
            if v * best_l < best_d * length:
                best_d, best_l, best_s = v, length, s
        s = Q1[j - m1]
        if s >= 0:
            v = P1[j - m1] + c * w1
            length = i + j - s
            if v * best_l < best_d * length:
                best_d, best_l, best_s = v, length, s
        s = Q2[j - m2]
        if s >= 0:
            v = P2[j - m2] + c * w2
            length = i + j - s
            if v * best_l < best_d * length:
                best_d, best_l, best_s = v, length, s
        Dc[j] = best_d
        Sc[j] = best_s
        if best_d * best_len < best_cost * best_l:
            best_cost = best_d
            best_len = best_l
            best_j = j
    return best_j
