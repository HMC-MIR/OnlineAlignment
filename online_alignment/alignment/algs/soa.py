"""Numba kernels for Simple Online Alignment (SOA).

SOA uses the steps (1,1), (1,2), (2,1) as (query, reference) increments, so row
``i`` of the accumulated cost matrix depends only on rows ``i - 1`` and ``i - 2``.
The matrix is kept as a ring buffer of three rows: row ``i`` lives in
``D[i % 3]``. Rows for frames before the first hold ``inf`` (and a start of -1),
which is exactly "no predecessor", so the first frames need no special case.

Every cell of a new row is independent of the others, so each kernel is a single
loop over the reference with branch-free selects that compiles to SIMD
instructions. Normalized costs are written to a separate array, and the position
is taken with ``np.argmin``, which returns the first minimum.

Exactness: candidates are float32 and path lengths are integers. With a fixed
start, every candidate into a cell shares the path length, so the cheapest one is
chosen by raw cost. With a flexible start, candidates v_a / l_a and v_b / l_b are
compared by cross-multiplying, v_a * l_b < v_b * l_a, which is exact in float64
(24 + 30 bits < 53). Two different quotients of this form never round to the same
float64, so these choices, and the argmin over the divided scores, are the same as
comparing exact normalized costs. Steps are tried in order, so the first wins ties.
"""

# library imports
import numpy as np
from numba import njit


@njit(cache=True)
def soa_update_fixed(
    costs: np.ndarray,
    D: np.ndarray,
    cur: int,
    r1: int,
    r2: int,
    w0: float,
    w1: float,
    w2: float,
):
    """Fixed start: compute a row of the accumulated cost matrix.

    Args:
        costs: Local costs of the current query frame. Shape (ref_length,), float32
        D: Ring buffer of accumulated cost rows. Shape (3, ref_length), float32
        cur: Ring row to write (current frame).
        r1: Ring row of the previous frame.
        r2: Ring row of the frame before that.
        w0, w1, w2: Weights of the steps (1,1), (1,2), (2,1), float32.
    """
    ref_length = D.shape[1]
    row, P1, P2 = D[cur], D[r1], D[r2]
    row[0] = np.inf
    if ref_length > 1:
        c = costs[1]
        a = P1[0] + c * w0
        b = P2[0] + c * w2
        row[1] = b if b < a else a
    for j in range(2, ref_length):
        c = costs[j]
        a = P1[j - 1] + c * w0  # (1,1)
        b = P1[j - 2] + c * w1  # (1,2)
        a = b if b < a else a
        b = P2[j - 1] + c * w2  # (2,1)
        row[j] = b if b < a else a


@njit(cache=True)
def soa_scores_fixed(i: int, row: np.ndarray, scores: np.ndarray):
    """Fixed start: accumulated cost over path length, ``row[j] / (i + j + 2)``."""
    for j in range(row.shape[0]):
        scores[j] = row[j] / (i + 1 + j + 1)


@njit(cache=True)
def soa_update_flexible(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    S: np.ndarray,
    cur: int,
    r1: int,
    r2: int,
    w0: float,
    w1: float,
    w2: float,
    scores: np.ndarray,
):
    """Flexible start: compute a row of D and S, and its normalized scores.

    ``S`` holds the reference frame where each cell's best path began (-1 if the
    cell is unreachable, exactly when D is inf). Candidates are compared by
    accumulated cost over path length ``i + j - start``.

    Args:
        i: Current query frame index.
        costs: Local costs of the current query frame. Shape (ref_length,), float32
        D: Ring buffer of accumulated cost rows. Shape (3, ref_length), float32
        S: Ring buffer of path starts, same shape as ``D``, int32.
        cur, r1, r2: Ring rows of the current, previous and second previous frame.
        w0, w1, w2: Weights of the steps (1,1), (1,2), (2,1), float32.
        scores: Output normalized costs. Shape (ref_length,), float64
    """
    ref_length = D.shape[1]
    Dc, Sc = D[cur], S[cur]
    P1, P2, Q1, Q2 = D[r1], D[r2], S[r1], S[r2]
    Dc[0] = np.inf
    Sc[0] = -1
    scores[0] = np.inf
    for j in range(1, ref_length):
        c = costs[j]
        # (1,1) from (i-1, j-1)
        best_d = P1[j - 1] + c * w0
        best_s = Q1[j - 1]
        best_l = i + j - best_s
        # (1,2) from (i-1, j-2)
        if j >= 2:
            s = Q1[j - 2]
            v = P1[j - 2] + c * w1
            length = i + j - s
            better = np.float64(v) * best_l < np.float64(best_d) * length
            best_d = v if better else best_d
            best_s = s if better else best_s
            best_l = length if better else best_l
        # (2,1) from (i-2, j-1)
        s = Q2[j - 1]
        v = P2[j - 1] + c * w2
        length = i + j - s
        better = np.float64(v) * best_l < np.float64(best_d) * length
        best_d = v if better else best_d
        best_s = s if better else best_s
        best_l = length if better else best_l
        Dc[j] = best_d
        Sc[j] = best_s
        scores[j] = best_d / best_l
