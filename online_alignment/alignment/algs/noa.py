# library imports
import numpy as np
from numba import njit


@njit(cache=True)
def noa_row_update(
    i: int,
    prev_j: int,
    costs: np.ndarray,
    D: np.ndarray,
    B: np.ndarray,
    dn: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
    ref_length: int,
) -> int:
    """
    Update alignment row using numba for optimized performance.

    Args:
        i: Current row index in the alignment matrix.
        prev_j: Previous column index in the alignment matrix.
        costs: Cost vector for the current feature row.
        D: Alignment cost matrix.
        B: Backtrace matrix.
        dn: Row step sizes.
        dm: Column step sizes.
        dw: Weights for each step.
        ref_length: Length of the reference features.

    Returns:
        Index of the best match in the reference features.
    """
    best_j = 0
    best_cost = np.inf

    # iterate over all possible column indices
    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step = -1

        # iterate over steps to find min cost step
        for k, (di, dj, w) in enumerate(zip(dn, dm, dw)):  # type: ignore
            prev_i, prev_j = i - di, j - dj

            # check if step is out of bounds
            if prev_i < 0 or prev_j < 0 or prev_j >= ref_length:
                continue

            # calculate cost of step
            cur_cost = D[prev_i, prev_j] + costs[j] * w

            # update best step if cost is lower
            if cur_cost < best_step_cost:
                best_step_cost = cur_cost
                best_step = k

        # update alignment cost matrix and backtrace matrix if best step is found
        if best_step != -1:
            D[i, j] = best_step_cost
            B[i, j] = best_step

            # update best cost if current cost is lower
            if best_step_cost < best_cost:
                best_cost = best_step_cost
                best_j = j

    # return best column index
    return best_j
