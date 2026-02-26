import numpy as np
from numba import njit


@njit
def normalize_by_path_length(D: np.ndarray) -> np.ndarray:
    """
    Normalize an accumulated cost matrix D by its Manhattan distance from the starting point (0, 0).
    For each element D[i, j], the normalization factor is (i + j + 1) to avoid division by zero at (0, 0).

    Args:
        D (np.ndarray): Accumulated cost matrix.

    Returns:
        np.ndarray: The normalized accumulated cost matrix.
    """
    n_rows, n_cols = D.shape
    D_normalized = np.zeros_like(D, dtype=D.dtype)
    for i in range(n_rows):
        for j in range(n_cols):
            path_length = i + j + 1  # add 1 so we don't divide by 0 at (0,0)
            D_normalized[i, j] = D[i, j] / path_length
    return D_normalized
