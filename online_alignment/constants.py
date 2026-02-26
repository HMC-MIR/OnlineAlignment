"""Constants for the online_alignment package."""

import numpy as np

# default DTW steps and weights
DEFAULT_DTW_STEPS: np.ndarray = np.array([1, 1, 1, 2, 2, 1]).reshape((-1, 2))
DEFAULT_DTW_WEIGHTS: np.ndarray = np.array([1, 1, 2])
OLTW_STEPS: np.ndarray = np.array([1, 0, 0, 1, 1, 1]).reshape((-1, 2))
OLTW_WEIGHTS: np.ndarray = np.array([1, 1, 1])

# NOA default steps and weights (matches template default: [[1,1],[1,2],[2,1]])
NOA_STEPS: np.ndarray = np.array([1, 1, 1, 2, 2, 1]).reshape((-1, 2))
NOA_WEIGHTS: np.ndarray = np.array([1, 1, 2])
