"""Constants for the online_alignment package."""

import numpy as np

# SOA (Simple Online Alignment) default DTW steps and weights.
# Each step row is (query_increment, reference_increment).
SOA_STEPS: np.ndarray = np.array([1, 1, 1, 2, 2, 1]).reshape((-1, 2))
SOA_WEIGHTS: np.ndarray = np.array([1, 1, 2])

# OLTW (Online Time Warping) default DTW steps and weights.
# Each step row is (reference_increment, query_increment).
OLTW_STEPS: np.ndarray = np.array([1, 0, 0, 1, 1, 1]).reshape((-1, 2))
OLTW_WEIGHTS: np.ndarray = np.array([1, 1, 1])

# OLTW default path transitions, ordered [BOTH, ROW, COLUMN].
OLTW_WINDOW_STEPS: np.ndarray = np.array([1, 1, 1, 0, 0, 1]).reshape((-1, 2))

# OLTW default band width (in frames) for the search window.
OLTW_BAND_WIDTH: int = 500
