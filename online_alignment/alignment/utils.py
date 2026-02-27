"""Utility functions for alignment algorithms."""

# library imports
import numpy as np

from typing import Tuple

def _arrange_oltw_steps(steps: np.ndarray) -> np.ndarray:
    """Arrange OLTW steps by comparing slopes.
    Highest slope = vertical, middle = diagonal, lowest = horizontal.

    Args:
        steps: OLTW steps array.

    Returns:
        np.ndarray: Arranged OLTW steps.
    """
    steps = np.asarray(steps)

    # Calculate slope as row increment divided by column increment
    # Handle division by zero: use np.inf so verticals (=0 col increment) sort highest
    slopes = np.zeros(steps.shape[0])
    for i, (r, c) in enumerate(steps):
        if c == 0 and r != 0:
            slopes[i] = np.inf  # vertical step (row step, no col step)
        elif r == 0 and c != 0:
            slopes[i] = 0.0     # horizontal step
        elif r == 0 and c == 0:
            slopes[i] = -np.inf # degenerate, shouldn't happen
        else:
            slopes[i] = r / c   # diagonal or irregular

    # Get sorted indices: vertical (inf), then diagonal, then horizontal (0)
    # Reverse so highest slopes (vertical) come first.
    sorted_indices = np.argsort(-slopes)

    arranged_steps = steps[sorted_indices]
    # Swap index 0 and 1 at the end
    # so that we have diagonal, vertical, horizontal order
    if arranged_steps.shape[0] > 1:
        arranged_steps[[0, 1]] = arranged_steps[[1, 0]]
    return arranged_steps


def _validate_dtw_steps_weights(steps: np.ndarray, weights: np.ndarray) -> None:
    """Validate DTW steps and weights.

    Args:
        steps: DTW steps array.
            Shape (n_steps, 2)
        weights: DTW weights array.
            Shape (n_steps, 1)

    Raises:
        ValueError: If DTW steps and weights have different shapes.
    """
    if steps.shape[0] != weights.shape[0]:
        raise ValueError("DTW steps and weights must have the same number of rows")

    if steps.shape[1] != 2:
        raise ValueError(
            f"DTW steps must have 2 columns for row and column steps. Got shape {steps.shape}"
        )
    if weights.ndim != 1:
        raise ValueError(f"DTW weights must be 1D array, got {weights.ndim}D array")


def _validate_prev_alignment_path(prev_alignment_path: np.ndarray, ref_length: int) -> None:
    """Validate previous alignment path.

    Args:
        prev_alignment_path: Previous alignment path.
            Should be 1D array with length no longer than the reference features.
        ref_length: Length of the reference features.

    Raises:
        ValueError: If previous alignment path has invalid shape.
    """
    # check if previous alignment path is 1D array
    if prev_alignment_path.ndim != 1:
        raise ValueError("Previous alignment path must be 1D array")

    # check if previous alignment path length is no longer than the reference features
    if len(prev_alignment_path) > ref_length:
        raise ValueError(
            f"Previous alignment path must be no longer than the reference features. "
            f"Got {len(prev_alignment_path)} > {ref_length}"
        )


def _validate_query_features_shape(query_features: np.ndarray):
    """Validate query features have the correct shape.

    Args:
        query_features (np.ndarray): complete query features input to an alignment algorithm
    """

    # check that query features have the right shape
