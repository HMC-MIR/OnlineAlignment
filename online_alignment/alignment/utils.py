"""Utility functions for alignment algorithms."""

# library imports
import numpy as np


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
