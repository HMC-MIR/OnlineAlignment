"""Utility functions for alignment algorithms."""

# library imports
import numpy as np


def _arrange_oltw_steps(steps: np.ndarray) -> np.ndarray:
    """Arrange OLTW path transitions into [BOTH, ROW, COLUMN] order by slope.

    Each step is (reference_increment, query_increment). The step with the
    highest slope (reference only) is ROW, the lowest (query only) is COLUMN,
    and the one in between is BOTH.

    Args:
        steps: OLTW transition steps. Shape (3, 2)

    Returns:
        np.ndarray: Steps ordered [BOTH, ROW, COLUMN].
    """
    steps = np.asarray(steps)

    # slope = reference increment / query increment; pure reference steps sort highest
    slopes = np.zeros(steps.shape[0])
    for i, (r, c) in enumerate(steps):
        if c == 0:
            slopes[i] = np.inf
        else:
            slopes[i] = r / c

    # highest slope first, then swap the first two for [BOTH, ROW, COLUMN]
    arranged_steps = steps[np.argsort(-slopes, kind="stable")]
    arranged_steps[[0, 1]] = arranged_steps[[1, 0]]
    return arranged_steps


def _validate_dtw_steps_weights(steps: np.ndarray, weights: np.ndarray) -> None:
    """Validate DTW steps and weights.

    Args:
        steps: DTW steps array. Shape (n_steps, 2)
        weights: DTW weights array. Shape (n_steps,)

    Raises:
        ValueError: If steps or weights have invalid shapes or values.
    """
    if steps.ndim != 2 or steps.shape[1] != 2:
        raise ValueError(
            f"DTW steps must have 2 columns for row and column steps. Got shape {steps.shape}"
        )
    if weights.ndim != 1:
        raise ValueError(f"DTW weights must be 1D array, got {weights.ndim}D array")
    if steps.shape[0] != weights.shape[0]:
        raise ValueError("DTW steps and weights must have the same number of rows")
    if np.any(steps < 0) or np.any(steps.sum(axis=1) == 0):
        raise ValueError(f"DTW steps must be non-negative and nonzero, got {steps.tolist()}")


def _validate_window_steps(window_steps: np.ndarray) -> None:
    """Validate OLTW path transition steps.

    Args:
        window_steps: Transition steps. Shape (3, 2)

    Raises:
        ValueError: If the steps have the wrong shape or are not one reference-only,
            one query-only, and one step that advances both.
    """
    if window_steps.shape != (3, 2):
        raise ValueError(f"window_steps must have shape (3, 2), got {window_steps.shape}")
    if np.any(window_steps < 0):
        raise ValueError(f"window_steps must be non-negative, got {window_steps.tolist()}")
    n_row_only = int(np.sum((window_steps[:, 0] > 0) & (window_steps[:, 1] == 0)))
    n_col_only = int(np.sum((window_steps[:, 0] == 0) & (window_steps[:, 1] > 0)))
    n_both = int(np.sum((window_steps[:, 0] > 0) & (window_steps[:, 1] > 0)))
    if (n_row_only, n_col_only, n_both) != (1, 1, 1):
        raise ValueError(
            "window_steps must contain one reference-only, one query-only and one "
            f"combined step, got {window_steps.tolist()}"
        )


def _validate_query_features_shape(query_features: np.ndarray, n_features: int) -> None:
    """Validate that query features are 2D with the reference's feature dimension.

    Args:
        query_features: Complete query features. Shape (n_features, n_frames)
        n_features: Feature dimension of the reference.

    Raises:
        ValueError: If the query features have the wrong shape.
    """
    if query_features.ndim != 2:
        raise ValueError(f"query_features must be 2D array, got {query_features.ndim}D")
    if query_features.shape[0] != n_features:
        raise ValueError(
            f"query_features must have {n_features} features to match the reference, "
            f"got {query_features.shape[0]}"
        )
    if query_features.shape[1] == 0:
        raise ValueError("query_features must contain at least one frame")


def _validate_query_frame(query_frame: np.ndarray, n_features: int) -> np.ndarray:
    """Validate a single query frame and return it as a 1D array.

    Args:
        query_frame: One query frame. Shape (n_features,) or (n_features, 1)
        n_features: Feature dimension of the reference.

    Returns:
        np.ndarray: The frame with shape (n_features,).

    Raises:
        ValueError: If the frame has the wrong shape.
    """
    query_frame = np.asarray(query_frame)
    if query_frame.ndim == 2 and query_frame.shape[1] == 1:
        query_frame = query_frame[:, 0]
    if query_frame.shape != (n_features,):
        raise ValueError(
            f"query_frame must have shape ({n_features},) or ({n_features}, 1), "
            f"got {query_frame.shape}"
        )
    return query_frame
