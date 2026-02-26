"""Define cosine cost metrics for two features. Written with optimized numpy & numba functions."""

# library imports
import numpy as np
from numba import njit, prange

# custom imports
from .cost_metric import CostMetric


# define v2v cost function
def cosine_dist_vec2vec(fv_1: np.ndarray, fv_2: np.ndarray, normalized: bool = None):
    """
    Calculates cosine distance between two feature frames fv_1 and fv_2.
        Assumes fv_1 and fv_2 are unnormalized.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
        normalized (bool): boolean indicating if fv_1 and fv_2 are __both__ L2 normalized

    Returns:
        Cosine distance (scalar) between normalized fv_1 and normalized fv_2
    """

    # check if normalized
    normalized_fv_1, normalized_fv_2 = normalized, normalized  # initialize
    if normalized is None:  # we don't know if the features are L2 normalized or not
        normalized_fv_1 = _check_l2_normalized_vec(fv_1)
        normalized_fv_2 = _check_l2_normalized_vec(fv_2)
        normalized = normalized_fv_1 & normalized_fv_2  # if both are normalized

    # normalize feature vectors if needed
    if not normalized_fv_1:
        fv_1 = fv_1 / (
            np.linalg.norm(fv_1, ord=2) + 1e-10
        )  # l2 normalization, let numpy handle div by 0 errors
    if not normalized_fv_2:
        fv_2 = fv_2 / (np.linalg.norm(fv_2, ord=2) + 1e-10)

    # now we are guaranteed normalization
    return _cosine_dist_vec2vec_normalized(fv_1, fv_2)


@njit
def _cosine_dist_vec2vec_normalized(fv_1: np.ndarray, fv_2: np.ndarray):
    """
    Calculates cosine distance between two normalized feature frames fv_1 and fv_2.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
    """
    return 1 - np.dot(fv_1.T, fv_2)  # optimize


@njit(parallel=True, cache=True)
def cosine_mat2mat_parallel(fm_1: np.ndarray, fm_2: np.ndarray) -> np.ndarray:
    """
    Cosine distance matrix in parallel over reference (row) indices.
    Columns of fm_1 and fm_2 are L2-normalized inside the function.
    Shape: (n_frames_1, n_frames_2). Use when BLAS is single-threaded.
    """
    n_feat, n_ref = fm_1.shape
    n_query = fm_2.shape[1]
    C = np.empty((n_ref, n_query), dtype=fm_1.dtype)
    # Normalize columns (broadcast-friendly in numba by row)
    norms_1 = np.empty(n_ref, dtype=fm_1.dtype)
    norms_2 = np.empty(n_query, dtype=fm_2.dtype)
    for i in range(n_ref):
        n1 = 0.0
        for k in range(n_feat):
            n1 += fm_1[k, i] * fm_1[k, i]
        norms_1[i] = np.sqrt(n1) + 1e-10
    for j in range(n_query):
        n2 = 0.0
        for k in range(n_feat):
            n2 += fm_2[k, j] * fm_2[k, j]
        norms_2[j] = np.sqrt(n2) + 1e-10
    for i in prange(n_ref):
        for j in range(n_query):
            dot = 0.0
            for k in range(n_feat):
                dot += (fm_1[k, i] / norms_1[i]) * (fm_2[k, j] / norms_2[j])
            C[i, j] = 1.0 - dot
    return C


# Helper Functions
def _check_l2_normalized_vec(fv_1: np.ndarray) -> bool:
    """
    Checks if a feature vector fv_1 is L2 normalized or not.

    Args:
        fv_1 (np.ndarray): feature vector to be checked

    Returns:
        bool: indicates if fv_1 is L2 normalized or not
    """
    l2_norm_squared = np.sum(fv_1**2)
    return np.isclose(l2_norm_squared, 1.0, atol=1e-6)


class CosineDistance(CostMetric):
    """Class for calculating cosine distance between feature vectors/matrices."""

    def __init__(self):
        super().__init__(v2v_cost=cosine_dist_vec2vec, name="cosine")

    # Matrix-Matrix Cosine Distance
    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray, normalized: bool = None):
        """
        Calculates cosine distance between two feature matrices fm_1 and fm_2.

        Optimized implementation using matrix operations. Assumes fm_1 and fm_2 are unnormalized
        unless normalized=True is specified.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames_1)
            fm_2: Query feature matrix, shape (n_features, n_frames_2)
            normalized: If True, assumes both matrices are L2 normalized column-wise.
                If None, checks normalization automatically.

        Returns:
            Cosine distance matrix, shape (n_frames_1, n_frames_2).
            Element (i, j) is the cosine distance between fm_1[:, i] and fm_2[:, j].
        """
        # Normalize matrices if needed
        if normalized is None:
            # Check if columns are normalized
            norms_1 = np.linalg.norm(fm_1, ord=2, axis=0, keepdims=True)
            norms_2 = np.linalg.norm(fm_2, ord=2, axis=0, keepdims=True)
            normalized = np.allclose(norms_1, 1.0, atol=1e-6) and np.allclose(
                norms_2, 1.0, atol=1e-6
            )

        if not normalized:
            # Normalize columns
            fm_1 = fm_1 / (np.linalg.norm(fm_1, ord=2, axis=0, keepdims=True) + 1e-10)
            fm_2 = fm_2 / (np.linalg.norm(fm_2, ord=2, axis=0, keepdims=True) + 1e-10)

        # Compute cosine distance
        return 1 - fm_1.T @ fm_2

    # Vector-Matrix Cosine Distance
    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray, normalized: bool = None):
        """
        Calculates cosine distance between a feature matrix fm_1 and a feature frame vector fv_2.

        Optimized implementation using matrix operations. Assumes fm_1 and fv_2 are unnormalized
        unless normalized=True is specified.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames)
            fv_2: Query feature frame, shape (n_features, 1)
            normalized: If True, assumes fm_1 columns and fv_2 are L2 normalized.
                If None, checks normalization automatically.

        Returns:
            Cosine distance vector, shape (n_frames,).
            Element i is the cosine distance between fm_1[:, i] and fv_2.
        """
        # Normalize if needed
        if normalized is None:
            norms_1 = np.linalg.norm(fm_1, ord=2, axis=0, keepdims=True)
            norm_2 = np.linalg.norm(fv_2, ord=2)
            normalized = np.allclose(norms_1, 1.0, atol=1e-6) and np.isclose(norm_2, 1.0, atol=1e-6)

        if not normalized:
            fm_1 = fm_1 / (np.linalg.norm(fm_1, ord=2, axis=0, keepdims=True) + 1e-10)
            fv_2 = fv_2 / (np.linalg.norm(fv_2, ord=2) + 1e-10)

        # Convert to cosine distance
        return 1 - (fm_1.T @ fv_2).flatten()

    # Vector-Vector Cosine Distance
    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray, normalized: bool = None):
        """
        Calculates cosine distance between two feature frame vectors fv_1 and fv_2.
            Assumes fv_1 and fv_2 are unnormalized.

        Args:
            fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
            normalized (bool): boolean indicating if fv_1 and fv_2 are __both__ L2 normalized

        Returns:
            Cosine distance (scalar) between normalized fv_1 and normalized fv_2
        """
        return self.v2v_cost(fv_1, fv_2, normalized)
