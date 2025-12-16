"""Define cosine cost metrics for two features. Written with optimized numpy & numba functions."""

# library imports
import numpy as np
from numba import njit

# custom imports
from .cost_metric import CostMetric


# define v2v cost function
def cosine_dist_vec2vec(fv_1: np.ndarray, fv_2: np.ndarray, normalized: bool = None):
    """Calculates cosine distance between two feature frames fv_1 and fv_2. Assumes fv_1 and fv_2 are unnormalized.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
        normalized (bool): boolean indicating if fv_1 and fv_2 are __both__ L2 normalized

    Returns:
        cosine distance between normalized fv_1 and normalized fv_2
    """

    # check if normalized
    normalized_fv_1, normalized_fv_2 = normalized, normalized  # initialize
    if normalized is None:  # we don't know if the features are L2 normalized or not
        normalized_fv_1 = _check_l2_normalized_vec(fv_1)
        normalized_fv_2 = _check_l2_normalized_vec(fv_2)
        normalized = normalized_fv_1 & normalized_fv_2  # if both are normalized

    # normalize feature vectors if needed
    if not normalized_fv_1:
        fv_1 = fv_1 / np.linalg.norm(
            fv_1, ord=2
        )  # l2 normalization, let numpy handle div by 0 errors
    if not normalized_fv_2:
        fv_2 = fv_2 / np.linalg.norm(fv_2, ord=2)

    # now we are guaranteed normalization
    return _cosine_dist_vec2vec_normalized(fv_1, fv_2)


@njit
def _cosine_dist_vec2vec_normalized(fv_1: np.ndarray, fv_2: np.ndarray):
    """Calculates cosine distance between two normalized feature frames fv_1 and fv_2.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
    """
    return 1 - fv_1.T @ fv_2  # optimize


### Helper Functions
def _check_l2_normalized_vec(fv_1: np.ndarray) -> bool:
    """Checks if a feature vector fv_1 is L2 normalized or not.

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

    ### Matrix-Matrix Cosine Distance
    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray, normalized: bool = None):
        """Calculates cosine distance between two feature matrices fm_1 and fm_2. Assumes fm_1 and fm_2 are unnormalized.

        Args:
            fm_1 (np.ndarray): reference feature matrix, shape (n_features, n_frames)
            fm_2 (np.ndarray): query feature matrix, shape (n_features, n_frames)
            normalized (bool): boolean indicating if fm_1 and fm_2 are __both__ L2 normalized

        Returns:
            cosine distance matrix between normalized fm_1 and normalized fm_2
        """
        raise NotImplementedError  # TODO: implement

    ### Vector-Matrix Cosine Distance
    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray, normalized: bool = None):
        """Calculates cosine distance between a feature matrix fm_1 and a feature frame vector fv_2. Assumes fm_1 and fv_2 are unnormalized.

        Args:
            fm_1 (np.ndarray): reference feature matrix, shape (n_features, n_frames)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
            normalized (bool): boolean indicating if fm_1 and fv_2 are __both__ L2 normalized

        Returns:
            cosine distance vector between normalized fm_1 and normalized fv_2
        """
        raise NotImplementedError  # TODO: implement

    ### Vector-Vector Cosine Distance
    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray, normalized: bool = None):
        """Calculates cosine distance between two feature frame vectors fv_1 and fv_2. Assumes fv_1 and fv_2 are unnormalized.

        Args:
            fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
            normalized (bool): boolean indicating if fv_1 and fv_2 are __both__ L2 normalized

        Returns:
            cosine distance vector between normalized fv_1 and normalized fv_2
        """
        return self.v2v_cost(fv_1, fv_2, normalized)
