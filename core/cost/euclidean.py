"""Define Euclidean cost metrics for two features. Written with optimized numpy and numba"""

# library imports
import numpy as np
from numba import njit

# custom imports
from .cost_metric import CostMetric


# define v2v cost function
@njit
def euclidean_dist_vec2vec(fv_1: np.ndarray, fv_2: np.ndarray):
    """Calculates euclidean distance between two feature frames fv_1 and fv_2.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)

    Returns:
        Euclidean distance between fv_1 and fv_2
    """

    diff = fv_1 - fv_2
    return np.sum(diff**2)


class EuclideanDistance(CostMetric):
    """Class for calculating Euclidean distance between feature vectors/matrices."""

    def __init__(self):
        super().__init__(v2v_cost=euclidean_dist_vec2vec, name="euclidean")

    ### Matrix-Matrix Euclidean Distance
    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates euclidean distance between two feature matrices fm_1 and fm_2.

        Args:
            fm_1 (np.ndarray): reference feature matrix, shape (n_features, n_frames)
            fm_2 (np.ndarray): query feature matrix, shape (n_features, n_frames)

        Returns:
            Euclidean distance matrix between fm_1 and fm_2
        """
        raise NotImplementedError

    ### Vector-Matrix Euclidean Distance
    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Euclidean distance between a feature matrix fm_1 and a feature frame vector fv_2.

        Args:
            fm_1 (np.ndarray): reference feature matrix, shape (n_features, n_frames)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)

        Returns:
            Euclidean distance vector between fm_1 and fv_2
        """
        raise NotImplementedError  # TODO: implement

    ### Vector-Vector Euclidean Distance
    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Euclidean distance between two feature frame vectors fv_1 and fv_2.

        Args:
            fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
            normalized (bool): boolean indicating if fv_1 and fv_2 are __both__ L2 normalized

        Returns:
            Euclidean distance vector between fv_1 and fv_2
        """
        return self.v2v_cost(fv_1, fv_2)
