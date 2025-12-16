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
    return np.sqrt(np.sum(diff**2))


class EuclideanDistance(CostMetric):
    """Class for calculating Euclidean distance between feature vectors/matrices."""

    def __init__(self):
        super().__init__(v2v_cost=euclidean_dist_vec2vec, name="euclidean")

    ### Matrix-Matrix Euclidean Distance
    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates Euclidean distance between two feature matrices fm_1 and fm_2.

        Optimized implementation using broadcasting and matrix operations.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames_1)
            fm_2: Query feature matrix, shape (n_features, n_frames_2)

        Returns:
            Euclidean distance matrix, shape (n_frames_1, n_frames_2).
            Element (i, j) is the Euclidean distance between fm_1[:, i] and fm_2[:, j].
        """
        # Use broadcasting: (n_features, n_frames_1, 1) - (n_features, 1, n_frames_2)
        # Results in (n_features, n_frames_1, n_frames_2)
        diff = fm_1[:, :, np.newaxis] - fm_2[:, np.newaxis, :]

        # Sum over features axis and return (n_frames_1, n_frames_2)
        return np.sqrt(np.sum(diff**2, axis=0))

    ### Vector-Matrix Euclidean Distance
    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Euclidean distance between a feature matrix fm_1 and a feature frame vector fv_2.

        Optimized implementation using broadcasting.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames)
            fv_2: Query feature frame, shape (n_features, 1)

        Returns:
            Euclidean distance vector, shape (n_frames,).
            Element i is the Euclidean distance between fm_1[:, i] and fv_2.
        """
        # Broadcast subtraction: (n_features, n_frames) - (n_features, 1)
        diff = fm_1 - fv_2

        # Sum over features axis
        return np.sqrt(np.sum(diff**2, axis=0))

    ### Vector-Vector Euclidean Distance
    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Euclidean distance between two feature frame vectors fv_1 and fv_2.

        Args:
            fv_1: Reference feature frame, shape (n_features, 1)
            fv_2: Query feature frame, shape (n_features, 1)

        Returns:
            Euclidean distance (scalar) between fv_1 and fv_2
        """
        return self.v2v_cost(fv_1, fv_2)
