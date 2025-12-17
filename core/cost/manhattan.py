"""Define Manhattan cost metrics for two features. Written with optimized numpy and numba""" #TODO: write tests for this file

# library imports
import numpy as np
from numba import njit

# custom imports
from .lpnorm import LpNormDistance


# define v2v cost function
@njit
def manhattan_dist_vec2vec(fv_1: np.ndarray, fv_2: np.ndarray):
    """Calculates manhattan distance between two feature frames fv_1 and fv_2.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)

    Returns:
        Manhattan distance between fv_1 and fv_2
    """

    diff = fv_1 - fv_2
    return np.sum(np.abs(diff))


class ManhattanDistance(LpNormDistance):
    """Class for calculating Manhattan distance between feature vectors/matrices."""

    def __init__(self):
        super().__init__(p=2)
        self.v2v_cost = manhattan_dist_vec2vec
        self.name = "manhattan"

    ### Matrix-Matrix Manhattan Distance
    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates Manhattan distance between two feature matrices fm_1 and fm_2.

        Optimized implementation using broadcasting and matrix operations.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames_1)
            fm_2: Query feature matrix, shape (n_features, n_frames_2)

        Returns:
            Manhattan distance matrix, shape (n_frames_1, n_frames_2).
            Element (i, j) is the Manhattan distance between fm_1[:, i] and fm_2[:, j].
        """
        # Use broadcasting: (n_features, n_frames_1, 1) - (n_features, 1, n_frames_2)
        # Results in (n_features, n_frames_1, n_frames_2)
        diff = fm_1[:, :, np.newaxis] - fm_2[:, np.newaxis, :]

        # Sum over features axis and return (n_frames_1, n_frames_2)
        return np.sum(np.abs(diff), axis=0)

    ### Vector-Matrix Manhattan Distance
    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Manhattan distance between a feature matrix fm_1 and a feature frame vector fv_2.

        Optimized implementation using broadcasting.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames)
            fv_2: Query feature frame, shape (n_features, 1)

        Returns:
            Manhattan distance vector, shape (n_frames,).
            Element i is the Manhattan distance between fm_1[:, i] and fv_2.
        """
        # Broadcast subtraction: (n_features, n_frames) - (n_features, 1)
        diff = fm_1 - fv_2

        # Sum over features axis
        return np.sum(np.abs(diff), axis=0)

    ### Vector-Vector Manhattan Distance
    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Manhattan distance between two feature frame vectors fv_1 and fv_2.

        Args:
            fv_1: Reference feature frame, shape (n_features, 1)
            fv_2: Query feature frame, shape (n_features, 1)

        Returns:
            Manhattan distance (scalar) between fv_1 and fv_2
        """
        return self.v2v_cost(fv_1, fv_2)
