"""Manhattan cost metrics. Optimized numpy and numba."""
# TODO: write tests for this file

# library imports
import numpy as np
from numba import njit

# custom imports
from .lpnorm import LpNormDistance


# define v2v cost function
@njit
def manhattan_dist_vec2vec(fv_1: np.ndarray, fv_2: np.ndarray):
    """Calculates manhattan distance between two feature frames fv_1 and fv_2."""
    diff = fv_1 - fv_2
    return np.sum(np.abs(diff))


class ManhattanDistance(LpNormDistance):
    """Class for calculating Manhattan distance between feature vectors/matrices."""

    def __init__(self):
        super().__init__(p=2)
        self.v2v_cost = manhattan_dist_vec2vec
        self.name = "manhattan"

    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates Manhattan distance between two feature matrices."""
        diff = fm_1[:, :, np.newaxis] - fm_2[:, np.newaxis, :]
        return np.sum(np.abs(diff), axis=0)

    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Manhattan distance between a feature matrix and a feature frame vector."""
        diff = fm_1 - fv_2
        return np.sum(np.abs(diff), axis=0)

    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """Calculates Manhattan distance between two feature frame vectors."""
        return self.v2v_cost(fv_1, fv_2)
