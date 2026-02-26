"""Lp-norm cost metrics. Optimized numpy and numba."""
# TODO: write tests for this file

# standard imports
import warnings

# library imports
import numpy as np
from numba import njit

# custom imports
from .cost_metric import CostMetric


# define v2v cost function
@njit
def lp_dist_vec2vec(fv_1: np.ndarray, fv_2: np.ndarray, p: int):
    r"""Calculates lp norm distance between two feature frames fv_1 and fv_2.
    The lp norm is defined as:

    .. math::
        d(fv_1, fv_2) = \left( \sum_{i=1}^{n} |fv_1[i] - fv_2[i]|^p \right)^{1/p}

    where :math:`n` is feature dimension and :math:`p` is the norm order.

    Args:
        fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
        fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
        p (int): dimension of the norm.

    Returns:
        lp norm distance between fv_1 and fv_2
    """

    diff = fv_1 - fv_2
    return np.power(np.sum(np.power(np.abs(diff), p)), 1 / p)


class LpNormDistance(CostMetric):
    """Class for calculating lp norm distance between feature vectors/matrices."""

    def __init__(self, p: int):
        super().__init__(v2v_cost=lp_dist_vec2vec, name=f"l{p}-norm")
        if p <= 0:
            raise ValueError(f"p must be a positive integer, got {p}")
        if p == 1:
            warnings.warn("p=1 is equivalent to Manhattan distance, use ManhattanDistance instead")
        elif p == 2:
            warnings.warn("p=2 is equivalent to Euclidean distance, use EuclideanDistance instead")
        self.p = p

    # Matrix-Matrix Lp Norm Distance
    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates lp norm distance between two feature matrices fm_1 and fm_2."""
        diff = fm_1[:, :, np.newaxis] - fm_2[:, np.newaxis, :]
        return np.power(np.sum(np.power(np.abs(diff), self.p), axis=0), 1 / self.p)

    # Vector-Matrix Lp Norm Distance
    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Lp norm distance between feature matrix fm_1 and feature vector fv_2."""
        diff = fm_1 - fv_2
        return np.power(np.sum(np.power(np.abs(diff), self.p), axis=0), 1 / self.p)

    # Vector-Vector Lp Norm Distance
    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """Calculates lp norm distance between two feature frame vectors fv_1 and fv_2."""
        return self.v2v_cost(fv_1, fv_2, self.p)
