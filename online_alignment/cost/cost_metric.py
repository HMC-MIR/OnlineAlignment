"""Template class for cost metrics"""

# standard imports
from typing import Callable

# library imports
import numpy as np


class CostMetric:
    """Template class for calculating costs between feature vectors/matrices"""

    def __init__(self, v2v_cost: Callable, name: str):
        """Initializes the cost metric with vector to vector distance.

        Args:
            v2v_cost (Callable): function to calculate distance between two feature vectors
            name (str): name of the cost metric
        """
        self.v2v_cost = v2v_cost
        self.name = name

    def vec2vec(self, fv_1: np.ndarray, fv_2: np.ndarray):
        """Calculates distance between two feature frames fv_1 and fv_2.

        Args:
            fv_1 (np.ndarray): reference feature frame, shape (n_features, 1)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)

        Returns:
            Distance (scalar) between fv_1 and fv_2
        """
        return self.v2v_cost(fv_1, fv_2)

    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Calculates distance between a feature matrix fm_1 and a feature frame vector fv_2.

        Generic implementation that computes distance between each column of fm_1 and fv_2.
        Subclasses can override this for optimized implementations.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames)
            fv_2: Query feature frame, shape (n_features, 1)

        Returns:
            Distance vector, shape (n_frames,). Element i is the distance between
            fm_1[:, i] and fv_2.
        """
        n_frames = fm_1.shape[1]
        distances = np.zeros(n_frames, dtype=np.float32)
        for i in range(n_frames):
            distances[i] = self.vec2vec(fm_1[:, i:i + 1], fv_2)
        return distances

    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates distance between two feature matrices fm_1 and fm_2.

        Generic implementation that computes distance between all pairs of columns.
        Subclasses can override this for optimized implementations.

        Args:
            fm_1: Reference feature matrix, shape (n_features, n_frames_1)
            fm_2: Query feature matrix, shape (n_features, n_frames_2)

        Returns:
            Distance matrix, shape (n_frames_1, n_frames_2). Element (i, j) is the
            distance between fm_1[:, i] and fm_2[:, j].
        """
        n_frames_1 = fm_1.shape[1]
        n_frames_2 = fm_2.shape[1]
        distances = np.zeros((n_frames_1, n_frames_2), dtype=np.float32)
        for i in range(n_frames_1):
            for j in range(n_frames_2):
                distances[i, j] = self.vec2vec(fm_1[:, i:i + 1], fm_2[:, j:j + 1])
        return distances
