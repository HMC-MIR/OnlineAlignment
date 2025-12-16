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
            Distance between normalized fv_1 and normalized fv_2
        """
        return self.v2v_cost(fv_1, fv_2)

    def mat2vec(self, fm_1: np.ndarray, fv_2: np.ndarray):
        """Calculates distance between a feature matrix fm_1 and a feature frame vector fv_2.

        Args:
            fm_1 (np.ndarray): reference feature matrix, shape (n_features, n_frames)
            fv_2 (np.ndarray): query feature frame, shape (n_features, 1)
            normalized (bool): boolean indicating if fm_1 and fv_2 are __both__ L2 normalized

        Returns:
            Distance vector between normalized fm_1 and normalized fv_2
        """
        raise NotImplementedError  # TODO: provide generic implementation based on v2v cost function

    def mat2mat(self, fm_1: np.ndarray, fm_2: np.ndarray):
        """Calculates distance between two feature matrices fm_1 and fm_2.

        Args:
            fm_1 (np.ndarray): reference feature matrix, shape (n_features, n_frames)
            fm_2 (np.ndarray): query feature matrix, shape (n_features, n_frames)

        Returns:
            Distance matrix between fm_1 and fm_2
        """
        raise NotImplementedError  # TODO: provide generic implementation based on v2v cost function
