"""Template class for online alignment algorithms of two signals"""

# standard imports
from typing import Any, Callable

# library imports
import numpy as np

# custom imports
from core.cost import CostMetric
from utils.backend import handle_cost


class OnlineAlignment:
    """Template class for online alignment algorithms of two signals"""

    def __init__(
        self,
        reference_features: np.ndarray,
        cost_metric: str | Callable | CostMetric,
    ):
        """Initialize the online alignment class.

        Args:
            reference_features (np.ndarray): features for the reference audio. Shape (n_features, n_frames)
        """
        # set up reference
        self.reference_features = reference_features

        # set up alignment costs
        self.cost_metric = handle_cost(cost_metric)

    def feed(self, query_frame: np.ndarray):
        """Feeds query frame features into the system"""
        raise NotImplementedError

    def process_frame(self):
        """Process incoming query frame."""
        raise NotImplementedError

    def align(self, query_features: np.ndarray):
        """Simulate online process with complete query features.

        Args:
            query_features (np.ndarray): Complete query feature matrix calculated ahead of time.
        """
        raise NotImplementedError
