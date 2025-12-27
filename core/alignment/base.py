"""Base class for alignment algorithms."""

# standard imports
from abc import ABC, abstractmethod
from typing import Callable

# library imports
import numpy as np

# custom imports
from ..cost import CostMetric, get_cost_metric


class AlignmentBase(ABC):
    """Base class for alignment algorithms of two signals.

    This class provides the common interface for both online and offline
    alignment algorithms. Subclasses should implement the specific alignment
    logic.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        cost_metric: str | Callable | CostMetric,
        **kwargs,
    ):
        """Initialize the alignment algorithm.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            **kwargs: Additional keyword arguments to pass to the cost metric constructor.
                For "lpnorm", use `p` to specify the norm order (defaults to 2).
        """
        # Validate input shape
        if reference_features.ndim != 2:
            raise ValueError(f"reference_features must be 2D array, got {reference_features.ndim}D")

        # set up reference
        self.reference_features = reference_features
        self.reference_length = reference_features.shape[1]

        # set up alignment costs
        self.cost_metric = get_cost_metric(cost_metric, **kwargs)

    @abstractmethod
    def align(self, query_features: np.ndarray):
        """Align query features to reference features.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Alignment path or result (implementation-specific).
        """
        pass


class OnlineAlignment(AlignmentBase):
    """Base class for online alignment algorithms.

    Online alignment algorithms process features frame-by-frame as they arrive,
    making them suitable for streaming or real-time applications.
    """

    @abstractmethod
    def feed(self, query_frame: np.ndarray):
        """Feed a single query frame into the alignment system.

        Args:
            query_frame: Single frame of query features. Shape (n_features, 1)
        """

    @abstractmethod
    def process_frame(self):
        """Process the most recently fed query frame.

        This method should update the internal alignment state based on the
        last frame fed via feed().
        """

    @abstractmethod
    def align(self, query_features: np.ndarray):
        """Simulate online process with complete query features.

        This is a convenience method that processes all query features
        frame-by-frame using feed() and process_frame().

        Args:
            query_features: Complete query feature matrix.
                Shape (n_features, n_frames)

        Returns:
            Alignment path or result (implementation-specific).
        """
