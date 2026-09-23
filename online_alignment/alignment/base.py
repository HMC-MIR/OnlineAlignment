"""Base class for alignment algorithms."""

# standard imports
from abc import ABC, abstractmethod
from typing import Callable, Union

# library imports
import numpy as np

# custom imports
from ..cost import CostMetric, get_cost_metric
from .utils import _validate_query_features_shape


class AlignmentBase(ABC):
    """Base class for alignment algorithms of two signals.

    This class provides the common interface for both online and offline
    alignment algorithms. Subclasses should implement the specific alignment
    logic.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        cost_metric: Union[str, Callable, CostMetric],
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
        reference_features = np.asarray(reference_features)
        if reference_features.ndim != 2:
            raise ValueError(f"reference_features must be 2D array, got {reference_features.ndim}D")
        if reference_features.shape[1] == 0:
            raise ValueError("reference_features must contain at least one frame")

        # set up reference
        self.reference_features = reference_features
        self.reference_length = reference_features.shape[1]
        self.n_features = reference_features.shape[0]

        # set up alignment costs
        self.cost_metric = get_cost_metric(cost_metric, **kwargs)

    @abstractmethod
    def align(self, query_features: np.ndarray) -> np.ndarray:
        """Align query features to reference features.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Warping path of integer frame indices. Shape (2, n_path_points),
            where ``path[0]`` is query frames and ``path[1]`` is reference frames.
        """


class OnlineAlignment(AlignmentBase):
    """Base class for online alignment algorithms.

    Online alignment algorithms process the query frame by frame as it arrives,
    using memory that does not grow with the query length. Call ``feed()`` for
    each new frame, then ``flush()`` once the query has ended.
    """

    @abstractmethod
    def reset(self) -> None:
        """Clear all alignment state so a new query can be aligned."""

    @abstractmethod
    def feed(self, query_frame: np.ndarray) -> int:
        """Feed the next query frame and advance the alignment.

        Args:
            query_frame: Single frame of query features.
                Shape (n_features,) or (n_features, 1)

        Returns:
            Current estimate of the reference frame index.
        """

    def flush(self) -> int:
        """Signal the end of the query and finish any pending path steps.

        Returns:
            Final estimate of the reference frame index.
        """
        return self.position

    @property
    @abstractmethod
    def position(self) -> int:
        """Current estimate of the reference frame index."""

    @property
    @abstractmethod
    def path(self) -> np.ndarray:
        """Warping path so far. Shape (2, n_path_points), rows [query, reference]."""

    def align(self, query_features: np.ndarray) -> np.ndarray:
        """Simulate the online process on a complete query.

        Resets the state, feeds every query frame in order, then flushes.

        Args:
            query_features: Complete query feature matrix.
                Shape (n_features, n_frames)

        Returns:
            Warping path of integer frame indices. Shape (2, n_path_points),
            where ``path[0]`` is query frames and ``path[1]`` is reference frames.
        """
        query_features = np.asarray(query_features)
        _validate_query_features_shape(query_features, self.n_features)
        self.reset()
        for i in range(query_features.shape[1]):
            self.feed(query_features[:, i])
        self.flush()
        return self.path
