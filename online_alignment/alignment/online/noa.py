"""Naive Online Alignment algorithm."""

# standard imports
from typing import Callable, Union

# library imports
import numpy as np

# core imports
from ...constants import DEFAULT_DTW_STEPS, DEFAULT_DTW_WEIGHTS
from ...cost import CostMetric

# local imports
from .base import OnlineAlignment
# from ..algs import noa_row_update
from ..utils import _validate_dtw_steps_weights, _validate_prev_alignment_path
# TODO: add validate query features


class NOA(OnlineAlignment):
    """Naive Online Alignment Algorithm without banding."""

    def __init__(
        self,
        reference_features: np.ndarray,
        steps: np.ndarray = DEFAULT_DTW_STEPS,
        weights: np.ndarray = DEFAULT_DTW_WEIGHTS,
        cost_metric: Union[str, Callable, CostMetric] = "cosine",
        prev_alignment_path: np.ndarray = None,
    ):
        """Initialize NOA algorithm.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            steps: DTW steps. Shape (n_steps, 2)
            weights: DTW weights. Shape (n_steps, 1)
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            prev_alignment_path: Previous alignment path. Shape (n_steps, 2)
                If None, a new alignment path will be created.

        """
        super().__init__(reference_features, cost_metric)

        # steps and weights
        _validate_dtw_steps_weights(steps, weights)
        self.steps = steps
        self.weights = weights

        # setup alignment path
        self._setup_alignment_path(prev_alignment_path)

        # setup alignment cost matrix
        self._setup_alignment_cost_matrix()

    def _get_max_valid_query_length(self) -> int:
        """Get maximum valid query length."""
        # find maximum possible valid ratio of reference to query steps
        max_slope_ratio = np.max(np.abs(self.steps[:, 0] / self.steps[:, 1]))
        return int(self.reference_length / max_slope_ratio)

    def _setup_alignment_path(self, prev_alignment_path: np.ndarray) -> None:
        """Setup alignment path.

        Args:
            prev_alignment_path: Previous alignment path. Shape (n_steps, 2)
                If None, a new alignment path will be created.
        """
        # reserve space for the path
        self.path = np.zeros(self._get_max_valid_query_length(), dtype=int)

        # populate path if previous alignment path is provided
        if prev_alignment_path is not None:
            _validate_prev_alignment_path(prev_alignment_path, self.reference_length)
            self.query_idx: int = len(prev_alignment_path)
            self.ref_idx: int = prev_alignment_path[-1]
            self.path[: self.query_idx] = prev_alignment_path
        else:  # otherwise, start from the beginning
            self.query_idx: int = 0
            self.ref_idx: int = 0

    def _setup_alignment_cost_matrix(self) -> None:
        """Setup alignment cost matrix."""
        # accumulated cost matrix
        self.D = np.full(
            (self._get_max_valid_query_length(), self.reference_length), np.inf, dtype=np.float32
        )

    def feed(self, query_frame: np.ndarray):
        """Feed a single query frame into the alignment system.

        Args:
            query_frame: Single frame of query features. Shape (n_features, 1)
        """
