"""Offline Online Time Warping (OLTW): the online algorithm run on a complete query."""

# standard imports
from typing import Callable, Optional, Union

# library imports
import numpy as np

# core imports
from ...constants import OLTW_BAND_WIDTH, OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS
from ...cost import CostMetric

# custom imports
from .base import OfflineAlignment
from ..online.oltw import OLTW


class OfflineOLTW(OfflineAlignment):
    """Online Time Warping on a complete query.

    Runs :class:`~online_alignment.alignment.online.OLTW` frame by frame over
    the whole query, so the path is identical to the online result. With
    ``c=None`` this is OLTW over the full DTW matrix.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        steps: np.ndarray = OLTW_STEPS,
        weights: np.ndarray = OLTW_WEIGHTS,
        window_steps: np.ndarray = OLTW_WINDOW_STEPS,
        cost_metric: Union[str, Callable, CostMetric] = "cosine",
        max_run_count: int = 3,
        c: Optional[int] = OLTW_BAND_WIDTH,
    ):
        """Initialize OfflineOLTW.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            steps: DTW step pattern for the accumulated cost. Shape (n_steps, 2)
                where each row is (reference_increment, query_increment).
            weights: Weight for each DTW step. Shape (n_steps,)
            window_steps: Path transitions. Shape (3, 2), one reference-only,
                one query-only and one combined step, in any order.
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            max_run_count: Maximum consecutive ROW or COLUMN transitions.
            c: Band width in frames, or None for an unbounded band.
        """
        super().__init__(reference_features, cost_metric)
        self._online = OLTW(
            reference_features,
            steps=steps,
            weights=weights,
            window_steps=window_steps,
            cost_metric=self.cost_metric,
            max_run_count=max_run_count,
            c=c,
        )
        self.steps = self._online.steps
        self.weights = self._online.weights
        self.window_steps = self._online.window_steps
        self.max_run_count = max_run_count
        self.c = self._online.c

        # path produced by the most recent align() call
        self.path: Optional[np.ndarray] = None

    def align(self, query_features: np.ndarray) -> np.ndarray:
        """Align query features to reference features using OLTW.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Warping path as integer frame indices. Shape (2, n_path_points),
            where ``path[0]`` is query frames and ``path[1]`` is reference
            frames. Multiply by ``hop_length / sample_rate`` for seconds.
        """
        self.path = self._online.align(query_features)
        return self.path


def run_offline_oltw(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    steps: np.ndarray = OLTW_STEPS,
    weights: np.ndarray = OLTW_WEIGHTS,
    window_steps: np.ndarray = OLTW_WINDOW_STEPS,
    cost_metric: Union[str, Callable, CostMetric] = "cosine",
    max_run_count: int = 3,
    c: Optional[int] = OLTW_BAND_WIDTH,
) -> np.ndarray:
    """Run offline OLTW alignment in a single call.

    Args:
        reference_features: Reference features. Shape (n_features, n_frames)
        query_features: Query features. Shape (n_features, n_frames)
        steps: DTW step pattern. Shape (n_steps, 2)
        weights: DTW step weights. Shape (n_steps,)
        window_steps: Path transitions. Shape (3, 2)
        cost_metric: Cost metric to use for computing distances.
            Can be a string name, callable function, or CostMetric instance.
        max_run_count: Maximum consecutive ROW or COLUMN transitions.
        c: Band width in frames, or None for an unbounded band.

    Returns:
        Warping path as integer frame indices. Shape (2, n_path_points).
    """
    oltw = OfflineOLTW(
        reference_features,
        steps=steps,
        weights=weights,
        window_steps=window_steps,
        cost_metric=cost_metric,
        max_run_count=max_run_count,
        c=c,
    )
    return oltw.align(query_features)
