"""Offline Simple Online Alignment (SOA): the online algorithm run on a complete query."""

# standard imports
from typing import Callable, Optional, Union

# library imports
import numpy as np

# core imports
from ...constants import SOA_STEPS, SOA_WEIGHTS
from ...cost import CostMetric

# custom imports
from .base import OfflineAlignment
from ..online.soa import SOA


class OfflineSOA(OfflineAlignment):
    """Simple Online Alignment on a complete query.

    Runs :class:`~online_alignment.alignment.online.SOA` frame by frame over the
    whole query, so the path is identical to the online result.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        steps: np.ndarray = SOA_STEPS,
        weights: np.ndarray = SOA_WEIGHTS,
        cost_metric: Union[str, Callable, CostMetric] = "cosine",
        normalize: bool = True,
        monotonic: bool = False,
        flexible_start: bool = False,
    ):
        """Initialize OfflineSOA.

        Args:
            reference_features: Reference audio features.
                Shape (n_features, n_frames)
            steps: DTW step pattern as (query_increment, reference_increment)
                rows. Must be ``[[1, 1], [1, 2], [2, 1]]``, the only pattern SOA
                supports.
            weights: Weight for each of the three steps. Shape (3,)
            cost_metric: Distance metric. Can be a string (``"cosine"``,
                ``"euclidean"``, …), a callable, or a :class:`CostMetric`
                instance.
            normalize: If ``True`` (default) use path-length-normalized cost
                when selecting the best reference frame per query frame. If
                ``False``, use raw accumulated cost.
            monotonic: If ``True``, never move backwards in the reference.
                Only applied when *normalize* is ``True``.
            flexible_start: If ``True``, the query may start at any reference
                frame instead of the first. Requires *normalize*.
        """
        super().__init__(reference_features, cost_metric)
        self._online = SOA(
            reference_features,
            steps=steps,
            weights=weights,
            cost_metric=self.cost_metric,
            normalize=normalize,
            monotonic=monotonic,
            flexible_start=flexible_start,
        )
        self.steps = self._online.steps
        self.weights = self._online.weights
        self.normalize = normalize
        self.monotonic = monotonic
        self.flexible_start = flexible_start

        # path produced by the most recent align() call
        self.path: Optional[np.ndarray] = None

    def align(self, query_features: np.ndarray) -> np.ndarray:
        """Align query features to reference features using SOA.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Warping path as integer frame indices. Shape (2, n_path_points),
            where ``path[0]`` is query frames and ``path[1]`` is reference
            frames. Multiply by ``hop_length / sample_rate`` for seconds.
        """
        self.path = self._online.align(query_features)
        return self.path


def run_offline_soa(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    steps: np.ndarray = SOA_STEPS,
    weights: np.ndarray = SOA_WEIGHTS,
    cost_metric: Union[str, Callable, CostMetric] = "cosine",
    normalize: bool = True,
    monotonic: bool = False,
    flexible_start: bool = False,
) -> np.ndarray:
    """Run offline SOA alignment in a single call.

    Args:
        reference_features: Reference features. Shape (n_features, n_frames)
        query_features: Query features. Shape (n_features, n_frames)
        steps: DTW step pattern; must be ``[[1, 1], [1, 2], [2, 1]]``.
        weights: Weights of the three steps. Shape (3,).
        cost_metric: Distance metric (string name, callable, or
            :class:`CostMetric` instance).
        normalize: Use path-length-normalized cost when tracking the best
            reference frame. Set to ``False`` for raw-cost behaviour.
        monotonic: Enforce monotonic reference progression.
            Only applied when *normalize* is ``True``.
        flexible_start: Let the query start at any reference frame instead of
            the first. Requires *normalize*.

    Returns:
        Warping path as integer frame indices. Shape (2, n_path_points).
    """
    soa = OfflineSOA(
        reference_features,
        steps=steps,
        weights=weights,
        cost_metric=cost_metric,
        normalize=normalize,
        monotonic=monotonic,
        flexible_start=flexible_start,
    )
    return soa.align(query_features)
