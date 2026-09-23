"""Simple Online Alignment (SOA) algorithm."""

# standard imports
from typing import Callable, List, Union

# library imports
import numpy as np

# core imports
from ...constants import SOA_STEPS, SOA_WEIGHTS
from ...cost import CostMetric

# local imports
from .base import OnlineAlignment
from ..algs import soa_row_update
from ..utils import _validate_dtw_steps_weights, _validate_query_frame


class SOA(OnlineAlignment):
    """Simple Online Alignment.

    For every query frame, SOA extends the accumulated cost matrix by one row
    over the whole reference and reports the reference frame with the lowest
    accumulated cost. It never backtracks, so each estimate is final as soon as
    it is made. Only the last ``max(row_steps) + 1`` rows of the matrix are kept.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        steps: np.ndarray = SOA_STEPS,
        weights: np.ndarray = SOA_WEIGHTS,
        cost_metric: Union[str, Callable, CostMetric] = "cosine",
        normalize: bool = True,
        monotonic: bool = False,
    ):
        """Initialize SOA.

        Args:
            reference_features: Reference audio features.
                Shape (n_features, n_frames)
            steps: DTW step pattern. Shape (n_steps, 2) where each row is
                (query_increment, reference_increment).
            weights: Weight for each step. Shape (n_steps,)
            cost_metric: Distance metric. Can be a string (``"cosine"``,
                ``"euclidean"``, …), a callable, or a :class:`CostMetric`
                instance.
            normalize: If ``True`` (default) use path-length-normalized cost
                when selecting the best reference frame per query frame. If
                ``False``, use raw accumulated cost.
            monotonic: If ``True``, never move backwards in the reference.
                Only applied when *normalize* is ``True``.
        """
        super().__init__(reference_features, cost_metric)

        steps = np.asarray(steps)
        weights = np.asarray(weights)
        _validate_dtw_steps_weights(steps, weights)
        self.steps = steps
        self.weights = weights
        self.normalize = normalize
        self.monotonic = monotonic

        # kernel inputs
        self._dn = steps[:, 0].astype(np.int64)
        self._dm = steps[:, 1].astype(np.int64)
        self._dw = weights.astype(np.float32)

        # ring buffer of accumulated cost rows
        self._D = np.empty((int(self._dn.max()) + 1, self.reference_length), dtype=np.float32)

        self.reset()

    def reset(self) -> None:
        """Clear all alignment state so a new query can be aligned."""
        self._D.fill(np.inf)
        self._D[0, 0] = 0.0
        self._n_frames = 0
        self._path: List[List[int]] = []

    @property
    def position(self) -> int:
        """Current estimate of the reference frame index."""
        return self._path[-1][1] if self._path else 0

    @property
    def path(self) -> np.ndarray:
        """Warping path so far. Shape (2, n_path_points), rows [query, reference]."""
        return np.array(self._path, dtype=np.int64).reshape(-1, 2).T

    @property
    def finished(self) -> bool:
        """True once the path has reached the last reference frame."""
        return self.position >= self.reference_length - 1

    def feed(self, query_frame: np.ndarray) -> int:
        """Feed the next query frame and advance the alignment.

        Args:
            query_frame: Single frame of query features.
                Shape (n_features,) or (n_features, 1)

        Returns:
            Estimated reference frame index for this query frame. Once the end
            of the reference is reached, further frames are ignored.
        """
        query_frame = _validate_query_frame(query_frame, self.n_features)
        i = self._n_frames
        self._n_frames += 1

        # the path always starts at (0, 0)
        if i == 0:
            self._path.append([0, 0])
            return 0
        if self.finished:
            return self.position

        costs = self.cost_metric.mat2vec(self.reference_features, query_frame)
        self._D[i % self._D.shape[0]].fill(np.inf)
        best_j = soa_row_update(i, costs, self._D, self._dn, self._dm, self._dw, self.normalize)

        if self.normalize and self.monotonic:
            best_j = max(best_j, self.position)

        self._path.append([i, int(best_j)])
        return int(best_j)
