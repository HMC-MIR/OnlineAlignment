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
from ..algs.soa import soa_scores_fixed, soa_update_fixed, soa_update_flexible
from ..utils import _validate_dtw_steps_weights, _validate_query_frame


class SOA(OnlineAlignment):
    """Simple Online Alignment.

    For every query frame, SOA extends the accumulated cost matrix by one row
    over the whole reference and reports the reference frame with the lowest
    accumulated cost. It never backtracks, so each estimate is final as soon as
    it is made. SOA uses the steps (1,1), (1,2), (2,1), so only the last three
    rows of the matrix are kept.

    By default the query starts at the first reference frame. With
    ``flexible_start=True`` it may start anywhere: each cell also records the
    reference frame where its path began, and paths are compared by accumulated
    cost divided by path length.
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
        """Initialize SOA.

        Args:
            reference_features: Reference audio features.
                Shape (n_features, n_frames)
            steps: DTW step pattern as (query_increment, reference_increment)
                rows. Must be ``[[1, 1], [1, 2], [2, 1]]``, the only pattern SOA
                supports; the argument is kept so the pattern stays explicit.
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
        if flexible_start and not normalize:
            raise ValueError(
                "flexible_start requires normalize=True: paths that start at different "
                "reference frames have different lengths, so raw costs are not comparable"
            )

        steps = np.asarray(steps)
        weights = np.asarray(weights)
        _validate_dtw_steps_weights(steps, weights)
        if steps.shape != SOA_STEPS.shape or not np.array_equal(steps, SOA_STEPS):
            raise ValueError(
                f"SOA supports only the steps {SOA_STEPS.tolist()}, in that order; "
                f"got {steps.tolist()}"
            )
        self.steps = steps
        self.weights = weights
        self.normalize = normalize
        self.monotonic = monotonic
        self.flexible_start = flexible_start

        # kernel inputs: weights of (1,1), (1,2), (2,1)
        self._w = tuple(np.float32(w) for w in weights)

        # local costs against the fixed reference
        self._costs = self.cost_metric.bind_reference(self.reference_features)

        # ring buffer of the last three accumulated cost rows, and normalized scores
        self._D = np.empty((3, self.reference_length), dtype=np.float32)
        self._scores = np.empty(self.reference_length, dtype=np.float64)

        # flexible start: ring buffer of the reference frame where each cell's path began
        self._S = np.empty(self._D.shape, dtype=np.int32) if flexible_start else None

        self.reset()

    def reset(self) -> None:
        """Clear all alignment state so a new query can be aligned."""
        self._D.fill(np.inf)
        self._D[0, 0] = 0.0
        if self._S is not None:
            self._S.fill(-1)
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

        # fixed start: the path starts at (0, 0)
        if i == 0 and self._S is None:
            self._path.append([0, 0])
            return 0

        # flexible start: every reference frame is a possible start (path length 0,
        # so the first estimate is the frame with the lowest local cost)
        if i == 0:
            costs = self._costs(query_frame)
            self._D[0] = costs
            self._S[0] = np.arange(self.reference_length)
            best_j = int(np.argmin(costs))
            self._path.append([0, best_j])
            return best_j

        if self.finished:
            return self.position

        costs = self._costs(query_frame)
        cur, r1, r2 = i % 3, (i - 1) % 3, (i - 2) % 3
        if self._S is not None:
            soa_update_flexible(i, costs, self._D, self._S, cur, r1, r2, *self._w, self._scores)
            best_j = int(np.argmin(self._scores))
        else:
            soa_update_fixed(costs, self._D, cur, r1, r2, *self._w)
            if self.normalize:
                soa_scores_fixed(i, self._D[cur], self._scores)
                best_j = int(np.argmin(self._scores))
            else:
                best_j = int(np.argmin(self._D[cur]))
        return self._append(i, best_j)

    def _append(self, i: int, best_j: int) -> int:
        """Apply the monotonic constraint and record the estimate for frame ``i``."""
        if self.normalize and self.monotonic:
            best_j = max(best_j, self.position)
        self._path.append([i, int(best_j)])
        return int(best_j)
