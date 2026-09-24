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
from ..algs.soa import (
    soa_min_default,
    soa_row_update,
    soa_row_update3,
    soa_row_update_flexible,
    soa_row_update_flexible3,
    soa_scores_fixed,
    soa_update_flexible_default,
)
from ..utils import _validate_dtw_steps_weights, _validate_query_frame


class SOA(OnlineAlignment):
    """Simple Online Alignment.

    For every query frame, SOA extends the accumulated cost matrix by one row
    over the whole reference and reports the reference frame with the lowest
    accumulated cost. It never backtracks, so each estimate is final as soon as
    it is made. Only the last ``max(row_steps) + 1`` rows of the matrix are kept.

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
        self.steps = steps
        self.weights = weights
        self.normalize = normalize
        self.monotonic = monotonic
        self.flexible_start = flexible_start

        # kernel inputs
        self._dn = steps[:, 0].astype(np.int64)
        self._dm = steps[:, 1].astype(np.int64)
        self._dw = weights.astype(np.float32)

        # local costs against the fixed reference
        self._costs = self.cost_metric.bind_reference(self.reference_features)

        # ring buffer of accumulated cost rows
        self._D = np.empty((int(self._dn.max()) + 1, self.reference_length), dtype=np.float32)

        # ring-buffer row of each step's predecessor, for the three-step kernels
        self._rows = np.empty(len(self._dn), dtype=np.int64)

        # the default steps and weights, in order, have vectorized kernels
        self._default_steps = (
            steps.shape == SOA_STEPS.shape
            and np.array_equal(steps, SOA_STEPS)
            and np.array_equal(weights, SOA_WEIGHTS)
        )
        self._scores = (
            np.empty(self.reference_length, dtype=np.float64) if self._default_steps else None
        )

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
        n_rows = self._D.shape[0]

        # default steps and weights: vectorized kernels, once frames i-1 and i-2 exist
        if self._default_steps and i >= 2:
            cur, r1, r2 = i % n_rows, (i - 1) % n_rows, (i - 2) % n_rows
            if self._S is not None:
                soa_update_flexible_default(
                    i, costs, self._D, self._S, cur, r1, r2, self._scores
                )
                best_j = int(np.argmin(self._scores))
            else:
                soa_min_default(costs, self._D, cur, r1, r2)
                if self.normalize:
                    soa_scores_fixed(i, self._D[cur], self._scores)
                    best_j = int(np.argmin(self._scores))
                else:
                    best_j = int(np.argmin(self._D[cur]))
            return self._append(i, best_j)

        # other three-step patterns: unrolled kernels that overwrite the whole row
        if len(self._dn) == 3:
            for k in range(3):
                prev_i = i - self._dn[k]
                self._rows[k] = prev_i % n_rows if prev_i >= 0 else -1
            if self._S is None:
                best_j = soa_row_update3(
                    i, costs, self._D, self._rows, self._dm, self._dw, self.normalize
                )
            else:
                best_j = soa_row_update_flexible3(
                    i, costs, self._D, self._S, self._rows, self._dm, self._dw
                )
            return self._append(i, best_j)

        row = i % n_rows
        self._D[row].fill(np.inf)
        if self._S is None:
            best_j = soa_row_update(
                i, costs, self._D, self._dn, self._dm, self._dw, self.normalize
            )
        else:
            self._S[row].fill(-1)
            best_j = soa_row_update_flexible(
                i, costs, self._D, self._S, self._dn, self._dm, self._dw
            )

        return self._append(i, best_j)

    def _append(self, i: int, best_j: int) -> int:
        """Apply the monotonic constraint and record the estimate for frame ``i``."""
        if self.normalize and self.monotonic:
            best_j = max(best_j, self.position)
        self._path.append([i, int(best_j)])
        return int(best_j)
