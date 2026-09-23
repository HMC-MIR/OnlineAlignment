"""Online Time Warping (OLTW) algorithm per Dixon (2005)."""

# standard imports
from typing import Callable, List, Optional, Union

# library imports
import numpy as np

# core imports
from ...constants import OLTW_BAND_WIDTH, OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS
from ...cost import CostMetric
from ...cost.cosine import cosine_mat2mat_prenormalized, cosine_normalize_columns

# local imports
from .base import OnlineAlignment
from ..algs import BOTH, oltw_fill_column, oltw_fill_row, oltw_get_inc
from ..utils import (
    _arrange_oltw_steps,
    _validate_dtw_steps_weights,
    _validate_query_features_shape,
    _validate_query_frame,
    _validate_window_steps,
)

# initial query capacity when the band is unbounded (c=None)
_INITIAL_QUERY_CAPACITY = 1024


class OLTW(OnlineAlignment):
    """Online Time Warping.

    OLTW follows a path through the accumulated cost matrix, computing only the
    cells within ``c`` frames of the current position. At each step it compares
    the path-length-normalized cost along the current reference row and query
    column, and advances the reference (ROW), the query (COLUMN), or both.

    With a finite ``c``, the cost matrix lives in a ring buffer of about
    ``c x c`` cells, so memory does not grow with the query length. With
    ``c=None`` the band is unbounded: every cell up to the current position is
    computed, which reproduces OLTW on the full DTW matrix, but memory grows as
    ``reference_length x query_length``.
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
        """Initialize OLTW.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            steps: DTW step pattern for the accumulated cost. Shape (n_steps, 2)
                where each row is (reference_increment, query_increment).
            weights: Weight for each DTW step. Shape (n_steps,)
            window_steps: Path transitions. Shape (3, 2), one reference-only step
                (ROW), one query-only step (COLUMN) and one combined step (BOTH),
                in any order.
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            max_run_count: Maximum consecutive ROW or COLUMN transitions.
            c: Band width in frames, or None for an unbounded band.
        """
        super().__init__(reference_features, cost_metric)

        steps = np.asarray(steps)
        weights = np.asarray(weights)
        window_steps = np.asarray(window_steps)
        _validate_dtw_steps_weights(steps, weights)
        _validate_window_steps(window_steps)
        if c is not None and (int(c) != c or c < 1):
            raise ValueError(f"c must be a positive integer or None, got {c}")
        if max_run_count < 1:
            raise ValueError(f"max_run_count must be at least 1, got {max_run_count}")

        self.steps = steps
        self.weights = weights
        self.window_steps = _arrange_oltw_steps(window_steps)
        self.max_run_count = max_run_count
        self.c = None if c is None else int(c)

        # kernel inputs
        self._steps = steps.astype(np.int64)
        self._weights = weights.astype(np.float64)
        self._window_steps = self.window_steps.astype(np.int64)
        self._c_int = -1 if self.c is None else self.c
        self._is_cosine = getattr(self.cost_metric, "name", None) == "cosine"
        # cosine costs use columns normalized once up front (same result as normalizing per block)
        self._reference = np.ascontiguousarray(self.reference_features, dtype=np.float64)
        if self._is_cosine:
            self._reference = cosine_normalize_columns(self._reference)

        # ring buffer sizes: the band, plus room for DTW predecessors and one transition
        max_dt = int(max(self._steps[:, 0].max(), self._window_steps[:, 0].max()))
        max_dj = int(max(self._steps[:, 1].max(), self._window_steps[:, 1].max()))
        if self.c is None:
            self._n_rows = self.reference_length
            n_cols = _INITIAL_QUERY_CAPACITY
        else:
            self._n_rows = min(self.reference_length, self.c + 2 * max_dt + 1)
            n_cols = self.c + 2 * max_dj + 1
        self._D = np.empty((self._n_rows, n_cols), dtype=np.float64)
        self._query = np.empty((self.n_features, n_cols), dtype=np.float64)

        self.reset()

    # ------------------------------------------------------------------
    # state
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all alignment state so a new query can be aligned."""
        self._D.fill(np.inf)
        self.t, self.j = 0, 0  # t is reference (row), j is query (column)
        self._last_frame = -1  # index of the newest query frame fed
        self._run_count = 0
        self._prev = -1  # last non-BOTH transition
        self._flushed = False
        self._path: List[List[int]] = []

    @property
    def position(self) -> int:
        """Current estimate of the reference frame index."""
        return self.t

    @property
    def path(self) -> np.ndarray:
        """Warping path so far. Shape (2, n_path_points), rows [query, reference]."""
        return np.array(self._path, dtype=np.int64).reshape(-1, 2).T

    @property
    def finished(self) -> bool:
        """True once the path has reached the last reference frame."""
        return self.t >= self.reference_length - 1

    def _reserve(self, n_frames: int) -> None:
        """Grow the unbounded-band buffers to hold at least ``n_frames`` query frames."""
        n_cols = self._D.shape[1]
        if self.c is not None or n_frames <= n_cols:
            return
        new_cols = max(n_frames, 2 * n_cols)
        D = np.full((self._n_rows, new_cols), np.inf, dtype=np.float64)
        D[:, :n_cols] = self._D
        query = np.empty((self.n_features, new_cols), dtype=np.float64)
        query[:, :n_cols] = self._query
        self._D, self._query = D, query

    # ------------------------------------------------------------------
    # online interface
    # ------------------------------------------------------------------

    def feed(self, query_frame: np.ndarray) -> int:
        """Feed the next query frame and advance the path as far as it can go.

        The path stops at the newest frame: ROW transitions from it are taken
        when the next frame arrives (or on ``flush()``), so the path matches
        the offline result exactly.

        Args:
            query_frame: Single frame of query features.
                Shape (n_features,) or (n_features, 1)

        Returns:
            Current estimate of the reference frame index.
        """
        if self._flushed:
            raise RuntimeError("Cannot feed after flush(); call reset() to start a new query")
        query_frame = _validate_query_frame(query_frame, self.n_features)

        self._last_frame += 1
        L = self._last_frame
        self._reserve(L + 1)
        if self._is_cosine:
            query_frame = cosine_normalize_columns(
                np.asarray(query_frame, dtype=np.float64).reshape(-1, 1)
            )[:, 0]
        self._query[:, L % self._query.shape[1]] = query_frame

        # the path always starts at (0, 0)
        if L == 0:
            self._D[0, 0] = self._costs(0, 1, 0, 1)[0, 0]
            self._path.append([0, 0])
            return self.t

        self._advance(final=False)
        return self.t

    def flush(self) -> int:
        """Signal the end of the query and finish the path.

        Returns:
            Final estimate of the reference frame index.
        """
        if not self._flushed and self._last_frame >= 0:
            self._advance(final=True)
        self._flushed = True
        return self.t

    def align(self, query_features: np.ndarray) -> np.ndarray:
        """Simulate the online process on a complete query.

        Args:
            query_features: Complete query feature matrix.
                Shape (n_features, n_frames)

        Returns:
            Warping path of integer frame indices. Shape (2, n_path_points),
            where ``path[0]`` is query frames and ``path[1]`` is reference frames.
        """
        query_features = np.asarray(query_features)
        _validate_query_features_shape(query_features, self.n_features)
        self._reserve(query_features.shape[1])
        return super().align(query_features)

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _costs(self, t_start: int, t_stop: int, j_start: int, j_stop: int) -> np.ndarray:
        """Local costs for reference frames [t_start, t_stop) and query frames [j_start, j_stop)."""
        n_cols = self._query.shape[1]
        if j_stop - j_start == 1 or self.c is None:
            query = self._query[:, j_start % n_cols:(j_stop - 1) % n_cols + 1]
        else:
            query = self._query[:, np.arange(j_start, j_stop) % n_cols]
        reference = self._reference[:, t_start:t_stop]

        # cosine is computed cell by cell so any block matches the full matrix exactly
        if self._is_cosine:
            return cosine_mat2mat_prenormalized(reference, query)
        return np.asarray(self.cost_metric.mat2mat(reference, query), dtype=np.float64)

    def _advance(self, final: bool) -> None:
        """Take path transitions until the path needs a query frame not yet fed.

        Args:
            final: True once the query has ended. Transitions past the last
                frame are then clamped to it, and the path stops there.
        """
        L = self._last_frame
        n_rows, n_cols = self._D.shape
        c = self.c
        while self.t < self.reference_length - 1 and self.j < L:
            inc = oltw_get_inc(
                self._D, self.t, self.j, self._c_int, self._run_count, self._prev,
                self.max_run_count,
            )
            dt, dj = self._window_steps[inc]
            if not final and self.j + dj > L:
                break  # wait for more query frames

            t_new = min(self.t + dt, self.reference_length - 1)
            j_new = min(self.j + dj, L)

            # new reference rows over the band of query frames
            j_lo = 0 if c is None else max(0, self.j - c + 1)
            for t in range(self.t + 1, t_new + 1):
                if n_rows < self.reference_length:
                    self._D[t % n_rows].fill(np.inf)
                costs = self._costs(t, t + 1, j_lo, self.j + 1)[0]
                oltw_fill_row(self._D, t, j_lo, costs, self._steps, self._weights)

            # new query columns over the band of reference frames
            t_lo = 0 if c is None else max(0, t_new - c + 1)
            for j in range(self.j + 1, j_new + 1):
                if c is not None:
                    self._D[:, j % n_cols].fill(np.inf)
                costs = self._costs(t_lo, t_new + 1, j, j + 1)[:, 0]
                oltw_fill_column(self._D, j, t_lo, costs, self._steps, self._weights)

            # update run count
            if inc == self._prev:
                self._run_count += 1
            else:
                self._run_count = 1
            if inc != BOTH:
                self._prev = inc

            self.t, self.j = t_new, j_new
            self._path.append([self.j, self.t])
