"""Fully offline version of Online Time Warping (OLTW) algorithm per Dixon et al."""

# standard imports
from typing import Callable

# library imports
import numpy as np
from librosa.sequence import dtw

# core imports
from ...constants import OLTW_STEPS, OLTW_WEIGHTS
from ...cost import CostMetric, normalize_by_path_length

# custom imports
from .base import OfflineAlignment
from ..utils import _validate_dtw_steps_weights, _validate_query_features_shape

# set constants for incrementing
BOTH = 0
ROW = 1
COLUMN = 2


class OfflineOLTW(OfflineAlignment):
    """Fully offline version of OLTW Algorithm without banding."""

    def __init__(
        self,
        reference_features: np.ndarray,
        DTW_steps: np.ndarray = OLTW_STEPS,
        DTW_weights: np.ndarray = OLTW_WEIGHTS,
        window_steps: np.ndarray = OLTW_STEPS,
        cost_metric: str | Callable | CostMetric = "cosine",
        max_run_count: int = 3,
        c: int = None,
    ):
        """Initialize OfflineOLTW algorithm.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            DTW_steps: cost matrix windowing steps. Shape (n_steps, 2)
            DTW_weights: cost matrix windowing weights. Shape (n_steps, 1)
            window_steps: OLTW transition steps. Shape (n_steps, 2)
                Index 0 corresponds to reference (row), index 1 corresponds to query (column), index 2 corresponds to both (row and column)
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            max_run_count: Maximum run count. Defaults to 3.
            c: band size for comparing costs. #TODO: optimize based on banding
        """
        super().__init__(reference_features, cost_metric)

        # initialize query and reference locations
        self.t, self.j = 0, 0  # t is reference (row), j is query (column)
        self.c = c

        # validate steps and weights
        _validate_dtw_steps_weights(DTW_steps, DTW_weights)
        self.DTW_steps = DTW_steps
        self.DTW_weights = DTW_weights
        
        # validate and store transition steps
        # window_steps should have shape (3, 2) for [BOTH, ROW, COLUMN] transitions
        # Each row specifies (row_increment, column_increment)
        if window_steps.shape[0] != 3 or window_steps.shape[1] != 2:
            raise ValueError(f"window_steps must have shape (3, 2), got {window_steps.shape}")
        self.window_steps = window_steps

        # initialize alignment parameters
        self.max_run_count = max_run_count
        self.cur_run_count = 0
        self.prev = None  # previous step taken

        # initialize path (0-indexed)
        self.path = [[0], [0]]  # TODO: provide optimizations

    def get_inc(self, D_normalized: np.ndarray):
        """Check which direction to increment based on normalized costs."""
        # handle initial period
        if self.c is not None and self.t < self.c:
            return BOTH

        # handle maximum run count
        if self.cur_run_count >= self.max_run_count:
            if self.prev == ROW:
                return COLUMN
            else:
                return ROW

        # calculate min cost and select which direction to increment
        x, y = self._get_min_cost_indices(D_normalized)

        if x < self.t:
            return COLUMN
        if y < self.j:
            return ROW
        else:
            return BOTH

    def _get_min_cost_indices(self, D_normalized: np.ndarray):
        """Calculate the min cost index in current row and column.

        Args:
            D_normalized (np.ndarray): Normalized accumulated cost matrix.
        """

        # Access the last c elements of the current row and column, or all history if c is None.
        # IMPORTANT: indices within `cur_row` / `cur_col` are *local* to the slice, so we track
        # the slice start offsets to convert the argmin back into global indices of D_normalized.
        if self.c is not None:
            row_start = max(0, self.j - self.c + 1)
            col_start = max(0, self.t - self.c + 1)
            cur_row = D_normalized[
                self.t, row_start : self.j + 1
            ]  # last c columns in row t
            cur_col = D_normalized[
                col_start : self.t + 1, self.j
            ]  # last c rows in column j
        else:
            row_start = 0
            col_start = 0
            cur_row = D_normalized[self.t, : self.j + 1]  # all previous columns in row t
            cur_col = D_normalized[: self.t + 1, self.j]  # all previous rows in column j

        # initialize min
        min_cost = np.inf
        min_cost_location = ROW  # index if we're at row or column
        min_cost_idx = -1  # index where we are on the row or column

        # loop through to find minimum indices
        for idx, cost in enumerate(cur_row):  # row first (reference row, query indices)
            if cost < min_cost:
                min_cost = cost
                min_cost_idx = idx
                min_cost_location = ROW

        for idx, cost in enumerate(cur_col):  # then column (query column, reference indices)
            if cost < min_cost:
                min_cost = cost
                min_cost_idx = idx
                min_cost_location = COLUMN

        if min_cost_location == ROW:
            # min_cost_idx is local to cur_row; convert to global query index
            return self.t, row_start + min_cost_idx  # (reference_idx, query_idx)
        else:
            # min_cost_idx is local to cur_col; convert to global reference index
            return col_start + min_cost_idx, self.j  # (reference_idx, query_idx)

    def align(self, query_features: np.ndarray):
        """Align query features to reference features.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Alignment path from OLTW. Shape (query_length, )
        """
        # validate input query features
        _validate_query_features_shape(query_features)

        # reset state for this alignment run
        self.t, self.j = 0, 0
        self.cur_run_count = 0
        self.prev = None
        self.path = [[0], [0]]

        # compute full cost matrix
        C = self.cost_metric.mat2mat(self.reference_features, query_features)

        # compute accumulated cost matrix up front
        D = dtw(
            backtrack=False,  # no backtrack since we only care about the accumulated cost matrix
            C=C,
            step_sizes_sigma=self.DTW_steps,
            weights_mul=self.DTW_weights,
        )

        # normalize by path length (manhattan distance)
        D_normalized = normalize_by_path_length(D)

        # dimensions for bounds checking (0-indexed)
        query_length = query_features.shape[1]
        ref_length = self.reference_length

        # main recursion loop
        # stop when we have reached the end of either query or reference
        while self.t < ref_length - 1 and self.j < query_length - 1:
            inc = self.get_inc(D_normalized)  # get increment direction

            # increment indices using transition steps while staying within bounds
            # window_steps[inc] gives [row_increment, column_increment]
            row_step, col_step = self.window_steps[inc]
            self.t += row_step  # increment reference (row) by transition step
            self.t = min(self.t, ref_length - 1)  # clamp to max
            self.j += col_step  # increment query (column) by transition step
            self.j = min(self.j, query_length - 1)  # clamp to max
            
            if inc == self.prev:
                self.cur_run_count += 1
            else:
                self.cur_run_count = 1
            if inc != BOTH:
                self.prev = inc

            self.path[0].append(self.j)  # query (column)
            self.path[1].append(self.t)  # reference (row)

        self.path = np.array(self.path)
        return self.path


def run_offline_oltw(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    DTW_steps: np.ndarray = OLTW_STEPS,
    DTW_weights: np.ndarray = OLTW_WEIGHTS,
    window_steps: np.ndarray = OLTW_STEPS,
    cost_metric: str | Callable | CostMetric = "cosine",
    max_run_count: int = 3,
    c: int = None,
):
    """Offline OLTW algorithm.

    Args:
        reference_features: Reference features. Shape (n_features, n_frames)
        query_features: Query features. Shape (n_features, n_frames)
        DTW_steps: DTW window steps. Shape (n_steps, 2)
        DTW_weights: DTW window weights. Shape (n_steps, 1)
        window_steps: OLTW transition steps. Shape (n_steps, 2)
        cost_metric: Cost metric to use for computing distances.
            Can be a string name, callable function, or CostMetric instance.
        max_run_count: Maximum run count. Defaults to 3.
        c: band size for comparing costs.
    """
    offline_oltw = OfflineOLTW(reference_features, DTW_steps, DTW_weights, window_steps, cost_metric, max_run_count, c)
    return offline_oltw.align(query_features)
