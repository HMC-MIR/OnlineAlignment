"""Fully offline version of Online Time Warping (OLTW) algorithm per Dixon et al."""

# standard imports
from typing import Callable

# library imports
import numpy as np
from librosa.sequence import dtw

# core imports
from core.constants import OLTW_STEPS, OLTW_WEIGHTS
from core.cost import CostMetric, normalize_by_path_length

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
        steps: np.ndarray = OLTW_STEPS,
        weights: np.ndarray = OLTW_WEIGHTS,
        cost_metric: str | Callable | CostMetric = "cosine",
        max_run_count: int = 3,
    ):
        """Initialize OfflineOLTW algorithm.

        Args:
            reference_features: Features for the reference audio.
                Shape (n_features, n_frames)
            steps: DTW steps. Shape (n_steps, 2)
            weights: DTW weights. Shape (n_steps, 1)
            cost_metric: Cost metric to use for computing distances.
                Can be a string name, callable function, or CostMetric instance.
            max_run_count: Maximum run count. Defaults to 3.
        """
        super().__init__(reference_features, cost_metric)

        # initialize query and reference locations
        self.t, self.j = 1  # t is query, j is reference

        # steps and weights
        _validate_dtw_steps_weights(steps, weights)
        self.steps = steps
        self.weights = weights

        # initialize alignment parameters
        self.max_run_count = max_run_count
        self.cur_run_count = 0
        self.prev = None  # previous step taken

        # initialize path
        self.path = [[0], [0]]  # TODO: provide optimizations

    def get_inc(self, D_normalized: np.ndarray):
        """Check which direction to increment.

        Args:
            t (int): query position
            j (int): reference position
        """
        # handle maximum run count
        if self.cur_run_count > self.max_run_count:
            if self.prev == ROW:
                return COLUMN
            else:
                return ROW

        # calculate min cost and select which direction to increment
        x, y = self._get_min_cost_indices(D_normalized)

        if x < self.t:
            return ROW
        if y < self.j:
            return COLUMN
        else:
            return BOTH

    def _get_min_cost_indices(self, D_noramlized: np.ndarray):
        """Calculate the min cost index in current row and column.

        Args:
            D_noramlized (np.ndarray): Normalized accumulated cost matrix.
        """

        # access current row and columns
        cur_col = D_noramlized[self.t, :]
        cur_row = D_noramlized[:, self.j]

        # initialize min
        min_cost = np.inf
        min_cost_location = ROW  # index if we're at row or column
        min_cost_idx = -1  # index where we are on the row or column

        # loop through to find minimum indices
        for idx, cost in enumerate(cur_row):  # row first
            if cost < min_cost:
                min_cost = cost
                min_cost_idx = idx

        for idx, cost in enumerate(cur_col):  # then column
            if cost < min_cost:
                min_cost = cost
                min_cost_idx = idx
                min_cost_location = ROW

        if min_cost_location == ROW:
            return min_cost_idx, self.j
        else:
            return self.t, min_cost_idx

    def align(self, query_features: np.ndarray):
        """Align query features to reference features.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Alignment path from OLTW. Shape (query_length, )
        """
        # validate input query features
        _validate_query_features_shape(query_features)

        # compute full cost matrix
        C = self.cost_metric.mat2mat(self.reference_features, query_features)

        # compute accumulated cost matrix
        D = dtw(
            backtrack=False,  # no backtrack since we only care about the accumulated cost matrix
            C=C,
            step_sizes_sigma=self.steps,
            weights_mul=self.weights,
        )

        # normalize by path length
        D_normalized = normalize_by_path_length(D)

        # main recursion loop
        while self.t < query_features.shape[1]:
            inc = self.get_inc(D_normalized)  # get increment direction

            if inc != COLUMN:
                self.t += 1
            if inc != ROW:
                self.j += 1
            if inc == self.prev:
                self.cur_run_count += 1
            else:
                self.cur_run_count = 1
            if inc != BOTH:
                self.prev = inc

            self.path[0].append(self.t)
            self.path[1].append(self.j)

        return self.path
