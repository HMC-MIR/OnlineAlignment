"""Tests for offline OLTW alignment: consistency of optimized vs reference implementation."""

# library imports
import numpy as np
import pytest
from librosa.sequence import dtw

# custom imports
from core.alignment.offline.oltw import OfflineOLTW, BOTH, ROW, COLUMN
from core.constants import OLTW_STEPS, OLTW_WEIGHTS
from core.cost import normalize_by_path_length
from core.alignment.utils import _validate_query_features_shape


def _reference_get_min_cost_indices(t, j, c, D_normalized):
    """Original loop-based _get_min_cost_indices for consistency testing."""
    if c is not None:
        row_start = max(0, j - c + 1)
        col_start = max(0, t - c + 1)
        cur_row = D_normalized[t, row_start : j + 1]
        cur_col = D_normalized[col_start : t + 1, j]
    else:
        row_start = 0
        col_start = 0
        cur_row = D_normalized[t, : j + 1]
        cur_col = D_normalized[: t + 1, j]

    min_cost = np.inf
    min_cost_location = ROW
    min_cost_idx = -1

    for idx, cost in enumerate(cur_row):
        if cost < min_cost:
            min_cost = cost
            min_cost_idx = idx
            min_cost_location = ROW
    for idx, cost in enumerate(cur_col):
        if cost < min_cost:
            min_cost = cost
            min_cost_idx = idx
            min_cost_location = COLUMN

    if min_cost_location == ROW:
        return t, row_start + min_cost_idx
    return col_start + min_cost_idx, j


class OfflineOLTWReference(OfflineOLTW):
    """OfflineOLTW with original loop-based _get_min_cost_indices and list-based path."""

    def _get_min_cost_indices(self, D_normalized):
        return _reference_get_min_cost_indices(
            self.t, self.j, self.c, D_normalized
        )

    def align(self, query_features: np.ndarray):
        _validate_query_features_shape(query_features)
        self.t, self.j = 0, 0
        self.cur_run_count = 0
        self.prev = None
        path = [[0], [0]]

        C = self.cost_metric.mat2mat(self.reference_features, query_features)
        D = dtw(
            backtrack=False,
            C=C,
            step_sizes_sigma=self.DTW_steps,
            weights_mul=self.DTW_weights,
        )
        D_normalized = normalize_by_path_length(D)
        query_length = query_features.shape[1]
        ref_length = self.reference_length

        while self.t < ref_length - 1 and self.j < query_length - 1:
            inc = self.get_inc(D_normalized)
            row_step, col_step = self.window_steps[inc]
            self.t += row_step
            self.t = min(self.t, ref_length - 1)
            self.j += col_step
            self.j = min(self.j, query_length - 1)
            if inc == self.prev:
                self.cur_run_count += 1
            else:
                self.cur_run_count = 1
            if inc != BOTH:
                self.prev = inc
            path[0].append(self.j)
            path[1].append(self.t)

        self.path = np.array(path)
        return self.path


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sequence_pair_a(rng):
    """First test sequence: ref (n_feat, 50), query (n_feat, 40)."""
    n_feat = 12
    ref = rng.standard_normal((n_feat, 50)).astype(np.float32)
    query = rng.standard_normal((n_feat, 40)).astype(np.float32)
    return ref, query


@pytest.fixture
def sequence_pair_b(rng):
    """Second test sequence: ref (n_feat, 80), query (n_feat, 60)."""
    n_feat = 12
    ref = rng.standard_normal((n_feat, 80)).astype(np.float32)
    query = rng.standard_normal((n_feat, 60)).astype(np.float32)
    return ref, query


def test_oltw_optimized_matches_reference_sequence_a(sequence_pair_a):
    """Optimized and reference implementations produce identical paths on first sequence."""
    ref, query = sequence_pair_a
    alg_ref = OfflineOLTWReference(ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS)
    alg_opt = OfflineOLTW(ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS)
    path_ref = alg_ref.align(query)
    path_opt = alg_opt.align(query)
    np.testing.assert_array_equal(path_opt, path_ref, err_msg="Path mismatch on sequence pair A")


def test_oltw_optimized_matches_reference_sequence_b(sequence_pair_b):
    """Optimized and reference implementations produce identical paths on second sequence."""
    ref, query = sequence_pair_b
    alg_ref = OfflineOLTWReference(ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS)
    alg_opt = OfflineOLTW(ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS)
    path_ref = alg_ref.align(query)
    path_opt = alg_opt.align(query)
    np.testing.assert_array_equal(path_opt, path_ref, err_msg="Path mismatch on sequence pair B")


def test_oltw_optimized_matches_reference_with_band_c(sequence_pair_a):
    """Consistency with band c set (e.g. c=5)."""
    ref, query = sequence_pair_a
    c = 5
    alg_ref = OfflineOLTWReference(
        ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS, c=c
    )
    alg_opt = OfflineOLTW(
        ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS, c=c
    )
    path_ref = alg_ref.align(query)
    path_opt = alg_opt.align(query)
    np.testing.assert_array_equal(path_opt, path_ref, err_msg="Path mismatch with band c=5")


def test_oltw_parallel_cost_matches_default_cosine(sequence_pair_a):
    """With use_parallel_cost=True (cosine), path matches default cost matrix."""
    ref, query = sequence_pair_a
    alg_default = OfflineOLTW(
        ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS
    )
    alg_parallel = OfflineOLTW(
        ref,
        DTW_steps=OLTW_STEPS,
        DTW_weights=OLTW_WEIGHTS,
        window_steps=OLTW_STEPS,
        use_parallel_cost=True,
    )
    path_default = alg_default.align(query)
    path_parallel = alg_parallel.align(query)
    np.testing.assert_array_equal(
        path_parallel, path_default, err_msg="Parallel cost path should match default (cosine)"
    )
