"""Tests for OfflineNOA: consistency of refactored implementation vs. inlined original logic.

The original NOA logic is reproduced here as two reference functions
(_align_noa_ref / _align_noa_no_norm_ref) so the test has no dependency on the
top-level noa.py script in the repo root.  Both functions mirror the original
alignNOA / alignNOA_no_norm exactly, using the same Numba kernels, so any
numerical difference in the refactored OfflineNOA will be caught immediately.

Run from OnlineAlignment/:
    pytest tests/core/alignment/offline/test_noa.py -v
"""

# library imports
import numpy as np
import pytest
from numba import njit

# custom imports
from core.alignment.offline.noa import OfflineNOA, run_offline_noa
from core.constants import NOA_STEPS, NOA_WEIGHTS


# ---------------------------------------------------------------------------
# Reference Numba kernels (inlined from original noa.py)
# ---------------------------------------------------------------------------


@njit(cache=True)
def _update_row_norm_ref(i, costs, D, B, dn, dm, dw, ref_length):
    """Original update_alignment_row_numba_norm logic."""
    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step_cost_norm = np.inf
        best_step = -1

        for k, (di, dj, w) in enumerate(zip(dn, dm, dw)):
            prev_i, prev_j = i - di, j - dj
            if prev_i < 0 or prev_j < 0 or prev_j >= ref_length:
                continue
            cur_cost = D[prev_i, prev_j] + costs[j] * w
            norm_cost = cur_cost / (i + 1 + j + 1)
            if norm_cost < best_step_cost_norm:
                best_step_cost_norm = norm_cost
                best_step = k
                best_step_cost = cur_cost

        if best_step != -1:
            D[i, j] = best_step_cost
            B[i, j] = best_step
            if best_step_cost_norm < best_cost:
                best_cost = best_step_cost_norm
                best_j = j

    return best_j


@njit(cache=True)
def _update_row_ref(i, costs, D, B, dn, dm, dw, ref_length):
    """Original update_alignment_row_numba logic (no normalisation)."""
    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step = -1

        for k, (di, dj, w) in enumerate(zip(dn, dm, dw)):
            prev_i, prev_j = i - di, j - dj
            if prev_i < 0 or prev_j < 0 or prev_j >= ref_length:
                continue
            cur_cost = D[prev_i, prev_j] + costs[j] * w
            if cur_cost < best_step_cost:
                best_step_cost = cur_cost
                best_step = k

        if best_step != -1:
            D[i, j] = best_step_cost
            B[i, j] = best_step
            if best_step_cost < best_cost:
                best_cost = best_step_cost
                best_j = j

    return best_j


@njit(cache=True)
def _cosine_dist_vec2mat_ref(feature_row, reference_features):
    """Original compute_cosine_distance logic."""
    costs = np.empty(reference_features.shape[1], dtype=np.float32)
    for j in range(reference_features.shape[1]):
        ref_col = reference_features[:, j]
        costs[j] = 1.0 - np.sum(feature_row * ref_col)
    return costs


# ---------------------------------------------------------------------------
# Reference alignment functions (exact copies of original alignNOA logic)
# ---------------------------------------------------------------------------


def _align_noa_ref(F1, F2, steps=NOA_STEPS, weights=NOA_WEIGHTS, monotonic=False):
    """Original alignNOA, returning integer frame indices (path[0]=query, path[1]=ref)."""
    path = [[0, 0]]
    ref_length = F2.shape[1]
    dn, dm = steps[:, 0], steps[:, 1]

    max_query_length = 2 * ref_length
    D = np.full((max_query_length, ref_length), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    B = np.full((max_query_length, ref_length), -1, dtype=np.int32)

    for i in range(1, F1.shape[1]):
        if path[-1][1] >= ref_length - 1:
            break
        costs = _cosine_dist_vec2mat_ref(F1[:, i], F2)
        best_j = _update_row_norm_ref(i, costs, D, B, dn, dm, weights, ref_length)
        if monotonic:
            best_j = max(best_j, path[-1][1])
        path.append([i, best_j])

    return np.array(path, dtype=np.int32).T


def _align_noa_no_norm_ref(F1, F2, steps=NOA_STEPS, weights=NOA_WEIGHTS):
    """Original alignNOA_no_norm, returning integer frame indices."""
    path = [[0, 0]]
    ref_length = F2.shape[1]
    dn, dm = steps[:, 0], steps[:, 1]

    max_query_length = 2 * ref_length
    D = np.full((max_query_length, ref_length), np.inf, dtype=np.float32)
    D[0, 0] = 0.0
    B = np.full((max_query_length, ref_length), -1, dtype=np.int32)

    for i in range(1, F1.shape[1]):
        if path[-1][1] >= ref_length - 1:
            break
        costs = _cosine_dist_vec2mat_ref(F1[:, i], F2)
        best_j = _update_row_ref(i, costs, D, B, dn, dm, weights, ref_length)
        path.append([i, best_j])

    return np.array(path, dtype=np.int32).T


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


def _normalized(rng, n_feat, n_frames):
    """L2-normalised random feature matrix (required by cosine kernel)."""
    F = rng.random((n_feat, n_frames)).astype(np.float32)
    norms = np.linalg.norm(F, axis=0, keepdims=True)
    return F / np.clip(norms, 1e-9, None)


@pytest.fixture
def sequence_pair_a(rng):
    """First test sequence: ref (12, 80), query (12, 60)."""
    return _normalized(rng, 12, 80), _normalized(rng, 12, 60)


@pytest.fixture
def sequence_pair_b(rng):
    """Second test sequence: ref (12, 100), query (12, 70)."""
    return _normalized(rng, 12, 100), _normalized(rng, 12, 70)


# ---------------------------------------------------------------------------
# Tests: return contract
# ---------------------------------------------------------------------------


def test_return_shape_and_dtype(sequence_pair_a):
    """align() returns shape (2, N) int32."""
    ref, query = sequence_pair_a
    path = run_offline_noa(ref, query)
    assert path.ndim == 2
    assert path.shape[0] == 2
    assert path.dtype == np.int32


def test_class_stores_path(sequence_pair_a):
    """OfflineNOA stores the result in self.path after align()."""
    ref, query = sequence_pair_a
    noa = OfflineNOA(ref)
    path = noa.align(query)
    assert path is noa.path


def test_path_rows_nondecreasing_query_column(sequence_pair_a):
    """Query frame index (row 0) is strictly increasing by construction."""
    ref, query = sequence_pair_a
    path = run_offline_noa(ref, query)
    # query indices should be 0, 1, 2, ... (incrementing by 1 each frame)
    assert np.all(np.diff(path[0]) == 1)


# ---------------------------------------------------------------------------
# Tests: equivalence with original (normalised, cosine)
# ---------------------------------------------------------------------------


def test_normalized_cosine_matches_reference_sequence_a(sequence_pair_a):
    """normalize=True, cosine matches original alignNOA on sequence pair A."""
    ref, query = sequence_pair_a
    path_ref = _align_noa_ref(query, ref)
    path_new = run_offline_noa(ref, query, normalize=True, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=True, cosine, pair A)"
    )


def test_normalized_cosine_matches_reference_sequence_b(sequence_pair_b):
    """normalize=True, cosine matches original alignNOA on sequence pair B."""
    ref, query = sequence_pair_b
    path_ref = _align_noa_ref(query, ref)
    path_new = run_offline_noa(ref, query, normalize=True, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=True, cosine, pair B)"
    )


def test_normalized_monotonic_matches_reference(sequence_pair_a):
    """normalize=True, monotonic=True matches original alignNOA(monotonic=True)."""
    ref, query = sequence_pair_a
    path_ref = _align_noa_ref(query, ref, monotonic=True)
    path_new = run_offline_noa(ref, query, normalize=True, monotonic=True, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=True, monotonic=True)"
    )


# ---------------------------------------------------------------------------
# Tests: equivalence with original (no normalisation, cosine)
# ---------------------------------------------------------------------------


def test_no_norm_cosine_matches_reference_sequence_a(sequence_pair_a):
    """normalize=False, cosine matches original alignNOA_no_norm on sequence pair A."""
    ref, query = sequence_pair_a
    path_ref = _align_noa_no_norm_ref(query, ref)
    path_new = run_offline_noa(ref, query, normalize=False, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=False, cosine, pair A)"
    )


def test_no_norm_cosine_matches_reference_sequence_b(sequence_pair_b):
    """normalize=False, cosine matches original alignNOA_no_norm on sequence pair B."""
    ref, query = sequence_pair_b
    path_ref = _align_noa_no_norm_ref(query, ref)
    path_new = run_offline_noa(ref, query, normalize=False, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=False, cosine, pair B)"
    )


# ---------------------------------------------------------------------------
# Tests: custom steps / weights
# ---------------------------------------------------------------------------


def test_custom_steps_weights(sequence_pair_a):
    """Custom steps and weights propagate correctly without error."""
    ref, query = sequence_pair_a
    custom_steps = np.array([1, 1, 1, 2, 2, 1]).reshape((-1, 2))
    custom_weights = np.array([1, 1, 2])
    path = run_offline_noa(
        ref, query,
        steps=custom_steps,
        weights=custom_weights,
        normalize=True,
    )
    assert path.shape[0] == 2
