"""Tests for SOA: the online and offline versions vs. the original dense-matrix logic.

The original SOA logic is reproduced here as two reference functions
(_align_soa_ref / _align_soa_no_norm_ref) that keep the full accumulated cost
matrix, so any numerical difference in the ring-buffer implementation will be
caught immediately.

Run from OnlineAlignment/:
    pytest tests/core/alignment/test_soa.py -v
"""

# library imports
import numpy as np
import pytest
from numba import njit

# custom imports
from online_alignment.alignment.offline.soa import OfflineSOA, run_offline_soa
from online_alignment.alignment.online.soa import SOA
from online_alignment.constants import SOA_STEPS, SOA_WEIGHTS


# ---------------------------------------------------------------------------
# Reference Numba kernels (inlined from the original SOA script)
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
# Reference alignment functions (exact copies of original alignSOA logic)
# ---------------------------------------------------------------------------


def _align_soa_ref(F1, F2, steps=SOA_STEPS, weights=SOA_WEIGHTS, monotonic=False):
    """Original alignSOA, returning integer frame indices (path[0]=query, path[1]=ref)."""
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


def _align_soa_no_norm_ref(F1, F2, steps=SOA_STEPS, weights=SOA_WEIGHTS):
    """Original alignSOA_no_norm, returning integer frame indices."""
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
    """align() returns shape (2, N) int64."""
    ref, query = sequence_pair_a
    path = run_offline_soa(ref, query)
    assert path.ndim == 2
    assert path.shape[0] == 2
    assert path.dtype == np.int64


def test_class_stores_path(sequence_pair_a):
    """OfflineSOA stores the result in self.path after align()."""
    ref, query = sequence_pair_a
    soa = OfflineSOA(ref)
    path = soa.align(query)
    assert path is soa.path


def test_path_rows_nondecreasing_query_column(sequence_pair_a):
    """Query frame index (row 0) is strictly increasing by construction."""
    ref, query = sequence_pair_a
    path = run_offline_soa(ref, query)
    # query indices should be 0, 1, 2, ... (incrementing by 1 each frame)
    assert np.all(np.diff(path[0]) == 1)


# ---------------------------------------------------------------------------
# Tests: equivalence with original (normalised, cosine)
# ---------------------------------------------------------------------------


def test_normalized_cosine_matches_reference_sequence_a(sequence_pair_a):
    """normalize=True, cosine matches original alignSOA on sequence pair A."""
    ref, query = sequence_pair_a
    path_ref = _align_soa_ref(query, ref)
    path_new = run_offline_soa(ref, query, normalize=True, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=True, cosine, pair A)"
    )


def test_normalized_cosine_matches_reference_sequence_b(sequence_pair_b):
    """normalize=True, cosine matches original alignSOA on sequence pair B."""
    ref, query = sequence_pair_b
    path_ref = _align_soa_ref(query, ref)
    path_new = run_offline_soa(ref, query, normalize=True, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=True, cosine, pair B)"
    )


def test_normalized_monotonic_matches_reference(sequence_pair_a):
    """normalize=True, monotonic=True matches original alignSOA(monotonic=True)."""
    ref, query = sequence_pair_a
    path_ref = _align_soa_ref(query, ref, monotonic=True)
    path_new = run_offline_soa(ref, query, normalize=True, monotonic=True, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=True, monotonic=True)"
    )


# ---------------------------------------------------------------------------
# Tests: equivalence with original (no normalisation, cosine)
# ---------------------------------------------------------------------------


def test_no_norm_cosine_matches_reference_sequence_a(sequence_pair_a):
    """normalize=False, cosine matches original alignSOA_no_norm on sequence pair A."""
    ref, query = sequence_pair_a
    path_ref = _align_soa_no_norm_ref(query, ref)
    path_new = run_offline_soa(ref, query, normalize=False, cost_metric="cosine")
    np.testing.assert_array_equal(
        path_new, path_ref, err_msg="Path mismatch (normalize=False, cosine, pair A)"
    )


def test_no_norm_cosine_matches_reference_sequence_b(sequence_pair_b):
    """normalize=False, cosine matches original alignSOA_no_norm on sequence pair B."""
    ref, query = sequence_pair_b
    path_ref = _align_soa_no_norm_ref(query, ref)
    path_new = run_offline_soa(ref, query, normalize=False, cost_metric="cosine")
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
    path = run_offline_soa(
        ref, query,
        steps=custom_steps,
        weights=custom_weights,
        normalize=True,
    )
    assert path.shape[0] == 2


# ---------------------------------------------------------------------------
# Tests: long queries and termination
# ---------------------------------------------------------------------------


def test_query_longer_than_twice_reference(rng):
    """Queries longer than 2x the reference no longer overflow the cost matrix."""
    ref = _normalized(rng, 12, 20)
    query = _normalized(rng, 12, 100)
    path = run_offline_soa(ref, query, monotonic=True)
    assert path.shape[0] == 2
    assert path[1, -1] <= ref.shape[1] - 1


def test_matches_reference_after_ring_buffer_wraps(rng):
    """Many more query frames than ring buffer rows still match the dense original."""
    ref = _normalized(rng, 12, 300)
    query = _normalized(rng, 12, 250)
    path_ref = _align_soa_ref(query, ref)
    path_new = run_offline_soa(ref, query)
    np.testing.assert_array_equal(path_new, path_ref)


# ---------------------------------------------------------------------------
# Tests: online SOA
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("normalize,monotonic", [(True, False), (True, True), (False, False)])
def test_online_feed_matches_offline(sequence_pair_a, normalize, monotonic):
    """Feeding frames one at a time gives the offline path, and feed() returns the position."""
    ref, query = sequence_pair_a
    offline = run_offline_soa(ref, query, normalize=normalize, monotonic=monotonic)

    soa = SOA(ref, normalize=normalize, monotonic=monotonic)
    positions = [soa.feed(query[:, i:i + 1]) for i in range(query.shape[1])]
    soa.flush()

    np.testing.assert_array_equal(soa.path, offline)
    n = offline.shape[1]
    np.testing.assert_array_equal(positions[:n], offline[1])
    assert all(p == offline[1, -1] for p in positions[n:])


def test_online_memory_is_bounded(sequence_pair_a):
    """The ring buffer holds max(row_steps) + 1 rows."""
    ref, _ = sequence_pair_a
    soa = SOA(ref)
    assert soa._D.shape == (SOA_STEPS[:, 0].max() + 1, ref.shape[1])


def test_online_reset_and_realign(sequence_pair_a, sequence_pair_b):
    """align() resets state, so reusing an instance gives the same result."""
    ref, query = sequence_pair_a
    soa = SOA(ref)
    first = soa.align(query)
    soa.align(sequence_pair_b[1][:, :30])
    np.testing.assert_array_equal(soa.align(query), first)


def test_online_rejects_bad_frame(sequence_pair_a):
    """Frames with the wrong feature dimension raise ValueError."""
    ref, _ = sequence_pair_a
    soa = SOA(ref)
    with pytest.raises(ValueError):
        soa.feed(np.zeros(5))


def test_offline_rejects_bad_query(sequence_pair_a):
    """Queries with the wrong shape raise ValueError."""
    ref, _ = sequence_pair_a
    with pytest.raises(ValueError):
        run_offline_soa(ref, np.zeros((5, 10), dtype=np.float32))
    with pytest.raises(ValueError):
        run_offline_soa(ref, np.zeros(12, dtype=np.float32))
