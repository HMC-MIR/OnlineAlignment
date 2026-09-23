"""Tests for OLTW: the ring-buffer online/offline versions vs. dense reference implementations.

Two independent references are used:

* ``_global_oltw_ref`` is the original full-matrix OLTW: the complete DTW
  matrix is computed with librosa and the path is walked over it. The banded
  implementation with ``c=None`` must reproduce it exactly.
* ``_banded_oltw_ref`` is a direct transcription of Dixon's banded OLTW on
  dense matrices, with no ring buffers. Finite ``c`` must reproduce it exactly.
"""

# library imports
import numpy as np
import pytest
from librosa.sequence import dtw

# custom imports
from online_alignment import OLTW, OfflineOLTW, run_offline_oltw
from online_alignment.alignment.algs import BOTH, ROW, COLUMN
from online_alignment.constants import OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS
from online_alignment.cost import get_cost_metric, normalize_by_path_length
from online_alignment.cost.cosine import cosine_mat2mat_parallel


# ---------------------------------------------------------------------------
# Reference implementations
# ---------------------------------------------------------------------------


def _full_cost_matrix(ref, query, cost_metric):
    """Full local cost matrix, shape (ref_length, query_length)."""
    ref = np.ascontiguousarray(ref, dtype=np.float64)
    query = np.ascontiguousarray(query, dtype=np.float64)
    if cost_metric == "cosine":
        return cosine_mat2mat_parallel(ref, query)
    return np.asarray(get_cost_metric(cost_metric).mat2mat(ref, query), dtype=np.float64)


def _get_inc_ref(D_norm, t, j, c, run_count, prev, max_run_count):
    """Original OLTW transition choice over a dense normalized cost matrix."""
    if c is not None and t < c:
        return BOTH
    if run_count >= max_run_count:
        return COLUMN if prev == ROW else ROW
    row_start = 0 if c is None else max(0, j - c + 1)
    col_start = 0 if c is None else max(0, t - c + 1)
    cur_row = D_norm[t, row_start:j + 1]
    cur_col = D_norm[col_start:t + 1, j]
    if np.min(cur_row) <= np.min(cur_col):
        x, y = t, row_start + int(np.argmin(cur_row))
    else:
        x, y = col_start + int(np.argmin(cur_col)), j
    if x < t:
        return COLUMN
    if y < j:
        return ROW
    return BOTH


def _global_oltw_ref(ref, query, steps, weights, window_steps, max_run_count=3, cost_metric="cosine"):
    """Original OfflineOLTW with c=None. window_steps are in [BOTH, ROW, COLUMN] order."""
    C = _full_cost_matrix(ref, query, cost_metric)
    D = dtw(backtrack=False, C=C, step_sizes_sigma=np.array(steps), weights_mul=np.array(weights))
    D_norm = normalize_by_path_length(D)
    R, Q = C.shape

    t, j, run_count, prev = 0, 0, 0, -1
    path = [[0, 0]]
    while t < R - 1 and j < Q - 1:
        inc = _get_inc_ref(D_norm, t, j, None, run_count, prev, max_run_count)
        t = min(t + window_steps[inc][0], R - 1)
        j = min(j + window_steps[inc][1], Q - 1)
        run_count = run_count + 1 if inc == prev else 1
        if inc != BOTH:
            prev = inc
        path.append([j, t])
    return np.array(path).T


def _banded_oltw_ref(ref, query, steps, weights, window_steps, c, max_run_count=3, cost_metric="cosine"):
    """Dixon's banded OLTW on dense matrices. window_steps are in [BOTH, ROW, COLUMN] order."""
    C = _full_cost_matrix(ref, query, cost_metric)
    R, Q = C.shape
    D = np.full((R, Q), np.inf)
    D[0, 0] = C[0, 0]

    def compute(tt, jj):
        best = np.inf
        for (dt, dj), w in zip(steps, weights):
            if tt - dt >= 0 and jj - dj >= 0:
                best = min(best, D[tt - dt, jj - dj] + float(w) * C[tt, jj])
        D[tt, jj] = best

    t, j, run_count, prev = 0, 0, 0, -1
    path = [[0, 0]]
    while t < R - 1 and j < Q - 1:
        D_norm = D / (np.arange(R)[:, None] + np.arange(Q)[None, :] + 1)
        inc = _get_inc_ref(D_norm, t, j, c, run_count, prev, max_run_count)
        t_new = min(t + window_steps[inc][0], R - 1)
        j_new = min(j + window_steps[inc][1], Q - 1)
        for tt in range(t + 1, t_new + 1):
            for jj in range(max(0, j - c + 1), j + 1):
                compute(tt, jj)
        for jj in range(j + 1, j_new + 1):
            for tt in range(max(0, t_new - c + 1), t_new + 1):
                compute(tt, jj)
        t, j = t_new, j_new
        run_count = run_count + 1 if inc == prev else 1
        if inc != BOTH:
            prev = inc
        path.append([j, t])
    return np.array(path).T


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def sequence_pair_a(rng):
    """Random sequences: ref (12, 50), query (12, 40)."""
    ref = rng.standard_normal((12, 50)).astype(np.float32)
    query = rng.standard_normal((12, 40)).astype(np.float32)
    return ref, query


@pytest.fixture
def sequence_pair_b(rng):
    """Random sequences: ref (12, 80), query (12, 60)."""
    ref = rng.standard_normal((12, 80)).astype(np.float32)
    query = rng.standard_normal((12, 60)).astype(np.float32)
    return ref, query


@pytest.fixture
def warped_pair(rng):
    """A query that is a time-warped, noisy copy of the reference: ref (12, 120), query (12, 150)."""
    ref = np.abs(rng.standard_normal((12, 120))).astype(np.float32)
    idx = np.clip(np.cumsum(rng.choice([0, 1, 1, 2], size=150)), 0, 119)
    query = ref[:, idx] + 0.1 * np.abs(rng.standard_normal((12, 150))).astype(np.float32)
    return ref, query


# DTW steps / weights / window steps (window steps in [BOTH, ROW, COLUMN] order)
CONFIGS = {
    "default": (OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS),
    "slopes": (np.array([[1, 1], [1, 2], [2, 1]]), np.array([2, 1, 1]), OLTW_WINDOW_STEPS),
    "big_both": (OLTW_STEPS, OLTW_WEIGHTS, np.array([[2, 1], [1, 0], [0, 1]])),
    "double_column": (OLTW_STEPS, OLTW_WEIGHTS, np.array([[1, 1], [1, 0], [0, 2]])),
}


# ---------------------------------------------------------------------------
# Tests: unbounded band reproduces full-matrix OLTW
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", ["sequence_pair_a", "sequence_pair_b", "warped_pair"])
@pytest.mark.parametrize("config", sorted(CONFIGS))
def test_global_matches_full_matrix_reference(request, pair, config):
    """c=None gives exactly the path of OLTW over the full librosa DTW matrix."""
    ref, query = request.getfixturevalue(pair)
    steps, weights, window_steps = CONFIGS[config]
    expected = _global_oltw_ref(ref, query, steps, weights, window_steps)
    path = run_offline_oltw(ref, query, steps=steps, weights=weights, window_steps=window_steps, c=None)
    np.testing.assert_array_equal(path, expected)


@pytest.mark.parametrize("cost_metric", ["euclidean", "manhattan"])
def test_global_matches_reference_other_metrics(warped_pair, cost_metric):
    """Non-cosine metrics also reproduce the full-matrix reference."""
    ref, query = warped_pair
    expected = _global_oltw_ref(
        ref, query, OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS, cost_metric=cost_metric
    )
    path = run_offline_oltw(ref, query, c=None, cost_metric=cost_metric)
    np.testing.assert_array_equal(path, expected)


# ---------------------------------------------------------------------------
# Tests: finite band reproduces dense banded OLTW
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("pair", ["sequence_pair_a", "sequence_pair_b", "warped_pair"])
@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("c", [1, 3, 5, 10, 1000])
def test_banded_matches_dense_reference(request, pair, config, c):
    """Finite c gives exactly the path of dense banded OLTW, including after the ring wraps."""
    ref, query = request.getfixturevalue(pair)
    steps, weights, window_steps = CONFIGS[config]
    expected = _banded_oltw_ref(ref, query, steps, weights, window_steps, c)
    path = run_offline_oltw(ref, query, steps=steps, weights=weights, window_steps=window_steps, c=c)
    np.testing.assert_array_equal(path, expected)


@pytest.mark.parametrize("max_run_count", [1, 2, 5])
def test_banded_max_run_count(warped_pair, max_run_count):
    """max_run_count is honoured the same way as in the reference."""
    ref, query = warped_pair
    expected = _banded_oltw_ref(
        ref, query, OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS, 8, max_run_count=max_run_count
    )
    path = run_offline_oltw(ref, query, c=8, max_run_count=max_run_count)
    np.testing.assert_array_equal(path, expected)


def test_banded_euclidean(warped_pair):
    """Non-cosine metrics work with a finite band."""
    ref, query = warped_pair
    expected = _banded_oltw_ref(
        ref, query, OLTW_STEPS, OLTW_WEIGHTS, OLTW_WINDOW_STEPS, 6, cost_metric="euclidean"
    )
    path = run_offline_oltw(ref, query, c=6, cost_metric="euclidean")
    np.testing.assert_array_equal(path, expected)


def test_window_steps_order_does_not_matter(warped_pair):
    """window_steps are arranged by slope, so any order gives the same path."""
    ref, query = warped_pair
    a = run_offline_oltw(ref, query, c=10, window_steps=[[1, 1], [1, 0], [0, 1]])
    b = run_offline_oltw(ref, query, c=10, window_steps=[[0, 1], [1, 1], [1, 0]])
    np.testing.assert_array_equal(a, b)


# ---------------------------------------------------------------------------
# Tests: online OLTW
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("config", sorted(CONFIGS))
@pytest.mark.parametrize("c", [None, 4, 20])
def test_online_feed_matches_offline(warped_pair, config, c):
    """Feeding frames one at a time then flushing gives the offline path exactly."""
    ref, query = warped_pair
    steps, weights, window_steps = CONFIGS[config]
    offline = run_offline_oltw(ref, query, steps=steps, weights=weights, window_steps=window_steps, c=c)

    oltw = OLTW(ref, steps=steps, weights=weights, window_steps=window_steps, c=c)
    for i in range(query.shape[1]):
        position = oltw.feed(query[:, i])
        assert position == oltw.path[1, -1]
        assert oltw.path[0, -1] <= i  # never uses a frame that has not arrived
    oltw.flush()
    np.testing.assert_array_equal(oltw.path, offline)


def test_online_memory_is_bounded(rng):
    """With finite c, buffer sizes depend on c, not on the query length."""
    ref = rng.standard_normal((12, 400)).astype(np.float32)
    query = rng.standard_normal((12, 2000)).astype(np.float32)
    oltw = OLTW(ref, c=10)
    shape = oltw._D.shape
    oltw.align(query)
    assert oltw._D.shape == shape
    assert shape[0] < 20 and shape[1] < 20


def test_online_unbounded_grows(rng):
    """With c=None, the buffers grow as frames arrive and the path still matches offline."""
    ref = rng.standard_normal((12, 30)).astype(np.float32)
    query = rng.standard_normal((12, 2500)).astype(np.float32)
    oltw = OLTW(ref, c=None)
    for i in range(query.shape[1]):
        oltw.feed(query[:, i])
    oltw.flush()
    np.testing.assert_array_equal(oltw.path, run_offline_oltw(ref, query, c=None))


def test_feed_after_flush_raises(sequence_pair_a):
    """The query cannot continue after flush() until reset()."""
    ref, query = sequence_pair_a
    oltw = OLTW(ref, c=5)
    oltw.feed(query[:, 0])
    oltw.flush()
    with pytest.raises(RuntimeError):
        oltw.feed(query[:, 1])
    oltw.reset()
    oltw.feed(query[:, 0])


def test_class_stores_path(sequence_pair_a):
    """OfflineOLTW stores the result in self.path after align()."""
    ref, query = sequence_pair_a
    oltw = OfflineOLTW(ref, c=5)
    path = oltw.align(query)
    assert path is oltw.path
    assert path.shape[0] == 2 and path.dtype == np.int64


def test_single_frame_inputs(rng):
    """One-frame reference or query gives the trivial path."""
    ref = rng.standard_normal((12, 1)).astype(np.float32)
    query = rng.standard_normal((12, 10)).astype(np.float32)
    np.testing.assert_array_equal(run_offline_oltw(ref, query), [[0], [0]])
    np.testing.assert_array_equal(run_offline_oltw(query, ref), [[0], [0]])


# ---------------------------------------------------------------------------
# Tests: validation
# ---------------------------------------------------------------------------


def test_invalid_parameters(sequence_pair_a):
    """Bad band widths, run counts and window steps raise ValueError."""
    ref, query = sequence_pair_a
    with pytest.raises(ValueError):
        OLTW(ref, c=0)
    with pytest.raises(ValueError):
        OLTW(ref, max_run_count=0)
    with pytest.raises(ValueError):
        OLTW(ref, window_steps=[[1, 1], [1, 2], [0, 1]])
    with pytest.raises(ValueError):
        OLTW(ref, window_steps=[[1, 1], [0, 1]])
    with pytest.raises(ValueError):
        run_offline_oltw(ref, query[:5])
