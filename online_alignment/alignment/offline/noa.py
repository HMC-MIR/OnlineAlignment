"""Fully offline version of the Naive Online Alignment (NOA) algorithm."""

# standard imports
from typing import Callable

# library imports
import numpy as np
from numba import njit

# core imports
from ...constants import NOA_STEPS, NOA_WEIGHTS
from ...cost import CostMetric

# custom imports
from .base import OfflineAlignment
from ..utils import _validate_dtw_steps_weights, _validate_query_features_shape


# ---------------------------------------------------------------------------
# Numba-compiled inner-loop helpers
# ---------------------------------------------------------------------------


@njit(cache=True)
def _update_alignment_row_norm(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    B: np.ndarray,
    dn: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
    ref_length: int,
) -> int:
    """Update one row of the alignment matrices with path-length normalisation.

    Selects the best predecessor cell for every reference column *j* by
    minimising the path-length-normalised accumulated cost, then returns the
    column index *best_j* with the globally lowest normalised cost (used as the
    current estimate of the reference position).

    Args:
        i: Current query frame index (row in D/B).
        costs: Local cost vector for frame *i*. Shape (ref_length,)
        D: Accumulated cost matrix. Shape (max_query_length, ref_length)
        B: Backtrace matrix.        Shape (max_query_length, ref_length)
        dn: Row step sizes.    Shape (n_steps,)
        dm: Column step sizes. Shape (n_steps,)
        dw: Step weights.      Shape (n_steps,)
        ref_length: Number of reference frames.

    Returns:
        best_j: Reference column index with the lowest normalised cost.
    """
    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step_cost_norm = np.inf
        best_step = -1

        for k in range(dn.shape[0]):
            di = dn[k]
            dj = dm[k]
            w = dw[k]
            prev_i, prev_j = i - di, j - dj

            if prev_i < 0 or prev_j < 0 or prev_j >= ref_length:
                continue

            cur_cost = D[prev_i, prev_j] + costs[j] * w
            norm_cost = cur_cost / (i + 1 + j + 1)  # normalise by path length

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
def _update_alignment_row(
    i: int,
    costs: np.ndarray,
    D: np.ndarray,
    B: np.ndarray,
    dn: np.ndarray,
    dm: np.ndarray,
    dw: np.ndarray,
    ref_length: int,
) -> int:
    """Update one row of the alignment matrices without path-length normalisation.

    Selects the best predecessor cell for every reference column *j* by
    minimising the raw accumulated cost, then returns the column index *best_j*
    with the globally lowest accumulated cost.

    Args:
        i: Current query frame index (row in D/B).
        costs: Local cost vector for frame *i*. Shape (ref_length,)
        D: Accumulated cost matrix. Shape (max_query_length, ref_length)
        B: Backtrace matrix.        Shape (max_query_length, ref_length)
        dn: Row step sizes.    Shape (n_steps,)
        dm: Column step sizes. Shape (n_steps,)
        dw: Step weights.      Shape (n_steps,)
        ref_length: Number of reference frames.

    Returns:
        best_j: Reference column index with the lowest accumulated cost.
    """
    best_j = 0
    best_cost = np.inf

    for j in range(min(costs.shape[0], ref_length)):
        best_step_cost = np.inf
        best_step = -1

        for k in range(dn.shape[0]):
            di = dn[k]
            dj = dm[k]
            w = dw[k]
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


# ---------------------------------------------------------------------------
# OfflineNOA class
# ---------------------------------------------------------------------------


class OfflineNOA(OfflineAlignment):
    """Fully offline Naive Online Alignment (NOA) algorithm.

    NOA builds the accumulated-cost matrix incrementally (one query frame at a
    time) and tracks the best reference position after each frame without
    performing a global backtrace.  The result is therefore an online-style
    warping path produced in a single offline pass.
    """

    def __init__(
        self,
        reference_features: np.ndarray,
        steps: np.ndarray = NOA_STEPS,
        weights: np.ndarray = NOA_WEIGHTS,
        cost_metric: str | Callable | CostMetric = "cosine",
        normalize: bool = True,
        monotonic: bool = False,
    ):
        """Initialise OfflineNOA.

        Args:
            reference_features: Reference audio features.
                Shape (n_features, n_frames)
            steps: DTW step pattern. Shape (n_steps, 2) where each row is
                (row_increment, column_increment).
            weights: Weight for each step. Shape (n_steps,)
            cost_metric: Distance metric. Can be a string (``"cosine"``,
                ``"euclidean"``, …), a callable, or a :class:`CostMetric`
                instance.
            normalize: If ``True`` (default) use path-length-normalised cost
                when selecting the best reference column per frame.  If
                ``False``, use raw accumulated cost (equivalent to
                ``alignNOA_no_norm`` from the original script).
            monotonic: If ``True``, force the warping path to be monotonically
                non-decreasing (i.e. never move backwards in the reference).
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

        # path produced by the most recent align() call
        self.path: np.ndarray | None = None

    def align(self, query_features: np.ndarray) -> np.ndarray:
        """Align query features to reference features using NOA.

        Args:
            query_features: Query feature matrix. Shape (n_features, n_frames)

        Returns:
            Warping path as **integer frame indices**. Shape (2, n_path_frames),
            where ``path[0]`` is query frame indices and ``path[1]`` is
            reference frame indices.  Converting to seconds is left to the
            caller (multiply by ``hop_length / sample_rate``).
        """
        _validate_query_features_shape(query_features)

        ref_length = self.reference_length
        dn = self.steps[:, 0].astype(np.int64)
        dm = self.steps[:, 1].astype(np.int64)
        dw = self.weights.astype(np.float32)

        # Pre-allocate cost and backtrace matrices
        max_query_length = 2 * ref_length
        D = np.full((max_query_length, ref_length), np.inf, dtype=np.float32)
        D[0, 0] = 0.0
        B = np.full((max_query_length, ref_length), -1, dtype=np.int32)

        path = [[0, 0]]

        update_fn = _update_alignment_row_norm if self.normalize else _update_alignment_row

        for i in range(1, query_features.shape[1]):
            if path[-1][1] >= ref_length - 1:
                break

            costs = self.cost_metric.mat2vec(self.reference_features, query_features[:, i])
            best_j = update_fn(i, costs, D, B, dn, dm, dw, ref_length)

            if self.normalize and self.monotonic:
                best_j = max(best_j, path[-1][1])

            path.append([i, best_j])

        # Return integer frame indices — caller converts to seconds
        self.path = np.array(path, dtype=np.int32).T
        return self.path


# ---------------------------------------------------------------------------
# Convenience wrapper
# ---------------------------------------------------------------------------


def run_offline_noa(
    reference_features: np.ndarray,
    query_features: np.ndarray,
    steps: np.ndarray = NOA_STEPS,
    weights: np.ndarray = NOA_WEIGHTS,
    cost_metric: str | Callable | CostMetric = "cosine",
    normalize: bool = True,
    monotonic: bool = False,
) -> np.ndarray:
    """Run offline NOA alignment in a single call.

    Args:
        reference_features: Reference features. Shape (n_features, n_frames)
        query_features: Query features.     Shape (n_features, n_frames)
        steps: DTW step pattern. Shape (n_steps, 2).
        weights: Step weights. Shape (n_steps,).
        cost_metric: Distance metric (string name, callable, or
            :class:`CostMetric` instance).
        normalize: Use path-length-normalised cost when tracking the best
            reference column.  Set to ``False`` for raw-cost behaviour.
        monotonic: Enforce monotonic reference-column progression.
            Only applied when *normalize* is ``True``.

    Returns:
        Warping path as integer frame indices. Shape (2, n_path_frames).
    """
    noa = OfflineNOA(
        reference_features,
        steps=steps,
        weights=weights,
        cost_metric=cost_metric,
        normalize=normalize,
        monotonic=monotonic,
    )
    return noa.align(query_features)
