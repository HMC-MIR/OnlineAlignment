#!/usr/bin/env python3
"""Time optimized OfflineOLTW vs reference (original) implementation on sequences.

Run from the OnlineAlignment directory:
    python scripts/time_oltw.py

Or from repo root with PYTHONPATH:
    PYTHONPATH=OnlineAlignment python OnlineAlignment/scripts/time_oltw.py
"""

from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

import numpy as np

# Suppress unrelated cost-metric warning during timing
warnings.filterwarnings("ignore", message="p=2 is equivalent to Euclidean")

# Ensure package root is on path when running script directly
_online_alignment = Path(__file__).resolve().parent.parent
if str(_online_alignment) not in sys.path:
    sys.path.insert(0, str(_online_alignment))

from core.alignment.offline.oltw import OfflineOLTW
from core.constants import OLTW_STEPS, OLTW_WEIGHTS
from tests.core.alignment.offline.test_oltw import OfflineOLTWReference


def make_sequence(rng: np.random.Generator, n_feat: int, ref_len: int, query_len: int):
    ref = rng.standard_normal((n_feat, ref_len)).astype(np.float32)
    query = rng.standard_normal((n_feat, query_len)).astype(np.float32)
    return ref, query


def time_align(align_fn, query: np.ndarray, n_warmup: int = 1, n_repeat: int = 5) -> tuple[float, float]:
    """Return (mean_seconds, std_seconds) over n_repeat runs after n_warmup."""
    for _ in range(n_warmup):
        align_fn(query)
    times: list[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        align_fn(query)
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def main():
    rng = np.random.default_rng(42)
    n_feat = 12
    n_warmup = 2
    n_repeat = 5

    # (ref_len, query_len) pairs to benchmark
    sizes = [
        (50, 40),
        (200, 150),
        (500, 400),
        (1000, 800),
        (2000, 1600),
        (4000, 3200),
        (8000, 6400),
    ]

    print("Offline OLTW: optimized vs reference (original)")
    print("=" * 70)
    print(f"{'ref×query':<16} {'reference (s)':<18} {'optimized (s)':<18} {'speedup':<10}")
    print("-" * 70)

    for ref_len, query_len in sizes:
        ref, query = make_sequence(rng, n_feat, ref_len, query_len)

        alg_ref = OfflineOLTWReference(
            ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS
        )
        alg_opt = OfflineOLTW(
            ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS
        )

        mean_ref, std_ref = time_align(
            lambda q: alg_ref.align(q), query, n_warmup=n_warmup, n_repeat=n_repeat
        )
        mean_opt, std_opt = time_align(
            lambda q: alg_opt.align(q), query, n_warmup=n_warmup, n_repeat=n_repeat
        )

        speedup = mean_ref / mean_opt if mean_opt > 0 else 0.0
        size_str = f"{ref_len}×{query_len}"
        print(f"{size_str:<16} {mean_ref:.4f} ± {std_ref:.4f}   {mean_opt:.4f} ± {std_opt:.4f}   {speedup:.2f}x")

    print("=" * 70)
    print("Done.")


def main_with_parallel():
    """Same as main() but also time optimized + use_parallel_cost=True (cosine)."""
    rng = np.random.default_rng(42)
    n_feat = 12
    n_warmup = 2
    n_repeat = 5
    sizes = [(50, 40), (200, 150), (500, 400), (1000, 800), (2000, 1600)]

    print("Offline OLTW: reference vs optimized vs optimized+parallel_cost (cosine)")
    print("=" * 85)
    print(f"{'ref×query':<12} {'reference':<14} {'optimized':<14} {'opt+par_c':<14} {'speedup':<8}")
    print("-" * 85)

    for ref_len, query_len in sizes:
        ref, query = make_sequence(rng, n_feat, ref_len, query_len)
        alg_ref = OfflineOLTWReference(
            ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS
        )
        alg_opt = OfflineOLTW(
            ref, DTW_steps=OLTW_STEPS, DTW_weights=OLTW_WEIGHTS, window_steps=OLTW_STEPS
        )
        alg_par = OfflineOLTW(
            ref,
            DTW_steps=OLTW_STEPS,
            DTW_weights=OLTW_WEIGHTS,
            window_steps=OLTW_STEPS,
            use_parallel_cost=True,
        )
        mean_ref, _ = time_align(
            lambda q: alg_ref.align(q), query, n_warmup=n_warmup, n_repeat=n_repeat
        )
        mean_opt, _ = time_align(
            lambda q: alg_opt.align(q), query, n_warmup=n_warmup, n_repeat=n_repeat
        )
        mean_par, _ = time_align(
            lambda q: alg_par.align(q), query, n_warmup=n_warmup, n_repeat=n_repeat
        )
        speedup = mean_ref / mean_opt if mean_opt > 0 else 0.0
        size_str = f"{ref_len}×{query_len}"
        print(f"{size_str:<12} {mean_ref:.4f} s     {mean_opt:.4f} s     {mean_par:.4f} s     {speedup:.2f}x")
    print("=" * 85)
    print("Done.")


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--parallel-cost":
        main_with_parallel()
    else:
        main()
