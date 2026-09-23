#!/usr/bin/env python3
"""Time SOA and OLTW on random sequences of increasing length.

Reports the total offline time for each algorithm, and the mean and worst-case
per-frame latency of ``feed()`` for the online versions.

Run from the OnlineAlignment directory after ``pip install -e .``:
    python scripts/time_alignment.py
"""

# standard imports
import time
from typing import Callable, List, Tuple

# library imports
import numpy as np

# custom imports
from online_alignment import OLTW, SOA, run_offline_oltw, run_offline_soa
from online_alignment.alignment import OnlineAlignment


def make_sequence(rng: np.random.Generator, n_feat: int, ref_len: int, query_len: int):
    ref = np.abs(rng.standard_normal((n_feat, ref_len))).astype(np.float32)
    query = np.abs(rng.standard_normal((n_feat, query_len))).astype(np.float32)
    return ref, query


def time_call(
    fn: Callable[[], object], n_warmup: int = 1, n_repeat: int = 3
) -> Tuple[float, float]:
    """Return (mean_seconds, std_seconds) over n_repeat runs after n_warmup."""
    for _ in range(n_warmup):
        fn()
    times: List[float] = []
    for _ in range(n_repeat):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return float(np.mean(times)), float(np.std(times))


def feed_latency(aligner: OnlineAlignment, query: np.ndarray) -> Tuple[float, float]:
    """Return (mean_seconds, max_seconds) per feed() call over the whole query."""
    aligner.reset()
    times = np.empty(query.shape[1])
    for i in range(query.shape[1]):
        t0 = time.perf_counter()
        aligner.feed(query[:, i])
        times[i] = time.perf_counter() - t0
    aligner.flush()
    return float(times.mean()), float(times.max())


def main():
    rng = np.random.default_rng(42)
    n_feat = 12
    c = 500
    sizes = [(500, 400), (1000, 800), (2000, 1600), (4000, 3200), (8000, 6400)]

    # compile the Numba kernels before timing
    warm_ref, warm_query = make_sequence(rng, n_feat, 50, 40)
    run_offline_soa(warm_ref, warm_query)
    run_offline_oltw(warm_ref, warm_query, c=5)

    print("Offline: total seconds per alignment")
    print(f"{'ref×query':<12} {'SOA':<12} {f'OLTW c={c}':<12} {'OLTW c=None':<12}")
    print("-" * 50)
    for ref_len, query_len in sizes:
        ref, query = make_sequence(rng, n_feat, ref_len, query_len)
        t_soa, _ = time_call(lambda: run_offline_soa(ref, query))
        t_band, _ = time_call(lambda: run_offline_oltw(ref, query, c=c))
        t_global, _ = time_call(lambda: run_offline_oltw(ref, query, c=None))
        print(f"{f'{ref_len}×{query_len}':<12} {t_soa:<12.4f} {t_band:<12.4f} {t_global:<12.4f}")

    print()
    print("Online: feed() latency per frame, mean / max (µs)")
    print(f"{'ref×query':<12} {'SOA':<22} {f'OLTW c={c}':<22}")
    print("-" * 56)
    for ref_len, query_len in sizes:
        ref, query = make_sequence(rng, n_feat, ref_len, query_len)
        soa_mean, soa_max = feed_latency(SOA(ref), query)
        oltw_mean, oltw_max = feed_latency(OLTW(ref, c=c), query)
        soa_str = f"{soa_mean * 1e6:.1f} / {soa_max * 1e6:.1f}"
        oltw_str = f"{oltw_mean * 1e6:.1f} / {oltw_max * 1e6:.1f}"
        print(f"{f'{ref_len}×{query_len}':<12} {soa_str:<22} {oltw_str:<22}")


if __name__ == "__main__":
    main()
