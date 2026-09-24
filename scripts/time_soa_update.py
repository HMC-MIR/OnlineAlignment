#!/usr/bin/env python3
"""Time one SOA update against a long reference, split into cost row and DP update.

Runs on a single CPU core with one BLAS and Numba thread. Before timing, the query
is fed until every reference frame is reachable: with a fixed start, query frame t
can only reach reference frames up to 2t, so early frames leave most of the row
at infinity and are much cheaper than a real performance in progress. A flexible
start reaches the whole reference from the first frame.

By default the features are random nonnegative 12-dimensional vectors,
L2-normalized like chroma. Real features time a little faster (their costs make
the DP comparisons more predictable), so pass real ones with ``--features`` for
numbers to quote: the files are concatenated and tiled to the reference length,
and the query is drawn from the last file.

Run from the OnlineAlignment directory after ``pip install -e .``:
    python scripts/time_soa_update.py                # 60-minute reference
    python scripts/time_soa_update.py --minutes 10 --frames 500
    python scripts/time_soa_update.py --features features/*.npy
"""

# standard imports
import argparse
import os

# one thread each, set before numpy and numba load their thread pools
for _var in ("MKL_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

import time  # noqa: E402
from typing import Tuple  # noqa: E402

# library imports
import numpy as np  # noqa: E402

# custom imports
from online_alignment import SOA  # noqa: E402

SR = 22050
HOP_LENGTH = 512
FRAME_PERIOD_MS = 1e3 * HOP_LENGTH / SR


def chroma_like(rng: np.random.Generator, n_frames: int) -> np.ndarray:
    features = np.abs(rng.standard_normal((12, n_frames))).astype(np.float32)
    return features / np.linalg.norm(features, axis=0, keepdims=True)


def real_features(paths, n_frames: int) -> Tuple[np.ndarray, np.ndarray]:
    """A reference of n_frames tiled from all but the last file, and the last file as query."""
    pool = np.concatenate([np.load(p) for p in paths[:-1]], axis=1)
    reps = -(-n_frames // pool.shape[1])
    reference = np.ascontiguousarray(np.tile(pool, (1, reps))[:, :n_frames], dtype=np.float32)
    return reference, np.ascontiguousarray(np.load(paths[-1]), dtype=np.float32)


def time_updates(
    reference: np.ndarray, query: np.ndarray, flexible: bool, n_frames: int
) -> Tuple[float, float, float]:
    """Median (cost row, DP update, whole feed) in ms over n_frames, once the row is reachable."""
    soa = SOA(reference, flexible_start=flexible)
    n_warmup = 2 if flexible else reference.shape[1] // 2 + 2

    cost_times = []
    bound_costs = soa._costs

    def timed_costs(frame: np.ndarray) -> np.ndarray:
        t0 = time.perf_counter()
        costs = bound_costs(frame)
        cost_times.append(time.perf_counter() - t0)
        return costs

    feed_times = []
    for i in range(n_warmup + n_frames):
        if i == n_warmup:
            soa._costs = timed_costs
        t0 = time.perf_counter()
        soa.feed(query[:, i % query.shape[1]])
        if i >= n_warmup:
            feed_times.append(time.perf_counter() - t0)
    if soa.finished:
        raise RuntimeError("the path reached the end of the reference; use a longer one")

    cost = 1e3 * float(np.median(cost_times))
    feed = 1e3 * float(np.median(feed_times))
    return cost, feed - cost, feed


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--minutes", type=float, default=60.0, help="reference length")
    parser.add_argument("--frames", type=int, default=1000, help="timed updates per mode")
    parser.add_argument("--core", type=int, default=0, help="CPU core to pin to")
    parser.add_argument(
        "--features", nargs="+", help="real feature .npy files, shape (12, n_frames); at least two"
    )
    args = parser.parse_args()

    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.core})

    n_ref = int(round(args.minutes * 60 * SR / HOP_LENGTH))
    if args.features:
        if len(args.features) < 2:
            parser.error("--features needs at least two files")
        reference, query = real_features(args.features, n_ref)
        source = f"{len(args.features)} feature files"
    else:
        rng = np.random.default_rng(0)
        reference, query = chroma_like(rng, n_ref), chroma_like(rng, 5000)
        source = "random chroma-like features"

    print(
        f"reference: {args.minutes:g} min, N = {n_ref} frames, {source}; "
        f"frame period {FRAME_PERIOD_MS:.1f} ms"
    )
    for flexible in (False, True):
        cost, dp, feed = time_updates(reference, query, flexible, args.frames)
        mode = "flexible start" if flexible else "fixed start"
        print(
            f"{mode:<15} cost row {cost:.2f} ms   DP update {dp:.2f} ms   "
            f"whole update {feed:.2f} ms  ({FRAME_PERIOD_MS / feed:.0f}x under real time)"
        )


if __name__ == "__main__":
    main()
