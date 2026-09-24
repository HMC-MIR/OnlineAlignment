# Changelog

## 0.4.1

Faster flexible start; paths are bit-identical to 0.4.0 (and to 0.3.0).

### Performance
Per update against a 60-minute reference (N = 155k frames) on one 2.4 GHz Xeon core, with
numba 0.63 and real chroma features:
- Flexible start: DP update 1.46 ms → 0.65 ms. The scores are now computed in their own
  loop, as they already were with a fixed start; writing them in the row loop made it
  over twice as slow.
- Fixed start is unchanged (DP update 0.64 ms, cost row about 0.6 ms).

### Changed
- `SOA` converts features to float32 (keeping their memory layout), and the kernels have
  explicit signatures, so each is compiled once: float32 costs and rows, int32 starts,
  float64 scores. Float64 features now give the same paths as the same features in
  float32; previously their float64 costs were used.

### Fixed
- The 0.4.0 notes and docs described the SOA kernels as SIMD-vectorized. They compile to
  scalar, branch-free code; only the score and argmin passes use SIMD. Vectorized versions
  with separate row arrays, row blocking and interleaved (cost, start) storage were all
  slower on the test machine, because the update is limited by memory traffic.

## 0.4.0

SOA now has a single implementation: one branch-free pass over the reference per
query frame, for both fixed and flexible start. Paths are bit-identical to 0.3.0 for the
supported steps.

### Changed
- **SOA supports only the steps (1,1), (1,2), (2,1)**, in that order, as in the paper.
  `steps` is still accepted but must be that pattern; others raise `ValueError`. This
  includes patterns with a same-row step such as (0,1), which chain cells within a row.
  Weights stay configurable (default 1, 1, 2).
- Path starts for `flexible_start=True` are stored as int32.

### Performance
Per update against a 60-minute reference (N = 155k frames) on one 2.4 GHz Xeon core,
once the reachable part of the row is largest:
- Fixed start: DP update 1.80 ms → 0.63 ms.
- Flexible start: DP update 2.87 ms → 1.35 ms.
- The cost row is unchanged at about 0.85 ms.

### Added
- `scripts/time_soa_update.py`, which times the cost row and the DP update separately on
  one core, once the reachable part of the row is largest.

## 0.3.0

### Added
- `flexible_start` option for `SOA`, `OfflineSOA` and `run_offline_soa`: the query may
  start at any reference frame. Each cell also records where its path began, and paths are
  compared by accumulated cost divided by path length, as in Section 2.2 of the paper. The
  default (`False`) keeps the fixed start at the first reference frame, unchanged.

## 0.2.1

Faster SOA and OLTW. Paths are bit-identical to 0.2.0. On Mazurka chroma features:
- SOA: 10.8 s → 2.5 s on a 10k × 10k-frame pair (the reference is normalized once instead
  of on every frame).
- OLTW with cosine runs each step in a single Numba call: `c=500` 1.5 s → 0.6 s on the same
  pair (about 50 µs per frame online); `c=None` 18 s → 8.6 s on a 17.6k × 11k pair (0.1.5
  took 15.5 s).

### Added
- `CostMetric.bind_reference(reference)`, which returns a per-frame cost function and lets a
  metric precompute work on a fixed reference (cosine normalizes it once).

## 0.2.0

Breaking release: NOA is renamed to SOA, and OLTW is now Dixon's banded algorithm.

### Added
- Online `SOA` and `OLTW` classes with `feed(frame) -> reference_position`, `flush()`,
  `reset()`, `position` and `path`. Memory is bounded: SOA keeps `max(query_steps) + 1`
  cost rows, and OLTW with a finite `c` keeps a ring buffer of about `c × c` cells.
- Online and offline versions share one implementation, so they produce identical paths.
- `ManhattanDistance` and `LpNormDistance` are exported from the top-level package.
- `CHANGELOG.md`, usage documentation, and `scripts/time_alignment.py` for timing both
  algorithms offline and online.

### Changed
- **NOA (Naive Online Alignment) is renamed SOA (Simple Online Alignment):** `NOA` → `SOA`,
  `OfflineNOA` → `OfflineSOA`, `run_offline_noa` → `run_offline_soa`, `NOA_STEPS` /
  `NOA_WEIGHTS` → `SOA_STEPS` / `SOA_WEIGHTS`. SOA paths are unchanged.
- **OLTW computes only the cells within `c` frames of the path** (Dixon 2005). Results with
  a finite `c` differ from 0.1.x, which computed the full DTW matrix and used `c` only to
  limit the search. `c=None` still computes every cell and gives the same paths as 0.1.x.
- OLTW's default `c` is now 500 (was `None`).
- OLTW arguments `DTW_steps` / `DTW_weights` are renamed `steps` / `weights`, to match SOA.
- Paths are `int64` for both algorithms (SOA was `int32`).
- `librosa` and `scipy` are no longer runtime dependencies; `librosa` is a dev dependency
  used by the tests.

### Removed
- OLTW's `use_parallel_cost` option. Cosine costs always use the exact per-cell kernel.
- `DEFAULT_DTW_STEPS` / `DEFAULT_DTW_WEIGHTS` (duplicates of the SOA defaults).
- `OnlineAlignment.process_frame()`, and the unfinished `prev_alignment_path` option of the
  online NOA stub.

### Fixed
- `ManhattanDistance` used `p=2` and emitted the Euclidean warning.
- Importing the package printed `LpNormDistance` warnings.
- Query features were never validated.
- Offline SOA crashed on queries longer than twice the reference.
- `scripts/time_oltw.py` imported modules that no longer exist.
