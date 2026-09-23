# Changelog

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
