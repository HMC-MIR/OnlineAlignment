# online_alignment

Online audio-to-audio alignment in Python, with Numba-compiled inner loops. The package
implements two algorithms, each usable online (frame by frame, bounded memory) or offline
(on a complete query):

- **SOA** (Simple Online Alignment): for every query frame, extend the DTW accumulated
  cost matrix by one row over the whole reference and report the reference frame with the
  lowest (path-length-normalized) cost.
- **OLTW** (Online Time Warping, [Dixon 2005](https://www.eecs.qmul.ac.uk/~simond/pub/2005/dafx05.pdf)):
  follow a path through the cost matrix, computing only the cells within `c` frames of the
  current position, and advance the reference, the query, or both at each step.

The offline versions run the online algorithm over the whole query, so an online run
produces exactly the same path as the offline one.

## Paper

This package implements SOA from *A Simple Alternative to Online Time Warping*.
The paper's experiments live in two companion repositories:

- [SimRealtimeMazurkaBenchmark (`vienna4x22` branch)](https://github.com/HMC-MIR/SimRealtimeMazurkaBenchmark/tree/vienna4x22):
  the Mazurka and Vienna 4×22 benchmarks.
- [PianoConcertoAccompaniment (`icassp` branch)](https://github.com/HMC-MIR/PianoConcertoAccompaniment/tree/icassp):
  the Piano Concerto benchmark and the real-time concerto accompaniment system.

The paper's systems correspond to these calls:

| Paper | Package |
|---|---|
| SOA | `run_offline_soa(reference, query)` or `SOA(reference)` |
| SOA-Mono | `run_offline_soa(reference, query, monotonic=True)` |
| SOA, free start (Section 2.2) | `run_offline_soa(reference, query, flexible_start=True)` |
| OLTW-Global | `run_offline_oltw(reference, query, c=None)` |

## Installation

```bash
pip install https://github.com/HMC-MIR/OnlineAlignment/releases/download/v0.4.0/online_alignment-0.4.0-py3-none-any.whl
```

Or from source, for development:

```bash
git clone https://github.com/HMC-MIR/OnlineAlignment.git
cd OnlineAlignment
pip install -e ".[dev]"      # or: conda env create -f environment-dev.yml
pytest
```

The only runtime dependencies are `numpy` and `numba`. Python 3.9+ is supported.

## Usage

Features are arrays of shape `(n_features, n_frames)`, e.g. chroma. Paths are integer
frame indices of shape `(2, n_path_points)`: `path[0]` is the query frame and `path[1]`
is the reference frame. Multiply by `hop_length / sample_rate` for seconds.

### Offline

```python
from online_alignment import run_offline_oltw, run_offline_soa

path = run_offline_oltw(reference, query, c=500)
path = run_offline_soa(reference, query, monotonic=True)
```

### Online

```python
from online_alignment import OLTW

oltw = OLTW(reference, c=500)
for frame in stream:              # frame shape (n_features,) or (n_features, 1)
    ref_frame = oltw.feed(frame)  # current estimate of the reference position
oltw.flush()                      # end of query: finish the path
path = oltw.path
```

`SOA` has the same interface. `align(query)` on either class runs `reset()`, `feed()`
for every frame, and `flush()`, then returns the path.

### Parameters

| | SOA | OLTW |
|---|---|---|
| `steps`, `weights` | Steps are always `[[1,1],[1,2],[2,1]]` as (query, reference) increments; weights default to `[1,1,2]` | DTW steps as (reference, query) increments. Default `[[1,0],[0,1],[1,1]]`, weights `[1,1,1]` |
| `cost_metric` | `"cosine"` (default), `"euclidean"`, `"manhattan"`, `"lpnorm"`, a function of two vectors, or a `CostMetric` | same |
| other | `normalize=True`: pick the best frame by path-length-normalized cost. `monotonic=False`: never move backwards (needs `normalize`). `flexible_start=False`: let the query start at any reference frame, tracking each path's start as in the paper (needs `normalize`) | `window_steps`: the three path transitions (reference-only, query-only, both), any order. `c=500`: band width, or `None` for unbounded. `max_run_count=3`: longest run of one transition |

### Memory

- **SOA** keeps three rows of the cost matrix (plus the same rows of path start frames with
  `flexible_start=True`): `O(reference_length)`. Each update is one vectorized pass over the
  reference.
- **OLTW** with a finite `c` keeps a ring buffer of about `c × c` cells, independent of
  both sequence lengths. With `c=None` every cell up to the current position is computed,
  which reproduces OLTW over the full DTW matrix, at `O(reference_length × query_length)`
  memory.

The warping path itself grows by one point per step.

## Package layout

```
online_alignment/
├── alignment/
│   ├── algs/       # Numba kernels shared by online and offline versions
│   ├── online/     # SOA, OLTW
│   └── offline/    # OfflineSOA, OfflineOLTW, run_offline_soa, run_offline_oltw
├── cost/           # cost metrics and the get_cost_metric registry
└── features/       # feature extractor base classes
```

`scripts/time_alignment.py` times the algorithms on random sequences, and
`scripts/time_soa_update.py` times one SOA update against a long reference on a single core.

## Releasing

Bump the version in `pyproject.toml` and `online_alignment/__init__.py` (a test checks they
match), add an entry to `CHANGELOG.md`, then push a tag:
`git tag v0.4.0 && git push --tags`. The publish workflow builds the wheel and creates a
GitHub release.
