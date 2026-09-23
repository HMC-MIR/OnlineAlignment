# Tests

This directory contains unit tests for the online_alignment package.

## Running Tests

Install test dependencies (librosa is used as a reference DTW implementation):

```bash
pip install -e ".[dev]"
```

Run all tests:

```bash
pytest
```

Run tests for a specific module:

```bash
pytest tests/core/cost/
```

Run with coverage (requires `pytest-cov`):

```bash
pytest --cov=online_alignment --cov-report=html
```

## Test Structure

Tests are organized to mirror the package structure:

- `tests/core/alignment/` - Alignment algorithms
  - `test_soa.py` - Online and offline SOA vs. the original dense-matrix implementation
  - `test_oltw.py` - Online and offline OLTW vs. full-matrix (librosa) and dense banded
    reference implementations
- `tests/core/cost/` - Cost metrics
  - `test_cosine.py`, `test_euclidean.py`, `test_manhattan.py`, `test_lpnorm.py` - Metrics
  - `test_cost_metric.py` - Base CostMetric class
  - `test_registry.py` - Cost metric registry

## Writing Tests

Follow pytest conventions:

- Test files should start with `test_`
- Test classes should start with `Test`
- Test functions should start with `test_`
- Use fixtures for common test data
