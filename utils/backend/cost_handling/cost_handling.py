"""API for checking inputs for cost metric"""

# standard imports
from typing import Callable

# custom imports
from core.cost import CostMetric, CosineDistance, EuclideanDistance

# Registry of supported cost metrics
_SUPPORTED_COST_METRICS: dict[str, CostMetric] = {
    "cosine": CosineDistance(),
    "euclidean": EuclideanDistance(),
}


def handle_cost(
    cost_metric: str | Callable | CostMetric,
) -> CostMetric:
    """Set cost function based on input cost metric.

    Args:
        cost_metric (str | Callable | CostMetric): input cost metric. Use string for standard cost metrics (cosine/euclidean) and function for custom cost functions.
    """
    # check cost metric type and direct to correct cost function setting  method
    if isinstance(cost_metric, CostMetric):
        return cost_metric  # Nothing to do since we already have a custom defined cost metric
    elif isinstance(cost_metric, str):
        return _handle_cost_str(cost_metric)
    elif isinstance(cost_metric, Callable):
        return _handle_cost_func(cost_metric)
    else:
        raise TypeError(
            f'Variable "cost_metric" must be str, Callable, or CostMetric, got {type(cost_metric).__name__}'
        )


def _handle_cost_str(cost_metric: str) -> CostMetric:
    """Set cost function based on string cost metric input.

    Args:
        cost_metric (str): string description of the cost metric.

    Returns:
        CostMetric: Desired cost metric.
    """
    if cost_metric not in _SUPPORTED_COST_METRICS:
        raise ValueError(
            f"Input cost metric '{cost_metric}' not yet supported. "
            f"Supported metrics: {list(_SUPPORTED_COST_METRICS.keys())}"
        )
    return _SUPPORTED_COST_METRICS[cost_metric]


def _handle_cost_func(cost_metric: Callable) -> CostMetric:
    """Set cost function based on v2v distance function input.

    Args:
        cost_metric (Callable): vector to vector distance function

    Returns:
        CostMetric: Custom-built cost metrics based on v2v distance function input.
    """
    # TODO: check cost_metric is a valid v2v cost function
    return CostMetric(v2v_cost=cost_metric, name=cost_metric.__name__)
