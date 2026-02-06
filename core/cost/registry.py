"""Cost metric registry and factory functions."""

# standard imports
from typing import Callable, Union
import inspect

# library imports
import numpy as np

# custom imports
from .cost_metric import CostMetric
from .cosine import CosineDistance
from .euclidean import EuclideanDistance
from .manhattan import ManhattanDistance
from .lpnorm import LpNormDistance


def _create_lpnorm_metric(**kwargs) -> CostMetric:
    """Create an LpNormDistance metric with the specified p value.

    Args:
        **kwargs: Keyword arguments. Accepts:
            - p (int): The norm order (defaults to 2).

    Returns:
        CostMetric: LpNormDistance instance with the specified p value.

    Raises:
        ValueError: If invalid keyword arguments are provided.
    """
    if len(kwargs) == 0:
        return LpNormDistance(p=2)  # Default p value
    elif len(kwargs) == 1 and "p" in kwargs:
        p = kwargs["p"]
        if not isinstance(p, int):
            raise ValueError(f"lpnorm parameter 'p' must be an integer, got {type(p).__name__}")
        return LpNormDistance(p=p)
    else:
        invalid_keys = [k for k in kwargs.keys() if k != "p"]
        if invalid_keys:
            raise ValueError(
                f"lpnorm only accepts 'p' as a keyword argument, "
                f"got unexpected keys: {invalid_keys}"
            )
        raise ValueError(f"lpnorm accepts at most 1 keyword argument (p), got {len(kwargs)}")


# Registry of supported cost metrics
# Values can be CostMetric instances or callables that return CostMetric instances
_COST_REGISTRY: dict[str, Union[CostMetric, Callable]] = {
    "cosine": CosineDistance(),
    "euclidean": EuclideanDistance(),
    "manhattan": ManhattanDistance(),
    "lpnorm": _create_lpnorm_metric,
}


def get_cost_metric(
    cost_metric: str | Callable | CostMetric,
    **kwargs,
) -> CostMetric:
    """Get a CostMetric instance from various input types.

    This function provides a convenient way to create CostMetric instances
    from string names, callable functions, or existing CostMetric instances.

    Args:
        cost_metric: Can be:
            - A string name ("cosine", "euclidean")
            - A callable function that computes vector-to-vector distance
            - An existing CostMetric instance
        **kwargs: Additional keyword arguments to pass to the cost metric constructor.
            For "lpnorm", use `p` to specify the norm order (defaults to 2).

    Returns:
        CostMetric instance ready to use.

    Raises:
        ValueError: If string name is not in the registry.
        TypeError: If input type is not supported.

    Examples:
        >>> metric = get_cost_metric("cosine")
        >>> metric = get_cost_metric(CosineDistance())
        >>> metric = get_cost_metric(my_custom_distance_function)
        >>> metric = get_cost_metric("lpnorm")  # Uses default p=2
        >>> metric = get_cost_metric("lpnorm", p=3)  # Uses p=3
    """
    if isinstance(cost_metric, CostMetric):
        return cost_metric
    if isinstance(cost_metric, str):
        return _get_cost_from_str(cost_metric, **kwargs)
    if isinstance(cost_metric, Callable):
        return _get_cost_from_callable(cost_metric)
    raise TypeError(
        f'Variable "cost_metric" must be str, Callable, or CostMetric, '
        f"got {type(cost_metric).__name__}"
    )


def _get_cost_from_str(cost_metric: str, **kwargs) -> CostMetric:
    """Get cost metric from string name.

    Args:
        cost_metric: String name of the cost metric.
        **kwargs: Additional keyword arguments to pass to the cost metric constructor.
            For "lpnorm", use `p` to specify the norm order (defaults to 2 if not provided).

    Returns:
        CostMetric: Desired cost metric instance.

    Raises:
        ValueError: If cost metric name is not supported or invalid arguments provided.
    """
    if cost_metric not in _COST_REGISTRY:
        raise ValueError(
            f"Input cost metric '{cost_metric}' not yet supported. "
            f"Supported metrics: {list[str](_COST_REGISTRY.keys())}"
        )

    registry_value = _COST_REGISTRY[cost_metric]

    # If it's already a CostMetric instance, return it directly
    if isinstance(registry_value, CostMetric):
        if kwargs:
            raise ValueError(
                f"Cost metric '{cost_metric}' does not accept arguments, "
                f"got unexpected kwargs: {list(kwargs.keys())}"
            )
        return registry_value

    # If it's a callable factory function, call it with kwargs
    if isinstance(registry_value, Callable):
        return registry_value(**kwargs)

    # Fallback (shouldn't reach here with current registry)
    raise TypeError(
        f"Registry value for '{cost_metric}' must be CostMetric or Callable, "
        f"got {type(registry_value).__name__}"
    )


def _get_cost_from_callable(cost_metric: Callable) -> CostMetric:
    """Create cost metric from a callable distance function.

    Args:
        cost_metric: Vector-to-vector distance function. Must accept two numpy arrays
            and return a scalar distance value.

    Returns:
        CostMetric: Custom-built cost metric based on the provided function.

    Raises:
        ValueError: If the callable doesn't appear to be a valid distance function.
    """
    # Basic validation: check if it's callable (already done by isinstance check)
    # Try to get function name
    func_name = getattr(cost_metric, "__name__", "unknown")

    # Optional: Try a test call with dummy arrays to validate signature
    try:
        sig = inspect.signature(cost_metric)
        n_params = len(sig.parameters)
        if n_params < 2:
            raise ValueError(
                f"Cost function '{func_name}' must accept at least 2 parameters "
                f"(two feature vectors), got {n_params}"
            )
    except (ValueError, TypeError):
        # If signature inspection fails, try a test call
        try:
            test_vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
            test_vec2 = np.array([[3.0], [4.0]], dtype=np.float32)
            result = cost_metric(test_vec1, test_vec2)
            # Check that result is a scalar or can be converted to one
            if not np.isscalar(result) and result.shape != ():
                raise ValueError(
                    f"Cost function '{func_name}' must return a scalar value, "
                    f"got shape {result.shape}"
                )
        except Exception as e:
            if isinstance(e, ValueError):
                raise
            # If test call fails, warn but don't fail (might be due to numba compilation, etc.)
            pass

    return CostMetric(v2v_cost=cost_metric, name=func_name)
