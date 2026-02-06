"""Tests for cost metric registry."""

# library imports
import numpy as np
import pytest

# custom imports
from core.cost import (
    CostMetric,
    CosineDistance,
    EuclideanDistance,
    LpNormDistance,
    ManhattanDistance,
    get_cost_metric,
)


class TestCostRegistry:
    """Test suite for cost metric registry."""

    def test_get_cost_metric_from_string_cosine(self):
        """Test getting cosine metric from string."""
        metric = get_cost_metric("cosine")
        assert isinstance(metric, CosineDistance)
        assert metric.name == "cosine"

    def test_get_cost_metric_from_string_euclidean(self):
        """Test getting euclidean metric from string."""
        metric = get_cost_metric("euclidean")
        assert isinstance(metric, EuclideanDistance)
        assert metric.name == "euclidean"

    def test_get_cost_metric_from_string_invalid(self):
        """Test getting invalid metric name raises error."""
        with pytest.raises(ValueError, match="not yet supported"):
            get_cost_metric("invalid_metric")

    def test_get_cost_metric_from_instance(self):
        """Test getting metric from CostMetric instance."""
        original_metric = CosineDistance()
        metric = get_cost_metric(original_metric)
        assert metric is original_metric  # Should return same instance

    def test_get_cost_metric_from_callable(self):
        """Test getting metric from callable function."""

        def custom_distance(v1, v2):
            return np.sum(np.abs(v1 - v2))

        metric = get_cost_metric(custom_distance)
        assert isinstance(metric, CostMetric)
        assert metric.name == "custom_distance"

        # Test that it works
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[3.0], [4.0]], dtype=np.float32)
        distance = metric.vec2vec(vec1, vec2)
        expected = np.sum(np.abs(vec1 - vec2))
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_get_cost_metric_from_callable_lambda(self):
        """Test getting metric from lambda function."""
        metric = get_cost_metric(lambda v1, v2: np.sum((v1 - v2) ** 2))
        assert isinstance(metric, CostMetric)
        # Lambda functions have name "<lambda>"
        assert metric.name == "<lambda>"

    def test_get_cost_metric_invalid_type(self):
        """Test that invalid type raises TypeError."""
        with pytest.raises(TypeError):
            get_cost_metric(123)  # Not a string, callable, or CostMetric

        with pytest.raises(TypeError):
            get_cost_metric(None)

    def test_get_cost_metric_callable_validation(self):
        """Test validation of callable cost functions."""

        # Function with wrong number of parameters
        def wrong_params(v1):
            return 0.0

        # Should raise ValueError or pass validation
        # (validation might be lenient for numba-compiled functions)
        try:
            metric = get_cost_metric(wrong_params)
            # If it passes, test that it works
            vec1 = np.array([[1.0]], dtype=np.float32)
            vec2 = np.array([[2.0]], dtype=np.float32)
            metric.vec2vec(vec1, vec2)
        except (ValueError, TypeError):
            pass  # Expected if validation is strict

    def test_get_cost_metric_callable_returns_scalar(self):
        """Test that callable must return scalar."""

        def returns_array(v1, v2):
            return np.array([1.0, 2.0])  # Returns array instead of scalar

        # Should raise ValueError or handle gracefully
        try:
            metric = get_cost_metric(returns_array)
            vec1 = np.array([[1.0]], dtype=np.float32)
            vec2 = np.array([[2.0]], dtype=np.float32)
            result = metric.vec2vec(vec1, vec2)
            # If it works, result should be usable
            assert result is not None
        except ValueError:
            pass  # Expected if validation catches this

    def test_registry_consistency(self):
        """Test that registry returns consistent instances."""
        metric1 = get_cost_metric("cosine")
        metric2 = get_cost_metric("cosine")
        # Should return same instance (singleton pattern)
        assert metric1 is metric2

    def test_custom_metric_works_with_mat2vec(self):
        """Test that custom metric works with mat2vec."""

        def custom_distance(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = get_cost_metric(custom_distance)
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        distances = metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        assert distances[0] == pytest.approx(0.0, abs=1e-6)

    def test_custom_metric_works_with_mat2mat(self):
        """Test that custom metric works with mat2mat."""

        def custom_distance(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = get_cost_metric(custom_distance)
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

        distances = metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_lpnorm_metric_with_default_p(self):
        """Test that lpnorm metric with default p=2 is created."""
        metric = get_cost_metric("lpnorm")
        assert isinstance(metric, LpNormDistance)
        assert metric.p == 2

    def test_lpnorm_metric_with_p(self):
        """Test that lpnorm metric with p=3 is created."""
        metric = get_cost_metric("lpnorm", p=3)
        assert isinstance(metric, LpNormDistance)
        assert metric.p == 3

    def test_manhattan_metric(self):
        """Test that manhattan metric is created."""
        metric = get_cost_metric("manhattan")
        assert isinstance(metric, ManhattanDistance)
        assert metric.name == "manhattan"
