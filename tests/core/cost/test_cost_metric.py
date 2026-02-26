"""Tests for CostMetric base class."""

# library imports
import numpy as np
import pytest

# custom imports
from online_alignment.cost import CostMetric


class TestCostMetric:
    """Test suite for CostMetric base class."""

    def test_init(self):
        """Test CostMetric initialization."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")
        assert metric.name == "test"
        assert callable(metric.v2v_cost)

    def test_vec2vec(self):
        """Test vec2vec method."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[3.0], [4.0]], dtype=np.float32)

        distance = metric.vec2vec(vec1, vec2)
        expected = np.sum((vec1 - vec2) ** 2)
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_mat2vec_generic(self):
        """Test generic mat2vec implementation."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        distances = metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        # First column should be identical to vec
        assert distances[0] == pytest.approx(0.0, abs=1e-6)

    def test_mat2mat_generic(self):
        """Test generic mat2mat implementation."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

        distances = metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        # Diagonal should be 0 (identical columns)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_mat2vec_consistency(self):
        """Test that mat2vec is consistent with vec2vec."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        # Compute using mat2vec
        distances_mat = metric.mat2vec(mat, vec)

        # Compute using vec2vec for each column
        distances_vec = np.array(
            [metric.vec2vec(mat[:, i : i + 1], vec) for i in range(mat.shape[1])]
        )

        np.testing.assert_array_almost_equal(distances_mat, distances_vec)

    def test_mat2mat_consistency(self):
        """Test that mat2mat is consistent with vec2vec."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

        # Compute using mat2mat
        distances_mat = metric.mat2mat(mat1, mat2)

        # Compute using vec2vec for each pair
        distances_vec = np.zeros((2, 2))
        for i in range(2):
            for j in range(2):
                distances_vec[i, j] = metric.vec2vec(mat1[:, i : i + 1], mat2[:, j : j + 1])

        np.testing.assert_array_almost_equal(distances_mat, distances_vec)

    def test_different_shapes(self):
        """Test with different input shapes."""

        def dummy_cost(v1, v2):
            return np.sum((v1 - v2) ** 2)

        metric = CostMetric(v2v_cost=dummy_cost, name="test")

        # Test mat2vec with different sizes
        mat = np.random.randn(10, 5).astype(np.float32)
        vec = np.random.randn(10, 1).astype(np.float32)
        distances = metric.mat2vec(mat, vec)
        assert distances.shape == (5,)

        # Test mat2mat with different sizes
        mat1 = np.random.randn(10, 5).astype(np.float32)
        mat2 = np.random.randn(10, 7).astype(np.float32)
        distances = metric.mat2mat(mat1, mat2)
        assert distances.shape == (5, 7)
