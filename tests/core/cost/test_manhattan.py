"""Tests for Manhattan distance cost metric."""

# library imports
import numpy as np
import pytest

# custom imports
from online_alignment.cost import ManhattanDistance


class TestManhattanDistance:
    """Test suite for ManhattanDistance class."""

    @pytest.fixture
    def manhattan_metric(self):
        """Create a ManhattanDistance instance."""
        return ManhattanDistance()

    @pytest.fixture
    def vec1(self):
        """Create a test vector."""
        return np.array([[1.0], [2.0]], dtype=np.float32)

    @pytest.fixture
    def vec2(self):
        """Create another test vector."""
        return np.array([[4.0], [6.0]], dtype=np.float32)

    def test_vec2vec_basic(self, manhattan_metric, vec1, vec2):
        """Test basic vec2vec computation."""
        distance = manhattan_metric.vec2vec(vec1, vec2)
        # Manhattan distance: |4-1| + |6-2| = 3 + 4 = 7.0
        expected = abs(4 - 1) + abs(6 - 2)
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_vec2vec_identical(self, manhattan_metric, vec1):
        """Test vec2vec with identical vectors."""
        distance = manhattan_metric.vec2vec(vec1, vec1)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_vec2vec_zero_vector(self, manhattan_metric, vec1):
        """Test vec2vec with zero vector."""
        zero_vec = np.array([[0.0], [0.0]], dtype=np.float32)
        distance = manhattan_metric.vec2vec(vec1, zero_vec)
        expected = abs(1.0) + abs(2.0)
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_vec2vec_negative_values(self, manhattan_metric):
        """Test vec2vec with negative values."""
        vec1 = np.array([[-1.0], [-2.0]], dtype=np.float32)
        vec2 = np.array([[1.0], [2.0]], dtype=np.float32)
        distance = manhattan_metric.vec2vec(vec1, vec2)
        expected = abs(1.0 - (-1.0)) + abs(2.0 - (-2.0))
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_vec2vec_mixed_signs(self, manhattan_metric):
        """Test vec2vec with mixed positive and negative values."""
        vec1 = np.array([[-1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[3.0], [-4.0]], dtype=np.float32)
        distance = manhattan_metric.vec2vec(vec1, vec2)
        expected = abs(3.0 - (-1.0)) + abs(-4.0 - 2.0)
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_mat2vec_basic(self, manhattan_metric):
        """Test basic mat2vec computation."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        distances = manhattan_metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        # First column should be identical to vec, so distance should be 0
        assert distances[0] == pytest.approx(0.0, abs=1e-6)
        # Second column distance should be |3-1| + |4-2| = 2 + 2 = 4
        expected = abs(3 - 1) + abs(4 - 2)
        assert distances[1] == pytest.approx(expected, abs=1e-6)

    def test_mat2vec_shape(self, manhattan_metric):
        """Test mat2vec output shape."""
        n_features = 5
        n_frames = 10
        mat = np.random.randn(n_features, n_frames).astype(np.float32)
        vec = np.random.randn(n_features, 1).astype(np.float32)

        distances = manhattan_metric.mat2vec(mat, vec)
        assert distances.shape == (n_frames,)

    def test_mat2mat_basic(self, manhattan_metric):
        """Test basic mat2mat computation."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

        distances = manhattan_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        # Diagonal should be 0 (identical columns)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_mat2mat_shape(self, manhattan_metric):
        """Test mat2mat output shape."""
        n_features = 5
        n_frames1 = 10
        n_frames2 = 7
        mat1 = np.random.randn(n_features, n_frames1).astype(np.float32)
        mat2 = np.random.randn(n_features, n_frames2).astype(np.float32)

        distances = manhattan_metric.mat2mat(mat1, mat2)
        assert distances.shape == (n_frames1, n_frames2)

    def test_mat2mat_symmetry(self, manhattan_metric):
        """Test that mat2mat is symmetric for identical matrices."""
        mat = np.random.randn(5, 10).astype(np.float32)
        distances = manhattan_metric.mat2mat(mat, mat)
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_mat2mat_different_sizes(self, manhattan_metric):
        """Test mat2mat with different sized matrices."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0], [2.0]], dtype=np.float32)

        distances = manhattan_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 1)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_mat2mat_calculation(self, manhattan_metric):
        """Test mat2mat with known values."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)

        distances = manhattan_metric.mat2mat(mat1, mat2)
        # Distance between mat1[:, 0] and mat2[:, 0]: |1-4| + |2-6| = 3 + 4 = 7
        assert distances[0, 0] == pytest.approx(7.0, abs=1e-6)
        # Distance between mat1[:, 0] and mat2[:, 1]: |1-5| + |2-7| = 4 + 5 = 9
        assert distances[0, 1] == pytest.approx(9.0, abs=1e-6)

    def test_name(self, manhattan_metric):
        """Test that metric has correct name."""
        assert manhattan_metric.name == "manhattan"

    def test_v2v_cost_attribute(self, manhattan_metric):
        """Test that v2v_cost is callable."""
        assert callable(manhattan_metric.v2v_cost)

    def test_consistency_with_vec2vec(self, manhattan_metric):
        """Test that mat2vec is consistent with vec2vec."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        # Compute using mat2vec
        distances_mat = manhattan_metric.mat2vec(mat, vec)

        # Compute using vec2vec for each column
        distances_vec = np.array(
            [manhattan_metric.vec2vec(mat[:, i : i + 1], vec) for i in range(mat.shape[1])]
        )

        np.testing.assert_array_almost_equal(distances_mat, distances_vec)

    def test_high_dimensional_vectors(self, manhattan_metric):
        """Test vec2vec with high-dimensional vectors."""
        n_features = 100
        vec1 = np.random.randn(n_features, 1).astype(np.float32)
        vec2 = np.random.randn(n_features, 1).astype(np.float32)

        distance = manhattan_metric.vec2vec(vec1, vec2)
        expected = np.sum(np.abs(vec1 - vec2))
        assert distance == pytest.approx(expected, abs=1e-4)

    def test_triangle_inequality(self, manhattan_metric):
        """Test that Manhattan distance satisfies triangle inequality."""
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[4.0], [6.0]], dtype=np.float32)
        vec3 = np.array([[7.0], [8.0]], dtype=np.float32)

        d12 = manhattan_metric.vec2vec(vec1, vec2)
        d23 = manhattan_metric.vec2vec(vec2, vec3)
        d13 = manhattan_metric.vec2vec(vec1, vec3)

        # Triangle inequality: d13 <= d12 + d23
        assert d13 <= d12 + d23 + 1e-6  # Add small epsilon for floating point
