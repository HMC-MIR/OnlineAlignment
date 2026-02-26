"""Tests for Euclidean distance cost metric."""

# library imports
import numpy as np
import pytest

# custom imports
from online_alignment.cost import EuclideanDistance


class TestEuclideanDistance:
    """Test suite for EuclideanDistance class."""

    @pytest.fixture
    def euclidean_metric(self):
        """Create a EuclideanDistance instance."""
        return EuclideanDistance()

    @pytest.fixture
    def vec1(self):
        """Create a test vector."""
        return np.array([[1.0], [2.0]], dtype=np.float32)

    @pytest.fixture
    def vec2(self):
        """Create another test vector."""
        return np.array([[4.0], [6.0]], dtype=np.float32)

    def test_vec2vec_basic(self, euclidean_metric, vec1, vec2):
        """Test basic vec2vec computation."""
        distance = euclidean_metric.vec2vec(vec1, vec2)
        # Returns squared distance: (4-1)^2 + (6-2)^2 = 9 + 16 = 25.0
        expected = np.sqrt((4 - 1) ** 2 + (6 - 2) ** 2)
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_vec2vec_identical(self, euclidean_metric, vec1):
        """Test vec2vec with identical vectors."""
        distance = euclidean_metric.vec2vec(vec1, vec1)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_vec2vec_zero_vector(self, euclidean_metric, vec1):
        """Test vec2vec with zero vector."""
        zero_vec = np.array([[0.0], [0.0]], dtype=np.float32)
        distance = euclidean_metric.vec2vec(vec1, zero_vec)
        expected = np.sqrt(np.sum(vec1**2))
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_vec2vec_negative_values(self, euclidean_metric):
        """Test vec2vec with negative values."""
        vec1 = np.array([[-1.0], [-2.0]], dtype=np.float32)
        vec2 = np.array([[1.0], [2.0]], dtype=np.float32)
        distance = euclidean_metric.vec2vec(vec1, vec2)
        expected = np.sqrt(np.sum((vec1 - vec2) ** 2))
        assert distance == pytest.approx(expected, abs=1e-6)

    def test_mat2vec_basic(self, euclidean_metric):
        """Test basic mat2vec computation."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        distances = euclidean_metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        # First column should be identical to vec, so distance should be 0
        assert distances[0] == pytest.approx(0.0, abs=1e-6)
        # Second column distance should be sqrt((3-1)^2 + (4-2)^2) = sqrt(4 + 4) = sqrt(8)
        expected = np.sqrt(np.sum((3 - 1) ** 2 + (4 - 2) ** 2))
        assert distances[1] == pytest.approx(expected, abs=1e-6)

    def test_mat2vec_shape(self, euclidean_metric):
        """Test mat2vec output shape."""
        n_features = 5
        n_frames = 10
        mat = np.random.randn(n_features, n_frames).astype(np.float32)
        vec = np.random.randn(n_features, 1).astype(np.float32)

        distances = euclidean_metric.mat2vec(mat, vec)
        assert distances.shape == (n_frames,)

    def test_mat2mat_basic(self, euclidean_metric):
        """Test basic mat2mat computation."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)

        distances = euclidean_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        # Diagonal should be 0 (identical columns)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-6)

    def test_mat2mat_shape(self, euclidean_metric):
        """Test mat2mat output shape."""
        n_features = 5
        n_frames1 = 10
        n_frames2 = 7
        mat1 = np.random.randn(n_features, n_frames1).astype(np.float32)
        mat2 = np.random.randn(n_features, n_frames2).astype(np.float32)

        distances = euclidean_metric.mat2mat(mat1, mat2)
        assert distances.shape == (n_frames1, n_frames2)

    def test_mat2mat_symmetry(self, euclidean_metric):
        """Test that mat2mat is symmetric for identical matrices."""
        mat = np.random.randn(5, 10).astype(np.float32)
        distances = euclidean_metric.mat2mat(mat, mat)
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_mat2mat_different_sizes(self, euclidean_metric):
        """Test mat2mat with different sized matrices."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0], [2.0]], dtype=np.float32)

        distances = euclidean_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 1)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)

    def test_name(self, euclidean_metric):
        """Test that metric has correct name."""
        assert euclidean_metric.name == "euclidean"

    def test_v2v_cost_attribute(self, euclidean_metric):
        """Test that v2v_cost is callable."""
        assert callable(euclidean_metric.v2v_cost)

    def test_consistency_with_vec2vec(self, euclidean_metric):
        """Test that mat2vec is consistent with vec2vec."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)

        # Compute using mat2vec
        distances_mat = euclidean_metric.mat2vec(mat, vec)

        # Compute using vec2vec for each column
        distances_vec = np.array(
            [euclidean_metric.vec2vec(mat[:, i : i + 1], vec) for i in range(mat.shape[1])]
        )

        np.testing.assert_array_almost_equal(distances_mat, distances_vec)
