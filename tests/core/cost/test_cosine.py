"""Tests for cosine distance cost metric."""

# library imports
import numpy as np
import pytest

# custom imports
from online_alignment.cost import CosineDistance


class TestCosineDistance:
    """Test suite for CosineDistance class."""

    @pytest.fixture
    def cosine_metric(self):
        """Create a CosineDistance instance."""
        return CosineDistance()

    @pytest.fixture
    def normalized_vec1(self):
        """Create a normalized test vector."""
        vec = np.array([[1.0], [0.0]], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    @pytest.fixture
    def normalized_vec2(self):
        """Create another normalized test vector."""
        vec = np.array([[0.0], [1.0]], dtype=np.float32)
        return vec / np.linalg.norm(vec)

    @pytest.fixture
    def unnormalized_vec1(self):
        """Create an unnormalized test vector."""
        return np.array([[3.0], [4.0]], dtype=np.float32)

    @pytest.fixture
    def unnormalized_vec2(self):
        """Create another unnormalized test vector."""
        return np.array([[5.0], [12.0]], dtype=np.float32)

    def test_vec2vec_normalized(self, cosine_metric, normalized_vec1, normalized_vec2):
        """Test vec2vec with normalized vectors."""
        distance = cosine_metric.vec2vec(normalized_vec1, normalized_vec2, normalized=True)
        # Accept scalar or (1, 1) array
        if isinstance(distance, np.ndarray):
            assert distance.shape == (1, 1)
            distance = distance.item()
        assert isinstance(distance, (float, np.floating))
        assert distance == pytest.approx(1.0, abs=1e-6)  # Orthogonal vectors

    def test_vec2vec_unnormalized(self, cosine_metric, unnormalized_vec1, unnormalized_vec2):
        """Test vec2vec with unnormalized vectors."""
        distance = cosine_metric.vec2vec(unnormalized_vec1, unnormalized_vec2)
        # Both vectors point in similar direction, so distance should be small
        if isinstance(distance, np.ndarray):
            assert distance.shape == (1, 1)
            distance = distance.item()
        assert isinstance(distance, (float, np.floating))
        assert 0.0 <= distance <= 1.0

    def test_vec2vec_identical(self, cosine_metric):
        """Test vec2vec with identical vectors."""
        vec = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
        distance = cosine_metric.vec2vec(vec, vec)
        assert distance == pytest.approx(0.0, abs=1e-6)

    def test_vec2vec_opposite(self, cosine_metric):
        """Test vec2vec with opposite vectors."""
        vec1 = np.array([[1.0], [0.0]], dtype=np.float32)
        vec2 = np.array([[-1.0], [0.0]], dtype=np.float32)
        distance = cosine_metric.vec2vec(vec1, vec2)
        assert distance == pytest.approx(2.0, abs=1e-6)  # 1 - (-1) = 2

    def test_vec2vec_zero_vector(self, cosine_metric):
        """Test vec2vec with zero vector."""
        vec1 = np.array([[1.0], [0.0]], dtype=np.float32)
        vec2 = np.array([[0.0], [0.0]], dtype=np.float32)
        # Should handle division by zero gracefully
        distance = cosine_metric.vec2vec(vec1, vec2)
        if isinstance(distance, np.ndarray):
            assert distance.shape == (1, 1)
            distance = distance.item()
        assert isinstance(distance, (float, np.floating))
        assert np.isfinite(distance)

    def test_mat2vec_normalized(self, cosine_metric):
        """Test mat2vec with normalized matrices."""
        # Create normalized matrix and vector
        mat = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mat = mat / np.linalg.norm(mat, axis=0, keepdims=True)
        vec = np.array([[1.0], [0.0]], dtype=np.float32)
        vec = vec / np.linalg.norm(vec)

        distances = cosine_metric.mat2vec(mat, vec, normalized=True)
        assert distances.shape == (2,)
        assert distances[0] == pytest.approx(0.0, abs=1e-6)  # Same as first column
        assert distances[1] == pytest.approx(1.0, abs=1e-6)  # Orthogonal to second column

    def test_mat2vec_unnormalized(self, cosine_metric):
        """Test mat2vec with unnormalized matrices."""
        mat = np.array([[3.0, 5.0], [4.0, 12.0]], dtype=np.float32)
        vec = np.array([[3.0], [4.0]], dtype=np.float32)

        distances = cosine_metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        assert len(distances) == 2
        # First column should be identical to vec, so distance should be ~0
        assert distances[0] == pytest.approx(0.0, abs=1e-5)

    def test_mat2vec_shape(self, cosine_metric):
        """Test mat2vec output shape."""
        n_features = 5
        n_frames = 10
        mat = np.random.randn(n_features, n_frames).astype(np.float32)
        vec = np.random.randn(n_features, 1).astype(np.float32)

        distances = cosine_metric.mat2vec(mat, vec)
        assert distances.shape == (n_frames,)

    def test_mat2mat_normalized(self, cosine_metric):
        """Test mat2mat with normalized matrices."""
        mat1 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mat1 = mat1 / np.linalg.norm(mat1, axis=0, keepdims=True)
        mat2 = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
        mat2 = mat2 / np.linalg.norm(mat2, axis=0, keepdims=True)

        distances = cosine_metric.mat2mat(mat1, mat2, normalized=True)
        assert distances.shape == (2, 2)
        # Diagonal should be 0 (identical), off-diagonal should be 1 (orthogonal)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-6)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-6)
        assert distances[0, 1] == pytest.approx(1.0, abs=1e-6)
        assert distances[1, 0] == pytest.approx(1.0, abs=1e-6)

    def test_mat2mat_unnormalized(self, cosine_metric):
        """Test mat2mat with unnormalized matrices."""
        mat1 = np.array([[3.0, 5.0], [4.0, 12.0]], dtype=np.float32)
        mat2 = np.array([[3.0, 5.0], [4.0, 12.0]], dtype=np.float32)

        distances = cosine_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        # Diagonal should be ~0 (identical columns)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-5)

    def test_mat2mat_shape(self, cosine_metric):
        """Test mat2mat output shape."""
        n_features = 5
        n_frames1 = 10
        n_frames2 = 7
        mat1 = np.random.randn(n_features, n_frames1).astype(np.float32)
        mat2 = np.random.randn(n_features, n_frames2).astype(np.float32)

        distances = cosine_metric.mat2mat(mat1, mat2)
        assert distances.shape == (n_frames1, n_frames2)

    def test_mat2mat_symmetry(self, cosine_metric):
        """Test that mat2mat is symmetric for identical matrices."""
        mat = np.random.randn(5, 10).astype(np.float32)
        distances = cosine_metric.mat2mat(mat, mat)
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_name(self, cosine_metric):
        """Test that metric has correct name."""
        assert cosine_metric.name == "cosine"

    def test_v2v_cost_attribute(self, cosine_metric):
        """Test that v2v_cost is callable."""
        assert callable(cosine_metric.v2v_cost)
