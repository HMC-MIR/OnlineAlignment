"""Tests for Lp norm distance cost metric."""

# library imports
import numpy as np
import pytest

# custom imports
from online_alignment.cost import LpNormDistance


class TestLpNormDistance:
    """Test suite for LpNormDistance class."""

    @pytest.fixture
    def vec1(self):
        """Create a test vector."""
        return np.array([[1.0], [2.0]], dtype=np.float32)

    @pytest.fixture
    def vec2(self):
        """Create another test vector."""
        return np.array([[4.0], [6.0]], dtype=np.float32)

    def test_vec2vec_p1(self, vec1, vec2):
        """Test vec2vec with p=1 (Manhattan distance)."""
        lp_metric = LpNormDistance(p=1)
        distance = lp_metric.vec2vec(vec1, vec2)
        # L1 norm: |4-1| + |6-2| = 3 + 4 = 7.0
        expected = abs(4 - 1) + abs(6 - 2)
        assert distance == pytest.approx(expected, abs=1e-5)

    def test_vec2vec_p2(self, vec1, vec2):
        """Test vec2vec with p=2 (Euclidean distance)."""
        lp_metric = LpNormDistance(p=2)
        distance = lp_metric.vec2vec(vec1, vec2)
        # L2 norm: sqrt((4-1)^2 + (6-2)^2) = sqrt(9 + 16) = 5.0
        expected = np.sqrt((4 - 1) ** 2 + (6 - 2) ** 2)
        assert distance == pytest.approx(expected, abs=1e-5)

    def test_vec2vec_p3(self, vec1, vec2):
        """Test vec2vec with p=3."""
        lp_metric = LpNormDistance(p=3)
        distance = lp_metric.vec2vec(vec1, vec2)
        # L3 norm: (|4-1|^3 + |6-2|^3)^(1/3) = (27 + 64)^(1/3) = 91^(1/3)
        expected = np.power(abs(4 - 1) ** 3 + abs(6 - 2) ** 3, 1.0 / 3)
        assert distance == pytest.approx(expected, abs=1e-5)

    def test_vec2vec_p4(self, vec1, vec2):
        """Test vec2vec with p=4."""
        lp_metric = LpNormDistance(p=4)
        distance = lp_metric.vec2vec(vec1, vec2)
        # L4 norm: (|4-1|^4 + |6-2|^4)^(1/4) = (81 + 256)^(1/4) = 337^(1/4)
        expected = np.power(abs(4 - 1) ** 4 + abs(6 - 2) ** 4, 1.0 / 4)
        assert distance == pytest.approx(expected, abs=1e-5)

    def test_vec2vec_identical(self, vec1):
        """Test vec2vec with identical vectors for different p values."""
        for p in [1, 2, 3, 4, 5]:
            lp_metric = LpNormDistance(p=p)
            distance = lp_metric.vec2vec(vec1, vec1)
            assert distance == pytest.approx(0.0, abs=1e-5)

    def test_vec2vec_zero_vector(self, vec1):
        """Test vec2vec with zero vector."""
        zero_vec = np.array([[0.0], [0.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)
        distance = lp_metric.vec2vec(vec1, zero_vec)
        expected = np.power(np.sum(np.abs(vec1) ** 2), 1.0 / 2)
        assert distance == pytest.approx(expected, abs=1e-5)

    def test_vec2vec_negative_values(self):
        """Test vec2vec with negative values."""
        vec1 = np.array([[-1.0], [-2.0]], dtype=np.float32)
        vec2 = np.array([[1.0], [2.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)
        distance = lp_metric.vec2vec(vec1, vec2)
        expected = np.sqrt((1.0 - (-1.0)) ** 2 + (2.0 - (-2.0)) ** 2)
        assert distance == pytest.approx(expected, abs=1e-5)

    def test_mat2vec_basic_p2(self):
        """Test basic mat2vec computation with p=2."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)

        distances = lp_metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        # First column should be identical to vec, so distance should be 0
        assert distances[0] == pytest.approx(0.0, abs=1e-5)
        # Second column distance should be sqrt((3-1)^2 + (4-2)^2) = sqrt(4 + 4) = sqrt(8)
        expected = np.sqrt((3 - 1) ** 2 + (4 - 2) ** 2)
        assert distances[1] == pytest.approx(expected, abs=1e-5)

    def test_mat2vec_basic_p1(self):
        """Test basic mat2vec computation with p=1."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=1)

        distances = lp_metric.mat2vec(mat, vec)
        assert distances.shape == (2,)
        # First column should be identical to vec, so distance should be 0
        assert distances[0] == pytest.approx(0.0, abs=1e-5)
        # Second column distance should be |3-1| + |4-2| = 2 + 2 = 4
        expected = abs(3 - 1) + abs(4 - 2)
        assert distances[1] == pytest.approx(expected, abs=1e-5)

    def test_mat2vec_shape(self):
        """Test mat2vec output shape."""
        n_features = 5
        n_frames = 10
        mat = np.random.randn(n_features, n_frames).astype(np.float32)
        vec = np.random.randn(n_features, 1).astype(np.float32)
        lp_metric = LpNormDistance(p=2)

        distances = lp_metric.mat2vec(mat, vec)
        assert distances.shape == (n_frames,)

    def test_mat2mat_basic_p2(self):
        """Test basic mat2mat computation with p=2."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)

        distances = lp_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        # Diagonal should be 0 (identical columns)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-5)

    def test_mat2mat_basic_p1(self):
        """Test basic mat2mat computation with p=1."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=1)

        distances = lp_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 2)
        # Diagonal should be 0 (identical columns)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)
        assert distances[1, 1] == pytest.approx(0.0, abs=1e-5)

    def test_mat2mat_shape(self):
        """Test mat2mat output shape."""
        n_features = 5
        n_frames1 = 10
        n_frames2 = 7
        mat1 = np.random.randn(n_features, n_frames1).astype(np.float32)
        mat2 = np.random.randn(n_features, n_frames2).astype(np.float32)
        lp_metric = LpNormDistance(p=2)

        distances = lp_metric.mat2mat(mat1, mat2)
        assert distances.shape == (n_frames1, n_frames2)

    def test_mat2mat_symmetry(self):
        """Test that mat2mat is symmetric for identical matrices."""
        mat = np.random.randn(5, 10).astype(np.float32)
        lp_metric = LpNormDistance(p=2)
        distances = lp_metric.mat2mat(mat, mat)
        # Should be symmetric
        np.testing.assert_array_almost_equal(distances, distances.T)

    def test_mat2mat_different_sizes(self):
        """Test mat2mat with different sized matrices."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[1.0], [2.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)

        distances = lp_metric.mat2mat(mat1, mat2)
        assert distances.shape == (2, 1)
        assert distances[0, 0] == pytest.approx(0.0, abs=1e-5)

    def test_mat2mat_calculation_p2(self):
        """Test mat2mat with known values for p=2."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)

        distances = lp_metric.mat2mat(mat1, mat2)
        # Distance between mat1[:, 0] and mat2[:, 0]: sqrt((1-4)^2 + (2-6)^2) = sqrt(9 + 16) = 5
        assert distances[0, 0] == pytest.approx(5.0, abs=1e-5)

    def test_mat2mat_calculation_p1(self):
        """Test mat2mat with known values for p=1."""
        mat1 = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
        mat2 = np.array([[4.0, 5.0], [6.0, 7.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=1)

        distances = lp_metric.mat2mat(mat1, mat2)
        # Distance between mat1[:, 0] and mat2[:, 0]: |1-4| + |2-6| = 3 + 4 = 7
        assert distances[0, 0] == pytest.approx(7.0, abs=1e-5)

    def test_name(self):
        """Test that metric has correct name."""
        lp_metric = LpNormDistance(p=3)
        assert lp_metric.name == "l3-norm"

    def test_v2v_cost_attribute(self):
        """Test that v2v_cost is callable."""
        lp_metric = LpNormDistance(p=2)
        assert callable(lp_metric.v2v_cost)

    def test_consistency_with_vec2vec(self):
        """Test that mat2vec is consistent with vec2vec."""
        mat = np.array([[1.0, 3.0], [2.0, 4.0]], dtype=np.float32)
        vec = np.array([[1.0], [2.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=2)

        # Compute using mat2vec
        distances_mat = lp_metric.mat2vec(mat, vec)

        # Compute using vec2vec for each column
        distances_vec = np.array(
            [lp_metric.vec2vec(mat[:, i : i + 1], vec) for i in range(mat.shape[1])]
        )

        np.testing.assert_array_almost_equal(distances_mat, distances_vec)

    def test_high_dimensional_vectors(self):
        """Test vec2vec with high-dimensional vectors."""
        n_features = 100
        vec1 = np.random.randn(n_features, 1).astype(np.float32)
        vec2 = np.random.randn(n_features, 1).astype(np.float32)
        lp_metric = LpNormDistance(p=2)

        distance = lp_metric.vec2vec(vec1, vec2)
        expected = np.power(np.sum(np.abs(vec1 - vec2) ** 2), 1.0 / 2)
        assert distance == pytest.approx(expected, abs=1e-4)

    def test_different_p_values(self):
        """Test that different p values produce different results."""
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[4.0], [6.0]], dtype=np.float32)

        distances = {}
        for p in [1, 2, 3, 4, 5]:
            lp_metric = LpNormDistance(p=p)
            distances[p] = lp_metric.vec2vec(vec1, vec2)

        # For the same vectors, different p values should give different distances
        # (except in special cases)
        assert distances[1] != pytest.approx(distances[2], abs=1e-5)
        assert distances[2] != pytest.approx(distances[3], abs=1e-5)

    def test_p_property(self):
        """Test that p property is correctly stored."""
        for p in [1, 2, 3, 4, 5]:
            lp_metric = LpNormDistance(p=p)
            assert lp_metric.p == p

    def test_large_p_value(self):
        """Test with a large p value (approaches Chebyshev/L∞ norm)."""
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[4.0], [6.0]], dtype=np.float32)
        lp_metric = LpNormDistance(p=10)

        distance = lp_metric.vec2vec(vec1, vec2)
        # For large p, the norm approaches max(|x_i - y_i|)
        max_diff = max(abs(4 - 1), abs(6 - 2))
        # Should be close to max_diff for large p
        assert distance >= max_diff * 0.9  # Should be at least close to max

    def test_odd_p_values(self):
        """Test with odd p values."""
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[4.0], [6.0]], dtype=np.float32)

        for p in [1, 3, 5, 7]:
            lp_metric = LpNormDistance(p=p)
            distance = lp_metric.vec2vec(vec1, vec2)
            expected = np.power(np.sum(np.abs(vec1 - vec2) ** p), 1.0 / p)
            assert distance == pytest.approx(expected, abs=1e-4)

    def test_even_p_values(self):
        """Test with even p values."""
        vec1 = np.array([[1.0], [2.0]], dtype=np.float32)
        vec2 = np.array([[4.0], [6.0]], dtype=np.float32)

        for p in [2, 4, 6, 8]:
            lp_metric = LpNormDistance(p=p)
            distance = lp_metric.vec2vec(vec1, vec2)
            expected = np.power(np.sum(np.abs(vec1 - vec2) ** p), 1.0 / p)
            assert distance == pytest.approx(expected, abs=1e-4)
