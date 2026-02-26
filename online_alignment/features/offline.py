"""Offline/batch feature extraction."""

# standard imports
from abc import abstractmethod

# library imports
import numpy as np

from .base import FeatureExtractor


class OfflineFeatureExtractor(FeatureExtractor):
    """Base class for offline feature extraction."""

    @abstractmethod
    def extract(
        self,
        signal: np.ndarray,
        frame_size: int | None = None,
        hop_size: int | None = None,
    ) -> np.ndarray:
        """Extract features from a complete signal. Returns shape (n_features, n_frames)."""
        pass
