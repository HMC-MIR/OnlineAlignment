"""Offline/batch feature extraction."""

# standard imports
from abc import abstractmethod
from typing import Optional

# library imports
import numpy as np

from .base import FeatureExtractor


class OfflineFeatureExtractor(FeatureExtractor):
    """Base class for offline feature extraction."""

    @abstractmethod
    def extract(
        self,
        signal: np.ndarray,
        frame_size: Optional[int] = None,
        hop_size: Optional[int] = None,
    ) -> np.ndarray:
        """Extract features from a complete signal. Returns shape (n_features, n_frames)."""
        pass
