"""Base class for feature extractors."""

# standard imports
from abc import ABC, abstractmethod

# library imports
import numpy as np


class FeatureExtractor(ABC):
    """Base class for feature extraction from audio signals."""

    def __init__(self, n_features: int | None = None):
        self.n_features = n_features

    @abstractmethod
    def extract(self, signal: np.ndarray) -> np.ndarray:
        """Extract features from a signal. Returns shape (n_features, n_frames)."""
        pass

    @abstractmethod
    def extract_frame(self, signal_frame: np.ndarray) -> np.ndarray:
        """Extract features from a single frame. Returns shape (n_features, 1)."""
        pass
