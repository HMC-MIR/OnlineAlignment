"""Online feature extraction for streaming signals."""

# standard imports
from abc import abstractmethod
from typing import Iterator, Optional

# library imports
import numpy as np

from .base import FeatureExtractor


class OnlineFeatureExtractor(FeatureExtractor):
    """Base class for online feature extraction."""

    def __init__(self, frame_size: int, hop_size: int, n_features: Optional[int] = None):
        super().__init__(n_features=n_features)
        self.frame_size = frame_size
        self.hop_size = hop_size
        self._buffer = np.zeros(frame_size, dtype=np.float32)
        self._buffer_idx = 0

    def reset(self):
        """Reset internal state for new signal."""
        self._buffer.fill(0)
        self._buffer_idx = 0

    @abstractmethod
    def feed(self, samples: np.ndarray) -> Iterator[np.ndarray]:
        """Feed new samples and yield features for complete frames."""
        pass

    @abstractmethod
    def flush(self) -> Iterator[np.ndarray]:
        """Flush remaining buffered samples and yield final features."""
        pass
