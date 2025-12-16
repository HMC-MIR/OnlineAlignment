"""Online audio alignment package.

This package provides scalable and extensible code for online audio alignment
with implementations of NOA (Naive Online Alignment) and OLTW (Online Time Warping).
"""

from alignment.online_alignment import OnlineAlignment
from alignment.noa import NOA
from alignment.oltw import OLTW
from core.cost import CostMetric, CosineDistance, EuclideanDistance
from utils.backend import handle_cost

__version__ = "0.1.0"

__all__ = [
    "OnlineAlignment",
    "NOA",
    "OLTW",
    "CostMetric",
    "CosineDistance",
    "EuclideanDistance",
    "handle_cost",
]
