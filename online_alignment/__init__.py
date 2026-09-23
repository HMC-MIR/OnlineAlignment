"""Online audio alignment package.

This package provides scalable and extensible code for online audio alignment
with implementations of SOA (Simple Online Alignment) and OLTW (Online Time Warping).

The package is organized into core modules:
    - online_alignment.alignment: Alignment algorithms (online and offline)
    - online_alignment.cost: Cost metrics for computing distances
    - online_alignment.features: Feature extraction (online and offline)
"""

# Submodules
from . import alignment, cost, features

# Alignment algorithms
from .alignment import AlignmentBase, OnlineAlignment
from .alignment.online import SOA, OLTW
from .alignment.offline import (
    OfflineAlignment,
    OfflineOLTW,
    OfflineSOA,
    run_offline_oltw,
    run_offline_soa,
)

# Cost metrics
from .cost import (
    CostMetric,
    CosineDistance,
    EuclideanDistance,
    LpNormDistance,
    ManhattanDistance,
    get_cost_metric,
)

# Feature extraction
from .features import (
    FeatureExtractor,
    OnlineFeatureExtractor,
    OfflineFeatureExtractor,
)

__version__ = "0.2.0"

__all__ = [
    # Submodules
    "alignment",
    "cost",
    "features",
    # Alignment
    "AlignmentBase",
    "OnlineAlignment",
    "SOA",
    "OLTW",
    "OfflineAlignment",
    "OfflineSOA",
    "run_offline_soa",
    "OfflineOLTW",
    "run_offline_oltw",
    # Cost metrics
    "CostMetric",
    "CosineDistance",
    "EuclideanDistance",
    "LpNormDistance",
    "ManhattanDistance",
    "get_cost_metric",
    # Features
    "FeatureExtractor",
    "OnlineFeatureExtractor",
    "OfflineFeatureExtractor",
]
