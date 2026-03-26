"""Kalman fusion engine for embedding spaces.

This module provides the core fusion algorithms for combining multiple
specialist embeddings using Kalman filtering. The main entry point is
the Panoramix class, which orchestrates routing and fusion.

The mathematical foundation is the Kalman filter, treating each specialist
as providing a noisy measurement of the true semantic state. Key innovations:
1. Diagonal covariance approximation for O(d) complexity
2. Sequential updates sorted by certainty for numerical stability
3. Support for arbitrary numbers of specialists
"""

from .kalman_fuser import (
    kalman_fuse_diagonal,
    fuse_with_prior,
    weighted_average_baseline,
    kalman_fuse_diagonal_batch,
    kalman_fuse_diagonal_ensemble_batch,
)
from .fuser import Panoramix
from .covariance import (
    CovarianceEstimator,
    DiagonalCovariance,
    EmpiricalCovariance,
    ConstantCovariance,
    ScalarCovariance,
    DistanceBasedCovariance,
    KNNBasedCovariance,
    DomainBasedCovariance,
)

__all__ = [
    "kalman_fuse_diagonal",
    "fuse_with_prior",
    "weighted_average_baseline",
    "kalman_fuse_diagonal_batch",
    "kalman_fuse_diagonal_ensemble_batch",
    "Panoramix",
    "CovarianceEstimator",
    "DiagonalCovariance",
    "EmpiricalCovariance",
    "ConstantCovariance",
    "ScalarCovariance",
    "DistanceBasedCovariance",
    "KNNBasedCovariance",
    "DomainBasedCovariance",
]
