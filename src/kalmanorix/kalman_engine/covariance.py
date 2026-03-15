"""Uncertainty estimation for embedding models.

This module provides methods for estimating the covariance matrix R that
represents a specialist model's uncertainty for a given input. The covariance
is a critical component of the Kalman fusion - it determines how much to
trust each specialist's embedding.

Key Concepts:
-------------
1. Covariance represents uncertainty: Higher values = less certain
2. We focus on diagonal covariance for computational efficiency
3. Uncertainty can be:
   - Fixed per model (global confidence)
   - Input-dependent (query difficulty)
   - Domain-dependent (distance from training data)

Mathematical Foundation:
-----------------------
For a model f that produces embeddings z = f(x), we want R(x) such that:
    z_true = z + ε, where ε ~ N(0, R(x))

That is, the true semantic state is the model's output plus zero-mean
Gaussian noise with covariance R(x). This is the measurement model assumed
by the Kalman filter.

Estimation Methods:
------------------
1. Empirical: Compute variance over validation set
   R = diag(Var[{f(x_i) for x_i in validation set}])

2. Distance-based: Scale by distance to training data
   R = R_base * exp(-d(x, X_train))

3. Model-based: Train a separate model to predict uncertainty
   R = g(x) where g is a small neural network

4. Ensemble: Use multiple forward passes with dropout
   R = Var[{f_dropout(x) for _ in range(n_samples)}]
"""

from typing import List, Optional, Callable, Union, Dict
from abc import ABC, abstractmethod
import logging

import numpy as np

logger = logging.getLogger(__name__)


class CovarianceEstimator(ABC):
    """Base class for uncertainty estimation strategies."""

    @abstractmethod
    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Estimate diagonal covariance for a model on given text.

        Args:
            model: Function that takes text and returns embedding
            text: Input text to estimate uncertainty for
            domain_hint: Optional domain tag to inform estimation

        Returns:
            covariance: Diagonal covariance vector (d,)
        """


class EmpiricalCovariance(CovarianceEstimator):
    """Fixed covariance estimated from validation set.

    This is the simplest approach: compute the variance of embeddings
    on a validation set and use that as a fixed uncertainty for all inputs.

    The assumption is that model uncertainty is constant across its domain.
    This is obviously false but serves as a baseline and works reasonably
    if the model's domain is narrow.

    Mathematical Form:
        R = diag(1/(n-1) * Σ_i (z_i - μ)²)

    Where:
        z_i are embeddings of validation samples
        μ is the mean embedding
    """

    def __init__(self, validation_embeddings: np.ndarray, epsilon: float = 1e-8):
        """Initialise with pre-computed validation embeddings.

        Args:
            validation_embeddings: Array of shape (n_samples, d) containing
                                  embeddings of validation texts.
            epsilon: Small constant added to covariance for numerical stability.
        """
        if validation_embeddings.ndim != 2:
            raise ValueError(
                f"Expected 2D array (n_samples, d), got shape {validation_embeddings.shape}"
            )

        self.dimension = validation_embeddings.shape[1]
        self.epsilon = epsilon

        # Compute empirical variance
        self.fixed_covariance = np.var(validation_embeddings, axis=0, ddof=1)
        self.fixed_covariance = np.maximum(self.fixed_covariance, epsilon)

        logger.info(
            "Empirical covariance computed: mean=%.6f, min=%.6f, max=%.6f",
            np.mean(self.fixed_covariance),
            np.min(self.fixed_covariance),
            np.max(self.fixed_covariance),
        )

    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Return fixed covariance regardless of input."""
        return self.fixed_covariance.copy()


class DistanceBasedCovariance(CovarianceEstimator):
    """Scale covariance by distance to training data.

    This method assumes that uncertainty increases as we move away from
    the model's training distribution. We approximate this by measuring
    distance to a reference set of training texts.

    Mathematical Form:
        R(x) = R_base * (1 + α * d(x, X_ref))

    Where:
        R_base is the empirical covariance on validation set
        d(x, X_ref) is distance to nearest reference point
        α is a scaling factor

    This implements the intuition that models are less certain on inputs
    unlike those seen during training.
    """

    def __init__(
        self,
        base_estimator: CovarianceEstimator,
        reference_texts: List[str],
        reference_embeddings: np.ndarray,
        alpha: float = 1.0,
        distance_metric: str = "cosine",
    ):
        """Initialise with reference set for distance computation.

        Args:
            base_estimator: Base uncertainty estimator (e.g., EmpiricalCovariance)
            reference_texts: List of texts representing training distribution
            reference_embeddings: Pre-computed embeddings of reference_texts,
                                 shape (n_ref, d)
            alpha: Scaling factor for distance (uncertainty = base * (1 + α*d))
            distance_metric: 'cosine' or 'euclidean'
        """
        self.base_estimator = base_estimator
        self.reference_texts = reference_texts
        self.reference_embeddings = reference_embeddings
        self.alpha = alpha
        self.distance_metric = distance_metric

        # Pre-normalise for cosine distance
        if distance_metric == "cosine":
            self.reference_embeddings = self.reference_embeddings / (
                np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True) + 1e-8
            )

    def _compute_distance(self, embedding: np.ndarray) -> float:
        """Compute minimum distance to reference set."""
        # Normalise for cosine
        if self.distance_metric == "cosine":
            embedding = embedding / (np.linalg.norm(embedding) + 1e-8)
            # Cosine distance = 1 - cosine similarity
            similarities = self.reference_embeddings @ embedding
            return 1.0 - np.max(similarities)
        # euclidean
        distances = np.linalg.norm(self.reference_embeddings - embedding, axis=1)
        return np.min(distances)

    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Estimate covariance scaled by distance to reference."""
        # Get base covariance
        base_cov = self.base_estimator.estimate(model, text, domain_hint)

        # Compute embedding and distance
        embedding = model(text)
        distance = self._compute_distance(embedding)

        # Scale covariance
        scaled_cov = base_cov * (1.0 + self.alpha * distance)

        return scaled_cov


class ConstantCovariance(CovarianceEstimator):
    """Fixed diagonal covariance specified directly.

    This is useful when the covariance is known a priori (e.g., from prior
    calibration) or when a simple constant uncertainty is desired.
    """

    def __init__(self, diagonal: np.ndarray, epsilon: float = 1e-8):
        """Initialise with fixed diagonal covariance.

        Args:
            diagonal: Diagonal covariance vector (d,)
            epsilon: Small constant for numerical stability (ensures positivity)
        """
        self.diagonal = np.asarray(diagonal).flatten()
        self.dimension = len(self.diagonal)
        self.epsilon = epsilon

        # Validate
        if np.any(self.diagonal < 0):
            raise ValueError(
                f"Covariance diagonal must be non-negative, got {self.diagonal}"
            )

        # Ensure positivity
        self.diagonal = np.maximum(self.diagonal, epsilon)

    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Return fixed diagonal covariance."""
        return self.diagonal.copy()


class ScalarCovariance(CovarianceEstimator):
    """Convert scalar uncertainty (sigma²) to diagonal covariance.

    This adapter takes a scalar variance value and replicates it across all
    dimensions, producing a diagonal covariance matrix sigma² * I.

    This is useful for integrating legacy uncertainty models that only provide
    a single variance estimate per query.
    """

    def __init__(
        self, sigma2: Union[float, Callable[[str], float]], epsilon: float = 1e-8
    ):
        """Initialise with scalar variance.

        Args:
            sigma2: Either a constant float or a callable taking text and returning variance
            epsilon: Small constant for numerical stability
        """
        self.sigma2 = sigma2
        self.epsilon = epsilon

    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Compute scalar variance and expand to diagonal."""
        if callable(self.sigma2):
            var = float(self.sigma2(text))
        else:
            var = float(self.sigma2)

        var = max(var, self.epsilon)
        d = model(text).shape[0]
        return np.full(d, var, dtype=np.float64)


class KNNBasedCovariance(CovarianceEstimator):
    """Estimate covariance using k‑nearest neighbors in embedding space.

    Given a reference set of embeddings with known diagonal covariances,
    estimate the covariance of a new embedding by averaging the covariances
    of its k nearest neighbors (weighted by inverse distance).

    This is a non‑parametric method that can capture local uncertainty patterns.
    """

    def __init__(
        self,
        reference_embeddings: np.ndarray,
        reference_covariances: np.ndarray,
        k: int = 5,
        distance_metric: str = "cosine",
        epsilon: float = 1e-8,
    ):
        """Initialise with reference set.

        Args:
            reference_embeddings: Array of shape (n_ref, d) containing reference embeddings
            reference_covariances: Array of shape (n_ref, d) containing diagonal covariances
                                   for each reference embedding
            k: Number of nearest neighbors to use
            distance_metric: 'cosine' or 'euclidean'
            epsilon: Small constant for numerical stability
        """
        self.reference_embeddings = reference_embeddings
        self.reference_covariances = reference_covariances
        self.k = k
        self.distance_metric = distance_metric
        self.epsilon = epsilon

        # Validate shapes
        if reference_embeddings.shape[0] != reference_covariances.shape[0]:
            raise ValueError(
                f"Number of reference embeddings ({reference_embeddings.shape[0]}) "
                f"must match number of reference covariances ({reference_covariances.shape[0]})"
            )
        if reference_embeddings.shape[1] != reference_covariances.shape[1]:
            raise ValueError(
                f"Embedding dimension ({reference_embeddings.shape[1]}) "
                f"must match covariance dimension ({reference_covariances.shape[1]})"
            )

        # Pre‑normalise for cosine distance
        if distance_metric == "cosine":
            self.reference_embeddings = self.reference_embeddings / (
                np.linalg.norm(self.reference_embeddings, axis=1, keepdims=True)
                + epsilon
            )

    def _compute_distances(self, embedding: np.ndarray) -> np.ndarray:
        """Compute distances to all reference embeddings."""
        if self.distance_metric == "cosine":
            embedding = embedding / (np.linalg.norm(embedding) + self.epsilon)
            similarities = self.reference_embeddings @ embedding
            return 1.0 - similarities  # cosine distance
        # euclidean
        return np.linalg.norm(self.reference_embeddings - embedding, axis=1)

    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Estimate covariance using kNN."""
        embedding = model(text)
        distances = self._compute_distances(embedding)

        # Get indices of k nearest neighbors
        if self.k >= len(distances):
            indices = np.arange(len(distances))
        else:
            indices = np.argpartition(distances, self.k)[: self.k]

        # Weight by inverse distance (add epsilon to avoid division by zero)
        weights = 1.0 / (distances[indices] + self.epsilon)
        weights = weights / np.sum(weights)

        # Weighted average of covariances
        weighted_cov = np.sum(
            self.reference_covariances[indices] * weights[:, np.newaxis], axis=0
        )

        return weighted_cov


class DomainBasedCovariance(CovarianceEstimator):
    """Scale covariance by domain hint.

    This estimator applies domain-specific scaling factors to a base covariance.
    Useful when uncertainty varies across domains (e.g., medical vs legal text).

    Example:
        >>> base = EmpiricalCovariance(embeddings)
        >>> domain_factors = {"medical": 0.5, "legal": 2.0}
        >>> estimator = DomainBasedCovariance(base, domain_factors, default_factor=1.0)
        >>> # For query with domain_hint="medical", covariance is halved
    """

    def __init__(
        self,
        base_estimator: CovarianceEstimator,
        domain_factors: Dict[str, float],
        default_factor: float = 1.0,
        epsilon: float = 1e-8,
    ):
        """Initialise with domain scaling factors.

        Args:
            base_estimator: Base covariance estimator
            domain_factors: Mapping from domain hint to scaling factor
            default_factor: Factor used when domain_hint is None or not in mapping
            epsilon: Small constant for numerical stability
        """
        self.base_estimator = base_estimator
        self.domain_factors = domain_factors
        self.default_factor = default_factor
        self.epsilon = epsilon

    def estimate(
        self,
        model: Callable[[str], np.ndarray],
        text: str,
        domain_hint: Optional[str] = None,
    ) -> np.ndarray:
        """Estimate covariance scaled by domain factor."""
        base_cov = self.base_estimator.estimate(model, text, domain_hint)

        if domain_hint is not None and domain_hint in self.domain_factors:
            factor = self.domain_factors[domain_hint]
        else:
            factor = self.default_factor

        # Apply scaling with epsilon clipping
        scaled = base_cov * max(factor, self.epsilon)
        return scaled


class DiagonalCovariance:
    """Container for diagonal covariance matrices.

    This class provides utility methods for working with diagonal covariances,
    including conversion, validation, and visualisation helpers.
    """

    def __init__(self, diagonal: np.ndarray):
        """Initialise with diagonal entries.

        Args:
            diagonal: Vector of diagonal entries, shape (d,)
        """
        self.diagonal = np.asarray(diagonal).flatten()
        self.dimension = len(self.diagonal)

        # Validate
        if np.any(self.diagonal < 0):
            raise ValueError(
                f"Covariance diagonal must be non-negative, got {self.diagonal}"
            )

    def to_full(self) -> np.ndarray:
        """Convert to full covariance matrix (for debugging only!)."""
        return np.diag(self.diagonal)

    def uncertainty_score(self) -> float:
        """Single scalar representing total uncertainty."""
        return float(np.sum(self.diagonal))

    def confidence_score(self) -> float:
        """Single scalar representing confidence (inverse uncertainty)."""
        return 1.0 / (1.0 + self.uncertainty_score())

    def __repr__(self) -> str:
        return (
            f"DiagonalCovariance(dim={self.dimension}, "
            f"uncertainty={self.uncertainty_score():.4f})"
        )
