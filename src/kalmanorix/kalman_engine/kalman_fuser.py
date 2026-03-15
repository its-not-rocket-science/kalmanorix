"""Kalman filter fusion for embedding vectors with diagonal covariance.

This module implements the core mathematical operation of KEFF: fusing multiple
embedding vectors using a Kalman update. Each specialist provides both an
embedding (measurement) and a diagonal covariance matrix (uncertainty).

Mathematical Specification:
----------------------------
For a single dimension i, the Kalman update is:

# pylint: disable=invalid-name  # Standard Kalman filter notation (x, P, R, K)

    K_i = P_i / (P_i + R_i)           # Kalman gain (scalar)
    x_i = x_i + K_i * (z_i - x_i)      # State update
    P_i = (1 - K_i) * P_i               # Covariance update

Where:
    x:  Prior state estimate (fused embedding so far)
    P:  Prior covariance (uncertainty of fused estimate)
    z:  New measurement (specialist embedding)
    R:  Measurement covariance (specialist's uncertainty)
    K:  Kalman gain (weights innovation by relative uncertainty)

For multiple measurements, we process sequentially. The order can affect the
result if covariances are not consistent, so we sort by uncertainty
(lowest R first) for numerical stability.

Assumptions:
------------
1. Linear Gaussian dynamics: The true semantic state evolves as a random walk
   with Gaussian noise (simplified: we assume no state transition between
   measurements, so prediction step is identity)
2. Diagonal covariance: All covariances are diagonal (no cross-dimension
   correlations). This reduces O(d³) to O(d) and is the only way to make
   this feasible for d=768+ dimensions.
3. Aligned spaces: All embeddings have been projected into the same reference
   space before fusion (handled by alignment module).
4. Independent dimensions: Each embedding dimension is treated independently,
   which assumes the embedding dimensions are orthogonal and uncorrelated.
   This is a strong assumption but necessary for computational tractability.

Numerical Considerations:
-------------------------
- Division by zero: If P_i + R_i = 0 (both uncertainties zero), we set K_i = 0
  to avoid numerical issues. This shouldn't happen with proper uncertainty
  estimation.
- Floating point: Use float64 for numerical stability during updates.
- Sorting: Process measurements from most certain (smallest R) to least
  certain to improve numerical stability.
"""

from typing import List, Tuple, Optional
import logging
import numpy as np

logger = logging.getLogger(__name__)


def kalman_fuse_diagonal(  # pylint: disable=too-many-arguments
    embeddings: List[np.ndarray],
    covariances: List[np.ndarray],
    initial_state: Optional[np.ndarray] = None,
    initial_covariance: Optional[np.ndarray] = None,
    sort_by_certainty: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse multiple embeddings using Kalman filter with diagonal covariance.

    Args:
        embeddings: List of embedding vectors, each shape (d,)
        covariances: List of diagonal covariance vectors, each shape (d,)
                    Must correspond 1:1 with embeddings.
        initial_state: Starting state vector (d,). If None, uses first embedding.
        initial_covariance: Starting covariance vector (d,). If None, uses first covariance.
        sort_by_certainty: If True, process measurements from lowest to highest
                          total uncertainty (sum of diagonal) for numerical stability.
        epsilon: Small constant to avoid division by zero.

    Returns:
        fused_embedding: Final state estimate after all updates, shape (d,)
        fused_covariance: Final covariance after all updates, shape (d,)

    Raises:
        ValueError: If embeddings and covariances lists have different lengths,
                   or if any vectors have incorrect shape/dimensions.
        TypeError: If inputs are not numpy arrays or have wrong dtype.

    Example:
        >>> tech_embed = np.array([0.1, 0.5, -0.2])
        >>> tech_cov = np.array([0.01, 0.02, 0.01])  # Very certain
        >>> cooking_embed = np.array([0.3, 0.1, 0.4])
        >>> cooking_cov = np.array([0.1, 0.15, 0.12])  # Less certain
        >>> fused, cov = kalman_fuse_diagonal(
        ...     [tech_embed, cooking_embed],
        ...     [tech_cov, cooking_cov]
        ... )
        >>> # Result will be closer to tech_embed where it's more certain
    """
    # Input validation
    _validate_inputs(embeddings, covariances)

    d = embeddings[0].shape[0]

    # Initialize state
    if initial_state is not None:
        if initial_state.shape != (d,):
            raise ValueError(
                f"initial_state must be shape ({d},), got {initial_state.shape}"
            )
        x = initial_state.copy().astype(np.float64)
    else:
        x = embeddings[0].copy().astype(np.float64)

    if initial_covariance is not None:
        if initial_covariance.shape != (d,):
            raise ValueError(
                f"initial_covariance must be shape ({d},), got {initial_covariance.shape}"
            )
        P = initial_covariance.copy().astype(np.float64)
    else:
        P = covariances[0].copy().astype(np.float64)

    # Create list of measurements to process
    measurements = list(zip(embeddings, covariances))

    # Optionally sort by total uncertainty (sum of diagonal)
    if sort_by_certainty and len(measurements) > 1:
        # Sort from most certain (smallest total covariance) to least certain
        measurements.sort(key=lambda m: np.sum(m[1]))
        logger.debug(
            "Sorted measurements by total uncertainty: %s",
            [np.sum(m[1]) for m in measurements],
        )

    # Sequential Kalman updates
    for i, (z, R) in enumerate(measurements):
        # Skip first if we used it as initial state
        if i == 0 and initial_state is None and initial_covariance is None:
            continue

        x, P = _kalman_update_diagonal(x, P, z, R, epsilon)

        logger.debug(
            "After measurement %d: state norm=%.4f, total uncertainty=%.4f",
            i,
            np.linalg.norm(x),
            np.sum(P),
        )

    return x, P


def _kalman_update_diagonal(  # pylint: disable=invalid-name
    x: np.ndarray,
    P: np.ndarray,
    z: np.ndarray,
    R: np.ndarray,
    epsilon: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Perform a single Kalman update step with diagonal matrices.

    This implements the core equations for one dimension at a time to avoid
    large matrix operations.

    Args:
        x: Prior state vector (d,)
        P: Prior diagonal covariance (d,)
        z: Measurement vector (d,)
        R: Measurement diagonal covariance (d,)
        epsilon: Small constant to avoid division by zero

    Returns:
        x_new: Updated state vector (d,)
        P_new: Updated diagonal covariance (d,)
    """
    # Ensure float64 for numerical stability
    x = x.astype(np.float64)
    P = P.astype(np.float64)
    z = z.astype(np.float64)
    R = R.astype(np.float64)

    # Kalman gain for each dimension independently
    # K_i = P_i / (P_i + R_i + epsilon)
    denominator = P + R + epsilon
    K = np.divide(P, denominator, where=denominator > epsilon)

    # Handle division by zero case (both P and R are zero)
    # If both uncertainties are zero, we keep the prior (K=0)
    K = np.where(denominator <= epsilon, 0.0, K)

    # State update: x = x + K * (z - x)
    innovation = z - x
    x_new = x + K * innovation

    # Covariance update: P = (1 - K) * P
    P_new = (1.0 - K) * P

    # Numerical safeguard: ensure covariance stays positive
    P_new = np.maximum(P_new, epsilon)

    return x_new, P_new


def _validate_inputs(
    embeddings: List[np.ndarray],
    covariances: List[np.ndarray],
) -> None:
    """Validate input shapes and types.

    Args:
        embeddings: List of embedding vectors
        covariances: List of covariance vectors

    Raises:
        ValueError: If validation fails
    """
    if len(embeddings) != len(covariances):
        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) must match "
            f"number of covariances ({len(covariances)})"
        )

    if len(embeddings) == 0:
        raise ValueError("At least one embedding required")

    # Check first embedding for dimension
    d = embeddings[0].shape[0]

    for i, (emb, cov) in enumerate(zip(embeddings, covariances)):
        if not isinstance(emb, np.ndarray) or not isinstance(cov, np.ndarray):
            raise TypeError(
                f"Item {i}: embeddings and covariances must be numpy arrays"
            )

        if emb.shape != (d,):
            raise ValueError(f"Embedding {i}: expected shape ({d},), got {emb.shape}")

        if cov.shape != (d,):
            raise ValueError(f"Covariance {i}: expected shape ({d},), got {cov.shape}")

        if np.any(cov < 0):
            raise ValueError(
                f"Covariance {i}: all values must be non-negative, got min {np.min(cov)}"
            )

        # Check for NaN or inf
        if not np.all(np.isfinite(emb)) or not np.all(np.isfinite(cov)):
            raise ValueError(f"Item {i}: embeddings and covariances must be finite")


def fuse_with_prior(
    embeddings: List[np.ndarray],
    covariances: List[np.ndarray],
    prior_mean: np.ndarray,
    prior_covariance: np.ndarray,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    """Fuse measurements with a prior belief.

    This is useful when we have a global prior model (e.g., a general-purpose
    embedding model) that we want to update with specialist information.

    Args:
        embeddings: List of specialist embeddings
        covariances: List of specialist covariances
        prior_mean: Prior state estimate (e.g., from general model)
        prior_covariance: Prior uncertainty (e.g., large for general model)
        **kwargs: Additional arguments passed to kalman_fuse_diagonal

    Returns:
        fused_embedding: Posterior state estimate
        fused_covariance: Posterior uncertainty

    Example:
        >>> general_embed = np.array([0.0, 0.0, 0.0])
        >>> general_cov = np.ones(3) * 0.5  # High uncertainty
        >>> specialist_embed = np.array([0.8, 0.7, 0.9])
        >>> specialist_cov = np.array([0.01, 0.01, 0.01])  # Low uncertainty
        >>> fused, cov = fuse_with_prior(
        ...     [specialist_embed], [specialist_cov],
        ...     general_embed, general_cov
        ... )
    """
    # Combine prior with measurements
    all_embeddings = [prior_mean] + embeddings
    all_covariances = [prior_covariance] + covariances

    return kalman_fuse_diagonal(
        all_embeddings,
        all_covariances,
        initial_state=None,  # Will use prior as initial
        initial_covariance=None,  # Will use prior as initial
        **kwargs,
    )


def weighted_average_baseline(
    embeddings: List[np.ndarray],
    weights: Optional[List[float]] = None,
) -> np.ndarray:
    """Simple weighted average baseline for comparison.

    This implements the naive fusion method that Kalman should outperform
    when uncertainties vary by query/domain.

    Args:
        embeddings: List of embedding vectors
        weights: Optional weights (if None, use equal weights)

    Returns:
        Weighted average embedding
    """
    if weights is None:
        weights = [1.0 / len(embeddings)] * len(embeddings)

    weights = np.array(weights) / np.sum(weights)  # Normalise

    result = np.zeros_like(embeddings[0])
    for emb, w in zip(embeddings, weights):
        result += w * emb

    return result
