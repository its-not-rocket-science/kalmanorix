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

from typing import List, Tuple, Optional, Any
import logging
import numpy as np
import numpy.typing as npt
from .structured_covariance import StructuredCovariance

logger = logging.getLogger(__name__)


def kalman_fuse_diagonal(  # pylint: disable=too-many-arguments
    embeddings: List[npt.NDArray[np.float64]],
    covariances: List[npt.NDArray[np.float64]],
    initial_state: Optional[npt.NDArray[np.float64]] = None,
    initial_covariance: Optional[npt.NDArray[np.float64]] = None,
    sort_by_certainty: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    x: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    R: npt.NDArray[np.float64],
    epsilon: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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
    # Initialize K array with zeros (default for denominator <= epsilon)
    K = np.zeros_like(P, dtype=np.float64)
    # Compute division only where denominator > epsilon
    np.divide(P, denominator, out=K, where=denominator > epsilon)

    # State update: x = x + K * (z - x)
    innovation = z - x
    x_new = x + K * innovation

    # Covariance update: P = (1 - K) * P
    P_new = (1.0 - K) * P

    # Numerical safeguard: ensure covariance stays positive
    P_new = np.maximum(P_new, epsilon)

    return x_new, P_new


def _validate_inputs(
    embeddings: List[npt.NDArray[np.float64]],
    covariances: List[npt.NDArray[np.float64]],
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
    embeddings: List[npt.NDArray[np.float64]],
    covariances: List[npt.NDArray[np.float64]],
    prior_mean: npt.NDArray[np.float64],
    prior_covariance: npt.NDArray[np.float64],
    **kwargs: Any,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
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

    # Disable sorting by default to ensure prior is used as initial state
    # (prior is first element in the list; sorting would reorder it)
    kwargs.setdefault("sort_by_certainty", False)

    return kalman_fuse_diagonal(
        all_embeddings,
        all_covariances,
        initial_state=None,  # Will use prior as initial
        initial_covariance=None,  # Will use prior as initial
        **kwargs,
    )


def kalman_fuse_diagonal_ensemble(
    embeddings: List[npt.NDArray[np.float64]],
    covariances: List[npt.NDArray[np.float64]],
    initial_state: Optional[npt.NDArray[np.float64]] = None,
    initial_covariance: Optional[npt.NDArray[np.float64]] = None,
    epsilon: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fuse multiple embeddings using ensemble Kalman filter with diagonal covariance.

    This implementation performs parallel fusion of all measurements simultaneously
    rather than sequential updates. For diagonal covariance and independent
    measurements, this produces the same result as sequential Kalman updates
    but can be more computationally efficient (single pass).

    Mathematical derivation:
        P_fused = (P0^{-1} + Σ R_i^{-1})^{-1}
        x_fused = P_fused ⊙ (P0^{-1} ⊙ x0 + Σ R_i^{-1} ⊙ z_i)

    where ⊙ denotes elementwise multiplication, and inverses are elementwise.

    Args:
        embeddings: List of embedding vectors, each shape (d,)
        covariances: List of diagonal covariance vectors, each shape (d,)
        initial_state: Prior state vector (d,). If None, uses non-informative prior.
        initial_covariance: Prior covariance vector (d,). If None, uses non-informative prior.
        epsilon: Small constant to avoid division by zero.

    Returns:
        fused_embedding: Final state estimate after all updates, shape (d,)
        fused_covariance: Final covariance after all updates, shape (d,)

    Raises:
        ValueError: If embeddings and covariances lists have different lengths,
                   or if any vectors have incorrect shape/dimensions.
        TypeError: If inputs are not numpy arrays or have wrong dtype.
    """
    # Input validation (reuse existing validation)
    _validate_inputs(embeddings, covariances)

    d = embeddings[0].shape[0]

    # Convert to float64 for numerical stability
    embeddings_f64 = [emb.astype(np.float64) for emb in embeddings]
    covariances_f64 = [cov.astype(np.float64) for cov in covariances]

    # Initialize prior terms
    if initial_state is not None:
        if initial_state.shape != (d,):
            raise ValueError(
                f"initial_state must be shape ({d},), got {initial_state.shape}"
            )
        x0 = initial_state.astype(np.float64)
    else:
        # Non-informative prior: zero contribution
        x0 = None

    if initial_covariance is not None:
        if initial_covariance.shape != (d,):
            raise ValueError(
                f"initial_covariance must be shape ({d},), got {initial_covariance.shape}"
            )
        P0 = initial_covariance.astype(np.float64)
    else:
        # Non-informative prior: infinite variance (zero precision)
        P0 = None

    # Compute sum of precision-weighted measurements: Σ R_i^{-1} ⊙ z_i
    # and sum of precisions: Σ R_i^{-1}
    sum_precision_weighted = np.zeros(d, dtype=np.float64)
    sum_precision = np.zeros(d, dtype=np.float64)

    for emb, cov in zip(embeddings_f64, covariances_f64):
        # Add epsilon to avoid division by zero
        inv_cov = 1.0 / (cov + epsilon)
        sum_precision_weighted += inv_cov * emb
        sum_precision += inv_cov

    # Add prior contribution if provided
    if x0 is not None and P0 is not None:
        inv_P0 = 1.0 / (P0 + epsilon)
        sum_precision_weighted += inv_P0 * x0
        sum_precision += inv_P0
    elif x0 is not None or P0 is not None:
        raise ValueError(
            "Both initial_state and initial_covariance must be provided together, or neither"
        )

    # Compute fused covariance: P = (sum_precision)^{-1}
    # Handle zero precision (should not happen with epsilon > 0)
    fused_covariance = 1.0 / (sum_precision + epsilon)

    # Compute fused state: x = P ⊙ sum_precision_weighted
    fused_embedding = fused_covariance * sum_precision_weighted

    return fused_embedding, fused_covariance


def _validate_batch_inputs(
    embeddings: npt.NDArray[np.float64],
    covariances: npt.NDArray[np.float64],
) -> None:
    """Validate batch input shapes and types.

    Args:
        embeddings: Array of shape (num_specialists, batch_size, d)
        covariances: Array of shape (num_specialists, batch_size, d)

    Raises:
        ValueError: If validation fails
    """
    if not isinstance(embeddings, np.ndarray) or not isinstance(
        covariances, np.ndarray
    ):
        raise TypeError("embeddings and covariances must be numpy arrays")

    if embeddings.ndim != 3 or covariances.ndim != 3:
        raise ValueError(
            f"embeddings and covariances must be 3D arrays, got shapes "
            f"{embeddings.shape} and {covariances.shape}"
        )

    if embeddings.shape != covariances.shape:
        raise ValueError(
            f"embeddings shape {embeddings.shape} must match "
            f"covariances shape {covariances.shape}"
        )

    num_specialists, batch_size, d = embeddings.shape

    if num_specialists == 0:
        raise ValueError("At least one specialist required")

    if batch_size == 0:
        raise ValueError("Batch size must be positive")

    if d == 0:
        raise ValueError("Embedding dimension must be positive")

    if np.any(covariances < 0):
        raise ValueError("All covariance values must be non-negative")

    if not np.all(np.isfinite(embeddings)) or not np.all(np.isfinite(covariances)):
        raise ValueError("embeddings and covariances must be finite")


def _kalman_update_diagonal_batch(
    x: npt.NDArray[np.float64],  # shape (batch_size, d)
    P: npt.NDArray[np.float64],  # shape (batch_size, d)
    z: npt.NDArray[np.float64],  # shape (batch_size, d)
    R: npt.NDArray[np.float64],  # shape (batch_size, d)
    epsilon: float,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform a single Kalman update step with diagonal matrices for a batch.

    Args:
        x: Prior state vectors (batch_size, d)
        P: Prior diagonal covariance (batch_size, d)
        z: Measurement vectors (batch_size, d)
        R: Measurement diagonal covariance (batch_size, d)
        epsilon: Small constant to avoid division by zero

    Returns:
        x_new: Updated state vectors (batch_size, d)
        P_new: Updated diagonal covariance (batch_size, d)
    """
    # Ensure float64 for numerical stability
    x = x.astype(np.float64)
    P = P.astype(np.float64)
    z = z.astype(np.float64)
    R = R.astype(np.float64)

    # Kalman gain for each dimension independently, broadcast across batch
    # K_i = P_i / (P_i + R_i + epsilon)
    denominator = P + R + epsilon
    # Initialize K array with zeros (default for denominator <= epsilon)
    K = np.zeros_like(P, dtype=np.float64)
    # Compute division only where denominator > epsilon
    np.divide(P, denominator, out=K, where=denominator > epsilon)

    # State update: x = x + K * (z - x)
    innovation = z - x
    x_new = x + K * innovation

    # Covariance update: P = (1 - K) * P
    P_new = (1.0 - K) * P

    # Numerical safeguard: ensure covariance stays positive
    P_new = np.maximum(P_new, epsilon)

    return x_new, P_new


def kalman_fuse_diagonal_batch(
    embeddings: npt.NDArray[np.float64],  # shape (num_specialists, batch_size, d)
    covariances: npt.NDArray[np.float64],  # shape (num_specialists, batch_size, d)
    initial_state: Optional[npt.NDArray[np.float64]] = None,  # shape (batch_size, d)
    initial_covariance: Optional[
        npt.NDArray[np.float64]
    ] = None,  # shape (batch_size, d)
    sort_by_certainty: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fuse multiple embeddings for a batch of queries using Kalman filter with diagonal covariance.

    Args:
        embeddings: Array of embedding vectors, shape (num_specialists, batch_size, d)
        covariances: Array of diagonal covariance vectors, shape (num_specialists, batch_size, d)
        initial_state: Starting state vectors (batch_size, d). If None, uses first specialist.
        initial_covariance: Starting covariance vectors (batch_size, d). If None, uses first covariance.
        sort_by_certainty: If True, process measurements from lowest to highest
                          total uncertainty (sum of diagonal) for numerical stability.
        epsilon: Small constant to avoid division by zero.

    Returns:
        fused_embedding: Final state estimate after all updates, shape (batch_size, d)
        fused_covariance: Final covariance after all updates, shape (batch_size, d)

    Raises:
        ValueError: If inputs have incorrect shapes or values.
    """
    # Input validation
    _validate_batch_inputs(embeddings, covariances)
    num_specialists, batch_size, d = embeddings.shape

    # Initialize state
    if initial_state is not None:
        if initial_state.shape != (batch_size, d):
            raise ValueError(
                f"initial_state must be shape ({batch_size}, {d}), got {initial_state.shape}"
            )
        x = initial_state.copy().astype(np.float64)
    else:
        # Use first specialist as initial state
        x = embeddings[0].copy().astype(np.float64)

    if initial_covariance is not None:
        if initial_covariance.shape != (batch_size, d):
            raise ValueError(
                f"initial_covariance must be shape ({batch_size}, {d}), got {initial_covariance.shape}"
            )
        P = initial_covariance.copy().astype(np.float64)
    else:
        P = covariances[0].copy().astype(np.float64)

    # Create list of measurements to process (specialists)
    measurements = list(zip(embeddings, covariances))

    # Optionally sort by total uncertainty (sum over dimensions) per batch element
    if sort_by_certainty and len(measurements) > 1:
        # Compute total uncertainty per specialist (batch_size,)
        # Sum over embedding dimension
        total_uncertainties = [np.sum(cov, axis=1) for cov in covariances]
        # Sort by average total uncertainty across batch (or could sort per batch element separately)
        # We'll sort by mean total uncertainty across batch (simple heuristic)
        avg_total = [np.mean(tot) for tot in total_uncertainties]
        sorted_indices = np.argsort(avg_total)
        measurements = [measurements[i] for i in sorted_indices]
        logger.debug(
            "Sorted measurements by average total uncertainty: %s",
            [avg_total[i] for i in sorted_indices],
        )

    # Sequential Kalman updates across specialists
    for i, (z, R) in enumerate(measurements):
        # Skip first if we used it as initial state
        if i == 0 and initial_state is None and initial_covariance is None:
            continue

        x, P = _kalman_update_diagonal_batch(x, P, z, R, epsilon)

        logger.debug(
            "After specialist %d: average state norm=%.4f, average total uncertainty=%.4f",
            i,
            np.mean(np.linalg.norm(x, axis=1)),
            np.mean(np.sum(P, axis=1)),
        )

    return x, P


def kalman_fuse_diagonal_ensemble_batch(
    embeddings: npt.NDArray[np.float64],  # shape (num_specialists, batch_size, d)
    covariances: npt.NDArray[np.float64],  # shape (num_specialists, batch_size, d)
    initial_state: Optional[npt.NDArray[np.float64]] = None,  # shape (batch_size, d)
    initial_covariance: Optional[
        npt.NDArray[np.float64]
    ] = None,  # shape (batch_size, d)
    epsilon: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fuse multiple embeddings for a batch of queries using ensemble Kalman filter.

    Parallel fusion of all measurements simultaneously (batch version).

    Args:
        embeddings: Array of embedding vectors, shape (num_specialists, batch_size, d)
        covariances: Array of diagonal covariance vectors, shape (num_specialists, batch_size, d)
        initial_state: Prior state vectors (batch_size, d). If None, uses non-informative prior.
        initial_covariance: Prior covariance vectors (batch_size, d). If None, uses non-informative prior.
        epsilon: Small constant to avoid division by zero.

    Returns:
        fused_embedding: Final state estimate after all updates, shape (batch_size, d)
        fused_covariance: Final covariance after all updates, shape (batch_size, d)
    """
    # Input validation
    _validate_batch_inputs(embeddings, covariances)
    num_specialists, batch_size, d = embeddings.shape

    # Convert to float64 for numerical stability
    embeddings_f64 = embeddings.astype(np.float64)
    covariances_f64 = covariances.astype(np.float64)

    # Initialize prior terms
    if initial_state is not None:
        if initial_state.shape != (batch_size, d):
            raise ValueError(
                f"initial_state must be shape ({batch_size}, {d}), got {initial_state.shape}"
            )
        x0 = initial_state.astype(np.float64)
    else:
        # Non-informative prior: zero contribution
        x0 = None

    if initial_covariance is not None:
        if initial_covariance.shape != (batch_size, d):
            raise ValueError(
                f"initial_covariance must be shape ({batch_size}, {d}), got {initial_covariance.shape}"
            )
        P0 = initial_covariance.astype(np.float64)
    else:
        # Non-informative prior: infinite variance (zero precision)
        P0 = None

    # Compute sum of precision-weighted measurements: Σ R_i^{-1} ⊙ z_i
    # and sum of precisions: Σ R_i^{-1}
    # Shape: (batch_size, d) for each sum
    sum_precision_weighted = np.zeros((batch_size, d), dtype=np.float64)
    sum_precision = np.zeros((batch_size, d), dtype=np.float64)

    for i in range(num_specialists):
        emb = embeddings_f64[i]
        cov = covariances_f64[i]
        # Add epsilon to avoid division by zero
        inv_cov = 1.0 / (cov + epsilon)
        sum_precision_weighted += inv_cov * emb
        sum_precision += inv_cov

    # Add prior contribution if provided
    if x0 is not None and P0 is not None:
        inv_P0 = 1.0 / (P0 + epsilon)
        sum_precision_weighted += inv_P0 * x0
        sum_precision += inv_P0
    elif x0 is not None or P0 is not None:
        raise ValueError(
            "Both initial_state and initial_covariance must be provided together, or neither"
        )

    # Compute fused covariance: P = (sum_precision)^{-1}
    # Handle zero precision (should not happen with epsilon > 0)
    fused_covariance = 1.0 / (sum_precision + epsilon)

    # Compute fused state: x = P ⊙ sum_precision_weighted
    fused_embedding = fused_covariance * sum_precision_weighted

    return fused_embedding, fused_covariance


def _structured_kalman_update_diagonal(
    x: npt.NDArray[np.float64],
    P: npt.NDArray[np.float64],
    z: npt.NDArray[np.float64],
    R: StructuredCovariance,
    epsilon: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Perform a single Kalman update step with structured covariance R = D + UUᵀ.

    Args:
        x: Prior state vector (d,)
        P: Prior diagonal covariance (d,)
        z: Measurement vector (d,)
        R: Structured covariance measurement uncertainty
        epsilon: Small constant to avoid division by zero

    Returns:
        x_new: Updated state vector (d,)
        P_new: Updated diagonal covariance (d,)
    """
    # Ensure float64 for numerical stability
    x = x.astype(np.float64)
    P = P.astype(np.float64)
    z = z.astype(np.float64)
    d = x.shape[0]

    # Innovation
    v = z - x

    # Solve (P + R) w = v using Woodbury identity
    # R.woodbury_solve(P, v) solves (P + D + UUᵀ) w = v
    w = R.woodbury_solve(P, v, epsilon)

    # Kalman gain applied to innovation: K v = P w (since K = P (P+R)⁻¹ and P diagonal)
    K_v = P * w

    # State update
    x_new = x + K_v

    # Approximate covariance update: P_new = P - diag(K) * P
    # Compute diag(K) = P * diag((P+R)⁻¹) ≈ P * w_e where w_e solves (P+R) w_e = e (vector of ones)
    e = np.ones(d, dtype=np.float64)
    w_e = R.woodbury_solve(P, e, epsilon)
    diag_K = P * w_e

    P_new = P - diag_K * P
    # Numerical safeguard: ensure covariance stays positive
    P_new = np.maximum(P_new, epsilon)

    return x_new, P_new


def kalman_fuse_structured(
    embeddings: List[npt.NDArray[np.float64]],
    structured_covariances: List[StructuredCovariance],
    initial_state: Optional[npt.NDArray[np.float64]] = None,
    initial_covariance: Optional[npt.NDArray[np.float64]] = None,
    sort_by_certainty: bool = True,
    epsilon: float = 1e-8,
) -> Tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Fuse multiple embeddings using Kalman filter with structured covariance.

    Supports low‑rank covariance R = D + UUᵀ where D is diagonal and U is a
    low‑rank factor matrix (d × k). Complexity O(d·k + k³) per measurement.

    Args:
        embeddings: List of embedding vectors, each shape (d,)
        structured_covariances: List of StructuredCovariance objects.
            Must correspond 1:1 with embeddings.
        initial_state: Starting state vector (d,). If None, uses first embedding.
        initial_covariance: Starting covariance vector (d,). If None, uses first covariance.
        sort_by_certainty: If True, process measurements from lowest to highest
                          total uncertainty (sum of diagonal + trace(UUᵀ)).
        epsilon: Small constant to avoid division by zero.

    Returns:
        fused_embedding: Final state estimate after all updates, shape (d,)
        fused_covariance: Final covariance after all updates, shape (d,)

    Raises:
        ValueError: If embeddings and covariances lists have different lengths,
                   or if any vectors have incorrect shape/dimensions.
        TypeError: If inputs are not numpy arrays or have wrong dtype.
    """
    # Input validation
    if len(embeddings) != len(structured_covariances):
        raise ValueError(
            f"Number of embeddings ({len(embeddings)}) must match "
            f"number of structured_covariances ({len(structured_covariances)})"
        )

    if len(embeddings) == 0:
        raise ValueError("At least one embedding required")

    # Check first embedding for dimension
    d = embeddings[0].shape[0]

    for i, (emb, cov) in enumerate(zip(embeddings, structured_covariances)):
        if not isinstance(emb, np.ndarray):
            raise TypeError(f"Item {i}: embedding must be numpy array")
        if emb.shape != (d,):
            raise ValueError(f"Embedding {i}: expected shape ({d},), got {emb.shape}")
        if cov.dimension != d:
            raise ValueError(
                f"Covariance {i}: dimension {cov.dimension} does not match embedding dimension {d}"
            )
        # Check for NaN or inf
        if not np.all(np.isfinite(emb)):
            raise ValueError(f"Item {i}: embedding must be finite")

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
        # Use first covariance diagonal as initial (ignore low-rank factor)
        P = structured_covariances[0].diagonal.copy().astype(np.float64)

    # Create list of measurements to process
    measurements = list(zip(embeddings, structured_covariances))

    # Optionally sort by total uncertainty (sum of diagonal + trace(UUᵀ))
    if sort_by_certainty and len(measurements) > 1:
        # Sort from most certain (smallest total covariance) to least certain
        measurements.sort(key=lambda m: m[1].uncertainty_score())
        logger.debug(
            "Sorted measurements by total uncertainty: %s",
            [m[1].uncertainty_score() for m in measurements],
        )

    # Sequential Kalman updates
    for i, (z, R) in enumerate(measurements):
        # Skip first if we used it as initial state
        if i == 0 and initial_state is None and initial_covariance is None:
            continue

        x, P = _structured_kalman_update_diagonal(x, P, z, R, epsilon)

        logger.debug(
            "After measurement %d: state norm=%.4f, total uncertainty=%.4f",
            i,
            np.linalg.norm(x),
            np.sum(P),
        )

    return x, P


def weighted_average_baseline(
    embeddings: List[npt.NDArray[np.float64]],
    weights: Optional[List[float]] = None,
) -> npt.NDArray[np.float64]:
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
