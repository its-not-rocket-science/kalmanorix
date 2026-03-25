"""Structured covariance representation R = D + UUᵀ.

This module provides the `StructuredCovariance` class for representing covariance
matrices as a diagonal plus low‑rank outer product:

    R = D + UUᵀ

where D is a diagonal matrix (stored as a vector of length d) and U is a low‑rank
factor matrix of shape (d × k), with k ≪ d.

The representation enables efficient Kalman updates via the Woodbury identity:

    (S + UUᵀ)⁻¹ = S⁻¹ - S⁻¹U(Iₖ + UᵀS⁻¹U)⁻¹UᵀS⁻¹

which reduces the cost from O(d³) to O(d·k + k³).

The class supports diagonal‑only covariances (U = None) as a special case, ensuring
backward compatibility with existing diagonal covariance code.
"""

from __future__ import annotations

from typing import Optional, Tuple
import logging

import numpy as np

logger = logging.getLogger(__name__)


class StructuredCovariance:
    """Covariance matrix represented as diagonal plus low‑rank outer product.

    Attributes
    ----------
    diagonal : np.ndarray, shape (d,)
        Diagonal entries of D (must be non‑negative).
    lowrank_factor : Optional[np.ndarray], shape (d, k) or None
        Low‑rank factor U. If None, the covariance is purely diagonal.
    rank : int
        Rank k of the low‑rank factor (0 if lowrank_factor is None).
    dimension : int
        Embedding dimension d.
    """

    def __init__(
        self,
        diagonal: np.ndarray,
        lowrank_factor: Optional[np.ndarray] = None,
    ) -> None:
        """Initialise structured covariance.

        Parameters
        ----------
        diagonal : np.ndarray, shape (d,)
            Diagonal entries of D. Must be non‑negative.
        lowrank_factor : Optional[np.ndarray], shape (d, k)
            Low‑rank factor U. If provided, must have the same first dimension
            as diagonal. If None, the covariance is diagonal‑only.

        Raises
        ------
        ValueError
            If diagonal contains negative values, or if lowrank_factor shape
            mismatches diagonal.
        """
        self.diagonal = np.asarray(diagonal).flatten()
        self.dimension = len(self.diagonal)

        if np.any(self.diagonal < 0):
            raise ValueError(
                f"Covariance diagonal must be non‑negative, got {self.diagonal}"
            )

        if lowrank_factor is None:
            self.lowrank_factor = None
            self.rank = 0
        else:
            self.lowrank_factor = np.asarray(lowrank_factor)
            if self.lowrank_factor.ndim != 2:
                raise ValueError(
                    f"lowrank_factor must be 2‑dimensional, got shape {self.lowrank_factor.shape}"
                )
            if self.lowrank_factor.shape[0] != self.dimension:
                raise ValueError(
                    f"lowrank_factor first dimension {self.lowrank_factor.shape[0]} "
                    f"must match diagonal length {self.dimension}"
                )
            self.rank = self.lowrank_factor.shape[1]

    @property
    def is_diagonal(self) -> bool:
        """True if covariance is diagonal‑only (lowrank_factor is None)."""
        return self.lowrank_factor is None

    def to_full(self) -> np.ndarray:
        """Convert to full covariance matrix (for debugging only!).

        Returns
        -------
        np.ndarray, shape (d, d)
            Full covariance matrix R = D + UUᵀ.
        """
        R = np.diag(self.diagonal)
        if self.lowrank_factor is not None:
            R += self.lowrank_factor @ self.lowrank_factor.T
        return R

    def diagonal_only(self) -> StructuredCovariance:
        """Return a diagonal‑only copy (drop low‑rank factor)."""
        return StructuredCovariance(self.diagonal, lowrank_factor=None)

    @classmethod
    def from_diagonal(cls, diagonal: np.ndarray) -> StructuredCovariance:
        """Create a diagonal‑only structured covariance."""
        return cls(diagonal, lowrank_factor=None)

    @classmethod
    def from_lowrank(
        cls,
        diagonal: np.ndarray,
        lowrank_factor: np.ndarray,
    ) -> StructuredCovariance:
        """Create structured covariance with low‑rank factor."""
        return cls(diagonal, lowrank_factor=lowrank_factor)

    def woodbury_solve(
        self,
        prior_diag: np.ndarray,
        v: np.ndarray,
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """Solve (prior_diag + D + UUᵀ) x = v using Woodbury identity.

        Parameters
        ----------
        prior_diag : np.ndarray, shape (d,)
            Diagonal entries of prior covariance P (must be positive).
        v : np.ndarray, shape (d,) or (d, m)
            Right‑hand side vector(s).
        epsilon : float
            Small constant added to diagonal to avoid division by zero.

        Returns
        -------
        x : np.ndarray, same shape as v
            Solution of (prior_diag + D + UUᵀ) x = v.

        Notes
        -----
        If U is None, the solution reduces to x = v / (prior_diag + D + epsilon).
        """
        # Total diagonal: prior + measurement diagonal + epsilon
        S = prior_diag + self.diagonal + epsilon
        Sinv = 1.0 / S

        if self.lowrank_factor is None:
            # Diagonal case
            return v * Sinv[:, None] if v.ndim == 2 else v * Sinv

        U = self.lowrank_factor
        k = self.rank

        # B = S⁻¹U  (d × k)
        B = Sinv[:, None] * U

        # M = Iₖ + UᵀB  (k × k)
        M = np.eye(k) + U.T @ B

        # Solve M z = Uᵀ (S⁻¹ v)
        if v.ndim == 1:
            UTSinv_v = U.T @ (Sinv * v)
            z = np.linalg.solve(M, UTSinv_v)
            x = Sinv * v - B @ z
        else:
            # v is (d, m)
            UTSinv_v = U.T @ (Sinv[:, None] * v)  # (k × m)
            z = np.linalg.solve(M, UTSinv_v)  # (k × m)
            x = Sinv[:, None] * v - B @ z  # (d × m)

        return x

    def uncertainty_score(self) -> float:
        """Single scalar representing total uncertainty."""
        total = np.sum(self.diagonal)
        if self.lowrank_factor is not None:
            # trace(UUᵀ) = sum of squares of all entries of U
            total += np.sum(self.lowrank_factor**2)
        return float(total)

    def confidence_score(self) -> float:
        """Single scalar representing confidence (inverse uncertainty)."""
        return 1.0 / (1.0 + self.uncertainty_score())

    def __repr__(self) -> str:
        if self.is_diagonal:
            return (
                f"StructuredCovariance(diagonal, d={self.dimension}, "
                f"uncertainty={self.uncertainty_score():.4f})"
            )
        return (
            f"StructuredCovariance(diagonal+lowrank, d={self.dimension}, k={self.rank}, "
            f"uncertainty={self.uncertainty_score():.4f})"
        )


def woodbury_inverse(
    S_diag: np.ndarray,
    U: Optional[np.ndarray],
    epsilon: float = 1e-8,
) -> Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """Compute components needed for Woodbury‑based Kalman gain.

    Returns (S_inv, B, M_inv) where:
        S_inv = 1 / (S_diag + epsilon)
        B = S_inv * U  (if U is not None)
        M_inv = (I + UᵀB)⁻¹  (if U is not None)

    This is a helper for implementations that want to reuse these pre‑computed
    quantities across multiple Kalman updates.
    """
    S = S_diag + epsilon
    S_inv = 1.0 / S

    if U is None:
        return S_inv, None, None

    # B = S⁻¹U  (d × k)
    B = S_inv[:, None] * U

    # M = Iₖ + UᵀB  (k × k)
    M = np.eye(U.shape[1]) + U.T @ B

    # Inverse of M
    try:
        M_inv = np.linalg.inv(M)
    except np.linalg.LinAlgError:
        # Fallback to pseudo‑inverse if M is singular
        logger.warning("Woodbury M matrix is singular, using pseudo‑inverse")
        M_inv = np.linalg.pinv(M)

    return S_inv, B, M_inv
