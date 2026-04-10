"""Correlation-aware utilities for specialist fusion.

This module adds simple, auditable adjustments for correlated specialist
residuals. The goal is conservative uncertainty handling: when specialists make
similar errors, we should reduce aggregate precision to avoid overconfidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import numpy.typing as npt


@dataclass(frozen=True)
class ResidualCorrelationProfile:
    """Global specialist residual-correlation profile estimated on validation data.

    Attributes:
        module_names: Ordered specialist names corresponding to matrix axes.
        correlation_matrix: Pairwise correlation of scalar residual magnitudes.
            Values are clipped to [-1, 1], diagonal fixed to 1.
    """

    module_names: list[str]
    correlation_matrix: npt.NDArray[np.float64]

    def index_for_modules(self, selected_names: Sequence[str]) -> npt.NDArray[np.int64]:
        """Return matrix indices for a selected subset of module names."""
        lookup = {name: idx for idx, name in enumerate(self.module_names)}
        return np.asarray([lookup[name] for name in selected_names], dtype=np.int64)

    def submatrix(self, selected_names: Sequence[str]) -> npt.NDArray[np.float64]:
        """Return correlation submatrix for selected modules."""
        idx = self.index_for_modules(selected_names)
        return self.correlation_matrix[np.ix_(idx, idx)]


def estimate_residual_correlation_profile(
    module_names: Sequence[str],
    residual_norms: npt.NDArray[np.float64],
    epsilon: float = 1e-12,
) -> ResidualCorrelationProfile:
    """Estimate pairwise residual correlations from validation residual norms.

    Args:
        module_names: Names of specialists (length n_modules).
        residual_norms: Array of shape (n_queries, n_modules), where each entry is
            a scalar residual magnitude (e.g., ||prediction - target||2).
        epsilon: Numerical floor for near-constant residual series.

    Returns:
        ResidualCorrelationProfile with an n_modules x n_modules matrix.
    """
    n_queries, n_modules = residual_norms.shape
    if n_modules != len(module_names):
        raise ValueError("residual_norms second dimension must match module_names")
    if n_queries < 2:
        raise ValueError("Need at least two validation queries to estimate correlation")

    x = np.asarray(residual_norms, dtype=np.float64)
    # Avoid NaNs from constant columns by adding tiny noise-free stabilizer.
    std = np.std(x, axis=0)
    safe_x = x.copy()
    for i in range(n_modules):
        if std[i] < epsilon:
            safe_x[:, i] = safe_x[:, i] + np.linspace(0.0, epsilon, n_queries)
    corr = np.corrcoef(safe_x, rowvar=False).astype(np.float64)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.clip(corr, -1.0, 1.0)
    np.fill_diagonal(corr, 1.0)
    return ResidualCorrelationProfile(
        module_names=list(module_names),
        correlation_matrix=corr,
    )


def correlation_inflation_factors(
    corr_submatrix: npt.NDArray[np.float64], alpha: float = 1.0
) -> npt.NDArray[np.float64]:
    """Per-expert covariance inflation from mean positive pairwise correlation.

    For specialist i with selected set S:
        inflate_i = 1 + alpha * mean_j!=i(max(rho_ij, 0))
    """
    n = corr_submatrix.shape[0]
    if n == 1:
        return np.ones(1, dtype=np.float64)
    clipped = np.clip(corr_submatrix, 0.0, 1.0)
    mask = ~np.eye(n, dtype=bool)
    mean_pos_corr = np.sum(clipped * mask, axis=1) / (n - 1)
    return 1.0 + alpha * mean_pos_corr


def effective_sample_size_discount(corr_submatrix: npt.NDArray[np.float64]) -> float:
    """Global precision discount from average positive off-diagonal correlation.

    Uses:
        rho_bar = mean_{i!=j}(max(rho_ij, 0))
        n_eff = n / (1 + (n - 1) * rho_bar)
        discount = n_eff / n in (0, 1]

    Properties:
    - independence (rho_bar=0) -> discount=1 (no change)
    - perfect correlation (rho_bar=1) -> discount=1/n
    """
    n = corr_submatrix.shape[0]
    if n <= 1:
        return 1.0
    clipped = np.clip(corr_submatrix, 0.0, 1.0)
    off_diag = clipped[~np.eye(n, dtype=bool)]
    rho_bar = float(np.mean(off_diag)) if off_diag.size else 0.0
    n_eff = n / (1.0 + (n - 1.0) * rho_bar)
    return float(np.clip(n_eff / n, 1.0 / n, 1.0))
