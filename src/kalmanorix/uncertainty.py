"""
Uncertainty models (sigma²) for Kalmanorix.

Sigma² callables map (query: str) -> float variance.
They are intentionally lightweight and deterministic for early phases.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable, List, Set

import numpy as np

EmbedderFn = Callable[[str], np.ndarray]


@dataclass(frozen=True)
class KeywordSigma2:
    """A simple query-dependent uncertainty heuristic.

    If a query contains any keyword, return in_domain_sigma2 (more confident).
    Otherwise return out_domain_sigma2 (less confident).
    """

    keywords: Set[str]
    in_domain_sigma2: float = 0.2
    out_domain_sigma2: float = 2.0

    def __call__(self, query: str) -> float:
        q = query.lower()
        if any(kw in q for kw in self.keywords):
            return float(self.in_domain_sigma2)
        return float(self.out_domain_sigma2)


@dataclass
class CentroidDistanceSigma2:
    """Query-dependent uncertainty based on distance to a module centroid.

    Steps:
      - Precompute centroid from calibration texts in the module's embedding space.
      - For a query, compute cosine similarity to centroid.
      - Map similarity to sigma2 with softplus scaling:
        higher similarity => lower sigma2.

    Attributes:
        embed: Embedder function mapping text to vector.
        centroid: Normalized centroid vector of shape (d,).
        base_sigma2: Minimum variance scale.
        beta: Temperature for distance scaling (higher => steeper growth).
        scale: Deprecated legacy parameter retained for backwards compatibility.
    """

    embed: EmbedderFn
    centroid: np.ndarray
    base_sigma2: float = 0.2
    beta: float = 1.0
    scale: float = 2.0

    @classmethod
    def from_calibration(
        cls,
        *,
        embed: EmbedderFn,
        calibration_texts: Iterable[str],
        base_sigma2: float = 0.2,
        beta: float = 1.0,
        scale: float = 2.0,
    ) -> "CentroidDistanceSigma2":
        """Construct from calibration texts to compute centroid.

        Args:
            embed: Embedder function mapping text to vector.
            calibration_texts: Sample texts from the specialist's domain.
            base_sigma2: Minimum variance scale.
            beta: Temperature for distance scaling (higher => steeper growth).
            scale: Deprecated legacy parameter retained for backwards compatibility.

        Returns:
            CentroidDistanceSigma2 instance with computed centroid.
        """
        embs = [embed(t) for t in calibration_texts]
        c = np.mean(np.stack(embs, axis=0), axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return cls(
            embed=embed,
            centroid=c,
            base_sigma2=base_sigma2,
            beta=beta,
            scale=scale,
        )

    def __call__(self, query: str) -> float:
        z = self.embed(query)
        z = z / (np.linalg.norm(z) + 1e-12)
        sim = float(z @ self.centroid)
        distance = 1.0 - sim  # sim in [-1, 1] => distance in [0, 2]
        uncertainty_multiplier = np.log1p(np.exp(self.beta * distance)) - np.log(2.0)
        length_norm = min(1.0, len(query.split()) / 20.0)
        sigma2 = self.base_sigma2 * (
            1.0 + uncertainty_multiplier + 0.2 * length_norm
        )
        return float(min(sigma2, self.base_sigma2 * 5.0))

    def calibrate(
        self, validation_queries: List[str], empirical_errors: List[float]
    ) -> None:
        """Calibrate beta and base_sigma2 using validation data.

        Uses 1-D optimization over beta to maximize correlation between
        predicted sigma² and empirical errors, then rescales base_sigma2 to
        match average error level.
        """
        if len(validation_queries) != len(empirical_errors):
            raise ValueError(
                "validation_queries and empirical_errors must have the same length."
            )
        if not validation_queries:
            raise ValueError("validation_queries must not be empty.")

        from scipy.optimize import minimize_scalar

        errors = np.asarray(empirical_errors, dtype=np.float64)
        if np.allclose(np.std(errors), 0.0):
            return

        def _predicted_for_beta(beta: float) -> np.ndarray:
            beta = max(float(beta), 1e-6)
            preds = []
            for query in validation_queries:
                z = self.embed(query)
                z = z / (np.linalg.norm(z) + 1e-12)
                sim = float(z @ self.centroid)
                distance = 1.0 - sim
                uncertainty_multiplier = np.log1p(np.exp(beta * distance)) - np.log(2.0)
                length_norm = min(1.0, len(query.split()) / 20.0)
                preds.append(1.0 + uncertainty_multiplier + 0.2 * length_norm)
            return np.asarray(preds, dtype=np.float64)

        def _objective(beta: float) -> float:
            preds = _predicted_for_beta(beta)
            if np.allclose(np.std(preds), 0.0):
                return 1.0
            corr = np.corrcoef(preds, errors)[0, 1]
            if np.isnan(corr):
                return 1.0
            return -float(corr)

        result = minimize_scalar(_objective, bounds=(0.05, 3.0), method="bounded")
        best_beta = float(result.x) if result.success else self.beta
        object.__setattr__(self, "beta", max(best_beta, 1e-6))

        preds = _predicted_for_beta(self.beta)
        denom = float(np.mean(preds))
        if denom > 1e-12:
            calibrated_base = float(np.mean(np.clip(errors, 0.0, None)) / denom)
            object.__setattr__(self, "base_sigma2", max(calibrated_base, 1e-6))


@dataclass(frozen=True)
class ConstantSigma2:
    """Fixed variance regardless of query (ablation study)."""

    value: float

    def __call__(self, query: str) -> float:
        return float(self.value)


@dataclass(frozen=True)
class ScaledSigma2:
    """Scale another sigma2 callable by constant factor.

    Useful for testing robustness to mis-specified uncertainties.
    scale > 1: over-confident (underestimates variance)
    scale < 1: under-confident (overestimates variance)
    """

    base_sigma2: Callable[[str], float]
    scale: float

    def __call__(self, query: str) -> float:
        return float(self.base_sigma2(query) * self.scale)
