"""
Uncertainty models (sigma²) for Kalmanorix.

Sigma² callables map (query: str) -> float variance.
They are intentionally lightweight and deterministic for early phases.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
import re
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Protocol, Set

import numpy as np

from .village import SEF

EmbedderFn = Callable[[str], np.ndarray]


class UncertaintyMethod(Protocol):
    """Protocol for pluggable query-dependent uncertainty estimators."""

    def __call__(self, query: str) -> float:
        """Return sigma² for query."""


@dataclass(frozen=True)
class UncertaintyMethodConfig:
    """Configuration for constructing uncertainty estimators via a common interface."""

    method: str
    base_sigma2: float = 0.2
    constant_value: float = 1.0
    keywords: Optional[Set[str]] = None
    keyword_in_domain_sigma2: float = 0.2
    keyword_out_domain_sigma2: float = 2.0
    stochastic_passes: int = 4


@dataclass(frozen=True)
class Sigma2Diagnostics:
    """Summary diagnostics for an uncertainty method over a query set."""

    n_queries: int
    min_sigma2: float
    max_sigma2: float
    mean_sigma2: float
    median_sigma2: float
    std_sigma2: float
    p10_sigma2: float
    p90_sigma2: float
    nonpositive_count: int


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
        sigma2 = self.base_sigma2 * (1.0 + uncertainty_multiplier + 0.2 * length_norm)
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


@dataclass(frozen=True)
class EmbeddingNormSigma2:
    """Cheap query-dependent sigma² from embedding norm / entropy-style proxy.

    Heuristic: lower embedding norm implies weaker or diffuse activation and therefore
    higher uncertainty. This method requires no retraining and one forward pass.
    """

    embed: EmbedderFn
    base_sigma2: float = 0.2
    alpha: float = 1.0
    norm_floor: float = 1e-6
    max_multiplier: float = 6.0

    def __call__(self, query: str) -> float:
        z = self.embed(query)
        norm = float(np.linalg.norm(z))
        safe_norm = max(norm, self.norm_floor)
        growth = np.log1p(self.alpha / safe_norm)
        sigma2 = self.base_sigma2 * (1.0 + growth)
        sigma2 = min(sigma2, self.base_sigma2 * self.max_multiplier)
        return float(max(sigma2, 1e-12))


@dataclass(frozen=True)
class SimilarityToCentroidSigma2:
    """Sigma² from cosine distance to specialist centroid.

    Similarity near 1.0 -> low uncertainty; farther queries -> higher uncertainty.
    """

    embed: EmbedderFn
    centroid: np.ndarray
    base_sigma2: float = 0.2
    slope: float = 2.0
    max_multiplier: float = 8.0

    @classmethod
    def from_calibration(
        cls,
        *,
        embed: EmbedderFn,
        calibration_texts: Iterable[str],
        base_sigma2: float = 0.2,
        slope: float = 2.0,
        max_multiplier: float = 8.0,
    ) -> "SimilarityToCentroidSigma2":
        embs = [embed(text) for text in calibration_texts]
        if not embs:
            raise ValueError("calibration_texts must not be empty")
        centroid = np.mean(np.stack(embs, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-12)
        return cls(
            embed=embed,
            centroid=centroid,
            base_sigma2=base_sigma2,
            slope=slope,
            max_multiplier=max_multiplier,
        )

    def __call__(self, query: str) -> float:
        z = self.embed(query)
        z = z / (np.linalg.norm(z) + 1e-12)
        sim = float(np.clip(z @ self.centroid, -1.0, 1.0))
        distance = 1.0 - sim
        sigma2 = self.base_sigma2 * (1.0 + self.slope * distance)
        sigma2 = min(sigma2, self.base_sigma2 * self.max_multiplier)
        return float(max(sigma2, 1e-12))


@dataclass(frozen=True)
class StochasticForwardSigma2:
    """Sigma² from stochastic forward passes (e.g., dropout-enabled embedder).

    The embed_stochastic callable should produce different vectors across calls when
    stochasticity is enabled; if deterministic, uncertainty falls back to base_sigma2.
    """

    embed_stochastic: EmbedderFn
    base_sigma2: float = 0.2
    n_passes: int = 4
    max_multiplier: float = 12.0

    def __call__(self, query: str) -> float:
        passes = max(int(self.n_passes), 2)
        samples = [self.embed_stochastic(query) for _ in range(passes)]
        stack = np.stack(samples, axis=0)
        per_dim_var = np.var(stack, axis=0, ddof=1)
        scalar_var = float(np.mean(np.clip(per_dim_var, 0.0, None)))
        sigma2 = self.base_sigma2 * (1.0 + scalar_var)
        sigma2 = min(sigma2, self.base_sigma2 * self.max_multiplier)
        return float(max(sigma2, 1e-12))


def apply_uncertainty_baseline_to_specialists(
    specialists: Iterable[SEF],
    *,
    method: str = "embedding_norm",
    calibration_texts_by_name: Optional[Mapping[str, Iterable[str]]] = None,
    base_sigma2: float = 0.2,
) -> List[SEF]:
    """Attach a query-dependent uncertainty method across specialists.

    This provides a practical baseline without retraining the base models and creates
    a hook to later swap more advanced methods.
    """
    updated: List[SEF] = []
    for sef in specialists:
        sigma2 = build_uncertainty_method(
            method=method,
            embed=sef.embed,
            base_sigma2=base_sigma2,
            calibration_texts=(
                calibration_texts_by_name.get(sef.name, [])
                if calibration_texts_by_name is not None
                else None
            ),
        )
        updated.append(replace(sef, sigma2=sigma2))
    return updated


def build_uncertainty_method(
    *,
    method: str,
    embed: EmbedderFn,
    base_sigma2: float = 0.2,
    calibration_texts: Optional[Iterable[str]] = None,
) -> UncertaintyMethod:
    """Factory hook for pluggable uncertainty estimators."""
    normalized = method.strip().lower()
    if normalized in ("constant", "constant_sigma2"):
        return ConstantSigma2(value=base_sigma2)
    if normalized in ("keyword", "keyword_based", "keyword_based_sigma2"):
        provided_keywords = set(calibration_texts or [])
        if len(provided_keywords) == 0:
            raise ValueError(
                "keyword_based_sigma2 requires non-empty calibration_texts."
            )
        return KeywordSigma2(
            keywords=provided_keywords,
            in_domain_sigma2=base_sigma2,
            out_domain_sigma2=max(base_sigma2 * 8.0, base_sigma2 + 1e-6),
        )
    if normalized in ("centroid_distance", "centroid_distance_sigma2"):
        texts = list(calibration_texts or [])
        if not texts:
            raise ValueError(
                "centroid_distance_sigma2 requires non-empty calibration_texts."
            )
        return CentroidDistanceSigma2.from_calibration(
            embed=embed,
            calibration_texts=texts,
            base_sigma2=base_sigma2,
        )
    if normalized == "embedding_norm":
        return EmbeddingNormSigma2(embed=embed, base_sigma2=base_sigma2)
    if normalized == "embedding_norm_sigma2":
        return EmbeddingNormSigma2(embed=embed, base_sigma2=base_sigma2)
    if normalized == "centroid_similarity":
        texts = list(calibration_texts or [])
        if not texts:
            raise ValueError(
                "centroid_similarity requires non-empty calibration_texts."
            )
        return SimilarityToCentroidSigma2.from_calibration(
            embed=embed, calibration_texts=texts, base_sigma2=base_sigma2
        )
    if normalized == "stochastic_forward":
        return StochasticForwardSigma2(embed_stochastic=embed, base_sigma2=base_sigma2)
    if normalized == "stochastic_forward_sigma2":
        return StochasticForwardSigma2(embed_stochastic=embed, base_sigma2=base_sigma2)
    raise ValueError(f"Unknown uncertainty method: {method}")


def create_uncertainty_method(
    *,
    config: UncertaintyMethodConfig,
    embed: EmbedderFn,
    calibration_texts: Optional[Iterable[str]] = None,
) -> UncertaintyMethod:
    """Common evaluation interface to build all uncertainty methods uniformly."""
    normalized = config.method.strip().lower()
    if normalized in ("constant", "constant_sigma2"):
        return ConstantSigma2(value=config.constant_value)
    if normalized in ("keyword", "keyword_based", "keyword_based_sigma2"):
        keywords = set(config.keywords or _extract_keywords(calibration_texts or []))
        if not keywords:
            raise ValueError(
                "keyword_based_sigma2 requires keywords or calibration_texts."
            )
        return KeywordSigma2(
            keywords=keywords,
            in_domain_sigma2=config.keyword_in_domain_sigma2,
            out_domain_sigma2=config.keyword_out_domain_sigma2,
        )
    if normalized in ("centroid_distance", "centroid_distance_sigma2"):
        texts = list(calibration_texts or [])
        if not texts:
            raise ValueError("centroid_distance_sigma2 requires calibration_texts.")
        return CentroidDistanceSigma2.from_calibration(
            embed=embed,
            calibration_texts=texts,
            base_sigma2=config.base_sigma2,
        )
    if normalized in ("embedding_norm", "embedding_norm_sigma2"):
        return EmbeddingNormSigma2(embed=embed, base_sigma2=config.base_sigma2)
    if normalized in ("stochastic_forward", "stochastic_forward_sigma2"):
        return StochasticForwardSigma2(
            embed_stochastic=embed,
            base_sigma2=config.base_sigma2,
            n_passes=config.stochastic_passes,
        )
    raise ValueError(f"Unknown uncertainty method: {config.method}")


def _extract_keywords(texts: Iterable[str], max_keywords: int = 24) -> List[str]:
    tokens: Dict[str, int] = {}
    for text in texts:
        for token in re.findall(r"[a-zA-Z]{3,}", text.lower()):
            tokens[token] = tokens.get(token, 0) + 1
    ranked = sorted(tokens.items(), key=lambda kv: (-kv[1], kv[0]))
    return [token for token, _ in ranked[:max_keywords]]


def summarize_uncertainty_distribution(
    sigma2_fn: Callable[[str], float], queries: Iterable[str]
) -> Sigma2Diagnostics:
    """Compute simple diagnostics to inspect sigma² distribution."""
    values = np.asarray([float(sigma2_fn(q)) for q in queries], dtype=np.float64)
    if values.size == 0:
        raise ValueError("queries must not be empty")
    return Sigma2Diagnostics(
        n_queries=int(values.size),
        min_sigma2=float(np.min(values)),
        max_sigma2=float(np.max(values)),
        mean_sigma2=float(np.mean(values)),
        median_sigma2=float(np.median(values)),
        std_sigma2=float(np.std(values)),
        p10_sigma2=float(np.percentile(values, 10)),
        p90_sigma2=float(np.percentile(values, 90)),
        nonpositive_count=int(np.sum(values <= 0.0)),
    )


def uncertainty_histogram(
    sigma2_fn: Callable[[str], float], queries: Iterable[str], bins: int = 10
) -> Dict[str, List[float]]:
    """Histogram diagnostics for sigma² distribution (for quick plotting/logging)."""
    values = np.asarray([float(sigma2_fn(q)) for q in queries], dtype=np.float64)
    if values.size == 0:
        raise ValueError("queries must not be empty")
    counts, edges = np.histogram(values, bins=max(int(bins), 2))
    return {"counts": counts.astype(float).tolist(), "bin_edges": edges.tolist()}
