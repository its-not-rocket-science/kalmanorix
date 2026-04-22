"""
Unified fusion orchestration for Kalmanorix.

This module provides:
- The `Potion` data structure (fused embedding + metadata)
- The `Fuser` abstraction for different fusion strategies
- Baseline fusers (Mean, Kalmanorix, DiagonalKalman, LearnedGate)
- The `Panoramix` orchestrator that combines routing and fusion

The design keeps routing, fusion, and uncertainty estimation cleanly separated.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

import numpy as np

from .scout import ScoutRouter
from .village import SEF, Village
from .types import Vec
from .kalman_engine.kalman_fuser import (
    kalman_fuse_diagonal,
    kalman_fuse_diagonal_ensemble,
    kalman_fuse_structured,
    kalman_fuse_diagonal_batch,
    kalman_fuse_diagonal_ensemble_batch,
)
from .kalman_engine.structured_covariance import StructuredCovariance
from .kalman_engine.correlation import (
    ResidualCorrelationProfile,
    correlation_inflation_factors,
    effective_sample_size_discount,
)


logger = logging.getLogger(__name__)


def _aligned_embedding(module: SEF, query: str) -> np.ndarray:
    """Compute one specialist embedding in shared aligned space."""
    emb = np.asarray(module.embed(query), dtype=np.float64)
    if module.alignment_matrix is not None:
        emb = np.asarray(module.alignment_matrix @ emb, dtype=np.float64)
    return emb


def _aligned_embeddings_for_query(modules: List[SEF], query: str) -> List[np.ndarray]:
    """Compute aligned embeddings once for all modules for a single query."""
    return [_aligned_embedding(module, query) for module in modules]


def _aligned_embeddings_for_batch(modules: List[SEF], queries: List[str]) -> np.ndarray:
    """Compute aligned embeddings once for all modules and queries.

    Returns:
        Array with shape (n_modules, n_queries, d).
    """
    embeddings = []
    for module in modules:
        module_embs = [_aligned_embedding(module, query) for query in queries]
        embeddings.append(np.stack(module_embs, axis=0))
    return np.stack(embeddings, axis=0).astype(np.float64, copy=False)


def _sigma2_for_embeddings(
    modules: List[SEF], query: str, embeddings: List[np.ndarray]
) -> np.ndarray:
    """Evaluate sigma² once per module for a single query using cached embeddings."""
    return np.asarray(
        [
            float(module.sigma2_for(query, query_embedding=emb))
            for module, emb in zip(modules, embeddings)
        ],
        dtype=np.float64,
    )


def _sigma2_for_batch_embeddings(
    modules: List[SEF], queries: List[str], embeddings: np.ndarray
) -> np.ndarray:
    """Evaluate sigma² once per (module, query) pair using cached embeddings.

    Args:
        modules: Specialist modules, shape ``(n_modules,)``.
        queries: Query strings, shape ``(n_queries,)``.
        embeddings: Precomputed aligned embeddings with shape
            ``(n_modules, n_queries, d)``.

    Returns:
        Sigma² matrix with shape ``(n_modules, n_queries)``.
    """
    n_modules = len(modules)
    n_queries = len(queries)
    sigma2 = np.empty((n_modules, n_queries), dtype=np.float64)
    for i, module in enumerate(modules):
        for j, query in enumerate(queries):
            sigma2[i, j] = float(
                module.sigma2_for(query, query_embedding=embeddings[i, j])
            )
    return sigma2


@dataclass(frozen=True)
class Potion:  # pylint: disable=too-few-public-methods
    """Result of a fusion operation.

    Attributes:
        vector: The fused embedding vector.
        weights: Per-module fusion weights.
        meta: Optional diagnostic metadata (e.g. gate values).
    """

    vector: Vec
    weights: Dict[str, float]
    meta: Optional[Dict[str, object]] = None


class Fuser(ABC):  # pylint: disable=too-few-public-methods
    """
    Abstract base class for fusion strategies.

    A Fuser takes embeddings from multiple modules and combines them into
    a single vector, optionally producing diagnostic metadata.
    """

    @abstractmethod
    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """Fuse embeddings from the given modules.

        Args:
            query: Input text query.
            modules: List of selected specialist modules.
            precomputed_embeddings: Optional aligned embeddings (same order as
                ``modules``). If provided, fusers should reuse these vectors
                rather than re-embedding the query.

        Returns:
            vector: Fused embedding.
            weights: Per-module contribution weights.
            meta: Optional metadata.
        """
        raise NotImplementedError

    def fuse_batch(
        self,
        queries: List[str],
        modules: List[SEF],
        *,
        precomputed_batch_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[Vec], List[Dict[str, float]], Optional[List[Dict[str, object]]]]:
        """Fuse embeddings for a batch of queries.

        Default implementation loops over queries and calls `fuse`.
        Subclasses may override with more efficient batch implementations.

        Args:
            queries: List of input text queries.
            modules: List of selected specialist modules.
            precomputed_batch_embeddings: Optional aligned batch embeddings with
                shape ``(n_modules, n_queries, d)``.

        Returns:
            vectors: List of fused embeddings.
            weights: List of per-module contribution weights.
            meta: Optional list of metadata dicts.
        """
        vectors = []
        weights_list = []
        meta_list = []
        for i, query in enumerate(queries):
            precomputed = (
                [
                    np.asarray(
                        precomputed_batch_embeddings[module_idx, i], dtype=np.float64
                    )
                    for module_idx in range(precomputed_batch_embeddings.shape[0])
                ]
                if precomputed_batch_embeddings is not None
                else None
            )
            vec, w, m = self.fuse(query, modules, precomputed_embeddings=precomputed)
            vectors.append(vec)
            weights_list.append(w)
            meta_list.append(m if m is not None else {})
        # If all metadata dicts are empty, return None
        if all(not m for m in meta_list):
            return vectors, weights_list, None
        return vectors, weights_list, meta_list


class MeanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    Uniform averaging baseline.

    All modules contribute equally, regardless of uncertainty or query content.
    """

    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        embeddings = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        z = np.stack(embeddings, axis=0)
        w = {m.name: 1.0 / len(modules) for m in modules}
        return z.mean(axis=0), w, None

    def fuse_batch(
        self,
        queries: List[str],
        modules: List[SEF],
        *,
        precomputed_batch_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[Vec], List[Dict[str, float]], Optional[List[Dict[str, object]]]]:
        """Batch fusion for MeanFuser."""
        n = len(modules)
        b = len(queries)
        if n == 0:
            raise ValueError("MeanFuser requires at least one module")
        if b == 0:
            return [], [], None
        # Collect embeddings: shape (n, b, d)
        embeddings_stack = (
            precomputed_batch_embeddings
            if precomputed_batch_embeddings is not None
            else _aligned_embeddings_for_batch(modules, queries)
        )
        # Mean across modules: (b, d)
        fused = np.mean(embeddings_stack, axis=0)
        # Uniform weights per query
        weight = 1.0 / n
        weights_list = [{module.name: weight for module in modules} for _ in range(b)]
        # No metadata
        return list(fused), weights_list, None


class KalmanorixFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    True Kalman fusion with diagonal covariance.

    Uses the core `kalman_fuse_diagonal` algorithm with per-dimension
    diagonal covariance. Converts each module's scalar sigma² to a
    diagonal covariance matrix (sigma² * I).
    """

    def __init__(
        self,
        sort_by_certainty: bool = True,
        epsilon: float = 1e-8,
        use_fast_scalar_path: bool = True,
    ):
        """
        Args:
            sort_by_certainty: Sort measurements by certainty before fusion
            epsilon: Small constant for numerical stability
            use_fast_scalar_path: Use optimized scalar-sigma² Kalman path that
                avoids constructing repeated diagonal covariance vectors.
        """
        self.sort_by_certainty = sort_by_certainty
        self.epsilon = epsilon
        self.use_fast_scalar_path = use_fast_scalar_path

    def _fuse_legacy(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """Reference implementation retained for regression benchmarking."""
        aligned = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        embeddings = [np.asarray(emb, dtype=np.float64) for emb in aligned]
        sigma2_values = _sigma2_for_embeddings(modules, query, embeddings)
        covariances = []
        for emb, sigma2 in zip(embeddings, sigma2_values):
            cov = np.full(emb.shape, sigma2, dtype=np.float64)
            covariances.append(cov)

        fused, fused_cov = kalman_fuse_diagonal(
            embeddings,
            covariances,
            sort_by_certainty=self.sort_by_certainty,
            epsilon=self.epsilon,
        )

        total_uncertainties = [float(np.sum(cov)) for cov in covariances]
        inv_uncertainties = [1.0 / (tu + self.epsilon) for tu in total_uncertainties]
        total_inv = sum(inv_uncertainties)
        weights = {
            module.name: float(inv_uncertainties[i] / total_inv)
            for i, module in enumerate(modules)
        }
        meta = {
            "fused_covariance": fused_cov,
            "total_uncertainties": total_uncertainties,
            "sort_by_certainty": self.sort_by_certainty,
            "variance": float(np.mean(fused_cov)),
            "implementation": "legacy",
        }
        return fused, weights, meta

    def _fuse_scalar_sigma2(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """Optimized implementation equivalent to sigma²*I covariance fusion."""
        aligned = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        embeddings = [np.asarray(emb, dtype=np.float64) for emb in aligned]
        sigma2_values = _sigma2_for_embeddings(modules, query, embeddings)

        order = np.arange(len(modules))
        if self.sort_by_certainty and len(modules) > 1:
            order = np.argsort(np.asarray(sigma2_values, dtype=np.float64))

        x = embeddings[int(order[0])].copy()
        p = max(float(sigma2_values[int(order[0])]), self.epsilon)
        for idx in order[1:]:
            r = max(float(sigma2_values[int(idx)]), self.epsilon)
            k = p / (p + r + self.epsilon)
            x += k * (embeddings[int(idx)] - x)
            p = max((1.0 - k) * p, self.epsilon)

        d = int(x.shape[0])
        fused_cov = np.full((d,), p, dtype=np.float64)
        total_uncertainties = [d * float(s) for s in sigma2_values.tolist()]
        inv_uncertainties = [1.0 / (tu + self.epsilon) for tu in total_uncertainties]
        total_inv = sum(inv_uncertainties)
        weights = {
            module.name: float(inv_uncertainties[i] / total_inv)
            for i, module in enumerate(modules)
        }
        meta = {
            "fused_covariance": fused_cov,
            "total_uncertainties": total_uncertainties,
            "sort_by_certainty": self.sort_by_certainty,
            "variance": float(p),
            "implementation": "optimized_scalar_sigma2",
        }
        return x, weights, meta

    def fuse(  # pylint: disable=too-many-locals
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("KalmanorixFuser requires at least one module")

        if self.use_fast_scalar_path:
            return self._fuse_scalar_sigma2(
                query=query,
                modules=modules,
                precomputed_embeddings=precomputed_embeddings,
            )
        return self._fuse_legacy(
            query=query,
            modules=modules,
            precomputed_embeddings=precomputed_embeddings,
        )

    def fuse_batch(
        self,
        queries: List[str],
        modules: List[SEF],
        *,
        precomputed_batch_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[Vec], List[Dict[str, float]], Optional[List[Dict[str, object]]]]:
        """Batch fusion for KalmanorixFuser."""
        n = len(modules)
        b = len(queries)
        if n == 0:
            raise ValueError("KalmanorixFuser requires at least one module")
        if b == 0:
            return [], [], None

        embeddings_arr = (
            precomputed_batch_embeddings
            if precomputed_batch_embeddings is not None
            else _aligned_embeddings_for_batch(modules, queries)
        )
        sigma2 = _sigma2_for_batch_embeddings(modules, queries, embeddings_arr)

        if self.use_fast_scalar_path:
            if self.sort_by_certainty and n > 1:
                avg_sigma2 = np.mean(sigma2, axis=1)
                order = np.argsort(avg_sigma2)
                embeddings_arr = embeddings_arr[order]
                sigma2 = sigma2[order]

            x = embeddings_arr[0].copy()
            p = np.maximum(sigma2[0].copy(), self.epsilon)
            for i in range(1, n):
                r = np.maximum(sigma2[i], self.epsilon)
                k = p / (p + r + self.epsilon)
                x += k[:, None] * (embeddings_arr[i] - x)
                p = np.maximum((1.0 - k) * p, self.epsilon)
            fused = x
            d = embeddings_arr.shape[2]
            fused_cov = np.repeat(p[:, None], d, axis=1)
            total_uncertainties = sigma2 * float(d)
        else:
            covariances = sigma2[:, :, None] * np.ones(
                (1, 1, embeddings_arr.shape[2]), dtype=np.float64
            )
            fused, fused_cov = kalman_fuse_diagonal_batch(
                embeddings_arr,
                covariances,
                sort_by_certainty=self.sort_by_certainty,
                epsilon=self.epsilon,
            )
            total_uncertainties = np.sum(covariances, axis=2)

        inv_uncertainties = 1.0 / (total_uncertainties + self.epsilon)
        total_inv = np.sum(inv_uncertainties, axis=0)  # shape (b,)
        weights_per_module = inv_uncertainties / total_inv  # shape (n, b)

        # Convert to list of dicts
        weights_list = []
        for j in range(b):
            w = {modules[i].name: float(weights_per_module[i, j]) for i in range(n)}
            weights_list.append(w)

        # Prepare metadata per query
        meta_list = []
        for j in range(b):
            meta = {
                "fused_covariance": fused_cov[j],
                "total_uncertainties": total_uncertainties[:, j].tolist(),
                "sort_by_certainty": self.sort_by_certainty,
                "variance": float(np.mean(fused_cov[j])),
                "implementation": (
                    "optimized_scalar_sigma2" if self.use_fast_scalar_path else "legacy"
                ),
            }
            meta_list.append(meta)

        return list(fused), weights_list, meta_list


class EnsembleKalmanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    Ensemble Kalman fusion with parallel updates.

    Uses the ensemble Kalman filter formulation that processes all measurements
    simultaneously (parallel updates). For diagonal covariance and independent
    measurements, this is mathematically equivalent to sequential Kalman updates
    but can be more computationally efficient.

    This implementation uses `kalman_fuse_diagonal_ensemble` for parallel fusion.
    """

    def __init__(self, epsilon: float = 1e-8):
        """
        Args:
            epsilon: Small constant for numerical stability
        """
        self.epsilon = epsilon

    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("EnsembleKalmanFuser requires at least one module")

        # Get embeddings and convert scalar sigma² to diagonal covariance
        embeddings = []
        covariances = []

        aligned = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        for module, emb in zip(modules, aligned):
            sigma2 = module.sigma2_for(query, query_embedding=emb)
            # Convert scalar variance to diagonal covariance vector
            cov = np.full(emb.shape, sigma2, dtype=np.float64)
            embeddings.append(emb)
            covariances.append(cov)

        # Perform ensemble Kalman fusion (parallel updates)
        fused, fused_cov = kalman_fuse_diagonal_ensemble(
            embeddings,
            covariances,
            epsilon=self.epsilon,
        )

        # Compute weights from precision contributions
        # Weight proportional to total precision (sum of 1/covariance)
        total_precisions = []
        for cov in covariances:
            inv_cov = 1.0 / (cov + self.epsilon)
            total_precisions.append(np.sum(inv_cov))

        total_precisions_sum = sum(total_precisions)
        weights = {
            module.name: float(prec / total_precisions_sum)
            for module, prec in zip(modules, total_precisions)
        }

        meta = {
            "fused_covariance": fused_cov,
            "total_precisions": total_precisions,
            "epsilon": self.epsilon,
            "variance": float(np.mean(fused_cov)),
        }

        return fused, weights, meta

    def fuse_batch(
        self,
        queries: List[str],
        modules: List[SEF],
        *,
        precomputed_batch_embeddings: Optional[np.ndarray] = None,
    ) -> Tuple[List[Vec], List[Dict[str, float]], Optional[List[Dict[str, object]]]]:
        """Batch fusion for EnsembleKalmanFuser."""
        n = len(modules)
        b = len(queries)
        if n == 0:
            raise ValueError("EnsembleKalmanFuser requires at least one module")
        if b == 0:
            return [], [], None

        # Build arrays: embeddings (n, b, d), covariances (n, b, d)
        embeddings_list = []
        covariances_list = []
        embeddings_arr = (
            precomputed_batch_embeddings
            if precomputed_batch_embeddings is not None
            else _aligned_embeddings_for_batch(modules, queries)
        )
        for i, module in enumerate(modules):
            module_embs = []
            module_covs = []
            for j, query in enumerate(queries):
                emb = embeddings_arr[i, j]
                sigma2 = module.sigma2_for(query, query_embedding=emb)
                cov = np.full(emb.shape, sigma2, dtype=np.float64)
                module_embs.append(emb)
                module_covs.append(cov)
            # Stack across queries: (b, d)
            embeddings_list.append(np.stack(module_embs, axis=0))
            covariances_list.append(np.stack(module_covs, axis=0))
        # Stack across modules: (n, b, d)
        embeddings = np.stack(embeddings_list, axis=0)
        covariances = np.stack(covariances_list, axis=0)

        # Perform batch ensemble Kalman fusion
        fused, fused_cov = kalman_fuse_diagonal_ensemble_batch(
            embeddings,
            covariances,
            epsilon=self.epsilon,
        )
        # fused shape (b, d), fused_cov shape (b, d)

        # Compute weights per module per query based on total precision
        # total precision per module per query: sum of 1/covariance over dimensions
        inv_cov = 1.0 / (covariances + self.epsilon)  # shape (n, b, d)
        total_precisions = np.sum(inv_cov, axis=2)  # shape (n, b)
        total_precisions_sum = np.sum(total_precisions, axis=0)  # shape (b,)
        # Normalize
        weights_per_module = total_precisions / total_precisions_sum  # shape (n, b)

        # Convert to list of dicts
        weights_list = []
        for j in range(b):
            w = {modules[i].name: float(weights_per_module[i, j]) for i in range(n)}
            weights_list.append(w)

        # Prepare metadata per query
        meta_list = []
        for j in range(b):
            meta = {
                "fused_covariance": fused_cov[j],
                "total_precisions": total_precisions[:, j].tolist(),
                "epsilon": self.epsilon,
                "variance": float(np.mean(fused_cov[j])),
            }
            meta_list.append(meta)

        return list(fused), weights_list, meta_list


class StructuredKalmanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    Kalman fusion with structured covariance (diagonal + low‑rank).

    Uses `kalman_fuse_structured` to fuse embeddings with structured covariance
    matrices of the form R = D + UUᵀ. If a module provides a structured covariance
    (via `get_structured_covariance`), it is used directly; otherwise a diagonal‑only
    structured covariance is created from the module's scalar sigma².
    """

    def __init__(self, sort_by_certainty: bool = True, epsilon: float = 1e-8):
        """
        Args:
            sort_by_certainty: Sort measurements by certainty before fusion
            epsilon: Small constant for numerical stability
        """
        self.sort_by_certainty = sort_by_certainty
        self.epsilon = epsilon

    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("StructuredKalmanFuser requires at least one module")

        embeddings = []
        structured_covs = []

        aligned = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        for module, emb in zip(modules, aligned):
            # Try to get structured covariance
            structured_cov = module.get_structured_covariance(query)
            if structured_cov is None:
                # Fall back to diagonal covariance from sigma²
                sigma2 = module.sigma2_for(query, query_embedding=emb)
                diagonal = np.full(emb.shape, sigma2, dtype=np.float64)
                structured_cov = StructuredCovariance.from_diagonal(diagonal)

            embeddings.append(emb)
            structured_covs.append(structured_cov)

        # Perform structured Kalman fusion
        fused, fused_cov = kalman_fuse_structured(
            embeddings,
            structured_covs,
            sort_by_certainty=self.sort_by_certainty,
            epsilon=self.epsilon,
        )

        # Compute weights from uncertainty scores (total variance)
        uncertainty_scores = [cov.uncertainty_score() for cov in structured_covs]
        inv_uncertainties = [1.0 / (u + self.epsilon) for u in uncertainty_scores]
        total_inv = sum(inv_uncertainties)
        weights = {
            module.name: float(inv_uncertainties[i] / total_inv)
            for i, module in enumerate(modules)
        }

        meta = {
            "fused_covariance": fused_cov,
            "uncertainty_scores": uncertainty_scores,
            "sort_by_certainty": self.sort_by_certainty,
            "variance": float(np.mean(fused_cov)),
            "has_lowrank": any(not cov.is_diagonal for cov in structured_covs),
        }

        return fused, weights, meta


class CorrelationAwareKalmanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """Kalman fusion that discounts redundant evidence from correlated experts.

    Variants:
    - ``covariance_inflation``: inflate each expert covariance by mean positive
      residual correlation to peers.
    - ``effective_sample_size``: apply a global precision discount
      ``discount=n_eff/n`` before ensemble fusion.
    """

    def __init__(
        self,
        correlation_profile: ResidualCorrelationProfile,
        mode: str = "covariance_inflation",
        inflation_alpha: float = 1.0,
        epsilon: float = 1e-8,
    ):
        if mode not in {"covariance_inflation", "effective_sample_size"}:
            raise ValueError(f"Unsupported mode: {mode}")
        self.correlation_profile = correlation_profile
        self.mode = mode
        self.inflation_alpha = inflation_alpha
        self.epsilon = epsilon

    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("CorrelationAwareKalmanFuser requires at least one module")

        embeddings = []
        base_covariances = []
        module_names = []
        aligned = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        for module, emb in zip(modules, aligned):
            sigma2 = module.sigma2_for(query, query_embedding=emb)
            cov = np.full(emb.shape, sigma2, dtype=np.float64)
            embeddings.append(emb)
            base_covariances.append(cov)
            module_names.append(module.name)

        corr_sub = self.correlation_profile.submatrix(module_names)

        adjusted_covariances = [cov.copy() for cov in base_covariances]
        ess_discount = 1.0
        inflation = np.ones(len(modules), dtype=np.float64)

        if self.mode == "covariance_inflation":
            inflation = correlation_inflation_factors(
                corr_submatrix=corr_sub, alpha=self.inflation_alpha
            )
            adjusted_covariances = [
                cov * inflation[i] for i, cov in enumerate(base_covariances)
            ]
        else:
            ess_discount = effective_sample_size_discount(corr_sub)
            adjusted_covariances = [cov / ess_discount for cov in base_covariances]

        fused, fused_cov = kalman_fuse_diagonal_ensemble(
            embeddings,
            adjusted_covariances,
            epsilon=self.epsilon,
        )

        total_precisions = []
        for cov in adjusted_covariances:
            total_precisions.append(float(np.sum(1.0 / (cov + self.epsilon))))
        precision_sum = sum(total_precisions)
        weights = {
            m.name: float(total_precisions[i] / precision_sum)
            for i, m in enumerate(modules)
        }
        meta = {
            "fused_covariance": fused_cov,
            "variance": float(np.mean(fused_cov)),
            "correlation_mode": self.mode,
            "inflation_factors": inflation.tolist(),
            "effective_sample_size_discount": float(ess_discount),
            "correlation_submatrix": corr_sub,
        }
        return fused, weights, meta


class DiagonalKalmanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    Sequential diagonal Kalman-style fusion with scalar prior variance.

    Legacy implementation (scalar prior variance).
    Maintains a scalar prior variance P (shared across dimensions).
    """

    def __init__(
        self, *, prior_sigma2: float = 1.0, sort_by_sigma2: bool = True
    ) -> None:
        """Initialize diagonal Kalman fuser with scalar prior variance.

        Args:
            prior_sigma2: Prior variance shared across dimensions.
            sort_by_sigma2: Sort measurements by variance before fusion.
        """
        self.prior_sigma2 = float(prior_sigma2)
        self.sort_by_sigma2 = bool(sort_by_sigma2)

    def _kalman_updates(
        self, order: List[str], z_by_name: Dict[str, Vec], r_by_name: Dict[str, float]
    ) -> Tuple[Vec, Dict[str, float], float]:
        """Perform sequential Kalman updates for diagonal covariance.

        Args:
            order: Module names in update order.
            z_by_name: Mapping from module name to embedding vector.
            r_by_name: Mapping from module name to scalar variance.

        Returns:
            Tuple of (fused embedding, Kalman gains per module, posterior variance).
        """
        x = np.zeros_like(z_by_name[order[0]])
        p = float(self.prior_sigma2)
        k_by_name: Dict[str, float] = {}
        for name in order:
            r = max(r_by_name[name], 1e-12)
            k = p / (p + r)
            x = x + (k * (z_by_name[name] - x))
            p = (1.0 - k) * p
            k_by_name[name] = float(k)
        return x, k_by_name, p

    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """Fuse embeddings using diagonal Kalman filter with scalar prior variance.

        Args:
            query: Input text query.
            modules: List of selected specialist modules.

        Returns:
            Tuple of (fused embedding, per-module weights, metadata).
        """
        if not modules:
            raise ValueError("DiagonalKalmanFuser requires at least one module")

        z_by_name: Dict[str, Vec] = {}
        r_by_name: Dict[str, float] = {}
        aligned = (
            precomputed_embeddings
            if precomputed_embeddings is not None
            else _aligned_embeddings_for_query(modules, query)
        )
        for m, emb in zip(modules, aligned):
            z_by_name[m.name] = emb
            r_by_name[m.name] = float(m.sigma2_for(query, query_embedding=emb))

        order = [m.name for m in modules]
        if self.sort_by_sigma2:
            order = sorted(order, key=lambda n: r_by_name[n])

        x, k_by_name, p = self._kalman_updates(order, z_by_name, r_by_name)

        # Convert K's into a stable weight readout (purely diagnostic)
        k_sum = sum(k_by_name.values()) + 1e-12
        weights = {name: float(k / k_sum) for name, k in k_by_name.items()}

        meta: Dict[str, object] = {
            "prior_sigma2": float(self.prior_sigma2),
            "post_sigma2": float(p),
            "variance": float(p),
            "kalman_gains": dict(k_by_name),
            "order": list(order),
            "sort_by_sigma2": self.sort_by_sigma2,
        }
        return x, weights, meta


class LearnedGateFuser(Fuser):
    """
    Learned two-way gating baseline.

    Predicts a scalar α(query) ∈ [0, 1] and returns:
        α * z_a + (1 - α) * z_b

    This serves as a critical learned baseline against which Kalman-style
    fusion should be compared.
    """

    def __init__(
        self,
        module_a: str,
        module_b: str,
        *,
        n_features: int = 256,
        lr: float = 0.3,
        l2: float = 1e-3,
        steps: int = 300,
    ) -> None:
        """Initialize learned gate fuser between two modules.

        Args:
            module_a: Name of first module.
            module_b: Name of second module.
            n_features: Dimension of hashed bag-of-words features.
            lr: Learning rate for logistic regression.
            l2: L2 regularization strength.
            steps: Number of training steps.
        """
        self.module_a = module_a
        self.module_b = module_b
        self.n_features = int(n_features)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.steps = int(steps)

        # Logistic regression weights (w[0] is bias)
        self.w = np.zeros(self.n_features + 1, dtype=np.float64)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Very simple alphanumeric tokenizer."""
        text = text.lower()
        out: List[str] = []
        buff: List[str] = []
        for ch in text:
            if ch.isalnum():
                buff.append(ch)
            else:
                if buff:
                    out.append("".join(buff))
                    buff.clear()
        if buff:
            out.append("".join(buff))
        return out

    @staticmethod
    def _stable_hash(token: str) -> int:
        """Stable hash (independent of Python's hash randomization)."""
        import hashlib

        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], byteorder="little", signed=False)

    def _featurize(self, text: str) -> Vec:
        """Convert text to a normalized hashed bag-of-words feature vector."""
        x = np.zeros(self.n_features + 1, dtype=np.float64)
        x[0] = 1.0  # bias

        for tok in self._tokenize(text):
            idx = 1 + (self._stable_hash(tok) % self.n_features)
            x[idx] += 1.0

        norm = np.linalg.norm(x[1:]) + 1e-12
        x[1:] /= norm
        return x

    @staticmethod
    def _sigmoid(t: float) -> float:
        """Numerically stable sigmoid."""
        if t >= 0:
            z = np.exp(-t)
            return float(1.0 / (1.0 + z))
        z = np.exp(t)
        return float(z / (1.0 + z))

    def fit(self, texts: List[str], y: List[int]) -> None:
        """
        Train the gate using binary labels:
          1 => prefer module_a
          0 => prefer module_b
        """
        if len(texts) != len(y) or not texts:
            raise ValueError("texts and y must be the same non-zero length")

        X = np.stack([self._featurize(t) for t in texts], axis=0)
        Y = np.array(y, dtype=np.float64)

        if not np.all((Y == 0.0) | (Y == 1.0)):
            raise ValueError("y must be binary labels 0/1")

        for _ in range(self.steps):
            logits = X @ self.w
            P = np.array([self._sigmoid(float(z)) for z in logits])
            grad = (X.T @ (P - Y)) / len(Y)
            grad[1:] += self.l2 * self.w[1:]
            self.w -= self.lr * grad

    def predict_alpha(self, text: str) -> float:
        """Predict α ∈ [0, 1], the probability of choosing module_a."""
        x = self._featurize(text)
        return self._sigmoid(float(x @ self.w))

    def fuse(
        self,
        query: str,
        modules: List[SEF],
        *,
        precomputed_embeddings: Optional[List[np.ndarray]] = None,
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """Fuse two modules using learned gating weight α(query).

        Args:
            query: Input text query.
            modules: List of selected specialist modules.

        Returns:
            Tuple of (fused embedding, per-module weights, metadata).

        Raises:
            ValueError: If required modules are not in the module list.
        """
        by_name = {m.name: m for m in modules}
        if self.module_a not in by_name or self.module_b not in by_name:
            raise ValueError(
                f"LearnedGateFuser expects modules '{self.module_a}' and "
                f"'{self.module_b}', got {list(by_name.keys())}"
            )

        a = by_name[self.module_a]
        b = by_name[self.module_b]

        if precomputed_embeddings is not None:
            by_name_emb = {
                m.name: emb for m, emb in zip(modules, precomputed_embeddings)
            }
            z_a = by_name_emb[a.name]
            z_b = by_name_emb[b.name]
        else:
            z_a = _aligned_embedding(a, query)
            z_b = _aligned_embedding(b, query)

        alpha = self.predict_alpha(query)
        x = (alpha * z_a) + ((1.0 - alpha) * z_b)

        weights = {a.name: float(alpha), b.name: float(1.0 - alpha)}
        meta = {"alpha": float(alpha), "gate": "hashed_logreg"}
        return x, weights, meta


@dataclass
class Panoramix:  # pylint: disable=too-few-public-methods
    """Orchestrator that combines routing and fusion.

    Panoramix:
      1. asks a ScoutRouter which modules to consult
      2. delegates fusion to a Fuser
      3. packages the result as a Potion

    Attributes:
        fuser: Fusion strategy implementing the Fuser interface.
        enable_nonlinear_alignment_fallback: Allow fallback if Procrustes fails.
        procrustes_failure_similarity_threshold: Failure threshold for mean similarity.
    """

    fuser: Fuser
    use_shared_embedding_path: bool = True
    enable_nonlinear_alignment_fallback: bool = False
    procrustes_failure_similarity_threshold: float = 0.5

    def should_use_nonlinear_alignment(
        self, mean_similarity_after_alignment: float
    ) -> bool:
        """Return whether nonlinear fallback should be used.

        Procrustes is considered failed when mean similarity after alignment is
        below `procrustes_failure_similarity_threshold`.
        """
        return (
            self.enable_nonlinear_alignment_fallback
            and mean_similarity_after_alignment
            < self.procrustes_failure_similarity_threshold
        )

    def brew(self, query: str, village: Village, scout: ScoutRouter) -> Potion:
        """Produce a fused embedding for a query.

        Args:
            query: Input text query.
            village: Collection of available specialist modules.
            scout: Routing strategy.

        Returns:
            Potion: The fused embedding and diagnostics.
        """
        chosen = scout.select(query, village)
        precomputed_embeddings = (
            _aligned_embeddings_for_query(chosen, query)
            if self.use_shared_embedding_path
            else None
        )
        vec, weights, fuser_meta = self.fuser.fuse(
            query,
            chosen,
            precomputed_embeddings=precomputed_embeddings,
        )
        meta: Dict[str, object] = {"selected_modules": [m.name for m in chosen]}
        if fuser_meta is not None:
            meta.update(fuser_meta)
        return Potion(vector=vec, weights=weights, meta=meta)

    def brew_batch(
        self,
        queries: List[str],
        village: Village,
        scout: ScoutRouter,
    ) -> List[Potion]:
        """Produce fused embeddings for a batch of queries.

        Args:
            queries: List of input text queries.
            village: Collection of available specialist modules.
            scout: Routing strategy.

        Returns:
            List[Potion]: List of fused embeddings and diagnostics.
        """
        # First, determine which modules are selected for each query
        chosen_list = [scout.select(query, village) for query in queries]
        # Check if all selections are identical (same modules in same order)
        first_chosen = chosen_list[0]
        all_same = all(
            len(chosen) == len(first_chosen)
            and all(c.name == fc.name for c, fc in zip(chosen, first_chosen))
            for chosen in chosen_list[1:]
        )
        if all_same:
            # Use batch fusion for efficiency
            precomputed_batch_embeddings = (
                _aligned_embeddings_for_batch(first_chosen, queries)
                if self.use_shared_embedding_path
                else None
            )
            vectors, weights_list, meta_list = self.fuser.fuse_batch(
                queries,
                first_chosen,
                precomputed_batch_embeddings=precomputed_batch_embeddings,
            )
            potions = []
            for i, query in enumerate(queries):
                meta: Dict[str, object] = {
                    "selected_modules": [m.name for m in first_chosen]
                }
                if (
                    meta_list is not None
                    and i < len(meta_list)
                    and meta_list[i] is not None
                ):
                    meta.update(meta_list[i])
                potions.append(
                    Potion(vector=vectors[i], weights=weights_list[i], meta=meta)
                )
            return potions
        else:
            # Fallback: loop over queries
            return [self.brew(query, village, scout) for query in queries]


__all__ = [
    "Potion",
    "Fuser",
    "MeanFuser",
    "KalmanorixFuser",
    "EnsembleKalmanFuser",
    "StructuredKalmanFuser",
    "DiagonalKalmanFuser",
    "LearnedGateFuser",
    "Panoramix",
]
