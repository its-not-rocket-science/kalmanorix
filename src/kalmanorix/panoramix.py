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
)
from .kalman_engine.structured_covariance import StructuredCovariance

# Re-export LearnedGateFuser from legacy module for now
# TODO: Integrate properly  # pylint: disable=fixme
from .panoramix_legacy import LearnedGateFuser  # noqa: F401, E402  # pylint: disable=unused-import

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Potion:  # pylint: disable=too-few-public-methods
    """
    Result of a fusion operation.

    Attributes
    ----------
    vector:
        The fused embedding vector.
    weights:
        Per-module fusion weights.
    meta:
        Optional diagnostic metadata (e.g. gate values).
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
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """
        Fuse embeddings from the given modules.

        Parameters
        ----------
        query:
            Input text query.
        modules:
            List of selected specialist modules.

        Returns
        -------
        vector:
            Fused embedding.
        weights:
            Per-module contribution weights.
        meta:
            Optional metadata.
        """
        raise NotImplementedError


class MeanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    Uniform averaging baseline.

    All modules contribute equally, regardless of uncertainty or query content.
    """

    def fuse(
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        embeddings = []
        for m in modules:
            emb = m.embed(query)
            if m.alignment_matrix is not None:
                emb = m.alignment_matrix @ emb
            embeddings.append(emb)
        z = np.stack(embeddings, axis=0)
        w = {m.name: 1.0 / len(modules) for m in modules}
        return z.mean(axis=0), w, None


class KalmanorixFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    True Kalman fusion with diagonal covariance.

    Uses the core `kalman_fuse_diagonal` algorithm with per-dimension
    diagonal covariance. Converts each module's scalar sigma² to a
    diagonal covariance matrix (sigma² * I).
    """

    def __init__(self, sort_by_certainty: bool = True, epsilon: float = 1e-8):
        """
        Args:
            sort_by_certainty: Sort measurements by certainty before fusion
            epsilon: Small constant for numerical stability
        """
        self.sort_by_certainty = sort_by_certainty
        self.epsilon = epsilon

    def fuse(  # pylint: disable=too-many-locals
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("KalmanorixFuser requires at least one module")

        # Get embeddings and convert scalar sigma² to diagonal covariance
        embeddings = []
        covariances = []

        for module in modules:
            emb = module.embed(query)
            # Apply alignment if available
            if module.alignment_matrix is not None:
                emb = module.alignment_matrix @ emb
            sigma2 = module.sigma2_for(query)
            # Convert scalar variance to diagonal covariance vector
            cov = np.full(emb.shape, sigma2, dtype=np.float64)
            embeddings.append(emb)
            covariances.append(cov)

        # Perform Kalman fusion
        fused, fused_cov = kalman_fuse_diagonal(
            embeddings,
            covariances,
            sort_by_certainty=self.sort_by_certainty,
            epsilon=self.epsilon,
        )

        # Compute weights from contributions (diagnostic)
        # Weight proportional to inverse total uncertainty
        total_uncertainties = [np.sum(cov) for cov in covariances]
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
        }

        return fused, weights, meta


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
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("EnsembleKalmanFuser requires at least one module")

        # Get embeddings and convert scalar sigma² to diagonal covariance
        embeddings = []
        covariances = []

        for module in modules:
            emb = module.embed(query)
            # Apply alignment if available
            if module.alignment_matrix is not None:
                emb = module.alignment_matrix @ emb
            sigma2 = module.sigma2_for(query)
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
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("StructuredKalmanFuser requires at least one module")

        embeddings = []
        structured_covs = []

        for module in modules:
            emb = module.embed(query)
            # Apply alignment if available
            if module.alignment_matrix is not None:
                emb = module.alignment_matrix @ emb

            # Try to get structured covariance
            structured_cov = module.get_structured_covariance(query)
            if structured_cov is None:
                # Fall back to diagonal covariance from sigma²
                sigma2 = module.sigma2_for(query)
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


class DiagonalKalmanFuser(Fuser):  # pylint: disable=too-few-public-methods
    """
    Sequential diagonal Kalman-style fusion with scalar prior variance.

    This is the legacy implementation from panoramix_legacy.py.
    Maintains a scalar prior variance P (shared across dimensions).
    """

    def __init__(
        self, *, prior_sigma2: float = 1.0, sort_by_sigma2: bool = True
    ) -> None:
        self.prior_sigma2 = float(prior_sigma2)
        self.sort_by_sigma2 = bool(sort_by_sigma2)

    def _kalman_updates(
        self, order: List[str], z_by_name: Dict[str, Vec], r_by_name: Dict[str, float]
    ) -> Tuple[Vec, Dict[str, float], float]:
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
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("DiagonalKalmanFuser requires at least one module")

        z_by_name: Dict[str, Vec] = {}
        r_by_name: Dict[str, float] = {}
        for m in modules:
            emb = m.embed(query)
            if m.alignment_matrix is not None:
                emb = m.alignment_matrix @ emb
            z_by_name[m.name] = emb
            r_by_name[m.name] = float(m.sigma2_for(query))

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


@dataclass
class Panoramix:  # pylint: disable=too-few-public-methods
    """
    Orchestrator that combines routing and fusion.

    Panoramix:
      1. asks a ScoutRouter which modules to consult
      2. delegates fusion to a Fuser
      3. packages the result as a Potion
    """

    fuser: Fuser

    def brew(self, query: str, village: Village, scout: ScoutRouter) -> Potion:
        """
        Produce a fused embedding for a query.

        Parameters
        ----------
        query:
            Input text query.
        village:
            Collection of available specialist modules.
        scout:
            Routing strategy.

        Returns
        -------
        Potion
            The fused embedding and diagnostics.
        """
        chosen = scout.select(query, village)
        vec, weights, fuser_meta = self.fuser.fuse(query, chosen)
        meta: Dict[str, object] = {"selected_modules": [m.name for m in chosen]}
        if fuser_meta is not None:
            meta.update(fuser_meta)
        return Potion(vector=vec, weights=weights, meta=meta)


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
