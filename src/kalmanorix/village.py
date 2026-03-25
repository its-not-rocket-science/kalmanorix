"""Kalmanorix public API."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Iterable, List, Optional, Union, TYPE_CHECKING
import numpy as np


from .types import Embedder, Vec

if TYPE_CHECKING:
    from .models.sef import SEFModel
    from .kalman_engine.structured_covariance import StructuredCovariance

Sigma2 = Union[float, Callable[[str], float]]


@dataclass(frozen=True)
class SEF:
    """Specialist Embedding Format (Phase 0/1).

    sigma2 can be:
      - float: constant uncertainty
      - Callable[[str], float]: query-dependent uncertainty

    If model is provided, it should be a SEFModel instance that can provide
    structured covariance (low‑rank) via get_structured_covariance().
    """

    name: str
    embed: Embedder
    sigma2: Sigma2
    meta: Optional[Dict[str, str]] = None
    alignment_matrix: Optional[np.ndarray] = None
    domain_centroid: Optional[Vec] = None
    model: Optional["SEFModel"] = None

    def sigma2_for(self, query: str) -> float:
        """Return uncertainty (variance) for a given query."""
        if callable(self.sigma2):
            val = float(self.sigma2(query))
        else:
            val = float(self.sigma2)

        # Safety: avoid zero/negative variances
        return max(val, 1e-12)

    def get_covariance(self, query: str) -> np.ndarray:
        """Return diagonal covariance vector for this query.

        If the SEF has an attached SEFModel, returns its diagonal covariance.
        Otherwise, returns sigma2_for(query) * ones(d).
        """
        if self.model is not None:
            # Use model's diagonal covariance
            from .models.sef import SEFModel  # lazy import to avoid circular

            assert isinstance(self.model, SEFModel)
            return self.model.get_covariance(query)
        # Fallback: scalar sigma2 converted to diagonal
        d = self.embed("dummy").shape[0]  # dimension
        sigma2 = self.sigma2_for(query)
        return np.full(d, sigma2, dtype=np.float64)

    def get_structured_covariance(self, query: str) -> Optional["StructuredCovariance"]:
        """Return structured covariance (diagonal + low‑rank) if available.

        If the SEF has an attached SEFModel that supports low‑rank covariance,
        returns a StructuredCovariance object. Otherwise returns None.
        """
        if self.model is not None:
            from .models.sef import SEFModel  # lazy import to avoid circular

            assert isinstance(self.model, SEFModel)
            return self.model.get_structured_covariance(query)
        return None

    def with_domain_centroid(self, calibration_texts: Iterable[str]) -> "SEF":
        """Return a new SEF with domain centroid computed from calibration texts.

        Args:
            calibration_texts: Sample texts from the specialist's domain.

        Returns:
            A new SEF with domain_centroid set to the normalized mean embedding.
        """
        centroid = compute_domain_centroid(self.embed, calibration_texts)
        return replace(self, domain_centroid=centroid)


def compute_domain_centroid(embed: Embedder, calibration_texts: Iterable[str]) -> Vec:
    """Compute normalized domain centroid from calibration texts.

    Args:
        embed: Embedder function.
        calibration_texts: Sample texts from the domain.

    Returns:
        Normalized centroid vector (unit length).
    """
    embeddings = [embed(text) for text in calibration_texts]
    if not embeddings:
        raise ValueError("calibration_texts must not be empty")
    centroid = np.mean(np.stack(embeddings, axis=0), axis=0)
    norm = np.linalg.norm(centroid)
    if norm == 0:
        return centroid
    return centroid / norm


@dataclass
class Village:
    """A simple container for specialists available at runtime."""

    modules: List[SEF]

    def __post_init__(self) -> None:
        if not self.modules:
            raise ValueError("Village must contain at least one SEF")

    def list(self) -> List[str]:
        """List names of available modules."""
        return [m.name for m in self.modules]
