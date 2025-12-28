from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Set

import numpy as np


@dataclass(frozen=True)
class KeywordSigma2:
    """A simple query-dependent uncertainty heuristic.

    If a query contains any keyword, return in_domain_sigma2 (more confident).
    Otherwise return out_domain_sigma2 (less confident).

    This is intentionally simple and deterministic, good for Phase 1.
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
      - Map similarity to sigma2: higher similarity => lower sigma2.

    This requires you to pass an embedder at construction time.
    """
    embed: callable
    centroid: np.ndarray
    base_sigma2: float = 0.2
    scale: float = 2.0

    @classmethod
    def from_calibration(
        cls,
        *,
        embed: callable,
        calibration_texts: Iterable[str],
        base_sigma2: float = 0.2,
        scale: float = 2.0,
    ) -> "CentroidDistanceSigma2":
        embs = [embed(t) for t in calibration_texts]
        c = np.mean(np.stack(embs, axis=0), axis=0)
        c = c / (np.linalg.norm(c) + 1e-12)
        return cls(embed=embed, centroid=c, base_sigma2=base_sigma2, scale=scale)

    def __call__(self, query: str) -> float:
        z = self.embed(query)
        z = z / (np.linalg.norm(z) + 1e-12)
        sim = float(z @ self.centroid)  # [-1, 1] typically
        sim01 = (sim + 1.0) / 2.0       # map to [0, 1]
        # Higher similarity -> lower uncertainty
        return float(self.base_sigma2 + self.scale * (1.0 - sim01))
