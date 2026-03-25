"""
Routing logic for Kalmanorix.

The ScoutRouter decides which specialists to consult for a given query.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import numpy as np

from .village import SEF, Village
from .types import Embedder

ThresholdFunction = Union[float, Callable[[str, np.ndarray, List[float]], float]]


@dataclass
class ScoutRouter:
    """
    Select which specialists to consult for a query.

    Modes
    -----
    all:
        Return all available modules (enables fusion).
    hard:
        Return only the single module with the lowest query-dependent sigma².
    semantic:
        Return modules whose domain centroid similarity with the query
        meets the similarity_threshold. similarity_threshold can be a float
        or a callable (query, query_vec, similarities) -> float for dynamic
        thresholding. Requires fast_embedder to be set.
        If no modules match, fall back to fallback_mode.
    """

    mode: str = "all"
    fast_embedder: Optional[Embedder] = None
    similarity_threshold: ThresholdFunction = 0.7
    fallback_mode: str = "all"

    def select(self, query: str, village: Village) -> List[SEF]:
        """Return the selected specialist modules for the given query."""
        if self.mode == "all":
            return village.modules
        if self.mode == "hard":
            q = query.lower()

            charge_like = bool(
                re.search(
                    r"\b(usb-?c|pd\b|power delivery|pps|pdo|rdo|watt|wattage|e-?marker|qc\b)\b",
                    q,
                )
            )

            if charge_like:
                for m in village.modules:
                    if m.name == "charge":
                        return [m]

            return [min(village.modules, key=lambda m: m.sigma2_for(query))]
        if self.mode == "semantic":
            if self.fast_embedder is None:
                raise ValueError("fast_embedder must be provided when mode='semantic'")
            # Encode query with fast embedder
            query_vec = self.fast_embedder(query)
            # Normalize query vector for cosine similarity
            query_norm = np.linalg.norm(query_vec)
            if query_norm == 0:
                # Fallback if query embedding is zero vector
                return self._fallback_selection(query, village)
            query_vec = query_vec / query_norm

            # Compute similarities for modules with centroids
            module_similarities = []
            for module in village.modules:
                if module.domain_centroid is None:
                    continue
                centroid = module.domain_centroid
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm == 0:
                    continue
                centroid = centroid / centroid_norm
                similarity = np.dot(query_vec, centroid)
                module_similarities.append((module, similarity))

            if not module_similarities:
                # No modules with valid centroids
                return self._fallback_selection(query, village)

            # Extract similarity values for threshold computation
            similarities = [sim for _, sim in module_similarities]

            # Determine threshold
            if callable(self.similarity_threshold):
                threshold = self.similarity_threshold(query, query_vec, similarities)
                print(f"[ScoutRouter] threshold={threshold:.6f} (dynamic)")
            else:
                threshold = self.similarity_threshold
                print(f"[ScoutRouter] threshold={threshold:.6f}")

            # Select modules meeting threshold
            selected = [
                module for module, sim in module_similarities if sim >= threshold
            ]
            print(f"[ScoutRouter] selected modules: {[m.name for m in selected]}")

            # Return selected modules or fallback if none
            return selected if selected else self._fallback_selection(query, village)
        raise ValueError("mode must be 'all', 'hard', or 'semantic'")

    def _fallback_selection(self, query: str, village: Village) -> List[SEF]:
        """Fallback selection when semantic routing finds no matches."""
        if self.fallback_mode == "all":
            return village.modules
        if self.fallback_mode == "hard":
            # Reuse hard mode logic but without charge detection
            return [min(village.modules, key=lambda m: m.sigma2_for(query))]
        raise ValueError("fallback_mode must be 'all' or 'hard'")
