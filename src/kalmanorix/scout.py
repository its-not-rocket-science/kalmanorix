"""
Routing logic for Kalmanorix.

The ScoutRouter decides which specialists to consult for a given query.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
import logging
from typing import Callable, List, Optional, Union, Dict, Tuple, Literal

import numpy as np

from .village import SEF, Village
from .types import Embedder

ThresholdFunction = Union[float, Callable[[str, np.ndarray, List[float]], float]]
RoutingMode = Literal["all", "hard", "semantic", "confidence"]
FallbackMode = Literal["all", "hard"]
ConfidenceMetric = Literal["similarity_gap", "top_similarity"]

logger = logging.getLogger(__name__)


@dataclass
class ScoutRouter:
    """Select which specialists to consult for a query.

    Attributes:
        mode: Routing mode: 'all', 'hard', 'semantic', or 'confidence'.
        fast_embedder: Fast embedder for semantic routing (required for 'semantic' and 'confidence' modes).
        similarity_threshold: Threshold for semantic routing. Can be float or callable
            (query, query_vec, similarities) -> float for dynamic thresholding.
        fallback_mode: Fallback mode when semantic routing finds no matches ('all' or 'hard').
        confidence_threshold: Threshold for confidence-based routing (default 0.8).
        confidence_metric: Metric for confidence calculation: 'similarity_gap' (difference between
            top and second similarity) or 'top_similarity' (absolute similarity of top specialist).

    Modes:
        - 'all': Return all available modules (enables fusion).
        - 'hard': Return only the single module with the lowest query-dependent sigma².
        - 'semantic': Return modules whose domain centroid similarity with the query
          meets the similarity_threshold. Requires fast_embedder to be set.
          If no modules match, fall back to fallback_mode.
        - 'confidence': Compute similarities, if confidence metric exceeds confidence_threshold,
          return only the top specialist; otherwise proceed with semantic routing.
          Requires fast_embedder.
    """

    mode: RoutingMode = "all"
    fast_embedder: Optional[Embedder] = None
    similarity_threshold: ThresholdFunction = 0.7
    fallback_mode: FallbackMode = "all"
    max_cache_size: int = 1000
    confidence_threshold: float = 0.8
    confidence_metric: ConfidenceMetric = "similarity_gap"
    hard_routing_heuristic: Optional[Callable[[str, Village], Optional[SEF]]] = None
    # Internal caches
    _centroid_cache: Dict[str, Optional[np.ndarray]] = field(
        default_factory=dict, init=False
    )
    _embedding_cache: OrderedDict[str, Tuple[np.ndarray, np.ndarray]] = field(
        default_factory=OrderedDict, init=False
    )

    def __post_init__(self) -> None:
        valid_modes = {"all", "hard", "semantic", "confidence"}
        if self.mode not in valid_modes:
            raise ValueError(f"mode must be one of {sorted(valid_modes)}")
        valid_fallback_modes = {"all", "hard"}
        if self.fallback_mode not in valid_fallback_modes:
            raise ValueError(
                f"fallback_mode must be one of {sorted(valid_fallback_modes)}"
            )
        valid_confidence_metrics = {"similarity_gap", "top_similarity"}
        if self.confidence_metric not in valid_confidence_metrics:
            raise ValueError(
                f"confidence_metric must be one of {sorted(valid_confidence_metrics)}"
            )

    def _get_query_embedding(self, query: str) -> np.ndarray:
        """Get normalized query embedding from fast_embedder with LRU caching."""
        # Check if embedder has its own cache (e.g., TfidfEmbedder)
        embedder_has_cache = self.fast_embedder is not None and hasattr(
            self.fast_embedder, "cache"
        )

        if not embedder_has_cache:
            # Use router-level cache only if embedder doesn't have its own cache
            if query in self._embedding_cache:
                # Move to end (most recently used)
                value = self._embedding_cache.pop(query)
                self._embedding_cache[query] = value
                return value[1]

        if self.fast_embedder is None:
            raise ValueError("fast_embedder must be provided when mode='semantic'")
        query_vec = self.fast_embedder(query)
        query_norm = np.linalg.norm(query_vec)
        if query_norm == 0:
            # Zero vector cannot be normalized; cache as zero vector
            normalized = query_vec
        else:
            normalized = query_vec / query_norm

        if not embedder_has_cache:
            # Store in router cache, evict oldest if needed
            self._embedding_cache[query] = (query_vec, normalized)
            if len(self._embedding_cache) > self.max_cache_size:
                self._embedding_cache.popitem(last=False)

        return normalized

    def _get_normalized_centroid(self, module: SEF) -> Optional[np.ndarray]:
        """Get normalized domain centroid for module with caching."""
        if module.name not in self._centroid_cache:
            centroid = module.domain_centroid
            if centroid is None:
                self._centroid_cache[module.name] = None
            else:
                centroid_norm = np.linalg.norm(centroid)
                if centroid_norm == 0:
                    self._centroid_cache[module.name] = None
                else:
                    self._centroid_cache[module.name] = centroid / centroid_norm
        return self._centroid_cache[module.name]

    def select(self, query: str, village: Village) -> List[SEF]:
        """Return the selected specialist modules for the given query.

        Args:
            query: Input text query.
            village: Village containing available specialist modules.

        Returns:
            List of selected SEF instances.

        Raises:
            ValueError: If mode='semantic' or 'confidence' but fast_embedder is not set.
        """
        if self.mode == "all":
            return village.modules
        if self.mode == "hard":
            heuristic_selection = self._select_with_hard_heuristic(query, village)
            if heuristic_selection is not None:
                return [heuristic_selection]

            return [min(village.modules, key=lambda m: m.sigma2_for(query))]
        if self.mode == "semantic":
            return self._semantic_selection(query, village)
        if self.mode == "confidence":
            return self._confidence_selection(query, village)
        raise ValueError("mode must be 'all', 'hard', 'semantic', or 'confidence'")

    def _confidence_selection(self, query: str, village: Village) -> List[SEF]:
        """Confidence-based routing logic.

        Args:
            query: Input text query.
            village: Village containing available specialist modules.

        Returns:
            List of selected SEF instances.
        """
        # Get normalized query embedding (cached)
        try:
            query_vec = self._get_query_embedding(query)
        except ValueError as e:
            raise ValueError(
                "fast_embedder must be provided when mode='confidence'"
            ) from e

        # Check for zero vector
        if np.linalg.norm(query_vec) == 0:
            # Fallback if query embedding is zero vector
            return self._fallback_selection(query, village)

        # Compute similarities for modules with centroids
        module_similarities = []
        for module in village.modules:
            centroid = self._get_normalized_centroid(module)
            if centroid is None:
                continue
            similarity = np.dot(query_vec, centroid)
            module_similarities.append((module, similarity))

        if not module_similarities:
            # No modules with valid centroids
            return self._fallback_selection(query, village)

        # Extract similarity values
        similarities = [sim for _, sim in module_similarities]

        # Compute confidence metric
        if self.confidence_metric == "similarity_gap":
            if len(similarities) >= 2:
                sorted_sims = sorted(similarities, reverse=True)
                confidence = sorted_sims[0] - sorted_sims[1]
            else:
                confidence = 1.0  # only one specialist, high confidence
        elif self.confidence_metric == "top_similarity":
            confidence = max(similarities)
        else:
            raise ValueError(f"Unknown confidence_metric: {self.confidence_metric}")

        logger.debug(
            "ScoutRouter confidence=%0.6f (metric=%s, threshold=%0.6f)",
            confidence,
            self.confidence_metric,
            self.confidence_threshold,
        )

        if confidence >= self.confidence_threshold:
            # Return only the top specialist
            top_module = max(module_similarities, key=lambda x: x[1])[0]
            logger.debug(
                "ScoutRouter confidence high; returning single specialist: %s",
                top_module.name,
            )
            return [top_module]
        else:
            # Confidence insufficient, proceed with semantic routing
            logger.debug("ScoutRouter confidence low; falling back to semantic routing")
            return self._semantic_selection(query, village)

    def warm_cache(self, queries: list[str]) -> None:
        """Pre‑compute embeddings for given queries to warm up caches.

        This improves latency for frequently‑seen queries by ensuring
        their embeddings are already cached when they arrive at runtime.

        Args:
            queries: List of query strings to pre‑compute.
        """
        if self.fast_embedder is None:
            return
        for query in queries:
            # This will populate both embedder cache (if any) and router cache
            _ = self._get_query_embedding(query)

    def _fallback_selection(self, query: str, village: Village) -> List[SEF]:
        """Fallback selection when semantic routing finds no matches.

        Args:
            query: Input text query.
            village: Village containing available specialist modules.

        Returns:
            List of selected SEF instances.
        """
        if self.fallback_mode == "all":
            return village.modules
        if self.fallback_mode == "hard":
            # Reuse hard mode logic but without charge detection
            return [min(village.modules, key=lambda m: m.sigma2_for(query))]
        raise ValueError("fallback_mode must be 'all' or 'hard'")

    def _semantic_selection(self, query: str, village: Village) -> List[SEF]:
        """Select modules via semantic similarity to domain centroids.

        This is the core semantic routing logic used by both 'semantic' and
        'confidence' modes.

        Args:
            query: Input text query.
            village: Village containing available specialist modules.

        Returns:
            List of selected SEF instances.
        """
        # Get normalized query embedding (cached)
        try:
            query_vec = self._get_query_embedding(query)
        except ValueError as e:
            raise ValueError(
                "fast_embedder must be provided when mode='semantic'"
            ) from e

        # Check for zero vector
        if np.linalg.norm(query_vec) == 0:
            # Fallback if query embedding is zero vector
            return self._fallback_selection(query, village)

        # Compute similarities for modules with centroids
        module_similarities = []
        for module in village.modules:
            centroid = self._get_normalized_centroid(module)
            if centroid is None:
                continue
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
            logger.debug("ScoutRouter threshold=%0.6f (dynamic)", threshold)
        else:
            threshold = self.similarity_threshold
            logger.debug("ScoutRouter threshold=%0.6f", threshold)

        # Select modules meeting threshold
        selected = [module for module, sim in module_similarities if sim >= threshold]
        logger.debug("ScoutRouter selected modules: %s", [m.name for m in selected])

        # Return selected modules or fallback if none
        return selected if selected else self._fallback_selection(query, village)

    def _select_with_hard_heuristic(
        self,
        query: str,
        village: Village,
    ) -> Optional[SEF]:
        """Apply optional hard-routing heuristic.

        Core routing remains sigma²-minimisation; heuristics are opt-in and isolated.
        """
        if self.hard_routing_heuristic is not None:
            return self.hard_routing_heuristic(query, village)
        return None
