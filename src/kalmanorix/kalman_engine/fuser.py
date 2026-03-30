"""High-level fusion orchestrator.

This module provides the Panoramix class, which coordinates the entire fusion
process: routing, alignment, uncertainty estimation, and Kalman fusion.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging

import numpy as np

from .kalman_fuser import kalman_fuse_diagonal
from .covariance import CovarianceEstimator
from ..village import Village
from ..models.sef import SEFModel
from ..scout import ScoutRouter

logger = logging.getLogger(__name__)


class Panoramix:
    """Main fusion engine orchestrating the Kalman update process.

    The Panoramix is responsible for:
    1. Receiving a query and selecting relevant specialists (via ScoutRouter)
    2. Retrieving embeddings and uncertainties from each specialist
    3. Ensuring all embeddings are in the same reference space (alignment)
    4. Performing Kalman fusion to produce a single unified embedding
    5. Returning the fused embedding with its uncertainty

    This class implements the core runtime of the KEFF framework.
    """

    def __init__(
        self,
        router: ScoutRouter,
        covariance_estimator: Optional[CovarianceEstimator] = None,
        alignment_method: str = "procrustes",
        use_prior: bool = False,
        prior_model: Optional[SEFModel] = None,
        sort_measurements: bool = True,
        epsilon: float = 1e-8,
    ):
        """Initialise the fusion engine.

        Args:
            router: ScoutRouter instance for selecting relevant models
            covariance_estimator: Strategy for estimating uncertainties.
                                 If None, uses EmpiricalCovariance with defaults.
            alignment_method: How to align embeddings: 'procrustes' (default),
                             'identity' (assume already aligned), or 'learned'.
            use_prior: Whether to use a prior model (e.g., general-purpose embedder)
            prior_model: SEFModel to use as prior (required if use_prior=True)
            sort_measurements: Sort measurements by certainty before fusion
            epsilon: Small constant for numerical stability
        """
        self.router = router
        self.covariance_estimator = covariance_estimator
        self.alignment_method = alignment_method
        self.use_prior = use_prior
        self.prior_model = prior_model
        self.sort_measurements = sort_measurements
        self.epsilon = epsilon

        if use_prior and prior_model is None:
            raise ValueError("prior_model required when use_prior=True")

        logger.info(
            "Panoramix initialised: alignment=%s, use_prior=%s, sort=%s",
            alignment_method,
            use_prior,
            sort_measurements,
        )

    def fuse(
        self,
        query: str,
        village: Village,
        _context: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Fuse embeddings from relevant specialists for a query.

        This is the main entry point for runtime fusion.

        Args:
            query: Input text to embed
            village: Village containing available specialist models
            _context: Optional context dict (e.g., domain hints for routing)

        Returns:
            fused_embedding: Final fused embedding vector (d,)
            fused_covariance: Final uncertainty diagonal (d,)

        Raises:
            RuntimeError: If no models selected or fusion fails
        """
        # Step 1: Select relevant models
        selected_models = self.router.select(query, village)  # type: ignore
        if not selected_models and not self.use_prior:
            raise RuntimeError(f"No models selected for query: {query[:50]}...")

        logger.debug("Selected %d models for fusion", len(selected_models))

        # Step 2: Get embeddings and uncertainties
        embeddings = []
        covariances = []

        for model in selected_models:
            # Get embedding
            emb = model.embed(query)

            # Align if needed (apply pre-computed Procrustes matrix)
            if self.alignment_method == "procrustes" and hasattr(
                model, "alignment_matrix"
            ):
                emb = model.alignment_matrix @ emb  # type: ignore

            # Estimate uncertainty
            if self.covariance_estimator is not None:
                cov = self.covariance_estimator.estimate(model.embed, query)  # type: ignore
            else:
                # Fall back to model's stored covariance
                cov = model.get_covariance(query)  # type: ignore

            embeddings.append(emb)
            covariances.append(cov)

        # Step 3: Add prior if used
        if self.use_prior and self.prior_model is not None:
            prior_emb = self.prior_model.embed(query)
            prior_cov = self.prior_model.get_covariance(query)
            # Prior becomes initial state
            fused, fused_cov = kalman_fuse_diagonal(
                embeddings,
                covariances,
                initial_state=prior_emb,
                initial_covariance=prior_cov,
                sort_by_certainty=self.sort_measurements,
                epsilon=self.epsilon,
            )
        else:
            # No prior, start from first measurement
            fused, fused_cov = kalman_fuse_diagonal(
                embeddings,
                covariances,
                sort_by_certainty=self.sort_measurements,
                epsilon=self.epsilon,
            )

        logger.debug("Fusion complete: uncertainty=%.4f", np.sum(fused_cov))

        return fused, fused_cov

    def fuse_batch(
        self,
        queries: List[str],
        village: Village,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Fuse multiple queries in batch (for efficiency).

        Args:
            queries: List of input texts
            village: Village containing available models
            context: Optional context

        Returns:
            List of (embedding, covariance) tuples for each query
        """
        results = []
        for query in queries:
            try:
                result = self.fuse(query, village, context)
                results.append(result)
            except RuntimeError as e:
                logger.error("Failed to fuse query '%s...': %s", query[:50], e)
                # Return zeros with high uncertainty on failure
                d = (
                    village.modules[0].embed("dummy").shape[0]
                    if village.modules
                    else 768
                )
                results.append((np.zeros(d), np.ones(d) * 1e6))

        return results
