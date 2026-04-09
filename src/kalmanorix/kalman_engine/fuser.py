"""High-level fusion orchestrator.

This module provides the Panoramix class, which coordinates the entire fusion
process: routing, alignment, uncertainty estimation, and Kalman fusion.
"""

from typing import List, Dict, Any, Optional, Tuple
import logging
import warnings

import numpy as np

from .kalman_fuser import (
    kalman_fuse_diagonal,
    kalman_fuse_diagonal_ensemble,
    kalman_fuse_diagonal_ensemble_batch,
)
from .covariance import CovarianceEstimator, estimate_covariance_for_query
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
        use_ensemble: bool = True,
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
            use_ensemble: If True, uses precision-weighted ensemble Kalman fusion.
                         If False, uses deprecated sequential Kalman updates.
            epsilon: Small constant for numerical stability
        """
        self.router = router
        self.covariance_estimator = covariance_estimator
        self.alignment_method = alignment_method
        self.use_prior = use_prior
        self.prior_model = prior_model
        self.sort_measurements = sort_measurements
        self.use_ensemble = use_ensemble
        self.epsilon = epsilon

        if use_prior and prior_model is None:
            raise ValueError("prior_model required when use_prior=True")

        logger.info(
            "Panoramix initialised: alignment=%s, use_prior=%s, sort=%s, ensemble=%s",
            alignment_method,
            use_prior,
            sort_measurements,
            use_ensemble,
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
                # Fall back to calibrated covariance profile when available.
                if getattr(model, "covariance_data", None):
                    try:
                        cov = estimate_covariance_for_query(
                            embedding=emb,
                            covariance_profile=model.covariance_data,  # type: ignore[arg-type]
                            epsilon=self.epsilon,
                        )
                    except (ValueError, TypeError):
                        cov = model.get_covariance(query)  # type: ignore
                else:
                    cov = model.get_covariance(query)  # type: ignore

            embeddings.append(emb)
            covariances.append(cov)

        # Step 3: Add prior if used
        if self.use_prior and self.prior_model is not None:
            prior_emb = self.prior_model.embed(query)
            prior_cov = self.prior_model.get_covariance(query)
            if self.use_ensemble:
                fused, fused_cov = kalman_fuse_diagonal_ensemble(
                    embeddings,
                    covariances,
                    initial_state=prior_emb,
                    initial_covariance=prior_cov,
                    epsilon=self.epsilon,
                )
            else:
                warnings.warn(
                    "Sequential Kalman fusion is deprecated. Use ensemble fusion "
                    "for better numerical stability. Set use_ensemble=True in "
                    "Panoramix constructor.",
                    DeprecationWarning,
                    stacklevel=2,
                )
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
            if self.use_ensemble:
                # No prior: precision-weighted fusion over all specialist measurements.
                fused, fused_cov = kalman_fuse_diagonal_ensemble(
                    embeddings,
                    covariances,
                    initial_state=None,
                    initial_covariance=None,
                    epsilon=self.epsilon,
                )
            else:
                warnings.warn(
                    "Sequential Kalman fusion is deprecated. Use ensemble fusion "
                    "for better numerical stability. Set use_ensemble=True in "
                    "Panoramix constructor.",
                    DeprecationWarning,
                    stacklevel=2,
                )
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
        if not queries:
            return results

        # Try efficient batch fusion path when all queries route to the same models.
        selected_models_per_query = [
            self.router.select(query, village) for query in queries
        ]  # type: ignore
        first_selected = selected_models_per_query[0]
        shared_selection = all(
            len(selected) == len(first_selected)
            and all(m.name == f.name for m, f in zip(selected, first_selected))
            for selected in selected_models_per_query[1:]
        )

        if shared_selection and (first_selected or self.use_prior):
            try:
                embeddings_by_model = []
                covariances_by_model = []

                for model in first_selected:
                    model_embeddings = []
                    model_covariances = []
                    for query in queries:
                        emb = model.embed(query)
                        if self.alignment_method == "procrustes" and hasattr(
                            model, "alignment_matrix"
                        ):
                            emb = model.alignment_matrix @ emb  # type: ignore

                        if self.covariance_estimator is not None:
                            cov = self.covariance_estimator.estimate(model.embed, query)  # type: ignore
                        else:
                            if getattr(model, "covariance_data", None):
                                try:
                                    cov = estimate_covariance_for_query(
                                        embedding=emb,
                                        covariance_profile=model.covariance_data,  # type: ignore[arg-type]
                                        epsilon=self.epsilon,
                                    )
                                except (ValueError, TypeError):
                                    cov = model.get_covariance(query)  # type: ignore
                            else:
                                cov = model.get_covariance(query)  # type: ignore

                        model_embeddings.append(emb)
                        model_covariances.append(cov)

                    embeddings_by_model.append(np.stack(model_embeddings, axis=0))
                    covariances_by_model.append(np.stack(model_covariances, axis=0))

                embeddings = np.stack(embeddings_by_model, axis=0)
                covariances = np.stack(covariances_by_model, axis=0)

                initial_state = None
                initial_covariance = None
                if self.use_prior and self.prior_model is not None:
                    prior_embeddings = [self.prior_model.embed(q) for q in queries]
                    prior_covariances = [
                        self.prior_model.get_covariance(q) for q in queries
                    ]
                    initial_state = np.stack(prior_embeddings, axis=0)
                    initial_covariance = np.stack(prior_covariances, axis=0)

                if self.use_ensemble:
                    fused, fused_cov = kalman_fuse_diagonal_ensemble_batch(
                        embeddings,
                        covariances,
                        initial_state=initial_state,
                        initial_covariance=initial_covariance,
                        epsilon=self.epsilon,
                    )
                else:
                    warnings.warn(
                        "Sequential Kalman fusion is deprecated. Use ensemble fusion "
                        "for better numerical stability. Set use_ensemble=True in "
                        "Panoramix constructor.",
                        DeprecationWarning,
                        stacklevel=2,
                    )
                    # Fall back to per-query sequential path below.
                    raise RuntimeError("Sequential batch fusion path disabled")

                for i in range(len(queries)):
                    results.append((fused[i], fused_cov[i]))
                return results
            except RuntimeError:
                # Fallback to per-query fusion below.
                pass

        for query in queries:
            try:
                result = self.fuse(query, village, context)
                results.append(result)
            except RuntimeError as e:
                logger.error("Failed to fuse query '%s...': %s", query[:50], e)
                # Return zeros with high uncertainty on failure
                d = self._infer_village_dimension(village)
                results.append((np.zeros(d), np.ones(d) * 1e6))

        return results

    @staticmethod
    def _infer_village_dimension(village: Village) -> int:
        """Infer embedding dimension without synthetic embed() probes."""
        if not village.modules:
            return 768
        module = village.modules[0]
        if hasattr(module, "infer_embedding_dimension"):
            try:
                return int(module.infer_embedding_dimension())
            except ValueError:
                pass
        if getattr(module, "domain_centroid", None) is not None:
            return int(module.domain_centroid.shape[0])
        if getattr(module, "alignment_matrix", None) is not None:
            return int(module.alignment_matrix.shape[0])
        return 768
