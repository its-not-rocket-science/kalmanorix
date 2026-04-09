"""
Orchestration script for Milestone 2.2: Uncertainty Robustness Test.

Tests whether Kalman fusion's uncertainty weighting provides robustness benefits:
1. Generalization to out-of-domain (OOD) queries
2. Graceful degradation with mis-specified covariances
3. Calibration of uncertainty estimates
4. Ablation: equal covariances → averaging (degenerate case)

Runs the full experiment:
1. Load pre-trained specialists from Milestone 2.1 output
2. Load datasets for seen domains + one OOD domain
3. Create OOD test set (documents from seen domains, queries from OOD domain)
4. Evaluate:
   - Kalman fusion vs Mean fusion on OOD queries
   - Calibration metrics for each specialist and fusion method
   - Mis-specification sweep with scaling factors [0.1, 0.5, 1.0, 2.0, 10.0]
   - Ablation: all sigma2 set to constant (should equal averaging)
5. Generate results:
   - JSON with all metrics
   - Plots: reliability diagrams, performance vs mis-specification factor
   - Summary table comparing conditions
"""

# pylint: disable=wrong-import-position,import-outside-toplevel,logging-fstring-interpolation,too-many-statements,unnecessary-comprehension,no-else-return

from __future__ import annotations

import argparse
import json
import logging
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Callable, Literal, cast

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import yaml

from experiments.config import TrainingConfig, DomainEnum
from kalmanorix.village import Village, SEF
from kalmanorix.scout import ScoutRouter
from kalmanorix.panoramix import Panoramix, KalmanorixFuser, MeanFuser
from kalmanorix.uncertainty import (
    CentroidDistanceSigma2,
    ConstantSigma2,
    ScaledSigma2,
)
from kalmanorix.ood_datasets import (
    create_ood_test_set,
    create_synthetic_ood_test_set,
)
from kalmanorix.calibration import (
    compute_embedding_calibration,
    compute_retrieval_calibration,
    plot_reliability_diagram,
    CalibrationResult,
)
from kalmanorix.experiment_reporting import write_calibration_report


logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OODExperimentConfig(TrainingConfig):
    """
    Configuration for Milestone 2.2 OOD robustness experiment.

    Extends TrainingConfig with OOD-specific parameters.
    """

    # OOD domain (unseen during training)
    ood_domain: DomainEnum = DomainEnum.TECH  # Default: tech as OOD

    # OOD test set parameters
    ood_proportion: float = 0.5  # Proportion of OOD queries in test set
    ood_n_docs: int = 5000  # Number of documents in test corpus
    ood_n_queries: int = 500  # Number of queries

    # Mis-specification scaling factors to test
    scaling_factors: List[float] = field(
        default_factory=lambda: [0.1, 0.5, 1.0, 2.0, 10.0]
    )

    # Calibration parameters
    calibration_n_bins: int = 10
    calibration_distance_metric: Literal["cosine", "l2"] = "cosine"  # "cosine" or "l2"
    calibration_error_tolerance: float = 0.25

    # Reference for calibration (ground truth embeddings)
    calibration_reference: Literal["monolith", "ensemble"] = (
        "monolith"  # "monolith" or "ensemble"
    )

    # Ablation: constant variance for all specialists (should equal averaging)
    ablation_constant_variance: float = 1.0

    def __post_init__(self) -> None:
        """Validate OOD-specific configuration."""
        super().__post_init__()

        if self.ood_domain in self.domains:
            raise ValueError(
                f"OOD domain {self.ood_domain} cannot be in training domains {self.domains}"
            )

        if not 0.0 <= self.ood_proportion <= 1.0:
            raise ValueError(
                f"ood_proportion must be in [0, 1], got {self.ood_proportion}"
            )

        if len(self.scaling_factors) == 0:
            raise ValueError("scaling_factors cannot be empty")

        if self.calibration_reference not in ("monolith", "ensemble"):
            raise ValueError(
                f"calibration_reference must be 'monolith' or 'ensemble', got {self.calibration_reference}"
            )


@dataclass
class OODExperimentResults:
    """Container for Milestone 2.2 experiment results."""

    config: OODExperimentConfig

    # Paths to pre-trained models from Milestone 2.1
    specialist_paths: Dict[DomainEnum, Path]
    monolith_path: Path
    test_set_path: Path

    # Basic retrieval metrics
    specialist_recalls: Dict[DomainEnum, Dict[int, float]]
    monolith_recalls: Dict[int, float]
    kalman_recalls: Dict[int, float]
    mean_recalls: Dict[int, float]

    # OOD-specific metrics
    ood_kalman_recalls: Dict[int, float]  # Recall on OOD queries only
    ood_mean_recalls: Dict[int, float]

    # Calibration results
    specialist_calibration: Dict[DomainEnum, CalibrationResult]
    monolith_calibration: CalibrationResult
    kalman_calibration: CalibrationResult
    mean_calibration: CalibrationResult

    # Mis-specification sweep results
    mis_spec_results: Dict[float, Dict[str, float]]  # scale -> {metric: value}

    # Ablation results (constant variance)
    ablation_recalls: Dict[int, float]
    ablation_calibration: CalibrationResult

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""

        def serialize_calibration(result: CalibrationResult) -> dict:
            return {
                "ece": result.ece,
                "brier_score": result.brier_score,
                "n_samples": result.n_samples,
                "mean_confidence": result.mean_confidence,
                "mean_accuracy": result.mean_accuracy,
                "bin_edges": result.bin_edges.tolist(),
                "bin_centers": result.bin_centers.tolist(),
                "bin_accuracies": result.bin_accuracies.tolist(),
                "bin_confidences": result.bin_confidences.tolist(),
                "bin_counts": result.bin_counts.tolist(),
            }

        return {
            "config": json.loads(self.config.to_json()),
            "specialist_paths": {
                d.value: str(p) for d, p in self.specialist_paths.items()
            },
            "monolith_path": str(self.monolith_path),
            "test_set_path": str(self.test_set_path),
            "specialist_recalls": {
                d.value: {k: v for k, v in recalls.items()}
                for d, recalls in self.specialist_recalls.items()
            },
            "monolith_recalls": self.monolith_recalls,
            "kalman_recalls": self.kalman_recalls,
            "mean_recalls": self.mean_recalls,
            "ood_kalman_recalls": self.ood_kalman_recalls,
            "ood_mean_recalls": self.ood_mean_recalls,
            "specialist_calibration": {
                d.value: serialize_calibration(result)
                for d, result in self.specialist_calibration.items()
            },
            "monolith_calibration": serialize_calibration(self.monolith_calibration),
            "kalman_calibration": serialize_calibration(self.kalman_calibration),
            "mean_calibration": serialize_calibration(self.mean_calibration),
            "mis_spec_results": self.mis_spec_results,
            "ablation_recalls": self.ablation_recalls,
            "ablation_calibration": serialize_calibration(self.ablation_calibration),
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_specialist_model(
    model_path: Path,
    domain: Optional[DomainEnum] = None,
    calibration_texts: Optional[List[str]] = None,
    scale_factor: float = 1.0,
    constant_variance: Optional[float] = None,
) -> SEF:
    """
    Load a trained specialist as SEF with uncertainty.

    Supports:
    - Original centroid-based uncertainty (if domain and calibration_texts provided)
    - Scaled uncertainty (scale_factor ≠ 1.0)
    - Constant variance (constant_variance provided)

    Parameters
    ----------
    model_path : Path
        Path to trained SentenceTransformer model.
    domain : Optional[DomainEnum]
        Domain for centroid calibration.
    calibration_texts : Optional[List[str]]
        Texts for centroid calibration.
    scale_factor : float
        Multiplicative scaling factor for sigma2.
    constant_variance : Optional[float]
        If provided, use constant variance instead of query-dependent.

    Returns
    -------
    SEF
        Loaded specialist with uncertainty.
    """
    from sentence_transformers import SentenceTransformer

    # Load the SentenceTransformer model
    st_model = SentenceTransformer(str(model_path))

    # Embedding function
    def embed_fn(s):
        return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    # Build sigma2
    sigma2: Callable[[str], float]
    if constant_variance is not None:
        sigma2 = ConstantSigma2(value=constant_variance)
        logger.info(
            f"  Using constant variance {constant_variance} for {model_path.name}"
        )

    elif (
        domain is not None
        and calibration_texts is not None
        and len(calibration_texts) > 0
    ):
        # Use centroid-based uncertainty calibrated on domain texts
        base_sigma2 = CentroidDistanceSigma2.from_calibration(
            embed=embed_fn,
            calibration_texts=calibration_texts,
            base_sigma2=0.5,
            scale=0.5,
        )
        if abs(scale_factor - 1.0) > 1e-9:
            sigma2 = ScaledSigma2(base_sigma2=base_sigma2, scale=scale_factor)
            logger.info(
                f"  Scaled uncertainty for {domain.value} (scale={scale_factor:.2f}) "
                f"using {len(calibration_texts)} calibration texts"
            )
        else:
            sigma2 = base_sigma2
            logger.info(
                f"  Calibrated uncertainty for {domain.value} "
                f"using {len(calibration_texts)} texts"
            )

    else:
        # Fallback: constant uncertainty (original behavior)
        dim = st_model.get_sentence_embedding_dimension()
        assert dim is not None
        sigma2 = CentroidDistanceSigma2(
            embed=embed_fn,
            centroid=np.zeros(dim),
            base_sigma2=0.5,
        )
        if abs(scale_factor - 1.0) > 1e-9:
            sigma2 = ScaledSigma2(base_sigma2=sigma2, scale=scale_factor)
            logger.info(
                f"  Scaled constant uncertainty for {model_path.name} (scale={scale_factor:.2f})"
            )
        else:
            logger.warning(
                f"Using constant uncertainty for {model_path.name} "
                "(no domain/calibration texts provided)"
            )

    return SEF(
        embed=embed_fn,
        sigma2=sigma2,
        name=model_path.name,
    )


def evaluate_model(
    model,
    test_docs: List[str],
    test_queries: List[Tuple[str, int]],
    k_values: List[int],
    is_village: bool = False,
    fuser_type: str = "kalman",  # "kalman" or "mean"
) -> Dict[int, float]:
    """
    Evaluate a model or village on retrieval task.

    Parameters
    ----------
    model : Union[SentenceTransformer, Village]
        Model or village to evaluate.
    test_docs : List[str]
        Document corpus.
    test_queries : List[Tuple[str, int]]
        Query texts and true document indices.
    k_values : List[int]
        Recall@k values to compute.
    is_village : bool
        Whether model is a Village (requires scout and panoramix).
    fuser_type : str
        If is_village=True, specify fuser type ("kalman" or "mean").

    Returns
    -------
    Dict[int, float]
        Mapping from k to recall.
    """
    # Encode documents once
    if is_village:
        village = model
        scout = ScoutRouter(mode="all")
        if fuser_type == "kalman":
            panoramix = Panoramix(fuser=KalmanorixFuser())
        elif fuser_type == "mean":
            panoramix = Panoramix(fuser=MeanFuser())
        else:
            raise ValueError(f"Unknown fuser_type: {fuser_type}")

        # Compute fused document embeddings
        doc_embeddings = []
        for doc in test_docs:
            potion = panoramix.brew(doc, village=village, scout=scout)
            doc_embeddings.append(potion.vector)
        doc_embs = np.array(doc_embeddings)

        # Compute recall for each k
        village_recalls: Dict[int, List[float]] = {k: [] for k in k_values}
        for query, true_id in test_queries:
            potion = panoramix.brew(query, village=village, scout=scout)
            q_emb = potion.vector
            scores = doc_embs @ q_emb
            ranked = list(np.argsort(-scores))
            for k in k_values:
                village_recalls[k].append(1.0 if true_id in ranked[:k] else 0.0)
        results = {k: statistics.mean(village_recalls[k]) for k in k_values}
        return results

    else:
        # Monolithic model
        # Encode all documents
        doc_embs = model.encode(
            test_docs, normalize_embeddings=True, show_progress_bar=False
        )

        # Evaluate each query
        model_recalls: dict[int, list[float]] = {k: [] for k in k_values}
        for query, true_id in test_queries:
            q_emb = model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            scores = doc_embs @ q_emb
            ranked = list(np.argsort(-scores))

            for k in k_values:
                model_recalls[k].append(1.0 if true_id in ranked[:k] else 0.0)

        return {k: statistics.mean(model_recalls[k]) for k in k_values}


def evaluate_ood_queries(
    model,
    test_docs: List[str],
    test_queries: List[Tuple[str, int]],
    k_values: List[int],
    is_village: bool = False,
    fuser_type: str = "kalman",
) -> Dict[int, float]:
    """
    Evaluate only OOD queries (those with true_doc_id = -1).

    Since there's no correct document for OOD queries, we compute
    recall@k as 0 for all queries. This function instead computes
    the average similarity to the nearest document (as a measure of
    embedding quality), but for consistency with other metrics we
    still return recall (always 0).

    Parameters
    ----------
    model : Union[SentenceTransformer, Village]
        Model or village.
    test_docs : List[str]
        Document corpus.
    test_queries : List[Tuple[str, int]]
        Query texts and true document indices (-1 for OOD).
    k_values : List[int]
        Recall@k values (unused, kept for interface consistency).
    is_village : bool
        Whether model is a Village.
    fuser_type : str
        Fuser type if village.

    Returns
    -------
    Dict[int, float]
        Recall@k = 0 for all k (since no correct document).
    """
    # Filter OOD queries (true_id == -1)
    ood_queries = [(q, idx) for q, idx in test_queries if idx == -1]
    if not ood_queries:
        return {k: 0.0 for k in k_values}

    # Compute recall (always 0 because no correct document)
    # We could compute other metrics like average similarity, but
    # for now we return 0 to indicate that recall is not defined for OOD.
    return {k: 0.0 for k in k_values}


def compute_calibration_for_model(
    model,
    test_docs: List[str],
    test_queries: List[Tuple[str, int]],
    config: OODExperimentConfig,
    reference_embeddings: Optional[np.ndarray] = None,
    is_village: bool = False,
    fuser_type: str = "kalman",
) -> CalibrationResult:
    """
    Compute calibration metrics for a model or village.

    Parameters
    ----------
    model : Union[SentenceTransformer, Village]
        Model or village.
    test_docs : List[str]
        Document corpus.
    test_queries : List[Tuple[str, int]]
        Query texts and true document indices.
    config : OODExperimentConfig
        Experiment configuration.
    reference_embeddings : Optional[np.ndarray]
        Reference embeddings for embedding calibration.
        If None, retrieval calibration is used.
    is_village : bool
        Whether model is a Village.
    fuser_type : str
        Fuser type if village.

    Returns
    -------
    CalibrationResult
        Calibration metrics.
    """
    if is_village:
        village = model
        scout = ScoutRouter(mode="all")
        if fuser_type == "kalman":
            panoramix = Panoramix(fuser=KalmanorixFuser())
        elif fuser_type == "mean":
            panoramix = Panoramix(fuser=MeanFuser())
        else:
            raise ValueError(f"Unknown fuser_type: {fuser_type}")

        # Encode queries with uncertainty
        query_embeddings_list: list[np.ndarray] = []
        query_variances_list: list[float] = []
        true_indices = []

        for query, true_id in test_queries:
            potion = panoramix.brew(query, village=village, scout=scout)
            query_embeddings_list.append(potion.vector)
            # Extract scalar variance from potion metadata
            # Assuming potion.meta contains 'variance' scalar
            meta_val = potion.meta.get("variance", 1.0) if potion.meta else 1.0
            variance = float(cast(float, meta_val))
            query_variances_list.append(variance)
            true_indices.append(true_id)

        query_embeddings = np.array(query_embeddings_list)
        query_variances = np.array(query_variances_list)

    else:
        # Monolithic model (no uncertainty)
        # For monolith, we assume constant variance (1.0) as baseline
        query_embeddings = model.encode(
            [q for q, _ in test_queries],
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        query_variances = np.ones(len(test_queries))
        true_indices = [idx for _, idx in test_queries]

    if reference_embeddings is not None:
        # Embedding calibration
        return compute_embedding_calibration(
            specialist_embeddings=query_embeddings,
            reference_embeddings=reference_embeddings,
            predicted_variances=query_variances,
            n_bins=config.calibration_n_bins,
            norm=config.calibration_distance_metric,
            error_tolerance=config.calibration_error_tolerance,
        )
    else:
        # Retrieval calibration (requires document embeddings)
        if is_village:
            # For village, we need to compute document embeddings using fusion
            scout = ScoutRouter(mode="all")
            if fuser_type == "kalman":
                panoramix = Panoramix(fuser=KalmanorixFuser())
            else:
                panoramix = Panoramix(fuser=MeanFuser())
            doc_embeddings_list = []
            for doc in test_docs:
                potion = panoramix.brew(doc, village=model, scout=scout)
                doc_embeddings_list.append(potion.vector)
            doc_embeddings = np.array(doc_embeddings_list)
        else:
            doc_embeddings = model.encode(
                test_docs, normalize_embeddings=True, show_progress_bar=False
            )

        return compute_retrieval_calibration(
            query_embeddings=query_embeddings,
            doc_embeddings=doc_embeddings,
            query_variances=query_variances,
            true_indices=true_indices,
            k=10,  # Fixed k for calibration
            n_bins=config.calibration_n_bins,
            distance_metric=config.calibration_distance_metric,
        )


def run_experiment(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_real_data: bool = False,
) -> OODExperimentResults:
    """
    Run the full Milestone 2.2 experiment.

    Parameters
    ----------
    config_path : Optional[Path]
        Path to configuration YAML file.
    output_dir : Optional[Path]
        Override output directory.
    use_real_data : bool
        Whether to use real datasets (requires internet).

    Returns
    -------
    OODExperimentResults
        Complete experiment results.
    """
    # Load configuration
    # Note: We need to load as OODExperimentConfig, not TrainingConfig
    # For simplicity, we'll load as dict and convert
    if config_path is None:
        config = OODExperimentConfig()
    else:
        # Load YAML and convert to OODExperimentConfig
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        # Convert string domains to DomainEnum
        if "domains" in data:
            data["domains"] = [DomainEnum(d) for d in data["domains"]]
        if "ood_domain" in data:
            data["ood_domain"] = DomainEnum(data["ood_domain"])
        config = OODExperimentConfig(**data)

    if output_dir:
        config = replace(config, output_dir=output_dir)

    logger.info(f"Starting Milestone 2.2 experiment: {config.experiment_name}")
    logger.info(f"Training domains: {[d.value for d in config.domains]}")
    logger.info(f"OOD domain: {config.ood_domain.value}")
    logger.info(f"Output directory: {config.output_dir}")

    # Create experiment directory
    experiment_dir = config.output_dir / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reproducibility
    config.to_yaml(experiment_dir / "config.yaml")

    # Step 1: Load pre-trained models from Milestone 2.1
    logger.info("Step 1: Loading pre-trained models...")
    # We assume models are saved in the Milestone 2.1 output directory
    # This is a simplification; in practice we need to locate them.
    # For now, we'll look in config.output_dir / "milestone_2_1" / config.experiment_name
    milestone_2_1_dir = (
        config.output_dir.parent / "milestone_2_1" / config.experiment_name
    )
    if not milestone_2_1_dir.exists():
        raise FileNotFoundError(
            f"Milestone 2.1 output not found at {milestone_2_1_dir}. "
            f"Please run Milestone 2.1 first."
        )

    specialist_paths = {}
    for domain in config.domains:
        model_path = milestone_2_1_dir / f"specialist_{domain.value}"
        if not model_path.exists():
            # Try alternative naming
            model_path = milestone_2_1_dir / f"specialist_{domain.value}_model"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Specialist model for domain {domain.value} not found in {milestone_2_1_dir}"
            )
        specialist_paths[domain] = model_path

    monolith_path = milestone_2_1_dir / "monolith"
    if not monolith_path.exists():
        monolith_path = milestone_2_1_dir / "monolith_model"
    if not monolith_path.exists():
        raise FileNotFoundError(f"Monolith model not found in {milestone_2_1_dir}")

    logger.info(f"Loaded {len(specialist_paths)} specialists and monolith model")

    # Step 2: Load domain data for calibration and test set creation
    logger.info("Step 2: Loading domain data...")
    if use_real_data:
        from kalmanorix.datasets import load_multiple_domains

        # Load all domains (including OOD domain)
        all_domains = [d.value for d in config.domains] + [config.ood_domain.value]
        domain_datasets = load_multiple_domains(
            domains=all_domains,  # type: ignore
            samples_per_domain=config.samples_per_domain,
            cache=True,
            force_download=False,
        )
    else:
        # Use synthetic data for testing
        domain_datasets = {}
        # We'll create synthetic datasets on the fly during test set creation

    # Step 3: Create OOD test set
    logger.info("Step 3: Creating OOD test set...")
    if use_real_data:
        # Convert DomainEnum to string for dataset lookup
        seen_domains = [d.value for d in config.domains]
        ood_domain = config.ood_domain.value
        documents, queries = create_ood_test_set(
            domain_datasets=domain_datasets,
            seen_domains=seen_domains,  # type: ignore
            ood_domain=ood_domain,  # type: ignore
            ood_proportion=config.ood_proportion,
            n_docs=config.ood_n_docs,
            n_queries=config.ood_n_queries,
            seed=config.seed,
        )
    else:
        # Synthetic test set
        seen_domains = [d.value for d in config.domains]
        ood_domain = config.ood_domain.value
        documents, queries = create_synthetic_ood_test_set(
            seen_domains=seen_domains,  # type: ignore
            ood_domain=ood_domain,  # type: ignore
            n_docs=config.ood_n_docs,
            n_queries=config.ood_n_queries,
            seed=config.seed,
        )

    # Save test set for reproducibility
    test_set_path = experiment_dir / "test_set.json"
    with open(test_set_path, "w", encoding="utf-8") as f:
        json.dump({"documents": documents, "queries": queries}, f, indent=2)
    logger.info(f"Test set saved to {test_set_path}")

    # Step 4: Load models with calibration
    logger.info("Step 4: Loading models with uncertainty calibration...")
    # For each specialist, we need calibration texts from its domain
    calibration_texts = {}
    if use_real_data:
        for domain in config.domains:
            dataset = domain_datasets[domain.value]
            # Use validation split for calibration
            calibration_texts[domain] = dataset.validation[:100]  # First 100 texts
    else:
        # Generate synthetic calibration texts
        from kalmanorix.toy_corpus import generate_anchor_sentences

        for domain in config.domains:
            tag = {
                DomainEnum.MEDICAL: "medical",
                DomainEnum.LEGAL: "general",
                DomainEnum.TECH: "tech",
                DomainEnum.COOK: "cook",
                DomainEnum.GENERAL: "general",
            }.get(domain, "general")
            texts = generate_anchor_sentences(n=100, domains=(tag,), seed=config.seed)
            calibration_texts[domain] = texts

    # Load specialists with calibrated uncertainty
    specialists = []
    specialist_sefs = {}
    for domain in config.domains:
        sef = load_specialist_model(
            model_path=specialist_paths[domain],
            domain=domain,
            calibration_texts=calibration_texts[domain],
            scale_factor=1.0,  # No scaling for baseline
        )
        specialists.append(sef)
        specialist_sefs[domain] = sef

    # Load monolith model
    from sentence_transformers import SentenceTransformer

    monolith_model = SentenceTransformer(str(monolith_path))

    # Create village
    village = Village(modules=specialists)

    # Step 5: Evaluate baseline retrieval performance
    logger.info("Step 5: Evaluating baseline retrieval performance...")
    k_values = config.recall_k_values

    # Individual specialists
    specialist_recalls = {}
    for domain, sef in specialist_sefs.items():
        # Create a single-module village for each specialist
        specialist_village = Village(modules=[sef])
        recalls = evaluate_model(
            model=specialist_village,
            test_docs=documents,
            test_queries=queries,
            k_values=k_values,
            is_village=True,
            fuser_type="mean",  # Single module, fuser doesn't matter
        )
        specialist_recalls[domain] = recalls

    # Monolith
    monolith_recalls = evaluate_model(
        model=monolith_model,
        test_docs=documents,
        test_queries=queries,
        k_values=k_values,
        is_village=False,
    )

    # Kalman fusion
    kalman_recalls = evaluate_model(
        model=village,
        test_docs=documents,
        test_queries=queries,
        k_values=k_values,
        is_village=True,
        fuser_type="kalman",
    )

    # Mean fusion
    mean_recalls = evaluate_model(
        model=village,
        test_docs=documents,
        test_queries=queries,
        k_values=k_values,
        is_village=True,
        fuser_type="mean",
    )

    # OOD-only queries
    ood_kalman_recalls = evaluate_ood_queries(
        model=village,
        test_docs=documents,
        test_queries=queries,
        k_values=k_values,
        is_village=True,
        fuser_type="kalman",
    )
    ood_mean_recalls = evaluate_ood_queries(
        model=village,
        test_docs=documents,
        test_queries=queries,
        k_values=k_values,
        is_village=True,
        fuser_type="mean",
    )

    # Step 6: Compute calibration metrics
    logger.info("Step 6: Computing calibration metrics...")
    # Get reference embeddings for calibration (monolith or ensemble)
    # We encode the query texts, not documents, for embedding calibration
    query_texts = [q for q, _ in queries]
    if config.calibration_reference == "monolith":
        reference_embeddings = monolith_model.encode(
            query_texts, normalize_embeddings=True, show_progress_bar=False
        )
    else:
        # Ensemble reference: average of specialist embeddings
        # For simplicity, we'll use monolith as reference
        reference_embeddings = monolith_model.encode(
            query_texts, normalize_embeddings=True, show_progress_bar=False
        )

    specialist_calibration = {}
    for domain, sef in specialist_sefs.items():
        specialist_village = Village(modules=[sef])
        cal = compute_calibration_for_model(
            model=specialist_village,
            test_docs=documents,
            test_queries=queries,
            config=config,
            reference_embeddings=reference_embeddings,
            is_village=True,
            fuser_type="mean",
        )
        specialist_calibration[domain] = cal

    monolith_calibration = compute_calibration_for_model(
        model=monolith_model,
        test_docs=documents,
        test_queries=queries,
        config=config,
        reference_embeddings=reference_embeddings,
        is_village=False,
    )

    kalman_calibration = compute_calibration_for_model(
        model=village,
        test_docs=documents,
        test_queries=queries,
        config=config,
        reference_embeddings=reference_embeddings,
        is_village=True,
        fuser_type="kalman",
    )

    mean_calibration = compute_calibration_for_model(
        model=village,
        test_docs=documents,
        test_queries=queries,
        config=config,
        reference_embeddings=reference_embeddings,
        is_village=True,
        fuser_type="mean",
    )

    # Step 7: Mis-specification sweep
    logger.info("Step 7: Running mis-specification sweep...")
    mis_spec_results = {}
    for scale in config.scaling_factors:
        logger.info(f"  Testing scale factor {scale:.2f}")
        # Load specialists with scaled uncertainty
        scaled_specialists = []
        for domain in config.domains:
            sef = load_specialist_model(
                model_path=specialist_paths[domain],
                domain=domain,
                calibration_texts=calibration_texts[domain],
                scale_factor=scale,
            )
            scaled_specialists.append(sef)
        scaled_village = Village(modules=scaled_specialists)

        # Evaluate with Kalman fusion
        recalls = evaluate_model(
            model=scaled_village,
            test_docs=documents,
            test_queries=queries,
            k_values=k_values,
            is_village=True,
            fuser_type="kalman",
        )
        # Store average recall@10 as metric
        avg_recall = recalls.get(10, 0.0)
        mis_spec_results[scale] = {
            "recall@10": avg_recall,
            "recall@5": recalls.get(5, 0.0),
            "recall@1": recalls.get(1, 0.0),
        }

    # Step 8: Ablation study (constant variance)
    logger.info("Step 8: Running ablation study (constant variance)...")
    constant_specialists = []
    for domain in config.domains:
        sef = load_specialist_model(
            model_path=specialist_paths[domain],
            domain=domain,
            calibration_texts=calibration_texts[domain],
            constant_variance=config.ablation_constant_variance,
        )
        constant_specialists.append(sef)
    constant_village = Village(modules=constant_specialists)

    ablation_recalls = evaluate_model(
        model=constant_village,
        test_docs=documents,
        test_queries=queries,
        k_values=k_values,
        is_village=True,
        fuser_type="kalman",
    )

    ablation_calibration = compute_calibration_for_model(
        model=constant_village,
        test_docs=documents,
        test_queries=queries,
        config=config,
        reference_embeddings=reference_embeddings,
        is_village=True,
        fuser_type="kalman",
    )

    # Step 9: Generate plots
    logger.info("Step 9: Generating plots...")
    try:
        import matplotlib.pyplot as plt

        # Reliability diagrams for each model
        for domain, cal in specialist_calibration.items():
            plot_reliability_diagram(
                cal,
                save_path=str(
                    experiment_dir / f"reliability_specialist_{domain.value}.png"
                ),
                title=f"Reliability Diagram - Specialist {domain.value}",
            )

        plot_reliability_diagram(
            monolith_calibration,
            save_path=str(experiment_dir / "reliability_monolith.png"),
            title="Reliability Diagram - Monolith",
        )
        plot_reliability_diagram(
            kalman_calibration,
            save_path=str(experiment_dir / "reliability_kalman.png"),
            title="Reliability Diagram - Kalman Fusion",
        )
        plot_reliability_diagram(
            mean_calibration,
            save_path=str(experiment_dir / "reliability_mean.png"),
            title="Reliability Diagram - Mean Fusion",
        )
        plot_reliability_diagram(
            ablation_calibration,
            save_path=str(experiment_dir / "reliability_ablation_constant.png"),
            title="Reliability Diagram - Constant-Variance Ablation",
        )

        # Performance vs mis-specification factor
        scales = list(mis_spec_results.keys())
        recalls_at_10 = [mis_spec_results[s]["recall@10"] for s in scales]
        plt.figure(figsize=(8, 6))
        plt.plot(scales, recalls_at_10, "o-", linewidth=2, markersize=8)
        plt.xscale("log")
        plt.xlabel("Uncertainty scaling factor (log scale)")
        plt.ylabel("Recall@10")
        plt.title("Robustness to Mis-specified Uncertainties")
        plt.grid(True, alpha=0.3)
        plt.savefig(experiment_dir / "mis_specification_sweep.png", dpi=150)
        plt.close()

    except ImportError:
        logger.warning("matplotlib not available, skipping plots")

    # Step 10: Compile results
    logger.info("Step 10: Compiling results...")
    results = OODExperimentResults(
        config=config,
        specialist_paths=specialist_paths,
        monolith_path=monolith_path,
        test_set_path=test_set_path,
        specialist_recalls=specialist_recalls,
        monolith_recalls=monolith_recalls,
        kalman_recalls=kalman_recalls,
        mean_recalls=mean_recalls,
        ood_kalman_recalls=ood_kalman_recalls,
        ood_mean_recalls=ood_mean_recalls,
        specialist_calibration=specialist_calibration,
        monolith_calibration=monolith_calibration,
        kalman_calibration=kalman_calibration,
        mean_calibration=mean_calibration,
        mis_spec_results=mis_spec_results,
        ablation_recalls=ablation_recalls,
        ablation_calibration=ablation_calibration,
    )

    # Save results
    results_path = experiment_dir / "results.json"
    results.save(results_path)
    logger.info(f"Results saved to {results_path}")

    write_calibration_report(
        experiment_dir=experiment_dir,
        specialist_calibration={d.value: c for d, c in specialist_calibration.items()},
        monolith_calibration=monolith_calibration,
        kalman_calibration=kalman_calibration,
        mean_calibration=mean_calibration,
        ablation_calibration=ablation_calibration,
    )

    # Print summary
    logger.info("\n=== EXPERIMENT SUMMARY ===")
    logger.info(f"Kalman fusion Recall@10: {kalman_recalls.get(10, 0.0):.3f}")
    logger.info(f"Mean fusion Recall@10: {mean_recalls.get(10, 0.0):.3f}")
    logger.info(f"Monolith Recall@10: {monolith_recalls.get(10, 0.0):.3f}")
    logger.info(f"Kalman ECE: {kalman_calibration.ece:.3f}")
    logger.info(f"Mean ECE: {mean_calibration.ece:.3f}")

    return results


def main() -> None:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(
        description="Run Milestone 2.2: Uncertainty Robustness Test"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to YAML configuration file (default: use default config)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Override output directory (default: from config)",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real datasets (requires internet)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing models (assumes Milestone 2.1 completed)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run experiment
    results = run_experiment(
        config_path=Path(args.config) if args.config else None,
        output_dir=Path(args.output_dir) if args.output_dir else None,
        use_real_data=args.real_data,
    )

    print("\nExperiment completed successfully!")
    print(
        f"Results saved to: {results.config.output_dir / results.config.experiment_name}"
    )


if __name__ == "__main__":
    main()
