"""
Orchestration script for Milestone 2.1: Specialists vs Monolith test.

Runs the full experiment:
1. Load configuration
2. Train specialists (1 epoch each domain) with compute tracking
3. Train monolith (2 epochs combined) with compute tracking
4. Generate mixed-domain test corpus
5. Evaluate: individual specialists, monolith, fused specialists
6. Compare Recall@k metrics
7. Log compute usage and results
"""

# pylint: disable=wrong-import-position,import-outside-toplevel,logging-fstring-interpolation,too-many-statements,unnecessary-comprehension,no-else-return

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from experiments.config import TrainingConfig, DomainEnum, load_config
from experiments.train_specialists_st import train_specialists_from_config
from experiments.train_monolith import train_monolith
from experiments.generate_test_set import generate_test_set

# Import Kalmanorix components for evaluation
from kalmanorix.village import Village, SEF
from kalmanorix.scout import ScoutRouter
from kalmanorix.panoramix import Panoramix
from kalmanorix.uncertainty import CentroidDistanceSigma2
from kalmanorix.panoramix import KalmanorixFuser
from kalmanorix.alignment import compute_alignments, align_sef_list


logger = logging.getLogger(__name__)


# pylint: disable=too-many-instance-attributes
@dataclass
class ExperimentResults:
    """Container for all experiment results."""

    config: TrainingConfig

    # Paths
    specialist_paths: Dict[DomainEnum, Path]
    monolith_path: Path
    test_set_path: Path

    # Compute metrics
    specialist_compute_paths: Dict[DomainEnum, Path]
    monolith_compute_path: Path

    # Evaluation metrics
    specialist_recalls: Dict[DomainEnum, Dict[int, float]]  # k -> recall
    monolith_recalls: Dict[int, float]
    fused_recalls: Dict[int, float]

    # Statistical significance (optional)
    p_values: Optional[Dict[int, float]] = None

    def to_dict(self) -> dict:
        """Convert to JSON-serializable dictionary."""
        return {
            "config": json.loads(self.config.to_json()),
            "specialist_paths": {
                d.value: str(p) for d, p in self.specialist_paths.items()
            },
            "monolith_path": str(self.monolith_path),
            "test_set_path": str(self.test_set_path),
            "specialist_compute_paths": {
                d.value: str(p) for d, p in self.specialist_compute_paths.items()
            },
            "monolith_compute_path": str(self.monolith_compute_path),
            "specialist_recalls": {
                d.value: {k: v for k, v in recalls.items()}
                for d, recalls in self.specialist_recalls.items()
            },
            "monolith_recalls": self.monolith_recalls,
            "fused_recalls": self.fused_recalls,
            "p_values": self.p_values,
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
) -> SEF:
    """Load a trained specialist as SEF with uncertainty.

    If domain and calibration_texts are provided, uses centroid-based uncertainty
    calibrated on the provided texts.
    Otherwise uses constant uncertainty (backward compatibility).
    """
    from sentence_transformers import SentenceTransformer

    # Load the SentenceTransformer model
    st_model = SentenceTransformer(str(model_path))

    # Embedding function for uncertainty estimation
    def embed_fn(s):
        return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    if (
        domain is not None
        and calibration_texts is not None
        and len(calibration_texts) > 0
    ):
        # Use centroid-based uncertainty calibrated on domain texts
        sigma2 = CentroidDistanceSigma2.from_calibration(
            embed=embed_fn,
            calibration_texts=calibration_texts,
            base_sigma2=0.5,  # Increased to reduce extreme variance ratios
            scale=0.5,
        )
        logger.info(
            f"  Calibrated uncertainty for {domain.value} using {len(calibration_texts)} texts"
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
        logger.warning(
            f"Using constant uncertainty for {model_path.name} (no domain/calibration texts provided)"
        )

    return SEF(
        embed=embed_fn,
        sigma2=sigma2,
        name=model_path.name,
    )


def evaluate_model(
    model_or_village,
    test_docs: List[str],
    test_queries: List[Tuple[str, int]],
    k_values: List[int],
    is_village: bool = False,
) -> Dict[int, float]:
    """
    Evaluate a model or village on retrieval task.

    Parameters
    ----------
    model_or_village : Union[SentenceTransformer, Village]
        Model or village to evaluate.
    test_docs : List[str]
        Document corpus.
    test_queries : List[Tuple[str, int]]
        Query texts and true document indices.
    k_values : List[int]
        Recall@k values to compute.
    is_village : bool
        Whether model_or_village is a Village (requires scout and panoramix).

    Returns
    -------
    Dict[int, float]
        Mapping from k to recall.
    """

    # Encode documents once
    if is_village:
        # For village, we need to encode with each specialist and fuse per query
        # We'll use Panoramix for evaluation
        village = model_or_village
        scout = ScoutRouter(mode="all")
        panoramix = Panoramix(fuser=KalmanorixFuser())

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
        model = model_or_village
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


def run_experiment(
    config_path: Optional[Path] = None,
    output_dir: Optional[Path] = None,
    use_real_data: bool = False,
) -> ExperimentResults:
    """
    Run the full Milestone 2.1 experiment.

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
    ExperimentResults
        Complete experiment results.
    """
    # Load configuration
    config = load_config(config_path)
    if output_dir:
        config = replace(config, output_dir=output_dir)

    logger.info(f"Starting Milestone 2.1 experiment: {config.experiment_name}")
    logger.info(f"Domains: {[d.value for d in config.domains]}")
    logger.info(f"Output directory: {config.output_dir}")

    # Create experiment directory
    experiment_dir = config.output_dir / config.experiment_name
    experiment_dir.mkdir(parents=True, exist_ok=True)

    # Save configuration for reproducibility
    config.to_yaml(experiment_dir / "config.yaml")

    # Step 1: Load domain data
    logger.info("Step 1: Loading domain data...")
    if use_real_data:
        from kalmanorix.datasets import load_multiple_domains

        domain_datasets = load_multiple_domains(
            domains=[d.value for d in config.domains],
            samples_per_domain=config.samples_per_domain,
            cache=True,
            force_download=False,
        )
        domain_data = {
            DomainEnum(d): dataset.train  # Use training split
            for d, dataset in domain_datasets.items()
        }
    else:
        # Use synthetic data
        from kalmanorix.toy_corpus import generate_anchor_sentences

        domain_data = {}
        for domain in config.domains:
            tag = {
                DomainEnum.MEDICAL: "medical",
                DomainEnum.LEGAL: "legal",
                DomainEnum.TECH: "tech",
                DomainEnum.COOK: "cook",
                DomainEnum.GENERAL: "general",
            }.get(domain, "general")

            sentences = generate_anchor_sentences(
                n=config.samples_per_domain,
                domains=(tag,),
                seed=config.seed + hash(domain),
            )
            domain_data[domain] = sentences

    # Step 2: Train specialists
    logger.info("Step 2: Training specialists...")
    specialist_dir = experiment_dir / "specialists"
    specialist_paths = train_specialists_from_config(
        config=config,
        domain_data=domain_data,
        output_dir=specialist_dir,
    )

    # Step 3: Train monolithic model
    logger.info("Step 3: Training monolithic model...")
    monolith_dir = experiment_dir / "monolith"
    monolith_result = train_monolith(
        config=config,
        domain_data=domain_data,
        output_dir=monolith_dir,
    )

    # Step 4: Generate test set
    logger.info("Step 4: Generating mixed-domain test set...")
    test_set_path = experiment_dir / "test_set.json"
    test_docs, test_queries = generate_test_set(
        config=config,
        output_path=test_set_path,
        use_real_data=use_real_data,
    )

    # Step 5: Evaluate specialists individually
    logger.info("Step 5: Evaluating specialists individually...")
    specialist_recalls = {}
    for domain, model_path in specialist_paths.items():
        logger.info(f"  Evaluating {domain.value}...")
        calibration_texts = domain_data[domain][
            :1000
        ]  # Use first 1000 training sentences
        model = load_specialist_model(
            model_path, domain=domain, calibration_texts=calibration_texts
        )

        # Convert SEF to SentenceTransformer-like interface
        class ModelWrapper:
            """Wrapper to make SEF compatible with SentenceTransformer API."""

            def __init__(self, sef):
                """Initialize wrapper."""
                self.sef = sef

            def encode(self, texts, normalize_embeddings=False, **kwargs):
                """Encode texts."""
                embs = [self.sef.embed(t) for t in texts]
                embs = np.array(embs)
                if normalize_embeddings:
                    norms = np.linalg.norm(embs, axis=1, keepdims=True)
                    embs = embs / (norms + 1e-12)
                return embs

        wrapper = ModelWrapper(model)
        recalls = evaluate_model(
            wrapper,
            test_docs,
            test_queries,
            k_values=config.recall_k_values,
        )
        specialist_recalls[domain] = recalls

    # Step 6: Evaluate monolith
    logger.info("Step 6: Evaluating monolithic model...")
    from sentence_transformers import SentenceTransformer

    monolith_model = SentenceTransformer(str(monolith_result.model_path))
    monolith_recalls = evaluate_model(
        monolith_model,
        test_docs,
        test_queries,
        k_values=config.recall_k_values,
    )

    # Step 7: Evaluate fused specialists
    logger.info("Step 7: Evaluating fused specialists...")
    # Build village from all specialists
    sef_list = []
    for domain, model_path in specialist_paths.items():
        calibration_texts = domain_data[domain][
            :1000
        ]  # Use first 1000 training sentences
        sef = load_specialist_model(
            model_path, domain=domain, calibration_texts=calibration_texts
        )
        sef_list.append(sef)

    # Compute Procrustes alignments
    logger.info("  Computing Procrustes alignments...")
    anchor_sentences = []
    for domain in config.domains:
        anchor_sentences.extend(
            domain_data[domain][:200]
        )  # 200 sentences per domain for alignment
    reference_sef_name = sef_list[0].name
    alignments = compute_alignments(
        sef_list=sef_list,
        anchor_sentences=anchor_sentences,
        reference_sef_name=reference_sef_name,
    )
    aligned_sefs = align_sef_list(sef_list, alignments)

    village = Village(aligned_sefs)

    # Evaluate fusion
    fused_recalls = evaluate_model(
        village,
        test_docs,
        test_queries,
        k_values=config.recall_k_values,
        is_village=True,
    )

    # Step 8: Collect compute metrics paths
    specialist_compute_paths = {}
    for domain in config.domains:
        path = specialist_dir / f"{domain.value}_compute_metrics.json"
        if path.exists():
            specialist_compute_paths[domain] = path

    # Step 9: Compile results
    results = ExperimentResults(
        config=config,
        specialist_paths=specialist_paths,
        monolith_path=monolith_result.model_path,
        test_set_path=test_set_path,
        specialist_compute_paths=specialist_compute_paths,
        monolith_compute_path=monolith_result.compute_metrics_path,
        specialist_recalls=specialist_recalls,
        monolith_recalls=monolith_recalls,
        fused_recalls=fused_recalls,
    )

    # Save results
    results_path = experiment_dir / "results.json"
    results.save(results_path)
    logger.info(f"Results saved to {results_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("MILESTONE 2.1 EXPERIMENT RESULTS")
    print("=" * 60)
    print(f"Domains: {[d.value for d in config.domains]}")
    print(f"Test set: {len(test_docs)} docs, {len(test_queries)} queries")
    print()

    print("Recall@k:")
    print(f"{'k':>4} {'Monolith':>10} {'Fused':>10} {'Improvement':>12}")
    for k in config.recall_k_values:
        mono = monolith_recalls.get(k, 0.0)
        fused = fused_recalls.get(k, 0.0)
        improvement = (fused - mono) / mono * 100 if mono > 0 else 0.0
        print(f"{k:4d} {mono:10.3f} {fused:10.3f} {improvement:11.1f}%")

    print()
    print("Individual specialist performance:")
    for domain, recalls in specialist_recalls.items():
        avg = statistics.mean(recalls.values())
        print(f"  {domain.value:10} Avg Recall: {avg:.3f}")

    print("\n" + "=" * 60)

    return results


def main() -> None:
    """Command-line entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Run Milestone 2.1 experiment: Specialists vs Monolith"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override output directory",
    )
    parser.add_argument(
        "--real-data",
        action="store_true",
        help="Use real datasets (requires internet)",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing models (not implemented)",
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
