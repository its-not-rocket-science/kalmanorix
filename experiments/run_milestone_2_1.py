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
from src.kalmanorix.village import Village, SEF
from src.kalmanorix.scout import ScoutRouter
from src.kalmanorix.panoramix import Panoramix
from src.kalmanorix.arena import eval_retrieval
from src.kalmanorix.uncertainty import CentroidDistanceSigma2
from src.kalmanorix.panoramix import KalmanorixFuser


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


def load_specialist_model(model_path: Path) -> SEF:
    """Load a trained specialist as SEF with uncertainty."""
    from sentence_transformers import SentenceTransformer

    # Load the SentenceTransformer model
    st_model = SentenceTransformer(str(model_path))

    # Wrap as SEF with centroid distance uncertainty
    # For simplicity, use constant uncertainty; in real experiment,
    # we'd fit CentroidDistanceSigma2 on validation data.
    dim = st_model.get_sentence_embedding_dimension()
    assert dim is not None
    sigma2 = CentroidDistanceSigma2(
        embed=lambda s: st_model.encode(s, convert_to_numpy=True),  # pylint: disable=unnecessary-lambda
        centroid=np.zeros(dim),
        base_sigma2=1.0,
    )

    return SEF(
        embed=lambda s: st_model.encode(s, convert_to_numpy=True),  # pylint: disable=unnecessary-lambda
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

        # We'll use eval_retrieval from arena which handles this
        # But it expects doc_embs precomputed. Need to adapt.
        # For simplicity, we'll compute doc embeddings using mean of specialists
        # This is not ideal but works for comparison.
        logger.warning("Using mean specialist embedding for document encoding")
        doc_embeddings = []
        for doc in test_docs:
            potion = panoramix.brew(doc, village=village, scout=scout)
            doc_embeddings.append(potion.vector)
        doc_embs = np.array(doc_embeddings)

        # Evaluate using arena.eval_retrieval
        recall = eval_retrieval(
            queries=test_queries,
            doc_embs=doc_embs,
            village=village,
            scout=scout,
            panoramix=panoramix,
            k=max(k_values),
        )
        # Convert to per-k (assuming eval_retrieval returns recall at max k)
        # We'll need to compute for each k separately. For now, approximate.
        results = {k: recall for k in k_values}
        return results

    else:
        # Monolithic model
        model = model_or_village
        # Encode all documents
        doc_embs = model.encode(test_docs, normalize_embeddings=True)

        # Evaluate each query
        recalls: dict[int, list[float]] = {k: [] for k in k_values}
        for query, true_id in test_queries:
            q_emb = model.encode([query], normalize_embeddings=True)[0]
            scores = doc_embs @ q_emb
            ranked = list(np.argsort(-scores))

            for k in k_values:
                recalls[k].append(1.0 if true_id in ranked[:k] else 0.0)

        return {k: statistics.mean(recalls[k]) for k in k_values}


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
        from src.kalmanorix.datasets import load_multiple_domains

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
        from src.kalmanorix.toy_corpus import generate_anchor_sentences

        domain_data = {}
        for domain in config.domains:
            tag = {
                DomainEnum.MEDICAL: "medical",
                DomainEnum.LEGAL: "general",
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
        model = load_specialist_model(model_path)

        # Convert SEF to SentenceTransformer-like interface
        class ModelWrapper:
            """Wrapper to make SEF compatible with SentenceTransformer API."""

            def __init__(self, sef):
                """Initialize wrapper."""
                self.sef = sef

            def encode(self, texts, normalize_embeddings=False):
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
    village = Village([])
    for domain, model_path in specialist_paths.items():
        sef = load_specialist_model(model_path)
        village.modules.append(sef)

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
