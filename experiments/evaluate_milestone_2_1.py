#!/usr/bin/env python3
"""
Evaluation-only script for Milestone 2.1: Specialists vs Monolith test.

Loads pre-trained models from experiments/outputs/milestone_2_1/ and evaluates:
1. Individual specialists
2. Monolithic model
3. Fused specialists (Kalmanorix fusion)

Outputs results.json with Recall@k metrics and compute comparison.
"""

from __future__ import annotations

import json
import logging
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from experiments.config import DomainEnum, load_config
from src.kalmanorix.village import Village, SEF
from src.kalmanorix.scout import ScoutRouter
from src.kalmanorix.panoramix import Panoramix
from src.kalmanorix.uncertainty import CentroidDistanceSigma2
from src.kalmanorix.panoramix import KalmanorixFuser
from src.kalmanorix.alignment import compute_alignments, align_sef_list


logger = logging.getLogger(__name__)


@dataclass
class ExperimentResults:
    """Container for all experiment results."""

    # Paths
    specialist_paths: Dict[DomainEnum, Path]
    monolith_path: Path
    test_set_path: Path

    # Compute metrics paths
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
                d.value: {k: float(v) for k, v in recalls.items()}
                for d, recalls in self.specialist_recalls.items()
            },
            "monolith_recalls": {k: float(v) for k, v in self.monolith_recalls.items()},
            "fused_recalls": {k: float(v) for k, v in self.fused_recalls.items()},
            "p_values": {k: float(v) for k, v in self.p_values.items()}
            if self.p_values
            else None,
        }

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def load_specialist_model(
    model_path: Path,
    domain: Optional[DomainEnum] = None,
    config: Optional[Any] = None,
) -> SEF:
    """Load a trained specialist as SEF with uncertainty.

    If domain and config are provided, uses centroid-based uncertainty
    calibrated on synthetic texts from the same domain.
    Otherwise uses constant uncertainty (backward compatibility).
    """
    from sentence_transformers import SentenceTransformer

    # Load the SentenceTransformer model
    st_model = SentenceTransformer(str(model_path))

    # Embedding function for uncertainty estimation
    def embed_fn(s):
        return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    if domain is not None and config is not None:
        # Generate calibration texts matching the domain's training distribution
        from src.kalmanorix.toy_corpus import generate_anchor_sentences

        # Map domain enum to tag used in synthetic data generation
        # Same mapping as in run_milestone_2_1.py
        tag = {
            DomainEnum.MEDICAL: "medical",
            DomainEnum.LEGAL: "legal",
            DomainEnum.TECH: "tech",
            DomainEnum.COOK: "cook",
            DomainEnum.GENERAL: "general",
        }.get(domain, "general")

        # Use same seed as training: config.seed + hash(domain)
        # Generate smaller set for calibration (1000 texts)
        calibration_texts = generate_anchor_sentences(
            n=1000,
            domains=(tag,),
            seed=config.seed + hash(domain),
        )

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
            f"Using constant uncertainty for {model_path.name} (no domain/config provided)"
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
        # Monolithic model or specialist wrapper
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


def run_evaluation(
    experiment_dir: Path,
    recall_k_values: Optional[List[int]] = None,
) -> ExperimentResults:
    """
    Run evaluation on pre-trained models.

    Parameters
    ----------
    experiment_dir : Path
        Directory containing the experiment (should have specialists/, monolith/, test_set.json).
    recall_k_values : List[int], optional
        Recall@k values to compute. Defaults to [1, 5, 10].

    Returns
    -------
    ExperimentResults
        Complete evaluation results.
    """
    if recall_k_values is None:
        recall_k_values = [1, 5, 10]

    logger.info("Starting evaluation of Milestone 2.1 experiment")
    logger.info(f"Experiment directory: {experiment_dir}")

    # Load experiment configuration
    config_path = experiment_dir / "config.yaml"
    if config_path.exists():
        config = load_config(config_path)
        logger.info(
            f"Loaded configuration: seed={config.seed}, domains={[d.value for d in config.domains]}"
        )
    else:
        logger.warning(f"Configuration not found at {config_path}, using default")
        config = load_config(None)

    # Step 1: Load test set
    test_set_path = experiment_dir / "test_set.json"
    if not test_set_path.exists():
        raise FileNotFoundError(f"Test set not found at {test_set_path}")

    logger.info("Step 1: Loading test set...")
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_docs = test_data["documents"]
    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    logger.info(f"  Loaded {len(test_docs)} documents, {len(test_queries)} queries")

    # Step 2: Locate models
    specialist_dir = experiment_dir / "specialists" / "specialists"
    if not specialist_dir.exists():
        # Try alternative structure
        specialist_dir = experiment_dir / "specialists"

    # Find specialist models
    specialist_paths = {}
    specialist_compute_paths = {}

    for domain in [DomainEnum.LEGAL, DomainEnum.MEDICAL]:
        # Look for model directory
        domain_model_dir = None
        for candidate in specialist_dir.glob(f"*{domain.value}*"):
            if candidate.is_dir() and (candidate / "model.safetensors").exists():
                domain_model_dir = candidate
                break

        if domain_model_dir is None:
            logger.warning(f"Could not find model for domain {domain.value}")
            continue

        specialist_paths[domain] = domain_model_dir

        # Look for compute metrics
        compute_path = specialist_dir / f"{domain.value}_compute_metrics.json"
        if compute_path.exists():
            specialist_compute_paths[domain] = compute_path
        else:
            logger.warning(f"Could not find compute metrics for {domain.value}")

    # Locate monolith
    monolith_dir = experiment_dir / "monolith"
    monolith_path = None
    for candidate in monolith_dir.glob("*"):
        if candidate.is_dir() and (candidate / "model.safetensors").exists():
            monolith_path = candidate
            break

    if monolith_path is None:
        raise FileNotFoundError(f"Monolith model not found in {monolith_dir}")

    monolith_compute_path = monolith_dir / "compute_metrics.json"
    if not monolith_compute_path.exists():
        logger.warning(
            f"Could not find monolith compute metrics at {monolith_compute_path}"
        )

    logger.info(
        f"Found {len(specialist_paths)} specialists and monolith at {monolith_path}"
    )

    # Step 3: Evaluate specialists individually
    logger.info("Step 2: Evaluating specialists individually...")
    specialist_recalls = {}
    for domain, model_path in specialist_paths.items():
        logger.info(f"  Evaluating {domain.value}...")
        model = load_specialist_model(model_path, domain=domain, config=config)

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
            k_values=recall_k_values,
        )
        specialist_recalls[domain] = recalls

    # Step 4: Evaluate monolith
    logger.info("Step 3: Evaluating monolithic model...")
    from sentence_transformers import SentenceTransformer

    monolith_model = SentenceTransformer(str(monolith_path))
    monolith_recalls = evaluate_model(
        monolith_model,
        test_docs,
        test_queries,
        k_values=recall_k_values,
    )

    # Step 5: Evaluate fused specialists
    logger.info("Step 4: Evaluating fused specialists...")
    # Build village from all specialists with Procrustes alignment
    sef_list = []
    for domain, model_path in specialist_paths.items():
        sef = load_specialist_model(model_path, domain=domain, config=config)
        sef_list.append(sef)

    # Compute alignments using first 100 test documents as anchors
    logger.info("  Computing Procrustes alignments...")
    anchor_sentences = test_docs[:100]
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
        k_values=recall_k_values,
        is_village=True,
    )

    # Step 6: Compile results
    results = ExperimentResults(
        specialist_paths=specialist_paths,
        monolith_path=monolith_path,
        test_set_path=test_set_path,
        specialist_compute_paths=specialist_compute_paths,
        monolith_compute_path=monolith_compute_path,
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
    print("MILESTONE 2.1 EVALUATION RESULTS")
    print("=" * 60)
    print(f"Domains: {[d.value for d in specialist_paths.keys()]}")
    print(f"Test set: {len(test_docs)} docs, {len(test_queries)} queries")
    print()

    print("Recall@k:")
    print(f"{'k':>4} {'Monolith':>10} {'Fused':>10} {'Improvement':>12}")
    for k in recall_k_values:
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
        description="Evaluate Milestone 2.1 experiment (specialists vs monolith)"
    )
    parser.add_argument(
        "--experiment-dir",
        type=str,
        default="experiments/outputs/milestone_2_1",
        help="Path to experiment directory (default: experiments/outputs/milestone_2_1)",
    )
    parser.add_argument(
        "--k-values",
        type=int,
        nargs="+",
        default=[1, 5, 10],
        help="Recall@k values to compute (default: 1 5 10)",
    )

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    # Run evaluation
    results = run_evaluation(
        experiment_dir=Path(args.experiment_dir),
        recall_k_values=args.k_values,
    )

    print("\nEvaluation completed successfully!")
    print(f"Results saved to: {results.test_set_path.parent / 'results.json'}")


if __name__ == "__main__":
    main()
