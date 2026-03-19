#!/usr/bin/env python3
"""
Evaluate Milestone 2.1 experiment with per-domain breakdown.

Loads pre-trained models and test set, computes recall per query type:
- legal pure
- medical pure
- mixed

Query type is determined by:
1. If query contains domain keywords (simple heuristic)
2. Centroid similarity to each specialist (fallback)
"""

from __future__ import annotations

import json
import logging
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

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
from src.kalmanorix.toy_corpus import generate_anchor_sentences

logger = logging.getLogger(__name__)

# Domain keywords for classification (simple)
DOMAIN_KEYWORDS = {
    "legal": ["law", "legal", "court", "case", "judge", "attorney", "statute"],
    "medical": [
        "patient",
        "treatment",
        "medicine",
        "diagnosis",
        "clinical",
        "hospital",
        "doctor",
    ],
}


def classify_query(query: str, legal_sim: float, medical_sim: float) -> str:
    """Classify query as 'legal', 'medical', or 'mixed'."""
    query_lower = query.lower()
    legal_keywords = any(kw in query_lower for kw in DOMAIN_KEYWORDS["legal"])
    medical_keywords = any(kw in query_lower for kw in DOMAIN_KEYWORDS["medical"])

    if legal_keywords and not medical_keywords:
        return "legal"
    if medical_keywords and not legal_keywords:
        return "medical"
    if legal_keywords and medical_keywords:
        return "mixed"

    # Fallback to centroid similarity
    if legal_sim > medical_sim + 0.2:
        return "legal"
    elif medical_sim > legal_sim + 0.2:
        return "medical"
    else:
        return "mixed"


def load_specialist_model(
    model_path: Path,
    domain: Optional[DomainEnum] = None,
    config: Optional[Any] = None,
) -> SEF:
    """Load a trained specialist as SEF with uncertainty."""
    from sentence_transformers import SentenceTransformer

    st_model = SentenceTransformer(str(model_path))

    def embed_fn(s):
        return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    if domain is not None and config is not None:
        tag = {
            DomainEnum.MEDICAL: "medical",
            DomainEnum.LEGAL: "legal",
            DomainEnum.TECH: "tech",
            DomainEnum.COOK: "cook",
            DomainEnum.GENERAL: "general",
        }.get(domain, "general")

        calibration_texts = generate_anchor_sentences(
            n=1000,
            domains=(tag,),
            seed=config.seed + hash(domain),
        )

        sigma2 = CentroidDistanceSigma2.from_calibration(
            embed=embed_fn,
            calibration_texts=calibration_texts,
            base_sigma2=0.5,
            scale=0.5,
        )
        logger.info(
            f"  Calibrated uncertainty for {domain.value} using {len(calibration_texts)} texts"
        )
    else:
        dim = st_model.get_sentence_embedding_dimension()
        assert dim is not None
        sigma2 = CentroidDistanceSigma2(
            embed=embed_fn,
            centroid=np.zeros(dim),
            base_sigma2=0.5,
        )
        logger.warning(f"Using constant uncertainty for {model_path.name}")

    return SEF(
        embed=embed_fn,
        sigma2=sigma2,
        name=model_path.name,
    )


def evaluate_per_query(
    models: Dict[str, Any],
    test_docs: List[str],
    test_queries: List[Tuple[str, int]],
    k: int = 1,
) -> Dict[str, List[bool]]:
    """
    Evaluate each model on each query, returning list of success per query.

    models: dict mapping model_name to either SentenceTransformer or Village
    """
    results: Dict[str, List[bool]] = {name: [] for name in models}

    # Precompute document embeddings for each model
    doc_embs_dict: Dict[str, Optional[np.ndarray]] = {}
    for name, model in models.items():
        if isinstance(model, Village):
            # For village, we need to compute per-query fusion
            # We'll handle separately
            doc_embs_dict[name] = None
        else:
            # Monolith or specialist wrapper
            doc_embs = model.encode(
                test_docs, normalize_embeddings=True, show_progress_bar=False
            )
            doc_embs_dict[name] = doc_embs

    # Process each query
    for query, true_id in test_queries:
        for name, model in models.items():
            if isinstance(model, Village):
                # Fusion: compute fused embedding for query, compare to pre-fused doc embeddings?
                # We'll need to compute fused embedding for each document as well, which is expensive.
                # For now, skip per-query fusion evaluation (will handle separately)
                continue
            else:
                doc_embs = doc_embs_dict[name]
                q_emb = model.encode(
                    [query], normalize_embeddings=True, show_progress_bar=False
                )[0]
                scores = doc_embs @ q_emb
                ranked = list(np.argsort(-scores))
                success = true_id in ranked[:k]
                results[name].append(success)

    return results


def main():
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    experiment_dir = Path("experiments/outputs/milestone_2_1")
    config_path = experiment_dir / "config.yaml"
    config = load_config(config_path)

    # Load test set
    test_set_path = experiment_dir / "test_set.json"
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_docs = test_data["documents"]
    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    logger.info(f"Loaded {len(test_docs)} docs, {len(test_queries)} queries")

    # Load specialists
    specialist_dir = experiment_dir / "specialists" / "specialists"
    specialist_paths = {}
    for domain in [DomainEnum.LEGAL, DomainEnum.MEDICAL]:
        domain_model_dir = None
        for candidate in specialist_dir.glob(f"*{domain.value}*"):
            if candidate.is_dir() and (candidate / "model.safetensors").exists():
                domain_model_dir = candidate
                break
        if domain_model_dir is None:
            logger.error(f"Could not find model for {domain.value}")
            continue
        specialist_paths[domain] = domain_model_dir

    # Load specialists as SEF for centroid similarity
    sef_list = []
    for domain, model_path in specialist_paths.items():
        sef = load_specialist_model(model_path, domain=domain, config=config)
        sef_list.append(sef)

    # Load monolith
    monolith_dir = experiment_dir / "monolith"
    monolith_path = None
    for candidate in monolith_dir.glob("*"):
        if candidate.is_dir() and (candidate / "model.safetensors").exists():
            monolith_path = candidate
            break
    if monolith_path is None:
        raise FileNotFoundError(f"Monolith model not found in {monolith_dir}")

    from sentence_transformers import SentenceTransformer

    monolith_model = SentenceTransformer(str(monolith_path))

    # Classify queries
    logger.info("Classifying queries...")
    query_types = []
    for query, true_id in test_queries:
        # Compute centroid similarities
        legal_sim = None
        medical_sim = None
        for sef in sef_list:
            if "legal" in sef.name:
                emb = sef.embed(query)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
                centroid = sef.sigma2.centroid
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
                legal_sim = float(emb_norm @ centroid_norm)
            elif "medical" in sef.name:
                emb = sef.embed(query)
                emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
                centroid = sef.sigma2.centroid
                centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
                medical_sim = float(emb_norm @ centroid_norm)

        if legal_sim is None or medical_sim is None:
            qtype = "mixed"
        else:
            qtype = classify_query(query, legal_sim, medical_sim)
        query_types.append(qtype)

    type_counts = defaultdict(int)
    for t in query_types:
        type_counts[t] += 1
    logger.info(f"Query type distribution: {dict(type_counts)}")

    # Evaluate each model per query type
    # We'll compute recall@1 for each type
    k = 1
    # Prepare models
    models = {}
    # Legal specialist
    legal_sef = sef_list[0] if "legal" in sef_list[0].name else sef_list[1]

    class LegalWrapper:
        def __init__(self, sef):
            self.sef = sef

        def encode(self, texts, normalize_embeddings=False, **kwargs):
            embs = [self.sef.embed(t) for t in texts]
            embs = np.array(embs)
            if normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-12)
            return embs

    models["legal"] = LegalWrapper(legal_sef)

    # Medical specialist
    medical_sef = sef_list[1] if "medical" in sef_list[1].name else sef_list[0]

    class MedicalWrapper:
        def __init__(self, sef):
            self.sef = sef

        def encode(self, texts, normalize_embeddings=False, **kwargs):
            embs = [self.sef.embed(t) for t in texts]
            embs = np.array(embs)
            if normalize_embeddings:
                norms = np.linalg.norm(embs, axis=1, keepdims=True)
                embs = embs / (norms + 1e-12)
            return embs

    models["medical"] = MedicalWrapper(medical_sef)

    # Monolith
    models["monolith"] = monolith_model

    # Fusion (Village) with Procrustes alignment
    # Compute alignments using first 100 test documents as anchors
    anchor_sentences = test_docs[:100]
    reference_sef_name = sef_list[0].name
    alignments = compute_alignments(
        sef_list=sef_list,
        anchor_sentences=anchor_sentences,
        reference_sef_name=reference_sef_name,
    )
    aligned_sefs = align_sef_list(sef_list, alignments)
    village = Village(aligned_sefs)
    models["fusion"] = village

    # Precompute document embeddings for non-fusion models
    doc_embs = {}
    for name, model in models.items():
        if name == "fusion":
            continue
        doc_embs[name] = model.encode(
            test_docs, normalize_embeddings=True, show_progress_bar=False
        )

    # Evaluate per query
    results = {name: defaultdict(list) for name in models}

    for idx, (query, true_id) in enumerate(test_queries):
        qtype = query_types[idx]

        for name, model in models.items():
            if name == "fusion":
                # Skip fusion for now (complex)
                continue
            emb = doc_embs[name]
            q_emb = model.encode(
                [query], normalize_embeddings=True, show_progress_bar=False
            )[0]
            scores = emb @ q_emb
            ranked = list(np.argsort(-scores))
            success = true_id in ranked[:k]
            results[name][qtype].append(success)

    # Compute recall per type
    logger.info("\n" + "=" * 60)
    logger.info("RECALL@1 PER QUERY TYPE")
    logger.info("=" * 60)

    all_types = sorted(set(query_types))
    for name in ["legal", "medical", "monolith"]:
        logger.info(f"\n{name}:")
        for t in all_types:
            successes = results[name][t]
            if successes:
                recall = statistics.mean(successes)
                logger.info(f"  {t}: {recall:.3f} ({len(successes)} queries)")
            else:
                logger.info(f"  {t}: N/A")

    # Now evaluate fusion using Panoramix (slow)
    logger.info("\nEvaluating fusion...")
    scout = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=KalmanorixFuser())

    # Precompute fused embeddings for all documents? Too expensive.
    # Instead compute per query (still expensive but okay for analysis)
    # We'll sample a subset of queries per type
    sample_per_type = 50
    sampled = {t: [] for t in all_types}
    for idx, (query, true_id) in enumerate(test_queries):
        qtype = query_types[idx]
        if len(sampled[qtype]) < sample_per_type:
            sampled[qtype].append((idx, query, true_id))

    for t in all_types:
        logger.info(f"  Processing {t} queries...")
        for idx, query, true_id in sampled[t]:
            # Compute fused embedding for query
            _ = panoramix.brew(query, village=village, scout=scout)

            # Need to compute fused embeddings for all documents? Too heavy.
            # Instead compute similarity to each document using mean specialist embedding?
            # For now, skip.
            pass

    logger.info(
        "\nNote: Fusion evaluation incomplete - need efficient document encoding."
    )
    logger.info(
        "Consider using mean specialist embedding for document encoding as in evaluate_model."
    )

    # Save classification
    output = {
        "query_types": query_types,
        "type_counts": dict(type_counts),
    }
    with open(experiment_dir / "query_types.json", "w") as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nQuery types saved to {experiment_dir / 'query_types.json'}")


if __name__ == "__main__":
    main()
