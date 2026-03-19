#!/usr/bin/env python3
"""
Compare different fusion strategies on full test set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from src.kalmanorix.village import Village, SEF
from src.kalmanorix.scout import ScoutRouter
from src.kalmanorix.panoramix import (
    Panoramix,
    KalmanorixFuser,
    MeanFuser,
    DiagonalKalmanFuser,
)
from src.kalmanorix.uncertainty import CentroidDistanceSigma2
from src.kalmanorix.alignment import compute_alignments, align_sef_list
from src.kalmanorix.toy_corpus import generate_anchor_sentences


def load_sef(model_path, domain_tag):
    """Load SEF with current experiment parameters."""
    st_model = SentenceTransformer(str(model_path))

    def embed_fn(s):
        return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    calibration_texts = generate_anchor_sentences(
        n=1000,
        domains=(domain_tag,),
        seed=42 + hash(domain_tag),
    )

    sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=embed_fn,
        calibration_texts=calibration_texts,
        base_sigma2=0.5,
        scale=0.5,
    )

    return SEF(embed=embed_fn, sigma2=sigma2, name=model_path.name)


def load_constant_sef(model_path, domain_tag):
    """Load SEF with constant sigma²."""
    st_model = SentenceTransformer(str(model_path))

    def embed_fn(s):
        return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    dim = st_model.get_sentence_embedding_dimension()
    sigma2 = CentroidDistanceSigma2(
        embed=embed_fn,
        centroid=np.zeros(dim),
        base_sigma2=1.0,
    )
    return SEF(embed=embed_fn, sigma2=sigma2, name=model_path.name + "_const")


def evaluate_strategy(
    fuser_class, fuser_kwargs, village, test_docs, test_queries, k_values=[1, 5, 10]
):
    """Evaluate a fusion strategy on all queries."""
    scout = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=fuser_class(**fuser_kwargs))

    # Compute fused document embeddings
    print("  Computing document embeddings...")
    doc_embeddings = []
    for doc in test_docs:
        potion = panoramix.brew(doc, village=village, scout=scout)
        doc_embeddings.append(potion.vector)
    doc_embs = np.array(doc_embeddings)

    # Evaluate recall@k
    recalls = {k: [] for k in k_values}

    print("  Evaluating queries...")
    for i, (query, true_id) in enumerate(test_queries):
        potion = panoramix.brew(query, village=village, scout=scout)
        q_emb = potion.vector

        # Cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        doc_norms = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
        scores = doc_norms @ q_norm

        ranked = list(np.argsort(-scores))
        for k in k_values:
            success = true_id in ranked[:k]
            recalls[k].append(success)

        # Print progress every 20 queries
        if (i + 1) % 20 == 0:
            print(f"    Processed {i + 1}/{len(test_queries)} queries")

    # Compute average recalls
    avg_recalls = {k: np.mean(recalls[k]) for k in k_values}
    return avg_recalls


def main():
    experiment_dir = Path("experiments/outputs/milestone_2_1")

    # Load test set
    print("Loading test set...")
    with open(experiment_dir / "test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_docs = test_data["documents"]
    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    print(f"Loaded {len(test_docs)} documents, {len(test_queries)} queries")

    # Load models
    specialist_dir = experiment_dir / "specialists" / "specialists"
    legal_path = specialist_dir / "legal-minilm"
    medical_path = specialist_dir / "medical-minilm"

    legal_sef = load_sef(legal_path, "legal")
    medical_sef = load_sef(medical_path, "medical")

    # Align specialists (current method)
    print("Aligning specialists...")
    sef_list = [legal_sef, medical_sef]
    anchor_sentences = test_docs[:100]
    alignments = compute_alignments(
        sef_list=sef_list,
        anchor_sentences=anchor_sentences,
        reference_sef_name=legal_sef.name,
    )
    aligned_sefs = align_sef_list(sef_list, alignments)
    village = Village(aligned_sefs)

    print("\n=== Comparing fusion strategies (100 queries) ===")

    # 1. KalmanorixFuser (current)
    print("\n1. KalmanorixFuser (current):")
    recalls = evaluate_strategy(
        KalmanorixFuser,
        {"sort_by_certainty": True},
        village,
        test_docs,
        test_queries,
        k_values=[1, 5, 10],
    )
    for k in [1, 5, 10]:
        print(f"  Recall@{k}: {recalls[k]:.3f}")

    # 2. DiagonalKalmanFuser (scalar variance)
    print("\n2. DiagonalKalmanFuser (scalar variance):")
    recalls = evaluate_strategy(
        DiagonalKalmanFuser,
        {"prior_sigma2": 1.0, "sort_by_sigma2": True},
        village,
        test_docs,
        test_queries,
        k_values=[1, 5, 10],
    )
    for k in [1, 5, 10]:
        print(f"  Recall@{k}: {recalls[k]:.3f}")

    # 3. MeanFuser (equal weights)
    print("\n3. MeanFuser (equal weights):")
    recalls = evaluate_strategy(
        MeanFuser, {}, village, test_docs, test_queries, k_values=[1, 5, 10]
    )
    for k in [1, 5, 10]:
        print(f"  Recall@{k}: {recalls[k]:.3f}")

    # 4. KalmanorixFuser with constant sigma²
    print("\n4. KalmanorixFuser with constant sigma² (equal uncertainty):")
    legal_const = load_constant_sef(legal_path, "legal")
    medical_const = load_constant_sef(medical_path, "medical")
    sef_list_const = [legal_const, medical_const]
    alignments_const = compute_alignments(
        sef_list=sef_list_const,
        anchor_sentences=anchor_sentences,
        reference_sef_name=legal_const.name,
    )
    aligned_const = align_sef_list(sef_list_const, alignments_const)
    village_const = Village(aligned_const)

    recalls = evaluate_strategy(
        KalmanorixFuser,
        {"sort_by_certainty": True},
        village_const,
        test_docs,
        test_queries,
        k_values=[1, 5, 10],
    )
    for k in [1, 5, 10]:
        print(f"  Recall@{k}: {recalls[k]:.3f}")

    print("\n=== Summary ===")
    print("If MeanFuser (equal weights) performs better than KalmanorixFuser,")
    print("the centroid-based uncertainty weighting may be harmful.")
    print("If constant sigma² performs better, the scale parameter may need tuning.")


if __name__ == "__main__":
    main()
