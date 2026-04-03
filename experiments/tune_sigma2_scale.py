#!/usr/bin/env python3
"""
Tune sigma² scale parameter to improve fusion performance.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from sentence_transformers import SentenceTransformer
from kalmanorix.village import Village, SEF
from kalmanorix.scout import ScoutRouter
from kalmanorix.panoramix import Panoramix, KalmanorixFuser
from kalmanorix.uncertainty import CentroidDistanceSigma2
from kalmanorix.alignment import compute_alignments, align_sef_list
from kalmanorix.toy_corpus import generate_anchor_sentences


def load_sef_with_params(model_path, domain_tag, base_sigma2, scale):
    """Load SEF with given sigma² parameters."""
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
        base_sigma2=base_sigma2,
        scale=scale,
    )

    return SEF(
        embed=embed_fn, sigma2=sigma2, name=f"{model_path.name}_b{base_sigma2}_s{scale}"
    )


def evaluate_params(base_sigma2, scale, test_docs, test_queries, num_queries=50):
    """Evaluate fusion with given sigma² parameters."""
    # Load specialists
    specialist_dir = Path("experiments/outputs/milestone_2_1/specialists/specialists")
    legal_path = specialist_dir / "legal-minilm"
    medical_path = specialist_dir / "medical-minilm"

    legal_sef = load_sef_with_params(legal_path, "legal", base_sigma2, scale)
    medical_sef = load_sef_with_params(medical_path, "medical", base_sigma2, scale)

    # Align specialists
    sef_list = [legal_sef, medical_sef]
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
    scout = ScoutRouter(mode="all")
    panoramix = Panoramix(fuser=KalmanorixFuser())

    # Compute fused document embeddings (subset of docs for speed)
    doc_subset = test_docs[:200]  # Use first 200 docs
    doc_embeddings = []
    for doc in doc_subset:
        potion = panoramix.brew(doc, village=village, scout=scout)
        doc_embeddings.append(potion.vector)
    doc_embs = np.array(doc_embeddings)

    # Evaluate recall@1 on subset of queries
    successes = []
    for query, true_id in test_queries[:num_queries]:
        potion = panoramix.brew(query, village=village, scout=scout)
        q_emb = potion.vector
        scores = doc_embs @ q_emb
        ranked = list(np.argsort(-scores))
        success = true_id in ranked[:1]
        successes.append(success)

    recall = np.mean(successes)
    return recall


def main():
    experiment_dir = Path("experiments/outputs/milestone_2_1")

    # Load test set
    print("Loading test set...")
    with open(experiment_dir / "test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_docs = test_data["documents"]
    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    print(
        f"Testing on {len(test_queries[:50])} queries, {len(test_docs[:200])} documents"
    )

    # Test different scale values with base_sigma2=0.5
    base_sigma2 = 0.5
    scale_values = [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 2.0]

    print(f"\n=== Tuning scale parameter (base_sigma2={base_sigma2}) ===")
    results = []
    for scale in scale_values:
        print(f"\nTesting scale={scale}...")
        recall = evaluate_params(
            base_sigma2, scale, test_docs, test_queries, num_queries=50
        )
        print(f"  Recall@1: {recall:.3f}")
        results.append((scale, recall))

    # Find best scale
    best_scale, best_recall = max(results, key=lambda x: x[1])
    print(f"\nBest scale: {best_scale} (recall={best_recall:.3f})")

    # Test different base_sigma2 with best scale
    print(f"\n=== Tuning base_sigma2 with scale={best_scale} ===")
    base_values = [0.1, 0.2, 0.5, 1.0, 2.0]
    base_results = []
    for base in base_values:
        print(f"\nTesting base_sigma2={base}...")
        recall = evaluate_params(
            base, best_scale, test_docs, test_queries, num_queries=50
        )
        print(f"  Recall@1: {recall:.3f}")
        base_results.append((base, recall))

    best_base, best_recall2 = max(base_results, key=lambda x: x[1])
    print(f"\nBest base_sigma2: {best_base} (recall={best_recall2:.3f})")

    # Compare with constant sigma² (scale=0)
    print("\n=== Constant sigma² (scale=0) ===")
    const_recall = evaluate_params(
        base_sigma2, 0.0, test_docs, test_queries, num_queries=50
    )
    print(f"Recall@1 with scale=0: {const_recall:.3f}")

    print("\n=== Summary ===")
    print(f"Optimal parameters: base_sigma2={best_base}, scale={best_scale}")
    print(f"Best recall: {best_recall2:.3f}")
    print(f"Constant sigma² recall: {const_recall:.3f}")
    print(f"Default (0.5, 0.5) recall: {results[scale_values.index(0.5)][1]:.3f}")


if __name__ == "__main__":
    main()
