#!/usr/bin/env python3
"""
Diagnose centroid similarity issue for medical specialist.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sentence_transformers import SentenceTransformer
from src.kalmanorix.uncertainty import CentroidDistanceSigma2
from src.kalmanorix.toy_corpus import generate_anchor_sentences


def main():
    base_dir = Path("experiments/outputs/milestone_2_1/specialists/specialists")
    legal_path = base_dir / "legal-minilm"
    medical_path = base_dir / "medical-minilm"

    legal_model = SentenceTransformer(str(legal_path))
    medical_model = SentenceTransformer(str(medical_path))

    # Generate texts with correct domain mapping (as in experiment)
    legal_texts = generate_anchor_sentences(
        n=1000, domains=("legal",), seed=42 + hash("legal")
    )
    medical_texts = generate_anchor_sentences(
        n=1000, domains=("medical",), seed=42 + hash("medical")
    )

    print("=== Centroid analysis with correct domain mapping ===")

    # Create sigma2 estimators with same params as experiment (base_sigma2=0.5, scale=0.5)
    def legal_embed_fn(s):
        return legal_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    def medical_embed_fn(s):
        return medical_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

    legal_sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=legal_embed_fn,
        calibration_texts=legal_texts[:1000],
        base_sigma2=0.5,
        scale=0.5,
    )

    medical_sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=medical_embed_fn,
        calibration_texts=medical_texts[:1000],
        base_sigma2=0.5,
        scale=0.5,
    )

    print(f"Legal centroid norm: {np.linalg.norm(legal_sigma2.centroid):.4f}")
    print(f"Medical centroid norm: {np.linalg.norm(medical_sigma2.centroid):.4f}")
    print(
        f"Centroid cosine similarity: {legal_sigma2.centroid @ medical_sigma2.centroid:.4f}"
    )

    # Test similarities of calibration texts to own centroid
    print("\n=== Similarity of calibration texts to own centroid ===")

    # Sample 50 legal texts
    legal_sample = legal_texts[:50]
    legal_embs = legal_model.encode(
        legal_sample, normalize_embeddings=False, show_progress_bar=False
    )
    legal_sims = []
    for emb in legal_embs:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
        sim = emb_norm @ legal_sigma2.centroid
        legal_sims.append(sim)

    print(
        f"Legal texts to legal centroid: mean={np.mean(legal_sims):.4f}, min={np.min(legal_sims):.4f}, max={np.max(legal_sims):.4f}"
    )

    # Sample 50 medical texts
    medical_sample = medical_texts[:50]
    medical_embs = medical_model.encode(
        medical_sample, normalize_embeddings=False, show_progress_bar=False
    )
    medical_sims = []
    for emb in medical_embs:
        emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
        sim = emb_norm @ medical_sigma2.centroid
        medical_sims.append(sim)

    print(
        f"Medical texts to medical centroid: mean={np.mean(medical_sims):.4f}, min={np.min(medical_sims):.4f}, max={np.max(medical_sims):.4f}"
    )

    # Test on example queries from test set
    print("\n=== Example query similarities ===")

    # Load test set
    import json

    with open(
        Path("experiments/outputs/milestone_2_1/test_set.json"), "r", encoding="utf-8"
    ) as f:
        test_data = json.load(f)

    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    # Test first 5 queries
    for i, (query, _) in enumerate(test_queries[:5]):
        sigma2_l = legal_sigma2(query)
        sigma2_m = medical_sigma2(query)

        # Compute similarities manually
        emb_l = legal_embed_fn(query)
        emb_l_norm = emb_l / (np.linalg.norm(emb_l) + 1e-12)
        sim_l = emb_l_norm @ legal_sigma2.centroid

        emb_m = medical_embed_fn(query)
        emb_m_norm = emb_m / (np.linalg.norm(emb_m) + 1e-12)
        sim_m = emb_m_norm @ medical_sigma2.centroid

        print(f"\nQuery {i}: '{query[:60]}...'")
        print(f"  Legal: similarity={sim_l:.4f}, sigma²={sigma2_l:.4f}")
        print(f"  Medical: similarity={sim_m:.4f}, sigma²={sigma2_m:.4f}")

        # Check if medical similarity is negative
        if sim_m < -0.5:
            print("  WARNING: Medical similarity strongly negative!")

    # Check embedding norms
    print("\n=== Embedding norm analysis ===")
    legal_norms = []
    medical_norms = []

    for text in legal_texts[:100]:
        emb_l = legal_embed_fn(text)
        legal_norms.append(np.linalg.norm(emb_l))

        emb_m = medical_embed_fn(text)
        medical_norms.append(np.linalg.norm(emb_m))

    print(
        f"Legal embeddings: mean norm={np.mean(legal_norms):.4f}, std={np.std(legal_norms):.4f}"
    )
    print(
        f"Medical embeddings: mean norm={np.mean(medical_norms):.4f}, std={np.std(medical_norms):.4f}"
    )

    # Check if centroid computation method matters
    print("\n=== Centroid computation method comparison ===")

    # Method 1: Current method (mean of raw embeddings, then normalize)
    embs_raw = medical_model.encode(
        medical_texts[:100], normalize_embeddings=False, show_progress_bar=False
    )
    centroid_raw = np.mean(embs_raw, axis=0)
    centroid1 = centroid_raw / (np.linalg.norm(centroid_raw) + 1e-12)

    # Method 2: Mean of normalized embeddings
    embs_norm = []
    for emb in embs_raw:
        embs_norm.append(emb / (np.linalg.norm(emb) + 1e-12))
    centroid2 = np.mean(embs_norm, axis=0)
    centroid2 = centroid2 / (np.linalg.norm(centroid2) + 1e-12)

    print(f"Medical centroid method 1 (current): norm={np.linalg.norm(centroid1):.4f}")
    print(
        f"Medical centroid method 2 (mean of normalized): norm={np.linalg.norm(centroid2):.4f}"
    )
    print(f"Cosine similarity between methods: {centroid1 @ centroid2:.4f}")

    # Compare similarities with both centroids
    test_emb = medical_embed_fn("patient treatment hospital")
    test_emb_norm = test_emb / (np.linalg.norm(test_emb) + 1e-12)
    sim1 = test_emb_norm @ centroid1
    sim2 = test_emb_norm @ centroid2
    print(f"Test medical query similarity to centroid1: {sim1:.4f}")
    print(f"Test medical query similarity to centroid2: {sim2:.4f}")


if __name__ == "__main__":
    main()
