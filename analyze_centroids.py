#!/usr/bin/env python3
"""
Analyze centroid similarities for specialists.
"""

import json
import logging
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
from experiments.config import DomainEnum, load_config
from src.kalmanorix.toy_corpus import generate_anchor_sentences
from src.kalmanorix.uncertainty import CentroidDistanceSigma2

logging.basicConfig(level=logging.INFO, format="%(message)s")


def main():
    experiment_dir = Path("experiments/outputs/milestone_2_1")
    config_path = experiment_dir / "config.yaml"
    config = load_config(config_path)

    # Load test queries
    test_set_path = experiment_dir / "test_set.json"
    with open(test_set_path, "r", encoding="utf-8") as f:
        test_data = json.load(f)
    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    # Generate calibration texts for both domains
    from sentence_transformers import SentenceTransformer

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
            print(f"Could not find model for {domain.value}")
            continue
        specialist_paths[domain] = domain_model_dir

    # For each specialist, compute centroid from synthetic data
    for domain, model_path in specialist_paths.items():
        print(f"\n=== {domain.value} specialist ===")

        # Load model
        st_model = SentenceTransformer(str(model_path))

        def embed_fn(s):
            return st_model.encode(s, convert_to_numpy=True, show_progress_bar=False)

        # Generate synthetic calibration texts
        tag = "general" if domain == DomainEnum.LEGAL else "medical"
        calibration_texts = generate_anchor_sentences(
            n=1000,
            domains=(tag,),
            seed=config.seed + hash(domain),
        )

        # Compute centroid
        sigma2 = CentroidDistanceSigma2.from_calibration(
            embed=embed_fn,
            calibration_texts=calibration_texts,
            base_sigma2=0.5,
            scale=0.5,
        )

        centroid = sigma2.centroid
        print(f"Centroid norm: {np.linalg.norm(centroid):.4f}")

        # Compute similarities for sample queries
        sample_queries = [q[0] for q in test_queries[:5]]
        print("Sample queries from test set:")
        for i, query in enumerate(sample_queries):
            emb = embed_fn(query)
            emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            sim = float(emb_norm @ centroid_norm)
            sigma2_val = sigma2(query)
            print(f"  Query {i + 1}: similarity={sim:.4f}, sigma²={sigma2_val:.4f}")

        # Also compute similarity of calibration texts themselves
        cal_embeddings = [embed_fn(t) for t in calibration_texts[:10]]
        cal_sims = []
        for emb in cal_embeddings:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            sim = float(emb_norm @ centroid_norm)
            cal_sims.append(sim)
        print(
            f"Avg similarity of calibration texts to centroid: {np.mean(cal_sims):.4f}"
        )

        # Check if queries are from same distribution
        # Generate fresh synthetic texts from same domain
        fresh_texts = generate_anchor_sentences(
            n=10,
            domains=(tag,),
            seed=config.seed + hash(domain) + 999,
        )
        fresh_embs = [embed_fn(t) for t in fresh_texts]
        fresh_sims = []
        for emb in fresh_embs:
            emb_norm = emb / (np.linalg.norm(emb) + 1e-12)
            centroid_norm = centroid / (np.linalg.norm(centroid) + 1e-12)
            sim = float(emb_norm @ centroid_norm)
            fresh_sims.append(sim)
        print(f"Avg similarity of fresh synthetic texts: {np.mean(fresh_sims):.4f}")


if __name__ == "__main__":
    main()
