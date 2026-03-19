#!/usr/bin/env python3
"""
Evaluate monolith model on test set.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
from sentence_transformers import SentenceTransformer


def main():
    experiment_dir = Path("experiments/outputs/milestone_2_1")

    # Load test set
    print("Loading test set...")
    with open(experiment_dir / "test_set.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    test_docs = test_data["documents"]
    test_queries = [(q["query"], q["true_doc_id"]) for q in test_data["queries"]]

    print(f"Loaded {len(test_docs)} documents, {len(test_queries)} queries")

    # Load monolith model
    monolith_path = (
        experiment_dir / "monolith" / "sentence-transformers-all-MiniLM-L6-v2"
    )
    print(f"Loading monolith from {monolith_path}...")
    monolith = SentenceTransformer(str(monolith_path))

    # Encode all documents
    print("Encoding documents...")
    doc_embeddings = monolith.encode(
        test_docs, convert_to_numpy=True, show_progress_bar=False
    )

    # Evaluate recall@k
    k_values = [1, 5, 10]
    recalls = {k: [] for k in k_values}

    print("Evaluating queries...")
    for i, (query, true_id) in enumerate(test_queries):
        q_emb = monolith.encode(query, convert_to_numpy=True, show_progress_bar=False)

        # Cosine similarity
        q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-12)
        doc_norms = doc_embeddings / (
            np.linalg.norm(doc_embeddings, axis=1, keepdims=True) + 1e-12
        )
        scores = doc_norms @ q_norm

        ranked = list(np.argsort(-scores))
        for k in k_values:
            success = true_id in ranked[:k]
            recalls[k].append(success)

        if (i + 1) % 20 == 0:
            print(f"  Processed {i + 1}/{len(test_queries)} queries")

    # Compute average recalls
    avg_recalls = {k: np.mean(recalls[k]) for k in k_values}

    print("\n=== Monolith Results ===")
    for k in k_values:
        print(f"Recall@{k}: {avg_recalls[k]:.3f}")

    return avg_recalls


if __name__ == "__main__":
    main()
