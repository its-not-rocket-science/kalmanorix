#!/usr/bin/env python3
"""Benchmark Kalman fusion vs simple averaging with statistical significance."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
from scipy import stats


def bootstrap_test(
    kalman_scores: List[float],
    mean_scores: List[float],
    n_bootstrap: int = 10000,
    alpha: float = 0.05,
) -> dict:
    """Paired bootstrap test for significance."""
    kalman = np.array(kalman_scores, dtype=np.float64)
    mean = np.array(mean_scores, dtype=np.float64)
    if kalman.shape != mean.shape:
        raise ValueError("kalman_scores and mean_scores must have the same length")
    if kalman.size == 0:
        raise ValueError("kalman_scores and mean_scores must not be empty")

    observed_diff = np.mean(kalman - mean)

    bootstrap_diffs = []
    n = len(kalman)
    for _ in range(n_bootstrap):
        idx = np.random.choice(n, n, replace=True)
        bootstrap_diffs.append(np.mean(kalman[idx] - mean[idx]))

    p_value = np.mean(np.array(bootstrap_diffs) <= 0)
    ci_lower = np.percentile(bootstrap_diffs, 100 * alpha / 2)
    ci_upper = np.percentile(bootstrap_diffs, 100 * (1 - alpha / 2))

    return {
        "observed_difference": float(observed_diff),
        "p_value": float(p_value),
        "confidence_interval_95": [float(ci_lower), float(ci_upper)],
        "kalman_wins": bool(p_value < alpha and observed_diff > 0),
        "mean_wins": bool(p_value < alpha and observed_diff < 0),
        "tie": bool(p_value >= alpha),
    }


def _recall_at_k(ranked_ids: List[int], target_id: int, k: int) -> float:
    return float(target_id in ranked_ids[:k])


def _mrr(ranked_ids: List[int], target_id: int) -> float:
    for idx, doc_id in enumerate(ranked_ids, start=1):
        if doc_id == target_id:
            return 1.0 / idx
    return 0.0


def _cosine_scores(query_emb: np.ndarray, doc_embs: np.ndarray) -> np.ndarray:
    q = query_emb / (np.linalg.norm(query_emb) + 1e-12)
    d = doc_embs / (np.linalg.norm(doc_embs, axis=1, keepdims=True) + 1e-12)
    return d @ q


def _load_or_create_dataset(path: Path) -> Dict[str, object]:
    if path.exists():
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)

    rng = np.random.default_rng(42)
    num_docs = 200
    num_queries = 80
    dim = 128

    docs = rng.normal(size=(num_docs, dim))
    queries = rng.normal(size=(num_queries, dim))
    targets = rng.integers(0, num_docs, size=num_queries)

    # Bias each query toward its relevant document to create learnable structure.
    queries = 0.7 * queries + 0.3 * docs[targets]

    dataset = {
        "doc_embeddings": docs.tolist(),
        "query_embeddings": queries.tolist(),
        "relevant_doc_ids": targets.tolist(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(dataset, handle)
    return dataset


def _fuse_kalman(query_emb: np.ndarray, noise_scale: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(abs(hash(query_emb.tobytes())) % (2**32))
    measurements = np.stack(
        [
            query_emb + rng.normal(0.0, noise_scale, size=query_emb.shape),
            query_emb + rng.normal(0.0, noise_scale * 1.15, size=query_emb.shape),
            query_emb + rng.normal(0.0, noise_scale * 1.3, size=query_emb.shape),
        ],
        axis=0,
    )
    variances = np.array([noise_scale**2, (noise_scale * 1.15) ** 2, (noise_scale * 1.3) ** 2])
    precisions = 1.0 / variances
    fused = np.average(measurements, axis=0, weights=precisions)
    return fused


def _fuse_mean(query_emb: np.ndarray, noise_scale: float = 0.03) -> np.ndarray:
    rng = np.random.default_rng(abs(hash((query_emb * 0.99).tobytes())) % (2**32))
    measurements = np.stack(
        [
            query_emb + rng.normal(0.0, noise_scale, size=query_emb.shape),
            query_emb + rng.normal(0.0, noise_scale * 1.15, size=query_emb.shape),
            query_emb + rng.normal(0.0, noise_scale * 1.3, size=query_emb.shape),
        ],
        axis=0,
    )
    return np.mean(measurements, axis=0)


def main() -> None:
    """Run benchmark and write CSV/JSON/plot artifacts."""
    root = Path(__file__).resolve().parents[1]
    results_dir = root / "results"
    dataset_path = results_dir / "milestone_1_3_mixed_domain_dataset.json"
    csv_path = results_dir / "milestone_1_3_kalman_vs_mean.csv"
    summary_path = results_dir / "milestone_1_3_summary.json"
    plot_path = results_dir / "kalman_vs_mean_plot.png"

    data = _load_or_create_dataset(dataset_path)
    doc_embs = np.asarray(data["doc_embeddings"], dtype=np.float64)
    query_embs = np.asarray(data["query_embeddings"], dtype=np.float64)
    relevant = np.asarray(data["relevant_doc_ids"], dtype=int)

    k_hit1: List[float] = []
    m_hit1: List[float] = []
    k_hit5: List[float] = []
    m_hit5: List[float] = []
    k_mrr: List[float] = []
    m_mrr: List[float] = []

    rows = []
    for idx, (query, target) in enumerate(zip(query_embs, relevant)):
        fused_kalman = _fuse_kalman(query)
        fused_mean = _fuse_mean(query)

        ranked_k = list(np.argsort(-_cosine_scores(fused_kalman, doc_embs)))
        ranked_m = list(np.argsort(-_cosine_scores(fused_mean, doc_embs)))

        kalman_hit1 = _recall_at_k(ranked_k, int(target), k=1)
        mean_hit1 = _recall_at_k(ranked_m, int(target), k=1)
        kalman_hit5 = _recall_at_k(ranked_k, int(target), k=5)
        mean_hit5 = _recall_at_k(ranked_m, int(target), k=5)
        kalman_mrr = _mrr(ranked_k, int(target))
        mean_mrr = _mrr(ranked_m, int(target))

        k_hit1.append(kalman_hit1)
        m_hit1.append(mean_hit1)
        k_hit5.append(kalman_hit5)
        m_hit5.append(mean_hit5)
        k_mrr.append(kalman_mrr)
        m_mrr.append(mean_mrr)

        rows.append(
            {
                "query_id": idx,
                "kalman_recall@1": kalman_hit1,
                "mean_recall@1": mean_hit1,
                "kalman_recall@5": kalman_hit5,
                "mean_recall@5": mean_hit5,
                "kalman_mrr": kalman_mrr,
                "mean_mrr": mean_mrr,
            }
        )

    results_dir.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    bootstrap_r1 = bootstrap_test(k_hit1, m_hit1)
    bootstrap_r5 = bootstrap_test(k_hit5, m_hit5)
    bootstrap_mrr = bootstrap_test(k_mrr, m_mrr)

    wilcoxon_mrr = stats.wilcoxon(np.array(k_mrr) - np.array(m_mrr), zero_method="wilcox")
    summary = {
        "overall_accuracy_comparison": {
            "kalman": {
                "recall@1": float(np.mean(k_hit1)),
                "recall@5": float(np.mean(k_hit5)),
                "mrr": float(np.mean(k_mrr)),
            },
            "mean": {
                "recall@1": float(np.mean(m_hit1)),
                "recall@5": float(np.mean(m_hit5)),
                "mrr": float(np.mean(m_mrr)),
            },
        },
        "p_values": {
            "recall@1": bootstrap_r1["p_value"],
            "recall@5": bootstrap_r5["p_value"],
            "mrr": bootstrap_mrr["p_value"],
            "wilcoxon_mrr": float(wilcoxon_mrr.pvalue),
        },
        "bootstrap": {
            "recall@1": bootstrap_r1,
            "recall@5": bootstrap_r5,
            "mrr": bootstrap_mrr,
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    metrics = ["Recall@1", "Recall@5", "MRR"]
    kalman_vals = [summary["overall_accuracy_comparison"]["kalman"]["recall@1"],
                   summary["overall_accuracy_comparison"]["kalman"]["recall@5"],
                   summary["overall_accuracy_comparison"]["kalman"]["mrr"]]
    mean_vals = [summary["overall_accuracy_comparison"]["mean"]["recall@1"],
                 summary["overall_accuracy_comparison"]["mean"]["recall@5"],
                 summary["overall_accuracy_comparison"]["mean"]["mrr"]]

    x = np.arange(len(metrics))
    width = 0.35
    plt.figure(figsize=(8, 4))
    plt.bar(x - width / 2, kalman_vals, width=width, label="Kalman")
    plt.bar(x + width / 2, mean_vals, width=width, label="Mean")
    plt.xticks(x, metrics)
    plt.ylim(0.0, 1.0)
    plt.ylabel("Score")
    plt.title("Kalman vs Mean Fusion")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plot_path, dpi=150)
    plt.close()


if __name__ == "__main__":
    main()
