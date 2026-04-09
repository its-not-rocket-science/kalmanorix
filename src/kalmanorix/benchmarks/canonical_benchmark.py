"""Canonical benchmark aggregation utilities for fusion method comparison."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from kalmanorix.benchmarks.statistical_testing import bootstrap_confidence_interval


@dataclass(frozen=True)
class QueryMetrics:
    """Per-query metrics used by the canonical benchmark."""

    ndcg10: float
    recall10: float
    mrr10: float


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(len(set(ranked[:k]).intersection(relevant)) / len(relevant))


def mrr_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return float(1.0 / idx)
    return 0.0


def ndcg_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(idx + 1)
    ideal_hits = min(k, len(relevant))
    idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, ideal_hits + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def evaluate_query(ranked: list[str], relevant: set[str]) -> QueryMetrics:
    return QueryMetrics(
        ndcg10=ndcg_at_k(ranked, relevant, 10),
        recall10=recall_at_k(ranked, relevant, 10),
        mrr10=mrr_at_k(ranked, relevant, 10),
    )


def aggregate_strategy_metrics(
    *,
    rankings: Mapping[str, list[str]],
    ground_truth: Mapping[str, set[str]],
    latency_ms: Mapping[str, float],
    flops_proxy: Mapping[str, float],
    seed: int,
    num_resamples: int,
) -> dict[str, Any]:
    """Aggregate query-level metrics and bootstrap CIs for one strategy."""
    ordered = sorted(rankings)
    ndcg10 = [evaluate_query(rankings[qid], ground_truth[qid]).ndcg10 for qid in ordered]
    recall10 = [evaluate_query(rankings[qid], ground_truth[qid]).recall10 for qid in ordered]
    mrr10 = [evaluate_query(rankings[qid], ground_truth[qid]).mrr10 for qid in ordered]
    latency = [float(latency_ms[qid]) for qid in ordered]
    flops = [float(flops_proxy[qid]) for qid in ordered]

    def _with_ci(values: list[float], metric_seed: int) -> dict[str, float]:
        ci = bootstrap_confidence_interval(
            values,
            np.zeros(len(values), dtype=float),
            num_resamples=num_resamples,
            seed=metric_seed,
        )
        return {
            "mean": float(np.mean(values)) if values else 0.0,
            "ci95_low": ci.lower,
            "ci95_high": ci.upper,
        }

    return {
        "num_queries": len(ordered),
        "metrics": {
            "ndcg@10": _with_ci(ndcg10, seed + 11),
            "recall@10": _with_ci(recall10, seed + 22),
            "mrr@10": _with_ci(mrr10, seed + 33),
            "latency_ms": _with_ci(latency, seed + 44),
            "flops_proxy": _with_ci(flops, seed + 55),
        },
        "query_level": {
            "ndcg@10": ndcg10,
            "recall@10": recall10,
            "mrr@10": mrr10,
        },
    }
