"""Evaluation adapters for benchmark registry."""

from __future__ import annotations

from typing import Any

import numpy as np

from kalmanorix.benchmarks import QueryRanking, evaluate_locked_protocol


def evaluate_synthetic_recall(
    corpus: Any, village: Any, strategies: dict[str, tuple[Any, Any]]
) -> dict[str, float]:
    """Evaluate synthetic toy recall@1."""
    metrics: dict[str, float] = {}
    for name, (scout, pan) in strategies.items():
        doc_mat = np.stack(
            [pan.brew(d, village=village, scout=scout).vector for d in corpus.docs]
        )
        hits = []
        for q, true_id in corpus.queries:
            qv = pan.brew(q, village=village, scout=scout).vector
            pred = int(np.argmax(doc_mat @ qv))
            hits.append(float(pred == true_id))
        metrics[f"{name}_recall@1"] = float(np.mean(hits))
    return metrics


def evaluate_locked(
    rows: list[dict[str, Any]],
    rankings_by_strategy: dict[str, dict[str, QueryRanking]],
    latencies_by_strategy: dict[str, dict[str, float]],
    flops_proxy_by_strategy: dict[str, dict[str, float]] | None = None,
    peak_memory_mb_by_strategy: dict[str, dict[str, float]] | None = None,
    specialist_count_by_strategy: dict[str, dict[str, float]] | None = None,
) -> dict[str, Any]:
    """Evaluate ranking outputs with locked protocol."""
    qrels = {
        r["query_id"]: {doc_id: 1.0 for doc_id in r["ground_truth_relevant_ids"]}
        for r in rows
    }
    query_domains = {r["query_id"]: r["domain_label"] for r in rows}
    reports: dict[str, Any] = {}
    for strategy, rankings in rankings_by_strategy.items():
        report = evaluate_locked_protocol(
            rankings=rankings,
            qrels=qrels,
            query_domains=query_domains,
            latency_ms=latencies_by_strategy[strategy],
            flops_proxy=None
            if flops_proxy_by_strategy is None
            else flops_proxy_by_strategy.get(strategy),
            peak_memory_mb=(
                None
                if peak_memory_mb_by_strategy is None
                else peak_memory_mb_by_strategy.get(strategy)
            ),
            specialist_count_selected=(
                None
                if specialist_count_by_strategy is None
                else specialist_count_by_strategy.get(strategy)
            ),
        )
        reports[strategy] = {
            "protocol_version": report.protocol_version,
            "protocol_sha256": report.protocol_sha256,
            "num_queries": report.num_queries,
            "global_primary": {
                metric: {"mean": stats.mean, "median": stats.median}
                for metric, stats in report.global_primary.items()
            },
            "global_secondary": {
                metric: {"mean": stats.mean, "median": stats.median}
                for metric, stats in report.global_secondary.items()
            },
            "per_domain_primary": {
                domain: {
                    metric: {"mean": stats.mean, "median": stats.median}
                    for metric, stats in domain_metrics.items()
                }
                for domain, domain_metrics in report.per_domain_primary.items()
            },
        }
    return reports
