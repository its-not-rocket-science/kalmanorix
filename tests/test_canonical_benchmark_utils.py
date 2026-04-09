from __future__ import annotations

import numpy as np

from kalmanorix.benchmarks.canonical_benchmark import (
    aggregate_strategy_metrics,
    mrr_at_k,
)
from kalmanorix.benchmarks.statistical_testing import generate_statistical_report


def test_mrr_at_k_respects_cutoff() -> None:
    ranked = ["d2", "d1", "d3"]
    relevant = {"d1"}
    assert mrr_at_k(ranked, relevant, 1) == 0.0
    assert mrr_at_k(ranked, relevant, 10) == 0.5


def test_aggregate_strategy_metrics_returns_ci_and_query_level() -> None:
    rankings = {"q1": ["d1", "d2"], "q2": ["d2", "d3"], "q3": ["d3", "d1"]}
    ground_truth = {"q1": {"d1"}, "q2": {"d2"}, "q3": {"d3"}}
    latency = {"q1": 1.0, "q2": 2.0, "q3": 3.0}
    flops = {"q1": 2.0, "q2": 2.0, "q3": 1.0}

    summary = aggregate_strategy_metrics(
        rankings=rankings,
        ground_truth=ground_truth,
        latency_ms=latency,
        flops_proxy=flops,
        seed=123,
        num_resamples=500,
    )

    assert summary["num_queries"] == 3
    assert set(summary["metrics"]) == {"ndcg@10", "recall@10", "mrr@10", "latency_ms", "flops_proxy"}
    assert len(summary["query_level"]["ndcg@10"]) == 3
    assert summary["metrics"]["latency_ms"]["mean"] == np.mean([1.0, 2.0, 3.0])


def test_paired_statistical_report_for_kalman_vs_mean_metrics() -> None:
    kalman = {
        "ndcg@10": [0.9, 0.8, 0.7, 0.6],
        "recall@10": [1.0, 1.0, 1.0, 0.5],
        "mrr@10": [1.0, 1.0, 0.5, 0.5],
    }
    mean = {
        "ndcg@10": [0.7, 0.7, 0.6, 0.4],
        "recall@10": [1.0, 1.0, 0.5, 0.5],
        "mrr@10": [1.0, 0.5, 0.5, 0.25],
    }

    report = generate_statistical_report(
        reference_method="kalman",
        candidate_method="mean",
        reference_metrics=kalman,
        candidate_metrics=mean,
        num_resamples=500,
        seed=7,
    )

    assert set(report.comparisons) == {"ndcg@10", "recall@10", "mrr@10"}
    assert report.comparisons["ndcg@10"].mean_difference > 0.0
