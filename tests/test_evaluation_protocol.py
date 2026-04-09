"""Tests for the locked pre-registered evaluation protocol."""

from __future__ import annotations

import pytest

from kalmanorix.benchmarks.evaluation_protocol import (
    PRIMARY_METRICS,
    SECONDARY_METRICS,
    QueryRanking,
    evaluate_locked_protocol,
)


def test_locked_protocol_metrics_and_aggregation() -> None:
    rankings = {
        "q1": QueryRanking(doc_ids=("d2", "d1", "d3"), scores=(0.9, 0.9, 0.1)),
        # q2 intentionally missing to verify empty-ranking behavior
    }
    qrels = {
        "q1": {"d1": 3.0, "d3": 1.0},
        "q2": {"d4": 2.0},
    }
    query_domains = {"q1": "finance", "q2": "biomedical"}
    latency_ms = {"q1": 10.0, "q2": 20.0}
    flops_proxy = {"q1": 1000.0}
    peak_memory_mb = {"q2": 50.0}
    specialist_count_selected = {"q1": 2.0, "q2": 1.0}

    report = evaluate_locked_protocol(
        rankings=rankings,
        qrels=qrels,
        query_domains=query_domains,
        latency_ms=latency_ms,
        flops_proxy=flops_proxy,
        peak_memory_mb=peak_memory_mb,
        specialist_count_selected=specialist_count_selected,
    )

    # q1 ties on score => lexical doc_id order: d1 before d2.
    # q1 recall@1 = 1/2, recall@5 = 1, mrr = 1.
    # q2 missing ranking => all primary metrics 0 for q2.
    assert report.num_queries == 2
    assert report.global_primary["recall@1"].mean == pytest.approx(0.25)
    assert report.global_primary["recall@5"].mean == pytest.approx(0.5)
    assert report.global_primary["recall@10"].mean == pytest.approx(0.5)
    assert report.global_primary["mrr"].mean == pytest.approx(0.5)
    assert report.global_secondary["latency_ms"].mean == pytest.approx(15.0)
    assert report.global_secondary["flops_proxy"].mean == pytest.approx(1000.0)
    assert report.global_secondary["peak_memory_mb"].mean == pytest.approx(50.0)
    assert report.global_secondary["specialist_count_selected"].mean == pytest.approx(1.5)
    assert set(report.global_primary) == set(PRIMARY_METRICS)
    assert set(report.global_secondary) == set(SECONDARY_METRICS)


def test_ndcg_known_toy_case() -> None:
    rankings = {"q1": QueryRanking(doc_ids=("d1", "d2", "d3"))}
    qrels = {"q1": {"d1": 3.0, "d2": 2.0}}
    query_domains = {"q1": "toy"}

    report = evaluate_locked_protocol(
        rankings=rankings,
        qrels=qrels,
        query_domains=query_domains,
    )
    assert report.global_primary["ndcg@10"].mean == pytest.approx(1.0)

    degraded = evaluate_locked_protocol(
        rankings={"q1": QueryRanking(doc_ids=("d3", "d2", "d1"))},
        qrels=qrels,
        query_domains=query_domains,
    )
    # Expected nDCG from DCG = 2/log2(3) + 7/log2(4) and ideal DCG = 7/log2(2) + 3/log2(3).
    assert degraded.global_primary["ndcg@10"].mean == pytest.approx(0.6064227, abs=1e-6)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="query_domains keys"):
        evaluate_locked_protocol(
            rankings={},
            qrels={"q1": {"d1": 1.0}},
            query_domains={},
        )

    with pytest.raises(ValueError, match="same length"):
        evaluate_locked_protocol(
            rankings={"q1": QueryRanking(doc_ids=("d1",), scores=(0.9, 0.8))},
            qrels={"q1": {"d1": 1.0}},
            query_domains={"q1": "general"},
        )
