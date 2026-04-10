"""Tests for mixed-domain benchmark build helpers."""

import pytest

from kalmanorix.benchmarks.mixed_domain import (
    SplitRatios,
    _build_query_records,
    _deterministic_split,
    build_mixed_domain_benchmark,
)


def test_deterministic_split_is_stable() -> None:
    queries = [
        {"query_id": f"nq:{idx}", "domain": "general_qa"} for idx in range(20)
    ] + [{"query_id": f"fiqa:{idx}", "domain": "finance"} for idx in range(20)]

    split_a = _deterministic_split(queries, seed=123, ratios=SplitRatios())
    split_b = _deterministic_split(queries, seed=123, ratios=SplitRatios())

    assert split_a == split_b


def test_deterministic_split_covers_all_labels() -> None:
    queries = [
        {"query_id": f"scifact:{idx}", "domain": "biomedical"} for idx in range(30)
    ]

    split_map = _deterministic_split(queries, seed=42, ratios=SplitRatios())

    labels = set(split_map.values())
    assert labels == {"train", "validation", "test"}
    assert len(split_map) == len(queries)


def test_build_query_records_can_include_cross_domain_negatives() -> None:
    query_rows = [
        {
            "query_id": "nq:q1",
            "query_text": "what is cpu thermal throttling",
            "domain": "general_qa",
            "source_dataset": "beir_nq",
        }
    ]
    doc_map = {
        "nq:d_pos": {
            "doc_id": "nq:d_pos",
            "title": "cpu throttling",
            "text": "A CPU may reduce clock speed under thermal load.",
            "domain": "general_qa",
            "source_dataset": "beir_nq",
        },
        "nq:d_neg": {
            "doc_id": "nq:d_neg",
            "title": "history facts",
            "text": "General history document",
            "domain": "general_qa",
            "source_dataset": "beir_nq",
        },
        "fiqa:d_neg": {
            "doc_id": "fiqa:d_neg",
            "title": "equity swap",
            "text": "Finance document",
            "domain": "finance",
            "source_dataset": "beir_fiqa",
        },
    }
    qrels_rows = [{"query_id": "nq:q1", "doc_id": "nq:d_pos", "relevance": 1}]

    rows = _build_query_records(
        query_rows=query_rows,
        doc_map=doc_map,
        qrels_rows=qrels_rows,
        split_map={"nq:q1": "test"},
        max_candidates=3,
        cross_domain_negative_ratio=0.5,
        seed=13,
    )
    assert rows
    domains = {doc["domain"] for doc in rows[0]["candidate_documents"]}
    assert "finance" in domains
    assert rows[0]["contains_cross_domain_hard_negatives"] is True


def test_cross_domain_ratio_validation() -> None:
    with pytest.raises(ValueError, match="cross_domain_negative_ratio"):
        build_mixed_domain_benchmark(cross_domain_negative_ratio=1.1)
