"""Tests for mixed-domain benchmark build helpers."""

from kalmanorix.benchmarks.mixed_domain import SplitRatios, _deterministic_split


def test_deterministic_split_is_stable() -> None:
    queries = [
        {"query_id": f"nq:{idx}", "domain": "general_qa"}
        for idx in range(20)
    ] + [
        {"query_id": f"fiqa:{idx}", "domain": "finance"}
        for idx in range(20)
    ]

    split_a = _deterministic_split(queries, seed=123, ratios=SplitRatios())
    split_b = _deterministic_split(queries, seed=123, ratios=SplitRatios())

    assert split_a == split_b


def test_deterministic_split_covers_all_labels() -> None:
    queries = [
        {"query_id": f"scifact:{idx}", "domain": "biomedical"}
        for idx in range(30)
    ]

    split_map = _deterministic_split(queries, seed=42, ratios=SplitRatios())

    labels = set(split_map.values())
    assert labels == {"train", "validation", "test"}
    assert len(split_map) == len(queries)
