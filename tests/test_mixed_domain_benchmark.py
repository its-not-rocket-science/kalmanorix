"""Tests for mixed-domain benchmark build helpers."""

import pytest

from kalmanorix.benchmarks.mixed_domain import (
    HardQueryConfig,
    SplitRatios,
    _augment_hard_queries,
    _build_query_records,
    _load_beir_triplet,
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


def test_augment_hard_queries_adds_required_categories_and_metadata() -> None:
    query_rows = [
        {
            "query_id": "nq:q1",
            "query_text": "cpu thermal throttling prevention",
            "domain": "general_qa",
            "source_dataset": "beir_nq",
        },
        {
            "query_id": "fiqa:q1",
            "query_text": "equity duration convexity risk",
            "domain": "finance",
            "source_dataset": "beir_fiqa",
        },
    ]
    qrels_rows = [
        {
            "query_id": "nq:q1",
            "doc_id": "nq:d1",
            "relevance": 1,
            "source_dataset": "beir_nq",
        },
        {
            "query_id": "fiqa:q1",
            "doc_id": "fiqa:d1",
            "relevance": 1,
            "source_dataset": "beir_fiqa",
        },
    ]
    split_map = {"nq:q1": "test", "fiqa:q1": "test"}

    augmented_queries, augmented_qrels, augmented_split_map = _augment_hard_queries(
        query_rows=query_rows,
        qrels_rows=qrels_rows,
        split_map=split_map,
        seed=5,
        config=HardQueryConfig(enabled=True, per_category_per_domain=1),
    )

    assert len(augmented_queries) > len(query_rows)
    synthetic = [row for row in augmented_queries if row.get("is_synthetic")]
    categories = {row["query_category"] for row in synthetic}
    assert "ambiguous_cross_domain" in categories
    assert "misleading_lexical_overlap" in categories
    assert "mixed_intent" in categories
    assert "adversarial_near_miss" in categories
    assert all(
        row.get("provenance_note", "").startswith("synthetic") for row in synthetic
    )
    assert all(augmented_split_map[row["query_id"]] == "test" for row in synthetic)
    synthetic_qrels = {
        row["query_id"] for row in augmented_qrels if row["query_id"].startswith("syn:")
    }
    assert synthetic_qrels


def test_load_beir_triplet_supports_component_specific_qrels(monkeypatch) -> None:
    captured_calls = []

    def _fake_load_split(dataset_name: str, config: str | None, split_name: str):
        captured_calls.append((dataset_name, config, split_name))
        return (dataset_name, config, split_name)

    monkeypatch.setattr(
        "kalmanorix.benchmarks.mixed_domain._load_split",
        _fake_load_split,
    )

    spec = {
        "hf_name": "BeIR/nq",
        "qrels_config": None,
        "qrels_split": "test",
    }

    corpus, queries, qrels = _load_beir_triplet(spec)
    assert corpus == ("BeIR/nq", "corpus", "corpus")
    assert queries == ("BeIR/nq", "queries", "queries")
    assert qrels == ("BeIR/nq", None, "test")
    assert captured_calls == [
        ("BeIR/nq", "corpus", "corpus"),
        ("BeIR/nq", "queries", "queries"),
        ("BeIR/nq", None, "test"),
    ]


def test_load_beir_triplet_reports_failed_component(monkeypatch) -> None:
    def _fake_load_split(dataset_name: str, config: str | None, split_name: str):
        if split_name == "test":
            raise ValueError("BuilderConfig 'qrels' not found")
        return {"dataset": dataset_name, "config": config, "split": split_name}

    monkeypatch.setattr(
        "kalmanorix.benchmarks.mixed_domain._load_split",
        _fake_load_split,
    )

    with pytest.raises(RuntimeError, match="Failed loading qrels"):
        _load_beir_triplet("BeIR/nq")


@pytest.mark.integration
def test_load_beir_triplet_smoke_beir_nq() -> None:
    datasets_module = pytest.importorskip("datasets")
    if not hasattr(datasets_module, "load_dataset"):
        pytest.skip("huggingface datasets package is unavailable in this environment")

    corpus_ds, queries_ds, qrels_ds = _load_beir_triplet(
        {"hf_name": "BeIR/nq", "qrels_config": None, "qrels_split": "test"}
    )

    assert len(corpus_ds) > 0
    assert len(queries_ds) > 0
    assert len(qrels_ds) > 0
