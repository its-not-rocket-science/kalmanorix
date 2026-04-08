"""Tests for fusion baseline strategy suite."""

from __future__ import annotations

import numpy as np

from kalmanorix.benchmarks.fusion_baselines import (
    EmbeddingDataset,
    MeanPoolingStrategy,
    FixedWeightedMeanStrategy,
    RouterOnlyStrategy,
    TopKSpecialistMeanStrategy,
    evaluate_strategy,
)


def _build_dataset() -> EmbeddingDataset:
    model_names = ["m1", "m2", "m3"]

    # Three docs where each model is strongest on one query.
    query_embeddings = {
        "m1": np.array([[1.0, 0.0], [0.2, 0.8]], dtype=np.float64),
        "m2": np.array([[0.8, 0.2], [0.0, 1.0]], dtype=np.float64),
        "m3": np.array([[0.7, 0.3], [0.1, 0.9]], dtype=np.float64),
    }
    doc_embeddings = {
        "m1": np.array([[1.0, 0.0], [0.0, 1.0], [0.7, 0.7]], dtype=np.float64),
        "m2": np.array([[1.0, 0.0], [0.0, 1.0], [0.6, 0.8]], dtype=np.float64),
        "m3": np.array([[1.0, 0.0], [0.0, 1.0], [0.8, 0.6]], dtype=np.float64),
    }
    router_scores = np.array(
        [
            [0.9, 0.2, 0.1],
            [0.1, 0.8, 0.7],
        ],
        dtype=np.float64,
    )

    return EmbeddingDataset(
        model_names=model_names,
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        relevant_doc_ids=np.array([0, 1], dtype=int),
        split_labels=["train", "test"],
        router_scores=router_scores,
    )


def test_weighted_mean_uniform_matches_mean_pooling() -> None:
    dataset = _build_dataset()
    mean_result = evaluate_strategy(dataset, MeanPoolingStrategy())
    weighted_result = evaluate_strategy(
        dataset,
        FixedWeightedMeanStrategy(weights={"m1": 1.0, "m2": 1.0, "m3": 1.0}),
    )

    assert mean_result.recall_at_1 == weighted_result.recall_at_1
    assert mean_result.recall_at_5 == weighted_result.recall_at_5
    assert mean_result.mrr == weighted_result.mrr


def test_router_only_equivalent_to_topk_one() -> None:
    dataset = _build_dataset()
    router_result = evaluate_strategy(dataset, RouterOnlyStrategy())
    topk_result = evaluate_strategy(dataset, TopKSpecialistMeanStrategy(top_k=1))

    assert router_result.recall_at_1 == topk_result.recall_at_1
    assert router_result.recall_at_5 == topk_result.recall_at_5
    assert router_result.mrr == topk_result.mrr


def test_topk_all_equivalent_to_mean_with_uniform_scores() -> None:
    dataset = _build_dataset()
    uniform_router_scores = np.ones_like(dataset.router_scores)
    dataset_uniform = EmbeddingDataset(
        model_names=dataset.model_names,
        query_embeddings=dataset.query_embeddings,
        doc_embeddings=dataset.doc_embeddings,
        relevant_doc_ids=dataset.relevant_doc_ids,
        split_labels=dataset.split_labels,
        router_scores=uniform_router_scores,
    )

    mean_result = evaluate_strategy(dataset_uniform, MeanPoolingStrategy())
    topk_all_result = evaluate_strategy(
        dataset_uniform,
        TopKSpecialistMeanStrategy(top_k=len(dataset_uniform.model_names)),
    )

    assert mean_result.recall_at_1 == topk_all_result.recall_at_1
    assert mean_result.recall_at_5 == topk_all_result.recall_at_5
    assert mean_result.mrr == topk_all_result.mrr
