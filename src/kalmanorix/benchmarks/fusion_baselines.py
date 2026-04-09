"""Configurable baseline suite for embedding fusion evaluation."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np


Array = np.ndarray


def _l2_normalize(vectors: Array) -> Array:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


@dataclass(frozen=True)
class EmbeddingDataset:
    """Precomputed embeddings for evaluating fusion strategies.

    Attributes:
        model_names: Ordered list of model names.
        query_embeddings: Per-model query embeddings with shape (n_queries, dim).
        doc_embeddings: Per-model document embeddings with shape (n_docs, dim).
        relevant_doc_ids: Relevant document indices per query.
        split_labels: Optional split label per query (e.g. train/test).
        router_scores: Optional router relevance scores per query per model.
    """

    model_names: list[str]
    query_embeddings: dict[str, Array]
    doc_embeddings: dict[str, Array]
    relevant_doc_ids: Array
    split_labels: list[str] | None = None
    router_scores: Array | None = None

    def __post_init__(self) -> None:
        if not self.model_names:
            raise ValueError("model_names must not be empty")
        missing_q = [
            name for name in self.model_names if name not in self.query_embeddings
        ]
        missing_d = [
            name for name in self.model_names if name not in self.doc_embeddings
        ]
        if missing_q or missing_d:
            raise ValueError(
                f"Missing embeddings for models: queries={missing_q}, docs={missing_d}"
            )

        n_queries = self.relevant_doc_ids.shape[0]
        for name in self.model_names:
            if self.query_embeddings[name].shape[0] != n_queries:
                raise ValueError(f"query_embeddings[{name}] query count mismatch")
            if (
                self.doc_embeddings[name].shape[1]
                != self.query_embeddings[name].shape[1]
            ):
                raise ValueError(f"embedding dimension mismatch for model {name}")

        if self.split_labels is not None and len(self.split_labels) != n_queries:
            raise ValueError("split_labels length must match relevant_doc_ids")

        if self.router_scores is not None:
            expected = (n_queries, len(self.model_names))
            if self.router_scores.shape != expected:
                raise ValueError(
                    f"router_scores shape must be {expected}, got {self.router_scores.shape}"
                )


class FusionStrategy(ABC):
    """Baseline strategy abstraction."""

    name: str

    def fit(self, dataset: EmbeddingDataset) -> None:
        """Optional fitting hook (default: no-op)."""

    @abstractmethod
    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        """Return per-model weights for one query."""


class MeanPoolingStrategy(FusionStrategy):
    """Uniform mean pooling across all models."""

    def __init__(self) -> None:
        self.name = "mean_pooling"

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        _ = query_idx
        return np.full(
            len(dataset.model_names), 1.0 / len(dataset.model_names), dtype=np.float64
        )


class FixedWeightedMeanStrategy(FusionStrategy):
    """Weighted mean using fixed, user-provided weights."""

    def __init__(self, weights: dict[str, float]) -> None:
        self.name = "weighted_mean"
        self._weights = weights

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        _ = query_idx
        values = np.array(
            [self._weights.get(name, 0.0) for name in dataset.model_names],
            dtype=np.float64,
        )
        denom = np.sum(values)
        if denom <= 0:
            raise ValueError("Fixed weights must sum to a positive value")
        return values / denom


class SingleGeneralPurposeStrategy(FusionStrategy):
    """Always uses one designated model."""

    def __init__(self, model_name: str) -> None:
        self.name = "single_general_purpose"
        self.model_name = model_name

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        _ = query_idx
        if self.model_name not in dataset.model_names:
            raise ValueError(f"Unknown model {self.model_name}")
        weights = np.zeros(len(dataset.model_names), dtype=np.float64)
        weights[dataset.model_names.index(self.model_name)] = 1.0
        return weights


class RouterOnlyStrategy(FusionStrategy):
    """No fusion: choose one specialist by router score."""

    def __init__(self) -> None:
        self.name = "router_only"

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        if dataset.router_scores is None:
            raise ValueError("router_scores are required for router-only baseline")
        top_idx = int(np.argmax(dataset.router_scores[query_idx]))
        weights = np.zeros(len(dataset.model_names), dtype=np.float64)
        weights[top_idx] = 1.0
        return weights


class TopKSpecialistMeanStrategy(FusionStrategy):
    """Semantic routing to top-k specialists, then uniform mean."""

    def __init__(self, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")
        self.name = "topk_specialist_mean"
        self.top_k = top_k

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        if dataset.router_scores is None:
            raise ValueError("router_scores are required for top-k specialist mean")
        n_models = len(dataset.model_names)
        k = min(self.top_k, n_models)
        ranked = np.argsort(-dataset.router_scores[query_idx])
        selected = ranked[:k]
        weights = np.zeros(n_models, dtype=np.float64)
        weights[selected] = 1.0 / k
        return weights


class OracleSingleBestStrategy(FusionStrategy):
    """Per-query oracle: choose single model with best rank for the ground truth."""

    def __init__(self) -> None:
        self.name = "oracle_single_best"

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        target_id = int(dataset.relevant_doc_ids[query_idx])
        best_model_idx = 0
        best_rank = np.inf

        for model_idx, model_name in enumerate(dataset.model_names):
            query = dataset.query_embeddings[model_name][query_idx]
            docs = dataset.doc_embeddings[model_name]
            scores = _l2_normalize(docs) @ (query / (np.linalg.norm(query) + 1e-12))
            rank = int(np.where(np.argsort(-scores) == target_id)[0][0])
            if rank < best_rank:
                best_rank = rank
                best_model_idx = model_idx

        weights = np.zeros(len(dataset.model_names), dtype=np.float64)
        weights[best_model_idx] = 1.0
        return weights


class LearnedLinearCombinationStrategy(FusionStrategy):
    """Learn non-negative linear weights from train split with ridge regression."""

    def __init__(self, ridge_lambda: float = 1e-3) -> None:
        self.name = "learned_linear_combination"
        self.ridge_lambda = ridge_lambda
        self._weights: Array | None = None

    def fit(self, dataset: EmbeddingDataset) -> None:
        if dataset.split_labels is None:
            raise ValueError("split_labels are required for learned linear combination")

        train_idx = [i for i, s in enumerate(dataset.split_labels) if s == "train"]
        if not train_idx:
            raise ValueError(
                "No train queries available for learned linear combination"
            )

        x_rows = []
        y_rows = []
        for query_idx in train_idx:
            target_id = int(dataset.relevant_doc_ids[query_idx])
            features = []
            for model_name in dataset.model_names:
                query = dataset.query_embeddings[model_name][query_idx]
                doc = dataset.doc_embeddings[model_name][target_id]
                q = query / (np.linalg.norm(query) + 1e-12)
                d = doc / (np.linalg.norm(doc) + 1e-12)
                features.append(float(np.dot(q, d)))
            x_rows.append(features)
            y_rows.append(1.0)

        x = np.asarray(x_rows, dtype=np.float64)
        y = np.asarray(y_rows, dtype=np.float64)
        n_features = x.shape[1]
        ridge = self.ridge_lambda * np.eye(n_features, dtype=np.float64)
        w = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        w = np.clip(w, 0.0, None)
        if np.sum(w) <= 0:
            w = np.full(n_features, 1.0 / n_features)
        else:
            w = w / np.sum(w)
        self._weights = w

    def select_weights(self, dataset: EmbeddingDataset, query_idx: int) -> Array:
        _ = query_idx
        if self._weights is None:
            raise ValueError("Learned strategy must be fit before evaluation")
        return self._weights


@dataclass(frozen=True)
class EvaluationResult:
    strategy_name: str
    recall_at_1: float
    recall_at_5: float
    mrr: float


def _fused_embeddings(
    dataset: EmbeddingDataset, weights: Array, query_idx: int
) -> tuple[Array, Array]:
    query_parts = []
    doc_parts = []
    for weight, model_name in zip(weights, dataset.model_names):
        query_parts.append(weight * dataset.query_embeddings[model_name][query_idx])
        doc_parts.append(weight * dataset.doc_embeddings[model_name])
    return np.sum(query_parts, axis=0), np.sum(doc_parts, axis=0)


def _evaluate_ranking(
    query_emb: Array, doc_embs: Array, target_id: int
) -> tuple[float, float, float]:
    scores = _l2_normalize(doc_embs) @ (query_emb / (np.linalg.norm(query_emb) + 1e-12))
    ranked = np.argsort(-scores)

    hit1 = float(target_id in ranked[:1])
    hit5 = float(target_id in ranked[:5])

    rank_idx = int(np.where(ranked == target_id)[0][0])
    mrr = 1.0 / (rank_idx + 1)
    return hit1, hit5, mrr


def evaluate_strategy(
    dataset: EmbeddingDataset, strategy: FusionStrategy
) -> EvaluationResult:
    """Evaluate one strategy with the shared retrieval pipeline."""
    strategy.fit(dataset)
    hit1_list = []
    hit5_list = []
    mrr_list = []

    for query_idx in range(dataset.relevant_doc_ids.shape[0]):
        weights = strategy.select_weights(dataset, query_idx)
        query_emb, doc_embs = _fused_embeddings(dataset, weights, query_idx)
        hit1, hit5, mrr = _evaluate_ranking(
            query_emb, doc_embs, int(dataset.relevant_doc_ids[query_idx])
        )
        hit1_list.append(hit1)
        hit5_list.append(hit5)
        mrr_list.append(mrr)

    return EvaluationResult(
        strategy_name=strategy.name,
        recall_at_1=float(np.mean(hit1_list)),
        recall_at_5=float(np.mean(hit5_list)),
        mrr=float(np.mean(mrr_list)),
    )


def run_experiment(config: dict[str, Any]) -> list[EvaluationResult]:
    """Run config-driven baseline experiments."""
    dataset = load_dataset_from_config(config)
    strategies = build_strategies(config)
    return [evaluate_strategy(dataset, strategy) for strategy in strategies]


def load_dataset_from_config(config: dict[str, Any]) -> EmbeddingDataset:
    dataset_cfg = config["dataset"]
    path = Path(dataset_cfg["path"])
    payload = json.loads(path.read_text(encoding="utf-8"))

    model_names = payload["model_names"]
    query_embeddings = {
        name: np.asarray(payload["query_embeddings"][name], dtype=np.float64)
        for name in model_names
    }
    doc_embeddings = {
        name: np.asarray(payload["doc_embeddings"][name], dtype=np.float64)
        for name in model_names
    }

    router_scores = payload.get("router_scores")
    return EmbeddingDataset(
        model_names=model_names,
        query_embeddings=query_embeddings,
        doc_embeddings=doc_embeddings,
        relevant_doc_ids=np.asarray(payload["relevant_doc_ids"], dtype=int),
        split_labels=payload.get("split_labels"),
        router_scores=(
            None
            if router_scores is None
            else np.asarray(router_scores, dtype=np.float64)
        ),
    )


def build_strategies(config: dict[str, Any]) -> list[FusionStrategy]:
    strategies = []
    for entry in config.get("strategies", []):
        strategy_type = entry["type"]
        if strategy_type == "oracle_single_best":
            strategies.append(OracleSingleBestStrategy())
        elif strategy_type == "single_general_purpose":
            strategies.append(
                SingleGeneralPurposeStrategy(model_name=entry["model_name"])
            )
        elif strategy_type == "mean_pooling":
            strategies.append(MeanPoolingStrategy())
        elif strategy_type == "weighted_mean":
            strategies.append(FixedWeightedMeanStrategy(weights=entry["weights"]))
        elif strategy_type == "router_only":
            strategies.append(RouterOnlyStrategy())
        elif strategy_type == "topk_specialist_mean":
            strategies.append(TopKSpecialistMeanStrategy(top_k=int(entry["top_k"])))
        elif strategy_type == "learned_linear_combination":
            strategies.append(
                LearnedLinearCombinationStrategy(
                    ridge_lambda=float(entry.get("ridge_lambda", 1e-3))
                )
            )
        else:
            raise ValueError(f"Unknown strategy type: {strategy_type}")
    return strategies
