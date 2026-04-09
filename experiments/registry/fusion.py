"""Unified fusion/retrieval strategy interface for benchmark registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
import hashlib
import time
from typing import Any

import numpy as np

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, Village


def _normalize(vec: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm <= 1e-12:
        return vec
    return vec / norm


def _resolve_embedding(module: Any, text: str) -> np.ndarray:
    vec = module.embed(text)
    if module.alignment_matrix is not None:
        vec = module.alignment_matrix @ vec
    return vec


def _fuse_and_rank(
    query_vecs: list[np.ndarray],
    doc_vecs: list[list[np.ndarray]],
    weights: np.ndarray,
    candidates: list[dict[str, Any]],
) -> list[str]:
    fused_query = np.sum([w * q for w, q in zip(weights, query_vecs, strict=True)], axis=0)
    q_norm = _normalize(fused_query)

    scored: list[tuple[str, float]] = []
    for idx, candidate in enumerate(candidates):
        fused_doc = np.sum([w * d[idx] for w, d in zip(weights, doc_vecs, strict=True)], axis=0)
        score = float(np.dot(_normalize(fused_doc), q_norm))
        scored.append((candidate["doc_id"], score))

    scored.sort(key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in scored]


def _deterministic_fold(query_id: str, train_fraction: float) -> str:
    digest = hashlib.sha256(query_id.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) / 0xFFFFFFFF
    return "train" if bucket < train_fraction else "eval"


def _reciprocal_rank(ranked_ids: list[str], relevant_ids: set[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_ids:
            return 1.0 / rank
    return 0.0


class RetrievalFusionStrategy(ABC):
    """Shared strategy interface for all fusion/retrieval baselines."""

    def __init__(self, name: str) -> None:
        self.name = name

    def fit(self, rows: list[dict[str, Any]], village: Village, options: dict[str, Any]) -> None:
        """Optional fitting step using the same benchmark rows."""

    @abstractmethod
    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        """Return per-module non-negative weights summing to 1."""


class UniformMeanStrategy(RetrievalFusionStrategy):
    def __init__(self) -> None:
        super().__init__(name="uniform_mean_fusion")

    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        _ = query_text
        n = len(modules)
        return np.full(n, 1.0 / n, dtype=np.float64)


class FixedWeightedMeanStrategy(RetrievalFusionStrategy):
    def __init__(self, weights: dict[str, float]) -> None:
        super().__init__(name="fixed_weighted_mean_fusion")
        self.weights = weights

    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        _ = query_text
        values = np.asarray([self.weights.get(module.name, 0.0) for module in modules], dtype=np.float64)
        values = np.clip(values, 0.0, None)
        total = float(np.sum(values))
        if total <= 0:
            return np.full(len(modules), 1.0 / len(modules), dtype=np.float64)
        return values / total


class SingleNamedModelStrategy(RetrievalFusionStrategy):
    def __init__(self, name: str, model_name: str) -> None:
        super().__init__(name=name)
        self.model_name = model_name

    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        _ = query_text
        names = [module.name for module in modules]
        if self.model_name not in names:
            raise ValueError(f"Model '{self.model_name}' unavailable. Available: {names}")
        weights = np.zeros(len(modules), dtype=np.float64)
        weights[names.index(self.model_name)] = 1.0
        return weights


class BestSingleSpecialistStrategy(SingleNamedModelStrategy):
    def __init__(self, train_fraction: float = 0.5) -> None:
        super().__init__(name="best_single_specialist", model_name="")
        self.train_fraction = train_fraction

    def fit(self, rows: list[dict[str, Any]], village: Village, options: dict[str, Any]) -> None:
        train_fraction = float(options.get("train_fraction", self.train_fraction))
        train_rows = [
            row for row in rows if _deterministic_fold(str(row["query_id"]), train_fraction) == "train"
        ]
        if not train_rows:
            train_rows = rows

        best_model_name = village.modules[0].name
        best_score = -1.0
        for module in village.modules:
            rr_values: list[float] = []
            for row in train_rows:
                qv = _normalize(_resolve_embedding(module, row["query_text"]))
                scores = []
                for cand in row["candidate_documents"]:
                    text = f"{cand.get('title', '')} {cand.get('text', '')}".strip()
                    dv = _normalize(_resolve_embedding(module, text))
                    scores.append((cand["doc_id"], float(np.dot(qv, dv))))
                scores.sort(key=lambda x: (-x[1], x[0]))
                ranked = [doc_id for doc_id, _ in scores]
                rr_values.append(_reciprocal_rank(ranked, set(row["ground_truth_relevant_ids"])))

            score = float(np.mean(rr_values)) if rr_values else 0.0
            if score > best_score:
                best_score = score
                best_model_name = module.name

        self.model_name = best_model_name


class RouterTop1Strategy(RetrievalFusionStrategy):
    def __init__(self) -> None:
        super().__init__(name="router_only_top1")

    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        scores = np.asarray([1.0 / module.sigma2_for(query_text) for module in modules], dtype=np.float64)
        weights = np.zeros(len(modules), dtype=np.float64)
        weights[int(np.argmax(scores))] = 1.0
        return weights


class RouterTopKMeanStrategy(RetrievalFusionStrategy):
    def __init__(self, top_k: int) -> None:
        if top_k <= 0:
            raise ValueError("top_k must be >= 1")
        super().__init__(name="router_only_topk_mean")
        self.top_k = top_k

    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        k = min(self.top_k, len(modules))
        scores = np.asarray([1.0 / module.sigma2_for(query_text) for module in modules], dtype=np.float64)
        idx = np.argsort(-scores)[:k]
        weights = np.zeros(len(modules), dtype=np.float64)
        weights[idx] = 1.0 / k
        return weights


class LearnedLinearCombinerStrategy(RetrievalFusionStrategy):
    def __init__(self, ridge_lambda: float = 1e-3, train_fraction: float = 0.5) -> None:
        super().__init__(name="learned_linear_combiner")
        self.ridge_lambda = ridge_lambda
        self.train_fraction = train_fraction
        self.weights = np.array([], dtype=np.float64)

    def fit(self, rows: list[dict[str, Any]], village: Village, options: dict[str, Any]) -> None:
        train_fraction = float(options.get("train_fraction", self.train_fraction))
        ridge_lambda = float(options.get("ridge_lambda", self.ridge_lambda))

        train_rows = [
            row for row in rows if _deterministic_fold(str(row["query_id"]), train_fraction) == "train"
        ]
        if not train_rows:
            train_rows = rows

        x_rows: list[list[float]] = []
        for row in train_rows:
            features: list[float] = []
            for module in village.modules:
                qv = _normalize(_resolve_embedding(module, row["query_text"]))
                rel_scores: list[float] = []
                for cand in row["candidate_documents"]:
                    if cand["doc_id"] not in row["ground_truth_relevant_ids"]:
                        continue
                    text = f"{cand.get('title', '')} {cand.get('text', '')}".strip()
                    dv = _normalize(_resolve_embedding(module, text))
                    rel_scores.append(float(np.dot(qv, dv)))
                features.append(float(np.mean(rel_scores)) if rel_scores else 0.0)
            x_rows.append(features)

        x = np.asarray(x_rows, dtype=np.float64)
        if x.size == 0:
            self.weights = np.full(len(village.modules), 1.0 / len(village.modules), dtype=np.float64)
            return
        y = np.ones(x.shape[0], dtype=np.float64)
        ridge = ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
        raw = np.linalg.solve(x.T @ x + ridge, x.T @ y)
        raw = np.clip(raw, 0.0, None)
        denom = float(np.sum(raw))
        if denom <= 0:
            self.weights = np.full(len(village.modules), 1.0 / len(village.modules), dtype=np.float64)
        else:
            self.weights = raw / denom

    def weights_for_query(self, query_text: str, modules: list[Any]) -> np.ndarray:
        _ = query_text
        if self.weights.size != len(modules):
            return np.full(len(modules), 1.0 / len(modules), dtype=np.float64)
        return self.weights


class GeneralistModelStrategy(SingleNamedModelStrategy):
    def __init__(self, model_name: str | None = None) -> None:
        super().__init__(name="single_generalist_model", model_name=model_name or "")

    def fit(self, rows: list[dict[str, Any]], village: Village, options: dict[str, Any]) -> None:
        _ = rows
        if self.model_name:
            return
        configured = options.get("generalist_model_name")
        if configured:
            self.model_name = str(configured)
            return
        for module in village.modules:
            lowered = module.name.lower()
            if "general" in lowered or "mpnet" in lowered:
                self.model_name = module.name
                return
        self.model_name = village.modules[0].name


@dataclass(frozen=True)
class PanoramixStrategyAdapter:
    name: str
    scout: ScoutRouter
    pan: Panoramix


def build_strategy(name: str, routing_mode: str) -> tuple[ScoutRouter, Panoramix]:
    """Backward-compatible Panoramix strategy constructor."""
    if name == "mean":
        fuser = MeanFuser()
    elif name == "kalman":
        fuser = KalmanorixFuser()
    else:
        raise ValueError(f"Unsupported fusion strategy: {name}")
    return ScoutRouter(mode=routing_mode), Panoramix(fuser=fuser)


def build_retrieval_baselines(options: dict[str, Any]) -> list[RetrievalFusionStrategy]:
    """Build required rigorous baseline suite with a unified interface."""
    fixed_weights = options.get("fixed_weights", {})
    top_k = int(options.get("router_top_k", 2))

    return [
        BestSingleSpecialistStrategy(train_fraction=float(options.get("train_fraction", 0.5))),
        GeneralistModelStrategy(model_name=options.get("generalist_model_name")),
        UniformMeanStrategy(),
        FixedWeightedMeanStrategy(weights={str(k): float(v) for k, v in fixed_weights.items()}),
        RouterTop1Strategy(),
        RouterTopKMeanStrategy(top_k=top_k),
        LearnedLinearCombinerStrategy(
            ridge_lambda=float(options.get("ridge_lambda", 1e-3)),
            train_fraction=float(options.get("train_fraction", 0.5)),
        ),
    ]


def rank_query_with_baseline(
    query_text: str,
    candidates: list[dict[str, Any]],
    village: Village,
    strategy: RetrievalFusionStrategy,
) -> tuple[list[str], float]:
    """Rank candidates with baseline strategy under shared retrieval pipeline."""
    start = time.perf_counter()
    modules = village.modules

    query_vecs = [_resolve_embedding(module, query_text) for module in modules]
    doc_vecs: list[list[np.ndarray]] = [[] for _ in modules]
    for candidate in candidates:
        doc_text = f"{candidate.get('title', '')} {candidate.get('text', '')}".strip()
        for idx, module in enumerate(modules):
            doc_vecs[idx].append(_resolve_embedding(module, doc_text))

    weights = strategy.weights_for_query(query_text=query_text, modules=modules)
    if len(weights) != len(modules):
        raise ValueError(f"Strategy {strategy.name} returned {len(weights)} weights for {len(modules)} modules")
    if np.any(weights < 0):
        raise ValueError(f"Strategy {strategy.name} returned negative weights")

    total = float(np.sum(weights))
    if total <= 0:
        weights = np.full(len(modules), 1.0 / len(modules), dtype=np.float64)
    else:
        weights = weights / total

    ranked = _fuse_and_rank(query_vecs=query_vecs, doc_vecs=doc_vecs, weights=weights, candidates=candidates)
    return ranked, (time.perf_counter() - start) * 1000.0


def rank_query(
    query_text: str,
    candidates: list[dict[str, Any]],
    village: Village,
    scout: ScoutRouter,
    pan: Panoramix,
) -> tuple[list[str], float]:
    """Rank candidate docs and return doc_ids + latency (ms)."""
    start = time.perf_counter()
    qv = pan.brew(query_text, village=village, scout=scout).vector
    scored = []
    for cand in candidates:
        doc_text = f"{cand.get('title', '')} {cand.get('text', '')}".strip()
        dv = pan.brew(doc_text, village=village, scout=scout).vector
        scored.append((cand["doc_id"], float(qv @ dv)))
    ranked = [doc_id for doc_id, _ in sorted(scored, key=lambda x: (-x[1], x[0]))]
    return ranked, (time.perf_counter() - start) * 1000.0
