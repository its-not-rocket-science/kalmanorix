"""Retrieval-aware Kalman fusion with a small learned correction layer."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np


Array = np.ndarray


def _softmax(logits: Array, axis: int = -1) -> Array:
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / (np.sum(exp, axis=axis, keepdims=True) + 1e-12)


def _l2_normalize(x: Array) -> Array:
    if x.ndim == 1:
        return x / (np.linalg.norm(x) + 1e-12)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


@dataclass(frozen=True)
class LearnedCorrectionConfig:
    random_seed: int = 13
    dimension: int = 48
    n_specialists: int = 4
    n_train: int = 220
    n_validation: int = 120
    n_test: int = 220
    n_noise_docs: int = 60
    kalman_anchor: float = 0.35
    model_type: Literal["linear", "mlp"] = "linear"
    ridge_lambda: float = 1e-2
    mlp_hidden_dim: int = 8
    mlp_steps: int = 350
    mlp_lr: float = 4e-2
    mlp_l2: float = 5e-3


@dataclass(frozen=True)
class CorrectionCheckpoint:
    model_type: str
    feature_dim: int
    n_specialists: int
    kalman_anchor: float
    linear_weights: list[float] | None
    linear_bias: float | None
    mlp_w1: list[list[float]] | None
    mlp_b1: list[float] | None
    mlp_w2: list[float] | None
    mlp_b2: float | None


class KalmanLearnedCorrection:
    """Predict specialist correction logits on top of Kalman precision weights."""

    def __init__(self, cfg: LearnedCorrectionConfig) -> None:
        self.cfg = cfg
        self._feature_dim: int | None = None
        self._w_linear: Array | None = None
        self._b_linear: float = 0.0
        self._w1: Array | None = None
        self._b1: Array | None = None
        self._w2: Array | None = None
        self._b2: float = 0.0

    def fit(
        self, features: Array, kalman_weights: Array, oracle_weights: Array
    ) -> None:
        n_queries, n_specialists, feature_dim = features.shape
        self._feature_dim = feature_dim
        target = np.log(oracle_weights + 1e-12) - np.log(kalman_weights + 1e-12)
        target = target * (1.0 - self.cfg.kalman_anchor)

        x = features.reshape(n_queries * n_specialists, feature_dim)
        y = target.reshape(n_queries * n_specialists)

        if self.cfg.model_type == "linear":
            ridge = self.cfg.ridge_lambda * np.eye(feature_dim, dtype=np.float64)
            self._w_linear = np.linalg.solve(x.T @ x + ridge, x.T @ y)
            self._b_linear = float(np.mean(y - x @ self._w_linear))
            return

        rng = np.random.default_rng(self.cfg.random_seed)
        hidden = self.cfg.mlp_hidden_dim
        w1 = rng.normal(scale=0.05, size=(feature_dim, hidden))
        b1 = np.zeros(hidden, dtype=np.float64)
        w2 = rng.normal(scale=0.05, size=(hidden,))
        b2 = 0.0

        m = x.shape[0]
        for _ in range(self.cfg.mlp_steps):
            h_pre = x @ w1 + b1
            h = np.tanh(h_pre)
            pred = h @ w2 + b2
            err = pred - y

            grad_pred = 2.0 * err / m
            grad_w2 = h.T @ grad_pred + self.cfg.mlp_l2 * w2
            grad_b2 = float(np.sum(grad_pred))

            grad_h = grad_pred[:, None] * w2[None, :]
            grad_hpre = grad_h * (1.0 - h * h)
            grad_w1 = x.T @ grad_hpre + self.cfg.mlp_l2 * w1
            grad_b1 = np.sum(grad_hpre, axis=0)

            w1 -= self.cfg.mlp_lr * grad_w1
            b1 -= self.cfg.mlp_lr * grad_b1
            w2 -= self.cfg.mlp_lr * grad_w2
            b2 -= self.cfg.mlp_lr * grad_b2

        self._w1 = w1
        self._b1 = b1
        self._w2 = w2
        self._b2 = float(b2)

    def predict_weights(self, features: Array, kalman_weights: Array) -> Array:
        if self._feature_dim is None:
            raise ValueError("Model must be fit before prediction")
        if features.shape[-1] != self._feature_dim:
            raise ValueError("Feature dimension mismatch")

        if self.cfg.model_type == "linear":
            if self._w_linear is None:
                raise ValueError("Linear model not initialized")
            correction = features @ self._w_linear + self._b_linear
        else:
            if self._w1 is None or self._b1 is None or self._w2 is None:
                raise ValueError("MLP model not initialized")
            h = np.tanh(features @ self._w1 + self._b1)
            correction = h @ self._w2 + self._b2

        logits = np.log(kalman_weights + 1e-12) + correction
        return _softmax(logits, axis=1)

    def to_checkpoint(self) -> CorrectionCheckpoint:
        return CorrectionCheckpoint(
            model_type=self.cfg.model_type,
            feature_dim=int(self._feature_dim or 0),
            n_specialists=self.cfg.n_specialists,
            kalman_anchor=self.cfg.kalman_anchor,
            linear_weights=(
                None if self._w_linear is None else self._w_linear.tolist()
            ),
            linear_bias=(None if self._w_linear is None else self._b_linear),
            mlp_w1=(None if self._w1 is None else self._w1.tolist()),
            mlp_b1=(None if self._b1 is None else self._b1.tolist()),
            mlp_w2=(None if self._w2 is None else self._w2.tolist()),
            mlp_b2=(None if self._w2 is None else self._b2),
        )

    @classmethod
    def from_checkpoint(
        cls, checkpoint: CorrectionCheckpoint
    ) -> "KalmanLearnedCorrection":
        cfg = LearnedCorrectionConfig(
            model_type=checkpoint.model_type,  # type: ignore[arg-type]
            n_specialists=checkpoint.n_specialists,
            kalman_anchor=checkpoint.kalman_anchor,
        )
        model = cls(cfg)
        model._feature_dim = checkpoint.feature_dim
        if checkpoint.linear_weights is not None:
            model._w_linear = np.asarray(checkpoint.linear_weights, dtype=np.float64)
            model._b_linear = float(checkpoint.linear_bias or 0.0)
        if checkpoint.mlp_w1 is not None:
            model._w1 = np.asarray(checkpoint.mlp_w1, dtype=np.float64)
            model._b1 = np.asarray(checkpoint.mlp_b1, dtype=np.float64)
            model._w2 = np.asarray(checkpoint.mlp_w2, dtype=np.float64)
            model._b2 = float(checkpoint.mlp_b2 or 0.0)
        return model


def _precision_weights(sigma2: Array) -> Array:
    precision = 1.0 / np.maximum(sigma2, 1e-8)
    return precision / (np.sum(precision, axis=1, keepdims=True) + 1e-12)


def _build_features(
    specialist_embeddings: Array,
    sigma2: Array,
    router_scores: Array,
    query_lengths: Array,
) -> Array:
    n, k, _ = specialist_embeddings.shape
    normalized_router = _softmax(router_scores, axis=1)
    router_entropy = -np.sum(
        normalized_router * np.log(normalized_router + 1e-12), axis=1
    ) / np.log(k + 1e-12)
    router_dispersion = np.std(router_scores, axis=1)
    sigma_dispersion = np.std(sigma2, axis=1)

    cosine = np.einsum(
        "nkd,njd->nkj",
        _l2_normalize(specialist_embeddings.reshape(n * k, -1)).reshape(n, k, -1),
        _l2_normalize(specialist_embeddings.reshape(n * k, -1)).reshape(n, k, -1),
    )
    agreement = (np.sum(cosine, axis=2) - 1.0) / np.maximum(k - 1, 1)

    feats = []
    for idx in range(k):
        per = np.stack(
            [
                sigma2[:, idx],
                1.0 / np.maximum(sigma2[:, idx], 1e-8),
                router_scores[:, idx],
                normalized_router[:, idx],
                agreement[:, idx],
                query_lengths,
                router_entropy,
                router_dispersion,
                sigma_dispersion,
            ],
            axis=1,
        )
        feats.append(per)
    features = np.stack(feats, axis=1)

    mu = np.mean(features, axis=(0, 1), keepdims=True)
    sd = np.std(features, axis=(0, 1), keepdims=True) + 1e-6
    return (features - mu) / sd


def _oracle_weights(specialist_embeddings: Array, true_docs: Array) -> Array:
    sims = np.einsum(
        "nkd,nd->nk",
        _l2_normalize(
            specialist_embeddings.reshape(-1, specialist_embeddings.shape[-1])
        ).reshape(specialist_embeddings.shape),
        _l2_normalize(true_docs),
    )
    return _softmax(6.0 * sims, axis=1)


def _fit_global_linear_combiner(
    specialists: Array, true_docs: Array, split_labels: list[str], ridge_lambda: float
) -> Array:
    train_idx = [i for i, s in enumerate(split_labels) if s == "train"]
    x = specialists[train_idx]
    y = true_docs[train_idx]
    n_train, k, d = x.shape
    design = x.transpose(0, 2, 1).reshape(n_train * d, k)
    target = y.reshape(n_train * d)
    ridge = ridge_lambda * np.eye(k, dtype=np.float64)
    w = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    w = np.clip(w, 0.0, None)
    return w / (np.sum(w) + 1e-12)


def _recall_mrr(
    fused_queries: Array, docs: Array, relevant_doc_ids: Array
) -> dict[str, float]:
    q = _l2_normalize(fused_queries)
    d = _l2_normalize(docs)
    scores = q @ d.T
    ranked = np.argsort(-scores, axis=1)

    r1: list[float] = []
    r5: list[float] = []
    mrr: list[float] = []
    for i, tgt in enumerate(relevant_doc_ids):
        row = ranked[i]
        r1.append(float(tgt in row[:1]))
        r5.append(float(tgt in row[:5]))
        pos = int(np.where(row == tgt)[0][0])
        mrr.append(1.0 / (pos + 1))
    return {
        "recall@1": float(np.mean(r1)),
        "recall@5": float(np.mean(r5)),
        "mrr": float(np.mean(mrr)),
    }


def _synthesize_problem(cfg: LearnedCorrectionConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    n_total = cfg.n_train + cfg.n_validation + cfg.n_test
    d = cfg.dimension
    k = cfg.n_specialists

    latent_queries = _l2_normalize(rng.normal(size=(n_total, d)))
    true_docs = _l2_normalize(
        latent_queries + rng.normal(scale=0.18, size=(n_total, d))
    )
    specialists = []
    sigma2_cols = []
    router_cols = []
    for s_idx in range(k):
        noise_scale = 0.09 + 0.06 * s_idx
        bias = rng.normal(scale=0.03 * (s_idx + 1), size=(d,))
        sigma2 = np.clip(
            noise_scale**2
            + 0.04 * rng.random(n_total)
            + 0.02 * (s_idx + 1) * np.abs(rng.normal(size=n_total)),
            1e-4,
            None,
        )
        obs = _l2_normalize(
            latent_queries
            + bias[None, :]
            + rng.normal(scale=np.sqrt(sigma2)[:, None], size=(n_total, d))
        )
        router = (
            np.einsum("nd,nd->n", obs, latent_queries)
            - 0.45 * sigma2
            + rng.normal(scale=0.03, size=n_total)
        )
        specialists.append(obs)
        sigma2_cols.append(sigma2)
        router_cols.append(router)

    specialist_arr = np.stack(specialists, axis=1)
    sigma2_arr = np.stack(sigma2_cols, axis=1)
    router_arr = np.stack(router_cols, axis=1)

    hard_negative_idx = rng.integers(0, n_total, size=cfg.n_noise_docs)
    hard_negatives = _l2_normalize(
        true_docs[hard_negative_idx]
        + rng.normal(scale=0.22, size=(cfg.n_noise_docs, d))
    )
    docs = np.concatenate([true_docs, hard_negatives], axis=0)
    relevant_doc_ids = np.arange(n_total, dtype=int)

    query_lengths = rng.integers(low=3, high=24, size=n_total).astype(np.float64)
    query_lengths = query_lengths / np.max(query_lengths)

    split_labels = (
        ["train"] * cfg.n_train
        + ["validation"] * cfg.n_validation
        + ["test"] * cfg.n_test
    )

    return {
        "specialists": specialist_arr,
        "sigma2": sigma2_arr,
        "router_scores": router_arr,
        "docs": docs,
        "relevant_doc_ids": relevant_doc_ids,
        "true_docs": true_docs,
        "query_lengths": query_lengths,
        "split_labels": split_labels,
    }


def run_kalman_learned_correction(
    output_dir: Path = Path("results/kalman_learned_correction"),
    config: LearnedCorrectionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or LearnedCorrectionConfig()
    problem = _synthesize_problem(cfg)

    specialists = problem["specialists"]
    sigma2 = problem["sigma2"]
    router_scores = problem["router_scores"]
    docs = problem["docs"]
    relevant_doc_ids = problem["relevant_doc_ids"]
    true_docs = problem["true_docs"]
    query_lengths = problem["query_lengths"]
    split_labels = problem["split_labels"]

    kalman_weights = _precision_weights(sigma2)
    features = _build_features(specialists, sigma2, router_scores, query_lengths)
    oracle = _oracle_weights(specialists, true_docs)

    train_idx = [i for i, s in enumerate(split_labels) if s == "train"]
    test_idx = [i for i, s in enumerate(split_labels) if s == "test"]

    correction = KalmanLearnedCorrection(cfg)
    correction.fit(features[train_idx], kalman_weights[train_idx], oracle[train_idx])
    corrected_weights = correction.predict_weights(features, kalman_weights)

    learned_linear_weights = _fit_global_linear_combiner(
        specialists, true_docs, split_labels, ridge_lambda=cfg.ridge_lambda
    )

    baselines = {
        "mean": np.full_like(kalman_weights, 1.0 / cfg.n_specialists),
        "kalman_precision": kalman_weights,
        "learned_linear_combiner": np.repeat(
            learned_linear_weights[None, :], len(split_labels), axis=0
        ),
        "kalman_learned_correction": corrected_weights,
    }

    metrics: dict[str, dict[str, float]] = {}
    for name, weights in baselines.items():
        fused = np.einsum("nk,nkd->nd", weights[test_idx], specialists[test_idx])
        metrics[name] = _recall_mrr(fused, docs, relevant_doc_ids[test_idx])

    checkpoint = correction.to_checkpoint()
    summary = {
        "config": asdict(cfg),
        "fit": {
            "train_size": len(train_idx),
            "test_size": len(test_idx),
            "checkpoint": asdict(checkpoint),
        },
        "metrics": metrics,
        "question": "Can task-aware learned correction recover Kalman fusion quality without abandoning precision weighting?",
        "answer": _render_answer(metrics),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_answer(metrics: dict[str, dict[str, float]]) -> str:
    kalman = metrics["kalman_precision"]["mrr"]
    corrected = metrics["kalman_learned_correction"]["mrr"]
    if corrected > kalman:
        return "Yes in this benchmark: Kalman + learned correction improves MRR while retaining Kalman precision as the anchor."
    return "Null result in this benchmark: learned correction did not improve over plain Kalman precision weighting."


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Kalman + Learned Correction Benchmark",
        "",
        summary["question"],
        "",
        f"**Answer:** {summary['answer']}",
        "",
        "## Retrieval metrics (test split)",
        "",
        "| Method | Recall@1 | Recall@5 | MRR |",
        "| --- | ---: | ---: | ---: |",
    ]
    for name, m in summary["metrics"].items():
        lines.append(
            f"| {name} | {m['recall@1']:.4f} | {m['recall@5']:.4f} | {m['mrr']:.4f} |"
        )
    return "\n".join(lines) + "\n"
