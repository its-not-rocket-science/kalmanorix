"""Correlation-focused synthetic benchmark for partially correlated experts.

This benchmark is intentionally **narrow**: it probes whether correlation-aware Kalman
fusion helps when experts are partially correlated and uncertainty quality differs
between experts.

Methods compared on the same synthetic-to-real-bridge protocol:
- mean fusion
- baseline Kalman
- correlation-aware Kalman (covariance inflation)
- weighted mean (validation-tuned inverse-variance weights)
- learned linear combiner (fit on validation)

Guardrails:
- all outputs are explicitly labeled as synthetic
- synthetic gains are reported as exploratory and are not headline proof
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import time
from typing import Any

import numpy as np

from kalmanorix.benchmarks.statistical_testing import (
    bootstrap_confidence_interval,
    paired_effect_size,
    paired_significance_test,
)
from kalmanorix.kalman_engine.correlation import (
    correlation_inflation_factors,
    estimate_residual_correlation_profile,
)
from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal_ensemble


@dataclass(frozen=True)
class CorrelationAwareFusionConfig:
    random_seed: int = 13
    dimension: int = 32
    n_docs: int = 256
    n_val: int = 280
    n_test: int = 420
    n_specialists: int = 4
    inflation_alpha: float = 1.0
    variance_floor: float = 1e-6
    ridge_lambda: float = 1e-3
    correlated_query_threshold: float = 0.70


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    mrr_at_10: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _query_level_retrieval(
    queries: np.ndarray, docs: np.ndarray, targets: np.ndarray
) -> dict[str, np.ndarray]:
    scores = _l2_normalize(queries) @ _l2_normalize(docs).T
    ranked = np.argsort(-scores, axis=1)

    hit1 = (ranked[:, 0] == targets).astype(np.float64)
    hit5 = np.array(
        [1.0 if t in ranked[i, :5] else 0.0 for i, t in enumerate(targets)],
        dtype=np.float64,
    )

    rr = np.zeros(len(targets), dtype=np.float64)
    for i, t in enumerate(targets):
        top10 = ranked[i, :10]
        loc = np.where(top10 == t)[0]
        rr[i] = 0.0 if loc.size == 0 else 1.0 / float(loc[0] + 1)

    return {"recall_at_1": hit1, "recall_at_5": hit5, "mrr_at_10": rr}


def _aggregate_metrics(per_query: dict[str, np.ndarray]) -> RetrievalMetrics:
    return RetrievalMetrics(
        recall_at_1=float(np.mean(per_query["recall_at_1"])),
        recall_at_5=float(np.mean(per_query["recall_at_5"])),
        mrr_at_10=float(np.mean(per_query["mrr_at_10"])),
    )


def _sample_specialist_noise(
    *,
    true_queries: np.ndarray,
    scales: np.ndarray,
    rho: float,
    rng: np.random.Generator,
) -> list[np.ndarray]:
    """Sample correlated specialist noise with shared+idiosyncratic Gaussian factors."""
    n_q, d = true_queries.shape
    n_specialists = int(scales.shape[0])
    sqrt_shared = np.sqrt(np.clip(rho, 0.0, 1.0))
    sqrt_indep = np.sqrt(max(1.0 - rho, 0.0))

    shared = rng.normal(size=(n_q, d))
    observations: list[np.ndarray] = []
    for i in range(n_specialists):
        indep = rng.normal(size=(n_q, d))
        noise = scales[i] * (sqrt_shared * shared + sqrt_indep * indep)
        observations.append(_l2_normalize(true_queries + noise))
    return observations


def _synthesize_problem(cfg: CorrelationAwareFusionConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    docs = _l2_normalize(rng.normal(size=(cfg.n_docs, cfg.dimension)))

    val_true = _l2_normalize(rng.normal(size=(cfg.n_val, cfg.dimension)))
    test_true = _l2_normalize(rng.normal(size=(cfg.n_test, cfg.dimension)))

    val_targets = np.argmax(val_true @ docs.T, axis=1)
    test_targets = np.argmax(test_true @ docs.T, axis=1)

    # Non-identical uncertainty quality across experts.
    scales = np.linspace(0.09, 0.22, cfg.n_specialists)

    # Validation is mixed, test is strengthened toward higher correlation.
    val_obs = _sample_specialist_noise(
        true_queries=val_true,
        scales=scales,
        rho=0.45,
        rng=rng,
    )

    half = cfg.n_test // 2
    test_obs_low = _sample_specialist_noise(
        true_queries=test_true[:half],
        scales=scales,
        rho=0.20,
        rng=rng,
    )
    test_obs_high = _sample_specialist_noise(
        true_queries=test_true[half:],
        scales=scales,
        rho=0.85,
        rng=rng,
    )
    test_obs = [np.vstack([a, b]) for a, b in zip(test_obs_low, test_obs_high)]

    test_buckets = np.array(
        ["low_correlation"] * half + ["high_correlation"] * (cfg.n_test - half)
    )

    return {
        "docs": docs,
        "val_true": val_true,
        "test_true": test_true,
        "val_targets": val_targets,
        "test_targets": test_targets,
        "val_obs": val_obs,
        "test_obs": test_obs,
        "test_buckets": test_buckets,
        "model_names": [f"specialist_{i}" for i in range(cfg.n_specialists)],
    }


def _fit_on_validation(
    *,
    val_true: np.ndarray,
    val_obs: list[np.ndarray],
    model_names: list[str],
    variance_floor: float,
) -> tuple[list[float], np.ndarray, Any]:
    residuals = [obs - val_true for obs in val_obs]
    sigma2 = [float(max(np.mean(r**2), variance_floor)) for r in residuals]

    residual_norms = np.column_stack(
        [np.linalg.norm(r, axis=1) for r in residuals]
    ).astype(np.float64)
    profile = estimate_residual_correlation_profile(model_names, residual_norms)
    return sigma2, residual_norms, profile


def _fit_learned_linear_weights(
    *,
    val_obs: list[np.ndarray],
    val_true: np.ndarray,
    ridge_lambda: float,
) -> list[float]:
    x = np.column_stack([obs.reshape(-1) for obs in val_obs]).astype(np.float64)
    y = val_true.reshape(-1).astype(np.float64)
    ridge = ridge_lambda * np.eye(x.shape[1], dtype=np.float64)
    w = np.linalg.solve(x.T @ x + ridge, x.T @ y)
    w = np.clip(w, 0.0, None)
    if float(np.sum(w)) <= 0.0:
        w = np.full_like(w, 1.0 / len(w))
    else:
        w = w / np.sum(w)
    return w.tolist()


def _correlation_score_for_query(residual_vectors: np.ndarray) -> float:
    """Return common-mode residual score in [0, 1].

    residual_vectors shape: (n_specialists, dim)
    score = ||mean residual|| / mean(||residual_i||)
    High score implies experts are moving together (shared error mode).
    """
    norms = np.linalg.norm(residual_vectors, axis=1)
    denom = float(np.mean(norms) + 1e-12)
    return float(np.linalg.norm(np.mean(residual_vectors, axis=0)) / denom)


def _is_correlated_query(
    residual_vectors: np.ndarray, *, threshold: float
) -> tuple[bool, float]:
    score = _correlation_score_for_query(residual_vectors)
    return bool(score >= threshold), score


def _fuse_queries(
    *,
    method_key: str,
    observations: list[np.ndarray],
    sigma2: list[float],
    corr_submatrix: np.ndarray,
    inflation_alpha: float,
    weighted_mean_weights: list[float],
    learned_linear_weights: list[float],
) -> tuple[np.ndarray, dict[str, float]]:
    n_q = observations[0].shape[0]
    fused = []
    per_query_ms = []

    inflation = correlation_inflation_factors(corr_submatrix, alpha=inflation_alpha)

    for q_idx in range(n_q):
        start = time.perf_counter()
        embs = [obs[q_idx] for obs in observations]
        if method_key == "mean_fusion":
            fused_vec = np.mean(np.stack(embs, axis=0), axis=0)
        elif method_key == "baseline_kalman":
            covs = [np.full_like(embs[0], s, dtype=np.float64) for s in sigma2]
            fused_vec, _ = kalman_fuse_diagonal_ensemble(embs, covs)
        elif method_key == "corr_kalman_cov_inflation":
            covs = [
                np.full_like(embs[0], s * inflation[i], dtype=np.float64)
                for i, s in enumerate(sigma2)
            ]
            fused_vec, _ = kalman_fuse_diagonal_ensemble(embs, covs)
        elif method_key == "weighted_mean":
            w = np.asarray(weighted_mean_weights, dtype=np.float64)
            fused_vec = np.sum(np.stack(embs, axis=0) * w[:, None], axis=0)
        elif method_key == "learned_linear_combiner":
            w = np.asarray(learned_linear_weights, dtype=np.float64)
            fused_vec = np.sum(np.stack(embs, axis=0) * w[:, None], axis=0)
        else:
            raise ValueError(f"Unknown method: {method_key}")
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        per_query_ms.append(elapsed_ms)
        fused.append(fused_vec)

    arr = np.asarray(per_query_ms, dtype=np.float64)
    return np.asarray(fused, dtype=np.float64), {
        "mean_ms": float(np.mean(arr)),
        "p95_ms": float(np.percentile(arr, 95)),
    }


def _paired_stats(candidate: np.ndarray, baseline: np.ndarray) -> dict[str, float]:
    ci = bootstrap_confidence_interval(candidate, baseline, seed=17)
    sig = paired_significance_test(candidate, baseline)
    eff = paired_effect_size(candidate, baseline)
    return {
        "delta_mean": float(np.mean(candidate - baseline)),
        "bootstrap_ci95": [float(ci.lower), float(ci.upper)],
        "wilcoxon_p": float(sig.p_value),
        "cohen_dz": float(eff.cohen_dz),
        "rank_biserial": float(eff.rank_biserial),
    }


def _bucket_metrics(
    *,
    per_query: dict[str, np.ndarray],
    buckets: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(buckets.tolist())):
        idx = np.where(buckets == bucket)[0]
        out[bucket] = {
            "recall_at_1": float(np.mean(per_query["recall_at_1"][idx])),
            "recall_at_5": float(np.mean(per_query["recall_at_5"][idx])),
            "mrr_at_10": float(np.mean(per_query["mrr_at_10"][idx])),
        }
    return out


def _render_report(summary: dict[str, Any]) -> str:
    metrics = summary["test_metrics"]
    deltas = summary["primary_metric_deltas_vs_kalman"]
    paired = summary["paired_stats_vs_kalman"]
    lines = [
        "# Correlated Experts Benchmark Slice (Synthetic, Narrow Hypothesis)",
        "",
        "**Label:** Synthetic-to-real bridge. Synthetic outputs are exploratory only and are not headline proof.",
        "",
        "Question: under partially correlated experts with non-identical uncertainty quality, does correlation-aware Kalman improve over baseline Kalman?",
        "",
        f"**Answer:** {summary['answer']}",
        "",
        "## Test metrics",
        "",
        "| Method | Recall@1 | Recall@5 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, vals in metrics.items():
        lines.append(
            f"| {method} | {vals['recall_at_1']:.4f} | {vals['recall_at_5']:.4f} | {vals['mrr_at_10']:.4f} |"
        )

    lines.extend(["", "## Primary metric deltas vs baseline Kalman", ""])
    lines.append("| Method | ΔMRR@10 |")
    lines.append("| --- | ---: |")
    for method, delta in deltas.items():
        lines.append(f"| {method} | {delta:+.4f} |")

    lines.extend(["", "## Paired statistics vs baseline Kalman (MRR@10 per query)", ""])
    lines.append(
        "| Method | ΔMean | 95% bootstrap CI | Wilcoxon p | Cohen dz | Rank-biserial |"
    )
    lines.append("| --- | ---: | --- | ---: | ---: | ---: |")
    for method, vals in paired.items():
        ci = vals["bootstrap_ci95"]
        lines.append(
            f"| {method} | {vals['delta_mean']:+.4f} | [{ci[0]:+.4f}, {ci[1]:+.4f}] | {vals['wilcoxon_p']:.4g} | {vals['cohen_dz']:+.3f} | {vals['rank_biserial']:+.3f} |"
        )

    lines.extend(
        [
            "",
            "## Fusion latency",
            "",
            "- Latency metrics are recorded in `summary.json` under `latency_ms` (mean and p95 ms/query).",
        ]
    )

    lines.extend(
        [
            "",
            "## Correlated slice definition",
            "",
            f"- Query-level correlated score: `||mean(residuals)|| / mean(||residual_i||)`.",
            f"- Correlated if score ≥ `{summary['correlated_definition']['threshold']:.2f}`.",
            f"- Correlated queries on test: `{summary['correlated_definition']['n_correlated_test_queries']}` / `{summary['test_split']['n_queries']}`.",
            "",
            "## Interpretation",
            "",
            "- This is a narrowed hypothesis regime, not a broad claim about general retrieval settings.",
            "- Any synthetic win is treated as directional evidence only; it does not close the headline Kalman-vs-mean claim.",
            "",
        ]
    )
    return "\n".join(lines)


def run_correlation_aware_fusion_benchmark(
    output_dir: Path = Path("results/correlation_aware_fusion"),
    config: CorrelationAwareFusionConfig | None = None,
) -> dict[str, Any]:
    cfg = config or CorrelationAwareFusionConfig()
    problem = _synthesize_problem(cfg)

    sigma2, residual_norms, corr_profile = _fit_on_validation(
        val_true=problem["val_true"],
        val_obs=problem["val_obs"],
        model_names=problem["model_names"],
        variance_floor=cfg.variance_floor,
    )
    corr_sub = corr_profile.correlation_matrix

    weighted_mean_weights = (
        np.asarray([1.0 / max(v, cfg.variance_floor) for v in sigma2], dtype=np.float64)
        / np.sum([1.0 / max(v, cfg.variance_floor) for v in sigma2])
    ).tolist()
    learned_linear_weights = _fit_learned_linear_weights(
        val_obs=problem["val_obs"],
        val_true=problem["val_true"],
        ridge_lambda=cfg.ridge_lambda,
    )

    method_specs = [
        ("mean_fusion", "MeanFuser"),
        ("baseline_kalman", "KalmanorixFuser"),
        (
            "corr_kalman_cov_inflation",
            "CorrelationAwareKalmanFuser",
        ),
        ("weighted_mean", "WeightedMeanFuser"),
        ("learned_linear_combiner", "LearnedLinearCombiner"),
    ]

    test_metrics: dict[str, dict[str, float]] = {}
    test_per_query: dict[str, dict[str, np.ndarray]] = {}
    bucket_metrics: dict[str, dict[str, dict[str, float]]] = {}
    latency_ms: dict[str, dict[str, float]] = {}

    for method_key, method_name in method_specs:
        fused, latency = _fuse_queries(
            method_key=method_key,
            observations=problem["test_obs"],
            sigma2=sigma2,
            corr_submatrix=corr_sub,
            inflation_alpha=cfg.inflation_alpha,
            weighted_mean_weights=weighted_mean_weights,
            learned_linear_weights=learned_linear_weights,
        )
        per_query = _query_level_retrieval(
            fused, problem["docs"], problem["test_targets"]
        )
        test_per_query[method_name] = per_query
        test_metrics[method_name] = asdict(_aggregate_metrics(per_query))
        bucket_metrics[method_name] = _bucket_metrics(
            per_query=per_query,
            buckets=problem["test_buckets"],
        )
        latency_ms[method_name] = latency

    baseline_name = "KalmanorixFuser"
    baseline_mrr = test_metrics[baseline_name]["mrr_at_10"]
    primary_deltas = {
        name: float(vals["mrr_at_10"] - baseline_mrr)
        for name, vals in test_metrics.items()
        if name != baseline_name
    }

    paired_stats_vs_kalman = {
        name: _paired_stats(
            test_per_query[name]["mrr_at_10"],
            test_per_query[baseline_name]["mrr_at_10"],
        )
        for name in test_metrics
        if name != baseline_name
    }

    best_alt = max(primary_deltas.items(), key=lambda kv: kv[1])
    if best_alt[1] > 0.002:
        answer = (
            f"In this synthetic narrowed regime, {best_alt[0]} outperformed baseline Kalman "
            f"(ΔMRR@10={best_alt[1]:+.4f}). Treat as exploratory only."
        )
    elif best_alt[1] < -0.002:
        answer = (
            "In this synthetic narrowed regime, alternatives underperformed baseline Kalman "
            f"(best ΔMRR@10={best_alt[1]:+.4f})."
        )
    else:
        answer = (
            "Null in this synthetic narrowed regime: no material gain over baseline Kalman "
            f"(best ΔMRR@10={best_alt[1]:+.4f})."
        )

    test_residuals = [obs - problem["test_true"] for obs in problem["test_obs"]]
    corr_flags: list[bool] = []
    corr_scores: list[float] = []
    for i in range(cfg.n_test):
        residual_vectors = np.stack([r[i] for r in test_residuals], axis=0)
        flag, score = _is_correlated_query(
            residual_vectors,
            threshold=cfg.correlated_query_threshold,
        )
        corr_flags.append(flag)
        corr_scores.append(score)

    summary: dict[str, Any] = {
        "label": "synthetic_narrowed_hypothesis_regime",
        "config": asdict(cfg),
        "validation_fit": {
            "sigma2": sigma2,
            "residual_norm_shape": list(residual_norms.shape),
            "correlation_matrix": corr_profile.correlation_matrix.tolist(),
            "module_names": corr_profile.module_names,
            "weighted_mean_weights": weighted_mean_weights,
            "learned_linear_combiner_weights": learned_linear_weights,
        },
        "correlated_definition": {
            "query_score": "norm(mean(residual_vectors))/mean(norm(residual_vectors_i))",
            "threshold": cfg.correlated_query_threshold,
            "n_correlated_test_queries": int(np.sum(corr_flags)),
            "mean_query_score": float(np.mean(corr_scores)),
        },
        "test_split": {
            "n_queries": cfg.n_test,
            "buckets": {
                "low_correlation": int(
                    np.sum(problem["test_buckets"] == "low_correlation")
                ),
                "high_correlation": int(
                    np.sum(problem["test_buckets"] == "high_correlation")
                ),
            },
        },
        "test_metrics": test_metrics,
        "bucket_metrics": bucket_metrics,
        "primary_metric_deltas_vs_kalman": primary_deltas,
        "paired_stats_vs_kalman": paired_stats_vs_kalman,
        "latency_ms": latency_ms,
        "interpretation": {
            "scope": "narrowed_hypothesis_regime",
            "synthetic_policy": "Synthetic outcomes are explicitly exploratory and do not count as headline proof.",
        },
        "answer": answer,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary
