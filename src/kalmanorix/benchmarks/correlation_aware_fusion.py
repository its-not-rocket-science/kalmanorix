"""Correlation-aware Kalman fusion benchmark with validation-only correlation fitting.

Compares four fusion methods:
- mean fusion
- baseline Kalman
- covariance-inflation correlation-aware Kalman
- effective-sample-size correlation-aware Kalman

The benchmark enforces split cleanliness:
- specialist noise scales and residual-correlation profile are estimated on validation
- all final metrics are measured on a strengthened test split
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from kalmanorix.kalman_engine.correlation import (
    correlation_inflation_factors,
    effective_sample_size_discount,
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


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    mrr_at_10: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def _retrieval_metrics(
    queries: np.ndarray, docs: np.ndarray, targets: np.ndarray
) -> RetrievalMetrics:
    scores = _l2_normalize(queries) @ _l2_normalize(docs).T
    ranked = np.argsort(-scores, axis=1)
    hit1 = np.mean(ranked[:, 0] == targets)
    hit5 = np.mean([t in ranked[i, :5] for i, t in enumerate(targets)])

    rr = []
    for i, t in enumerate(targets):
        top10 = ranked[i, :10]
        loc = np.where(top10 == t)[0]
        rr.append(0.0 if loc.size == 0 else 1.0 / float(loc[0] + 1))
    return RetrievalMetrics(float(hit1), float(hit5), float(np.mean(rr)))


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

    scales = np.linspace(0.10, 0.20, cfg.n_specialists)

    # Validation is mixed, test is intentionally strengthened toward higher correlation.
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

    # Correlation profile is estimated from scalar residual magnitudes on validation only.
    residual_norms = np.column_stack(
        [np.linalg.norm(r, axis=1) for r in residuals]
    ).astype(np.float64)
    profile = estimate_residual_correlation_profile(model_names, residual_norms)
    return sigma2, residual_norms, profile


def _fuse_queries(
    *,
    method_key: str,
    observations: list[np.ndarray],
    sigma2: list[float],
    corr_submatrix: np.ndarray,
    inflation_alpha: float,
) -> np.ndarray:
    n_q = observations[0].shape[0]
    fused = []

    inflation = correlation_inflation_factors(corr_submatrix, alpha=inflation_alpha)
    ess_discount = effective_sample_size_discount(corr_submatrix)

    for q_idx in range(n_q):
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
        elif method_key == "corr_kalman_effective_sample_size":
            safe_discount = max(ess_discount, 1.0 / len(sigma2))
            covs = [
                np.full_like(embs[0], s / safe_discount, dtype=np.float64)
                for s in sigma2
            ]
            fused_vec, _ = kalman_fuse_diagonal_ensemble(embs, covs)
        else:
            raise ValueError(f"Unknown method: {method_key}")
        fused.append(fused_vec)

    return np.asarray(fused, dtype=np.float64)


def _bucket_metrics(
    *,
    fused: np.ndarray,
    docs: np.ndarray,
    targets: np.ndarray,
    buckets: np.ndarray,
) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for bucket in sorted(set(buckets.tolist())):
        idx = np.where(buckets == bucket)[0]
        m = _retrieval_metrics(fused[idx], docs, targets[idx])
        out[bucket] = asdict(m)
    return out


def _render_report(summary: dict[str, Any]) -> str:
    metrics = summary["test_metrics"]
    per_bucket = summary["bucket_metrics"]
    lines = [
        "# Correlation-Aware Fusion Benchmark",
        "",
        "Question: does correlation adjustment improve Kalman fusion on a strengthened correlated-expert test split?",
        "",
        f"**Answer:** {summary['answer']}",
        "",
        "## Test metrics (strengthened split)",
        "",
        "| Method | Recall@1 | Recall@5 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]
    for method, vals in metrics.items():
        lines.append(
            f"| {method} | {vals['recall_at_1']:.4f} | {vals['recall_at_5']:.4f} | {vals['mrr_at_10']:.4f} |"
        )

    lines.extend(["", "## Per-bucket metrics", ""])
    for method, by_bucket in per_bucket.items():
        lines.append(f"### {method}")
        lines.append("")
        lines.append("| Bucket | Recall@1 | Recall@5 | MRR@10 |")
        lines.append("| --- | ---: | ---: | ---: |")
        for bucket, vals in by_bucket.items():
            lines.append(
                f"| {bucket} | {vals['recall_at_1']:.4f} | {vals['recall_at_5']:.4f} | {vals['mrr_at_10']:.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Notes",
            "",
            "- Correlation profile was estimated only from validation residuals.",
            "- Test split is strengthened by including a high-correlation half (ρ=0.85).",
            "- Null/negative outcomes are reported directly; no optimistic retuning on test.",
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

    method_specs = [
        ("mean_fusion", "MeanFuser"),
        ("baseline_kalman", "KalmanorixFuser"),
        (
            "corr_kalman_cov_inflation",
            "CorrelationAwareKalmanFuser (covariance_inflation)",
        ),
        (
            "corr_kalman_effective_sample_size",
            "CorrelationAwareKalmanFuser (effective_sample_size)",
        ),
    ]
    test_metrics: dict[str, dict[str, float]] = {}
    bucket_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for method_key, method_name in method_specs:
        fused = _fuse_queries(
            method_key=method_key,
            observations=problem["test_obs"],
            sigma2=sigma2,
            corr_submatrix=corr_sub,
            inflation_alpha=cfg.inflation_alpha,
        )
        test_metrics[method_name] = asdict(
            _retrieval_metrics(fused, problem["docs"], problem["test_targets"])
        )
        bucket_metrics[method_name] = _bucket_metrics(
            fused=fused,
            docs=problem["docs"],
            targets=problem["test_targets"],
            buckets=problem["test_buckets"],
        )

    baseline = test_metrics["KalmanorixFuser"]["mrr_at_10"]
    best_corr_method = max(
        [
            "CorrelationAwareKalmanFuser (covariance_inflation)",
            "CorrelationAwareKalmanFuser (effective_sample_size)",
        ],
        key=lambda m: test_metrics[m]["mrr_at_10"],
    )
    best_corr = test_metrics[best_corr_method]["mrr_at_10"]
    delta = best_corr - baseline
    if delta > 0.002:
        answer = (
            f"Correlation-aware Kalman improved over baseline Kalman (best: {best_corr_method}, "
            f"ΔMRR@10={delta:.4f})."
        )
    elif delta < -0.002:
        answer = (
            f"Correlation-aware Kalman regressed vs baseline Kalman on this run "
            f"(best corr method ΔMRR@10={delta:.4f})."
        )
    else:
        answer = (
            "Null result: correlation-aware adjustments did not materially improve baseline "
            f"Kalman (best ΔMRR@10={delta:.4f})."
        )

    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "validation_fit": {
            "sigma2": sigma2,
            "residual_norm_shape": list(residual_norms.shape),
            "correlation_matrix": corr_profile.correlation_matrix.tolist(),
            "module_names": corr_profile.module_names,
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
        "answer": answer,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary
