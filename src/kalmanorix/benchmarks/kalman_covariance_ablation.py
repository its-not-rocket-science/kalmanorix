"""Kalman covariance-structure ablation benchmark.

Compares four fusion variants on a shared synthetic retrieval benchmark:
- mean fusion
- scalar Kalman (single variance per specialist)
- diagonal Kalman (per-dimension variance)
- structured Kalman (diagonal + low-rank covariance)

Covariance fitting is performed on held-out validation residuals only.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np

from kalmanorix.kalman_engine.kalman_fuser import (
    kalman_fuse_diagonal,
    kalman_fuse_structured,
)
from kalmanorix.kalman_engine.structured_covariance import StructuredCovariance


@dataclass(frozen=True)
class CovarianceFitConfig:
    diagonal_floor: float = 1e-6
    shrinkage_to_diagonal: float = 0.1
    max_lowrank_fro_norm: float = 10.0
    lowrank_rank: int = 4


@dataclass(frozen=True)
class AblationConfig:
    benchmark_version: str = "kalman_covariance_ablation_v2_enlarged"
    random_seed: int = 0
    dimension: int = 32
    n_docs: int = 512
    n_val: int = 1200
    n_test: int = 1600
    n_specialists: int = 3
    n_domains: int = 4
    disagreement_quantile: float = 0.8
    uncertainty_skew_quantile: float = 0.8
    fit: CovarianceFitConfig = CovarianceFitConfig()


@dataclass(frozen=True)
class RetrievalMetrics:
    recall_at_1: float
    recall_at_5: float
    mrr_at_10: float


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def project_psd(matrix: np.ndarray, eigenvalue_floor: float = 0.0) -> np.ndarray:
    """Project a symmetric matrix to PSD cone by clipping eigenvalues."""
    sym = 0.5 * (matrix + matrix.T)
    evals, evecs = np.linalg.eigh(sym)
    evals_clipped = np.clip(evals, eigenvalue_floor, None)
    return (evecs * evals_clipped[None, :]) @ evecs.T


def _stable_diagonal(diagonal: np.ndarray, floor: float) -> np.ndarray:
    return np.maximum(np.asarray(diagonal, dtype=np.float64), floor)


def fit_scalar_variance(residuals: np.ndarray, diagonal_floor: float = 1e-6) -> float:
    sq = np.asarray(residuals, dtype=np.float64) ** 2
    return float(max(np.mean(sq), diagonal_floor))


def fit_diagonal_variance(residuals: np.ndarray, diagonal_floor: float = 1e-6) -> np.ndarray:
    var = np.var(np.asarray(residuals, dtype=np.float64), axis=0, ddof=1)
    return _stable_diagonal(var, diagonal_floor)


def fit_structured_covariance(
    residuals: np.ndarray,
    *,
    rank: int,
    diagonal_floor: float = 1e-6,
    shrinkage_to_diagonal: float = 0.1,
    max_lowrank_fro_norm: float = 10.0,
) -> StructuredCovariance:
    """Fit diagonal + low-rank covariance from residuals using PCA/SVD structure."""
    err = np.asarray(residuals, dtype=np.float64)
    if err.ndim != 2:
        raise ValueError("residuals must be (n_samples, d)")

    n_samples, d = err.shape
    if n_samples < 2:
        diagonal = np.full(d, diagonal_floor, dtype=np.float64)
        return StructuredCovariance.from_diagonal(diagonal)

    centered = err - np.mean(err, axis=0, keepdims=True)
    cov = (centered.T @ centered) / max(n_samples - 1, 1)
    cov = project_psd(cov, eigenvalue_floor=0.0)

    shrink = float(np.clip(shrinkage_to_diagonal, 0.0, 1.0))
    diag_cov = np.diag(np.diag(cov))
    cov_shrunk = (1.0 - shrink) * cov + shrink * diag_cov
    cov_shrunk = project_psd(cov_shrunk, eigenvalue_floor=0.0)

    diagonal = _stable_diagonal(np.diag(cov_shrunk), diagonal_floor)

    k = int(max(0, min(rank, d)))
    if k == 0:
        return StructuredCovariance.from_diagonal(diagonal)

    offdiag = cov_shrunk - np.diag(np.diag(cov_shrunk))
    offdiag = project_psd(offdiag, eigenvalue_floor=0.0)
    evals, evecs = np.linalg.eigh(offdiag)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]

    positive = evals > 1e-12
    if not np.any(positive):
        return StructuredCovariance.from_diagonal(diagonal)

    evals = evals[positive][:k]
    evecs = evecs[:, positive][:, :k]
    if evals.size == 0:
        return StructuredCovariance.from_diagonal(diagonal)

    lowrank = evecs * np.sqrt(evals)[None, :]
    fro = float(np.linalg.norm(lowrank, ord="fro"))
    if fro > max_lowrank_fro_norm and fro > 0:
        lowrank = lowrank * (max_lowrank_fro_norm / fro)

    return StructuredCovariance.from_lowrank(diagonal=diagonal, lowrank_factor=lowrank)


def _make_ground_truth_covariances(cfg: AblationConfig, rng: np.random.Generator) -> list[np.ndarray]:
    covs: list[np.ndarray] = []
    d = cfg.dimension
    base_scales = np.linspace(0.03, 0.12, cfg.n_specialists)

    for idx, base in enumerate(base_scales):
        diag = np.linspace(base, base * (1.0 + 0.7 * (idx + 1)), d)
        cov = np.diag(diag)
        if idx % 2 == 1:
            rank = min(3, d // 4)
            vecs = rng.normal(size=(d, rank))
            vecs /= np.linalg.norm(vecs, axis=0, keepdims=True) + 1e-12
            strengths = np.linspace(base * 0.5, base * 1.2, rank)
            cov += (vecs * strengths[None, :]) @ vecs.T
        covs.append(project_psd(cov, eigenvalue_floor=1e-8))

    return covs


def _sample_problem(cfg: AblationConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    d = cfg.dimension

    domain_centers = _l2_normalize(rng.normal(size=(cfg.n_domains, d)))
    doc_domains = rng.integers(0, cfg.n_domains, size=cfg.n_docs)
    docs = domain_centers[doc_domains] + 0.45 * rng.normal(size=(cfg.n_docs, d))
    docs = _l2_normalize(docs)

    def _sample_queries(n_q: int) -> tuple[np.ndarray, np.ndarray]:
        queries = []
        multi_mask = np.zeros(n_q, dtype=bool)
        for i in range(n_q):
            is_multi = bool(rng.random() < 0.35)
            multi_mask[i] = is_multi
            if is_multi:
                doms = rng.choice(cfg.n_domains, size=2, replace=False)
                latent = (
                    0.55 * domain_centers[doms[0]]
                    + 0.45 * domain_centers[doms[1]]
                    + 0.65 * rng.normal(size=d)
                )
            else:
                dom = int(rng.integers(0, cfg.n_domains))
                latent = domain_centers[dom] + 0.55 * rng.normal(size=d)
            queries.append(latent)
        return _l2_normalize(np.asarray(queries, dtype=np.float64)), multi_mask

    val_true, _ = _sample_queries(cfg.n_val)
    test_true, test_multi_domain_mask = _sample_queries(cfg.n_test)

    val_targets = np.argmax(val_true @ docs.T, axis=1)
    test_targets = np.argmax(test_true @ docs.T, axis=1)

    true_covs = _make_ground_truth_covariances(cfg, rng)
    query_noise_scale_val = rng.lognormal(mean=0.0, sigma=0.3, size=(cfg.n_val, cfg.n_specialists))
    query_noise_scale_test = rng.lognormal(mean=0.0, sigma=0.45, size=(cfg.n_test, cfg.n_specialists))

    val_obs: list[np.ndarray] = []
    test_obs: list[np.ndarray] = []
    for idx, cov in enumerate(true_covs):
        val_noise = rng.multivariate_normal(np.zeros(d), cov, size=cfg.n_val)
        test_noise = rng.multivariate_normal(np.zeros(d), cov, size=cfg.n_test)
        val_obs.append(_l2_normalize(val_true + query_noise_scale_val[:, [idx]] * val_noise))
        test_obs.append(_l2_normalize(test_true + query_noise_scale_test[:, [idx]] * test_noise))

    specialist_top1 = np.stack([np.argmax(obs @ docs.T, axis=1) for obs in test_obs], axis=1)
    disagreement = np.array([len(set(preds.tolist())) for preds in specialist_top1], dtype=np.float64) / float(
        cfg.n_specialists
    )
    skew = (
        np.max(query_noise_scale_test, axis=1)
        / (np.min(query_noise_scale_test, axis=1) + 1e-12)
    )
    disagreement_thr = float(np.quantile(disagreement, cfg.disagreement_quantile))
    skew_thr = float(np.quantile(skew, cfg.uncertainty_skew_quantile))

    return {
        "docs": docs,
        "doc_domains": doc_domains,
        "val_true": val_true,
        "test_true": test_true,
        "val_targets": val_targets,
        "test_targets": test_targets,
        "val_obs": val_obs,
        "test_obs": test_obs,
        "bucket_masks": {
            "all_queries": np.ones(cfg.n_test, dtype=bool),
            "high_disagreement": disagreement >= disagreement_thr,
            "multi_domain": test_multi_domain_mask,
            "uncertainty_skewed": skew >= skew_thr,
        },
        "bucket_metadata": {
            "disagreement_threshold": disagreement_thr,
            "uncertainty_skew_threshold": skew_thr,
        },
    }


def _retrieval_metrics(queries: np.ndarray, docs: np.ndarray, targets: np.ndarray) -> RetrievalMetrics:
    if queries.shape[0] == 0 or targets.shape[0] == 0:
        return RetrievalMetrics(0.0, 0.0, 0.0)
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


def _fuse_queries(
    method: str,
    observations: list[np.ndarray],
    scalar_vars: list[float],
    diag_vars: list[np.ndarray],
    structured_covs: list[StructuredCovariance],
) -> np.ndarray:
    n_q = observations[0].shape[0]
    fused = []

    for q_idx in range(n_q):
        embs = [obs[q_idx] for obs in observations]
        if method == "mean_fusion":
            fused_vec = np.mean(np.stack(embs, axis=0), axis=0)
        elif method == "scalar_kalman":
            covs = [np.full_like(embs[0], s, dtype=np.float64) for s in scalar_vars]
            fused_vec, _ = kalman_fuse_diagonal(embs, covs)
        elif method == "diagonal_kalman":
            fused_vec, _ = kalman_fuse_diagonal(embs, diag_vars)
        elif method == "structured_kalman":
            fused_vec, _ = kalman_fuse_structured(embs, structured_covs)
        else:
            raise ValueError(f"Unknown fusion method: {method}")
        fused.append(fused_vec)

    return np.asarray(fused, dtype=np.float64)


def _profile_method(
    method: str,
    observations: list[np.ndarray],
    scalar_vars: list[float],
    diag_vars: list[np.ndarray],
    structured_covs: list[StructuredCovariance],
) -> tuple[np.ndarray, dict[str, float]]:
    tracemalloc.start()
    start = time.perf_counter()
    fused = _fuse_queries(method, observations, scalar_vars, diag_vars, structured_covs)
    elapsed_ms = (time.perf_counter() - start) * 1000.0
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    n_queries = max(1, observations[0].shape[0])
    return fused, {
        "latency_ms_total": float(elapsed_ms),
        "latency_ms_per_query": float(elapsed_ms / n_queries),
        "peak_memory_kib": float(peak / 1024.0),
    }


def run_kalman_covariance_ablation(
    output_dir: Path = Path("results/kalman_covariance_ablation_v2"),
    config: AblationConfig | None = None,
) -> dict[str, Any]:
    """Run covariance-structure ablation and write summary/report artifacts."""
    cfg = config or AblationConfig()
    problem = _sample_problem(cfg)

    val_true = problem["val_true"]
    val_obs: list[np.ndarray] = problem["val_obs"]

    scalar_vars = []
    diag_vars = []
    structured_covs = []
    fit_cfg = cfg.fit

    for specialist_obs in val_obs:
        residuals = specialist_obs - val_true
        scalar_vars.append(fit_scalar_variance(residuals, fit_cfg.diagonal_floor))
        diag_vars.append(fit_diagonal_variance(residuals, fit_cfg.diagonal_floor))
        structured_covs.append(
            fit_structured_covariance(
                residuals,
                rank=fit_cfg.lowrank_rank,
                diagonal_floor=fit_cfg.diagonal_floor,
                shrinkage_to_diagonal=fit_cfg.shrinkage_to_diagonal,
                max_lowrank_fro_norm=fit_cfg.max_lowrank_fro_norm,
            )
        )

    methods = ["mean_fusion", "scalar_kalman", "diagonal_kalman", "structured_kalman"]
    metrics: dict[str, dict[str, float]] = {}
    bucket_metrics: dict[str, dict[str, dict[str, float]]] = {}
    efficiency: dict[str, dict[str, float]] = {}
    for method in methods:
        fused, perf = _profile_method(
            method,
            problem["test_obs"],
            scalar_vars,
            diag_vars,
            structured_covs,
        )
        efficiency[method] = perf
        m = _retrieval_metrics(fused, problem["docs"], problem["test_targets"])
        metrics[method] = asdict(m)
        bucket_metrics[method] = {}
        for bucket_name, mask in problem["bucket_masks"].items():
            bucket_targets = problem["test_targets"][mask]
            bucket_fused = fused[mask]
            bucket_metrics[method][bucket_name] = asdict(
                _retrieval_metrics(bucket_fused, problem["docs"], bucket_targets)
            )

    summary = {
        "config": asdict(cfg),
        "covariance_fit": {
            "scalar_variances": scalar_vars,
            "diagonal_shapes": [list(v.shape) for v in diag_vars],
            "structured_ranks": [cov.rank for cov in structured_covs],
        },
        "metrics": metrics,
        "bucket_metrics": bucket_metrics,
        "efficiency": efficiency,
        "bucket_metadata": problem["bucket_metadata"],
        "benchmark_artifact": str(output_dir),
        "prior_artifact_for_comparison": "results/kalman_covariance_ablation",
        "question": "Do richer uncertainty families justify complexity?",
        "answer": _answer(metrics, bucket_metrics),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")

    return summary


def _answer(metrics: dict[str, dict[str, float]], bucket_metrics: dict[str, dict[str, dict[str, float]]]) -> str:
    baseline_global = metrics["scalar_kalman"]["recall_at_1"]
    richer_global = max(metrics["diagonal_kalman"]["recall_at_1"], metrics["structured_kalman"]["recall_at_1"])
    global_gain = richer_global - baseline_global
    local_gain = max(
        bucket_metrics["diagonal_kalman"][bucket]["recall_at_1"] - bucket_metrics["scalar_kalman"][bucket]["recall_at_1"]
        for bucket in ["high_disagreement", "multi_domain", "uncertainty_skewed"]
    )
    local_gain = max(
        local_gain,
        max(
            bucket_metrics["structured_kalman"][bucket]["recall_at_1"]
            - bucket_metrics["scalar_kalman"][bucket]["recall_at_1"]
            for bucket in ["high_disagreement", "multi_domain", "uncertainty_skewed"]
        ),
    )
    if global_gain >= 0.01:
        return "Richer covariance is globally useful: diagonal/structured variants improve overall recall@1 by at least 1 point."
    if local_gain >= 0.02:
        return "Richer covariance is niche useful: global gains are small, but targeted buckets show meaningful improvements."
    return "Richer covariance is not worth it in this setup: no global or bucket-level gain cleared practical thresholds."


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Kalman Covariance Ablation",
        "",
        f"Benchmark version: `{summary['config']['benchmark_version']}`",
        "",
        summary["question"],
        "",
        f"**Answer:** {summary['answer']}",
        "",
        "## Retrieval Metrics",
        "",
        "| Method | Recall@1 | Recall@5 | MRR@10 |",
        "| --- | ---: | ---: | ---: |",
    ]

    for method, vals in summary["metrics"].items():
        lines.append(
            f"| {method} | {vals['recall_at_1']:.4f} | {vals['recall_at_5']:.4f} | {vals['mrr_at_10']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Per-bucket Recall@1",
            "",
            "| Method | All | High Disagreement | Multi-domain | Uncertainty-skewed |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
    )
    for method, vals in summary["bucket_metrics"].items():
        lines.append(
            "| "
            f"{method} | "
            f"{vals['all_queries']['recall_at_1']:.4f} | "
            f"{vals['high_disagreement']['recall_at_1']:.4f} | "
            f"{vals['multi_domain']['recall_at_1']:.4f} | "
            f"{vals['uncertainty_skewed']['recall_at_1']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Efficiency Trade-offs",
            "",
            "| Method | Total latency (ms) | Latency/query (ms) | Peak memory (KiB) |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method, vals in summary["efficiency"].items():
        lines.append(
            f"| {method} | {vals['latency_ms_total']:.2f} | {vals['latency_ms_per_query']:.4f} | {vals['peak_memory_kib']:.1f} |"
        )

    return "\n".join(lines) + "\n"
