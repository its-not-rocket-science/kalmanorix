"""Benchmark correlation-aware Kalman fusion on synthetic specialists.

Outputs artifacts under ``results/correlation_aware_fusion/``:
- residual_correlation_matrix.json
- metrics.json
- report.md
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from kalmanorix.kalman_engine.correlation import (
    correlation_inflation_factors,
    effective_sample_size_discount,
    estimate_residual_correlation_profile,
)
from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal_ensemble


RESULTS_DIR = Path("results/correlation_aware_fusion")


@dataclass(frozen=True)
class SyntheticBatch:
    truths: np.ndarray  # (n_queries, d)
    docs: np.ndarray  # (n_queries, n_docs, d)
    relevant_idx: np.ndarray  # (n_queries,)
    specialist_preds: np.ndarray  # (n_queries, n_specialists, d)
    specialist_covs: np.ndarray  # (n_queries, n_specialists, d)


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1e-12)


def make_synthetic(
    *,
    n_queries: int,
    d: int = 64,
    n_specialists: int = 4,
    n_docs: int = 24,
    shared_noise_scale: float = 0.22,
    private_noise_scale: float = 0.15,
    seed: int = 7,
) -> SyntheticBatch:
    rng = np.random.default_rng(seed)
    truths = _l2norm(rng.standard_normal((n_queries, d)))

    docs = np.zeros((n_queries, n_docs, d), dtype=np.float64)
    relevant_idx = np.zeros(n_queries, dtype=np.int64)
    for q in range(n_queries):
        rel = rng.integers(0, n_docs)
        relevant_idx[q] = rel
        docs[q] = _l2norm(rng.standard_normal((n_docs, d)))
        docs[q, rel] = _l2norm(truths[q] + 0.05 * rng.standard_normal(d))

    shared = shared_noise_scale * rng.standard_normal((n_queries, d))
    specialist_preds = np.zeros((n_queries, n_specialists, d), dtype=np.float64)
    specialist_covs = np.full(
        (n_queries, n_specialists, d), private_noise_scale**2, dtype=np.float64
    )
    specialist_offsets = 0.03 * rng.standard_normal((n_specialists, d))
    for s in range(n_specialists):
        private = private_noise_scale * rng.standard_normal((n_queries, d))
        specialist_preds[:, s, :] = (
            truths + shared + private + specialist_offsets[s][None, :]
        )
    return SyntheticBatch(
        truths=truths,
        docs=docs,
        relevant_idx=relevant_idx,
        specialist_preds=specialist_preds,
        specialist_covs=specialist_covs,
    )


def retrieval_recall_at_1(vectors: np.ndarray, docs: np.ndarray, rel_idx: np.ndarray) -> float:
    qn = _l2norm(vectors)
    dn = _l2norm(docs)
    scores = np.einsum("qd,qnd->qn", qn, dn)
    pred = np.argmax(scores, axis=1)
    return float(np.mean(pred == rel_idx))


def calibration_gap(errors: np.ndarray, posterior_vars: np.ndarray) -> float:
    mse = np.mean(np.sum(errors**2, axis=1))
    mean_var = np.mean(np.sum(posterior_vars, axis=1))
    return float(abs(mse - mean_var))


def run_fusion(batch: SyntheticBatch, corr_matrix: np.ndarray) -> dict[str, dict[str, float]]:
    n_queries, n_specialists, _ = batch.specialist_preds.shape

    fused: dict[str, list[np.ndarray]] = {
        "mean": [],
        "kalman": [],
        "kalman_corr_inflation": [],
        "kalman_corr_ess": [],
    }
    posterior: dict[str, list[np.ndarray]] = {k: [] for k in fused}

    for q in range(n_queries):
        embs = [batch.specialist_preds[q, s] for s in range(n_specialists)]
        covs = [batch.specialist_covs[q, s] for s in range(n_specialists)]

        mean_vec = np.mean(np.stack(embs), axis=0)
        mean_var = np.var(np.stack(embs), axis=0) + 1e-8
        fused["mean"].append(mean_vec)
        posterior["mean"].append(mean_var)

        kal_vec, kal_cov = kalman_fuse_diagonal_ensemble(embs, covs)
        fused["kalman"].append(kal_vec)
        posterior["kalman"].append(kal_cov)

        infl = correlation_inflation_factors(corr_matrix, alpha=1.0)
        infl_covs = [covs[s] * infl[s] for s in range(n_specialists)]
        inf_vec, inf_cov = kalman_fuse_diagonal_ensemble(embs, infl_covs)
        fused["kalman_corr_inflation"].append(inf_vec)
        posterior["kalman_corr_inflation"].append(inf_cov)

        ess_discount = effective_sample_size_discount(corr_matrix)
        ess_covs = [cov / ess_discount for cov in covs]
        ess_vec, ess_cov = kalman_fuse_diagonal_ensemble(embs, ess_covs)
        fused["kalman_corr_ess"].append(ess_vec)
        posterior["kalman_corr_ess"].append(ess_cov)
    metrics: dict[str, dict[str, float]] = {}
    for name, vectors in fused.items():
        vec_arr = np.stack(vectors)
        post_arr = np.stack(posterior[name])
        errors = vec_arr - batch.truths
        metrics[name] = {
            "recall_at_1": retrieval_recall_at_1(
                vec_arr, batch.docs, batch.relevant_idx
            ),
            "mean_posterior_variance": float(np.mean(np.sum(post_arr, axis=1))),
            "calibration_gap_abs_mse_minus_var": calibration_gap(errors, post_arr),
            "embedding_mse": float(np.mean(np.sum(errors**2, axis=1))),
        }
    return metrics


def main() -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    val = make_synthetic(n_queries=300, seed=11)
    test = make_synthetic(n_queries=300, seed=19)

    val_residual_norms = np.linalg.norm(
        val.specialist_preds - val.truths[:, None, :], axis=2
    )  # (q, specialists)
    module_names = [f"specialist_{i}" for i in range(val.specialist_preds.shape[1])]
    corr_profile = estimate_residual_correlation_profile(
        module_names=module_names,
        residual_norms=val_residual_norms,
    )
    corr = corr_profile.correlation_matrix

    metrics = run_fusion(test, corr)

    (RESULTS_DIR / "residual_correlation_matrix.json").write_text(
        json.dumps(
            {
                "module_names": module_names,
                "correlation_matrix": corr.tolist(),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    (RESULTS_DIR / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    report_lines = [
        "# Correlation-aware Fusion Report",
        "",
        "Estimated residual correlation matrix is validation-derived.",
        "",
        "## Metrics",
    ]
    for name, row in metrics.items():
        report_lines.append(
            f"- **{name}**: recall@1={row['recall_at_1']:.3f}, "
            f"posterior_var={row['mean_posterior_variance']:.4f}, "
            f"calibration_gap={row['calibration_gap_abs_mse_minus_var']:.4f}"
        )
    (RESULTS_DIR / "report.md").write_text("\n".join(report_lines), encoding="utf-8")


if __name__ == "__main__":
    main()
