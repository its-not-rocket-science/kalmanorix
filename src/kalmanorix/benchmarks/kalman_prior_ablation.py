"""Prior-strength ablation for Kalman fusion.

This benchmark studies whether stronger priors make Kalman fusion materially
better than mean fusion and the current default Kalman initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal


@dataclass(frozen=True)
class PriorAblationConfig:
    random_seed: int = 7
    dimension: int = 48
    n_specialists: int = 4
    n_train: int = 220
    n_validation: int = 180
    n_test: int = 260
    latency_base_ms: float = 1.0
    latency_per_specialist_ms: float = 0.35
    latency_kalman_overhead_ms: float = 0.20


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def fit_learned_linear_prior(
    specialist_embeddings: np.ndarray,
    true_embeddings: np.ndarray,
    split_labels: list[str],
    fit_split: str = "train",
    ridge_lambda: float = 1e-3,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Fit linear combiner weights for prior without split leakage.

    specialist_embeddings shape: (n_queries, n_specialists, d)
    true_embeddings shape: (n_queries, d)
    """
    fit_idx = [i for i, split in enumerate(split_labels) if split == fit_split]
    if not fit_idx:
        raise ValueError(f"No samples found for fit_split={fit_split!r}")

    x = specialist_embeddings[fit_idx]  # (n_fit, n_specialists, d)
    y = true_embeddings[fit_idx]  # (n_fit, d)

    n_fit, n_specialists, d = x.shape
    design = x.transpose(0, 2, 1).reshape(n_fit * d, n_specialists)
    target = y.reshape(n_fit * d)

    ridge = ridge_lambda * np.eye(n_specialists, dtype=np.float64)
    weights = np.linalg.solve(design.T @ design + ridge, design.T @ target)
    weights = np.clip(weights, 0.0, None)
    if float(np.sum(weights)) <= 0.0:
        weights = np.full(n_specialists, 1.0 / n_specialists, dtype=np.float64)
    else:
        weights = weights / np.sum(weights)

    meta = {
        "fit_split": fit_split,
        "fit_indices": fit_idx,
        "n_fit": len(fit_idx),
    }
    return weights.astype(np.float64), meta


def kalman_fuse_with_prior_modes(
    embeddings: list[np.ndarray],
    covariances: list[np.ndarray],
    *,
    prior_mode: str,
    generalist_prior: np.ndarray | None = None,
    learned_prior_weights: np.ndarray | None = None,
    router_scores: np.ndarray | None = None,
    residual_mode: bool = False,
    prior_only: bool = False,
    epsilon: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray]:
    """Kalman fusion with configurable prior source.

    prior_mode options:
    - "zero" / "current_default": no explicit prior, use current Kalman init.
    - "single_generalist": use provided generalist prior embedding.
    - "learned_linear_combiner": weighted specialist prior.
    - "router_top_specialist": top router specialist as prior.
    """
    if not embeddings or not covariances:
        raise ValueError("embeddings and covariances must be non-empty")

    d = embeddings[0].shape[0]
    prior_mean: np.ndarray | None = None
    prior_cov = np.mean(np.stack(covariances, axis=0), axis=0)

    if prior_mode in {"zero", "current_default"}:
        prior_mean = None
    elif prior_mode == "single_generalist":
        if generalist_prior is None:
            raise ValueError("generalist_prior is required for single_generalist")
        prior_mean = generalist_prior.astype(np.float64)
    elif prior_mode == "learned_linear_combiner":
        if learned_prior_weights is None:
            raise ValueError(
                "learned_prior_weights is required for learned_linear_combiner"
            )
        w = learned_prior_weights.astype(np.float64)
        prior_mean = np.sum(np.stack(embeddings, axis=0) * w[:, None], axis=0)
    elif prior_mode == "router_top_specialist":
        if router_scores is None:
            raise ValueError("router_scores is required for router_top_specialist")
        idx = int(np.argmax(router_scores))
        prior_mean = embeddings[idx].astype(np.float64)
        prior_cov = covariances[idx].astype(np.float64)
    else:
        raise ValueError(f"Unknown prior_mode: {prior_mode}")

    if prior_only:
        if prior_mean is None:
            prior_mean = np.zeros(d, dtype=np.float64)
        return prior_mean.copy(), np.maximum(prior_cov.copy(), epsilon)

    if residual_mode:
        base = np.zeros(d, dtype=np.float64) if prior_mean is None else prior_mean
        residuals = [emb - base for emb in embeddings]
        fused_residual, fused_cov = kalman_fuse_diagonal(
            residuals,
            covariances,
            sort_by_certainty=True,
            epsilon=epsilon,
        )
        return base + fused_residual, fused_cov

    if prior_mean is None:
        return kalman_fuse_diagonal(
            embeddings,
            covariances,
            sort_by_certainty=True,
            epsilon=epsilon,
        )

    return kalman_fuse_diagonal(
        embeddings,
        covariances,
        initial_state=prior_mean,
        initial_covariance=prior_cov,
        sort_by_certainty=True,
        epsilon=epsilon,
    )


def _sample_problem(cfg: PriorAblationConfig) -> dict[str, Any]:
    rng = np.random.default_rng(cfg.random_seed)
    n_total = cfg.n_train + cfg.n_validation + cfg.n_test
    d = cfg.dimension
    k = cfg.n_specialists

    truth = _l2_normalize(rng.normal(size=(n_total, d)))
    generalist_noise = rng.normal(scale=0.20, size=(n_total, d))
    generalist = _l2_normalize(truth + generalist_noise)

    specialist = []
    router_scores = []
    for idx in range(k):
        domain_bias = rng.normal(scale=0.03 * (idx + 1), size=(d,))
        noise_scale = 0.08 + 0.05 * idx
        obs = _l2_normalize(
            truth
            + domain_bias[None, :]
            + rng.normal(scale=noise_scale, size=(n_total, d))
        )
        specialist.append(obs)
        router_scores.append(
            -np.linalg.norm(obs - truth, axis=1) + rng.normal(scale=0.02, size=n_total)
        )

    specialist_arr = np.stack(specialist, axis=1)  # (n, k, d)
    router_arr = np.stack(router_scores, axis=1)  # (n, k)

    split_labels = (
        (["train"] * cfg.n_train)
        + (["validation"] * cfg.n_validation)
        + (["test"] * cfg.n_test)
    )
    return {
        "truth": truth,
        "generalist": generalist,
        "specialist": specialist_arr,
        "router_scores": router_arr,
        "split_labels": split_labels,
    }


def _fit_diag_covariances(problem: dict[str, Any]) -> list[np.ndarray]:
    truth = problem["truth"]
    splits = problem["split_labels"]
    val_idx = [i for i, s in enumerate(splits) if s == "validation"]
    specialist = problem["specialist"]
    covs = []
    for s_idx in range(specialist.shape[1]):
        residual = specialist[val_idx, s_idx, :] - truth[val_idx]
        var = np.var(residual, axis=0, ddof=1)
        covs.append(np.maximum(var, 1e-6))
    return covs


def _error_and_calibration(
    errors: np.ndarray, uncertainties: np.ndarray
) -> dict[str, float]:
    confidence = np.exp(-uncertainties)
    observed = np.exp(-errors)
    bins = np.linspace(0.0, 1.0, 11)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:], strict=True):
        mask = (confidence >= lo) & (confidence < hi)
        if not np.any(mask):
            continue
        ece += float(np.mean(mask)) * abs(
            float(np.mean(confidence[mask]) - np.mean(observed[mask]))
        )
    return {
        "mean_error": float(np.mean(errors)),
        "calibration_ece": float(ece),
    }


def run_kalman_prior_ablation(
    output_dir: Path = Path("results/kalman_prior_ablation"),
    config: PriorAblationConfig | None = None,
) -> dict[str, Any]:
    cfg = config or PriorAblationConfig()
    problem = _sample_problem(cfg)
    specialist = problem["specialist"]
    truth = problem["truth"]
    generalist = problem["generalist"]
    router_scores = problem["router_scores"]
    split_labels = problem["split_labels"]

    covariances = _fit_diag_covariances(problem)
    learned_w, fit_meta = fit_learned_linear_prior(
        specialist, truth, split_labels, fit_split="train"
    )

    test_idx = [i for i, s in enumerate(split_labels) if s == "test"]

    methods = {
        "mean_fusion": {"prior_mode": "zero", "residual_mode": False},
        "kalman_current": {"prior_mode": "current_default", "residual_mode": False},
        "kalman_generalist_prior": {
            "prior_mode": "single_generalist",
            "residual_mode": False,
        },
        "kalman_learned_linear_prior": {
            "prior_mode": "learned_linear_combiner",
            "residual_mode": False,
        },
        "kalman_residuals": {"prior_mode": "single_generalist", "residual_mode": True},
    }

    results: dict[str, dict[str, float]] = {}
    for name, mode in methods.items():
        errs = []
        unc = []
        lat = []
        for q_idx in test_idx:
            embs = [specialist[q_idx, s_idx, :] for s_idx in range(cfg.n_specialists)]
            covs = covariances

            if name == "mean_fusion":
                fused = np.mean(np.stack(embs, axis=0), axis=0)
                fused_cov = np.mean(np.stack(covs, axis=0), axis=0)
                latency = (
                    cfg.latency_base_ms
                    + cfg.n_specialists * cfg.latency_per_specialist_ms
                )
            else:
                fused, fused_cov = kalman_fuse_with_prior_modes(
                    embs,
                    covs,
                    prior_mode=mode["prior_mode"],
                    generalist_prior=generalist[q_idx],
                    learned_prior_weights=learned_w,
                    router_scores=router_scores[q_idx],
                    residual_mode=mode["residual_mode"],
                )
                latency = (
                    cfg.latency_base_ms
                    + cfg.n_specialists * cfg.latency_per_specialist_ms
                    + cfg.latency_kalman_overhead_ms
                )

            e = 1.0 - float(
                np.dot(fused, truth[q_idx])
                / ((np.linalg.norm(fused) * np.linalg.norm(truth[q_idx])) + 1e-12)
            )
            errs.append(e)
            unc.append(float(np.mean(fused_cov)))
            lat.append(latency)

        err_arr = np.asarray(errs, dtype=np.float64)
        unc_arr = np.asarray(unc, dtype=np.float64)
        lat_arr = np.asarray(lat, dtype=np.float64)
        metrics = _error_and_calibration(err_arr, unc_arr)
        metrics["latency_ms"] = float(np.mean(lat_arr))
        metrics["latency_normalized_error"] = float(np.mean(err_arr * lat_arr))
        results[name] = metrics

    best_lne = min(results, key=lambda k: results[k]["latency_normalized_error"])
    summary = {
        "config": asdict(cfg),
        "learned_prior": {
            "weights": learned_w.tolist(),
            **fit_meta,
        },
        "metrics": results,
        "question": "Does a stronger prior make Kalman fusion materially more useful?",
        "answer": _render_answer(results, best_lne),
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def _render_answer(results: dict[str, dict[str, float]], best: str) -> str:
    baseline = results["kalman_current"]["latency_normalized_error"]
    improved = results[best]["latency_normalized_error"] < baseline
    if not improved:
        return "Null result: stronger priors did not beat the current Kalman baseline on latency-normalized error."
    return f"Yes in this benchmark: {best} achieved the best latency-normalized error and should be preferred over current Kalman."


def _render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# Kalman Prior Ablation",
        "",
        summary["question"],
        "",
        f"**Answer:** {summary['answer']}",
        "",
        "## Metrics",
        "",
        "| Method | Mean Error | Latency (ms) | Latency-Normalized Error | Calibration ECE |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, m in summary["metrics"].items():
        lines.append(
            f"| {name} | {m['mean_error']:.5f} | {m['latency_ms']:.3f} | {m['latency_normalized_error']:.5f} | {m['calibration_ece']:.5f} |"
        )

    lines.extend(
        [
            "",
            "## Learned Prior Fit",
            "",
            f"- fit split: `{summary['learned_prior']['fit_split']}`",
            f"- n_fit: {summary['learned_prior']['n_fit']}",
            f"- weights: {summary['learned_prior']['weights']}",
        ]
    )
    return "\n".join(lines) + "\n"
