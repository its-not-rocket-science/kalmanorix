"""Assumption-oriented stress-test slice for Kalman fusion mechanisms.

This benchmark intentionally builds a *separate* synthetic slice to pressure-test when
Kalman-style fusion should or should not help. It is not part of canonical headline
benchmarking and should be used for mechanism-level hypothesis checks.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from kalmanorix.kalman_engine.correlation import (
    correlation_inflation_factors,
    estimate_residual_correlation_profile,
)
from kalmanorix.kalman_engine.kalman_fuser import (
    kalman_fuse_diagonal_ensemble,
    kalman_fuse_structured,
)
from kalmanorix.kalman_engine.structured_covariance import StructuredCovariance


@dataclass(frozen=True)
class AssumptionStressConfig:
    random_seed: int = 41
    dimension: int = 48
    n_docs: int = 300
    n_specialists: int = 4
    n_per_case_type: int = 120
    variance_floor: float = 1e-6
    inflation_alpha: float = 1.0


@dataclass(frozen=True)
class TaggedCase:
    case_id: int
    assumption_type: str
    rationale: str


def _l2_normalize(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-12
    return x / norms


def _retrieval_metrics(queries: np.ndarray, docs: np.ndarray, targets: np.ndarray) -> dict[str, float]:
    scores = _l2_normalize(queries) @ _l2_normalize(docs).T
    ranked = np.argsort(-scores, axis=1)
    hit1 = float(np.mean(ranked[:, 0] == targets))
    hit5 = float(np.mean([t in ranked[i, :5] for i, t in enumerate(targets)]))

    rr = []
    for i, t in enumerate(targets):
        rank_pos = int(np.where(ranked[i] == t)[0][0])
        rr.append(1.0 / float(rank_pos + 1))
    return {"recall_at_1": hit1, "recall_at_5": hit5, "mrr": float(np.mean(rr))}


def _sample_case_type(
    *,
    case_type: str,
    true_queries: np.ndarray,
    rng: np.random.Generator,
) -> tuple[list[np.ndarray], np.ndarray, str]:
    """Return specialist observations + router scores + rationale for one assumption case type."""
    n_q, d = true_queries.shape
    n_specialists = 4

    if case_type == "strong_specialist_asymmetry":
        scales = np.array(
            [
                [0.05, 0.16, 0.20, 0.23],
                [0.16, 0.05, 0.20, 0.23],
                [0.18, 0.20, 0.05, 0.16],
                [0.20, 0.22, 0.17, 0.05],
            ]
        )
        dominant = rng.integers(0, n_specialists, size=n_q)
        rationale = (
            "One specialist is clearly better per-query while others are much noisier; "
            "tests if uncertainty-aware fusion and routing exploit heterogeneity."
        )
    elif case_type == "conflicting_specialists":
        scales = np.full((n_specialists, n_specialists), 0.12)
        dominant = rng.integers(0, n_specialists, size=n_q)
        rationale = (
            "Specialists are similarly noisy, but one specialist may carry an adversarial bias; "
            "tests robustness when experts disagree with structured conflict."
        )
    elif case_type == "high_redundancy":
        scales = np.full((n_specialists, n_specialists), 0.11)
        dominant = rng.integers(0, n_specialists, size=n_q)
        rationale = (
            "Experts are near-identical and share uncertainty; Kalman should offer little gain "
            "over mean when assumptions of complementary information are weak."
        )
    elif case_type == "low_redundancy_complementary":
        scales = np.array(
            [
                [0.08, 0.10, 0.15, 0.18],
                [0.10, 0.08, 0.18, 0.15],
                [0.16, 0.14, 0.08, 0.10],
                [0.14, 0.16, 0.10, 0.08],
            ]
        )
        dominant = rng.integers(0, n_specialists, size=n_q)
        rationale = (
            "No single specialist dominates all dimensions of query space; signals are complementary "
            "with lower redundancy, where Kalman-style weighting should help."
        )
    else:
        raise ValueError(f"Unknown case type: {case_type}")

    observations: list[np.ndarray] = []
    router_scores = np.zeros((n_q, n_specialists), dtype=np.float64)

    shared = rng.normal(size=(n_q, d))
    for specialist_idx in range(n_specialists):
        sigma = scales[dominant, specialist_idx][:, None]

        if case_type == "high_redundancy":
            indep = rng.normal(size=(n_q, d))
            noise = sigma * (0.95 * shared + 0.05 * indep)
        else:
            noise = sigma * rng.normal(size=(n_q, d))

        obs = true_queries + noise

        if case_type == "conflicting_specialists":
            conflict_mask = dominant == specialist_idx
            if np.any(conflict_mask):
                wrong_direction = rng.normal(size=(int(np.sum(conflict_mask)), d))
                obs[conflict_mask] += 0.25 * wrong_direction

        if case_type == "low_redundancy_complementary":
            block_size = d // n_specialists
            start = specialist_idx * block_size
            stop = d if specialist_idx == n_specialists - 1 else (specialist_idx + 1) * block_size
            obs[:, start:stop] += 0.10 * rng.normal(size=(n_q, stop - start))

        observations.append(_l2_normalize(obs))

        router_scores[:, specialist_idx] = 1.0 / (scales[dominant, specialist_idx] + 1e-6)

    router_scores += 0.2 * rng.normal(size=router_scores.shape)
    return observations, router_scores, rationale


def _fit_validation_sigma_and_corr(
    *,
    val_true: np.ndarray,
    val_observations: list[np.ndarray],
    model_names: list[str],
    variance_floor: float,
) -> tuple[list[float], np.ndarray]:
    residuals = [obs - val_true for obs in val_observations]
    sigma2 = [float(max(np.mean(r**2), variance_floor)) for r in residuals]
    residual_norms = np.column_stack([np.linalg.norm(r, axis=1) for r in residuals]).astype(np.float64)
    corr_profile = estimate_residual_correlation_profile(model_names, residual_norms)
    return sigma2, corr_profile.correlation_matrix


def _fuse_query(
    *,
    method: str,
    embs: list[np.ndarray],
    sigma2: list[float],
    corr_matrix: np.ndarray,
    inflation_alpha: float,
    router_row: np.ndarray,
) -> np.ndarray:
    if method == "mean":
        return np.mean(np.stack(embs, axis=0), axis=0)

    if method == "hard_routing":
        chosen = int(np.argmax(router_row))
        return embs[chosen]

    if method == "scalar_kalman":
        covs = [np.full_like(embs[0], s, dtype=np.float64) for s in sigma2]
        fused, _ = kalman_fuse_diagonal_ensemble(embs, covs)
        return fused

    if method == "correlation_aware_kalman":
        inflation = correlation_inflation_factors(corr_matrix, alpha=inflation_alpha)
        covs = [np.full_like(embs[0], s * inflation[i], dtype=np.float64) for i, s in enumerate(sigma2)]
        fused, _ = kalman_fuse_diagonal_ensemble(embs, covs)
        return fused

    if method == "structured_kalman":
        lowrank = np.sqrt(0.10) * np.eye(embs[0].shape[0], 1)
        structured_covs = [
            StructuredCovariance(
                np.full_like(embs[0], s, dtype=np.float64),
                lowrank_factor=lowrank,
            )
            for s in sigma2
        ]
        fused, _ = kalman_fuse_structured(embs, structured_covs)
        return fused

    raise ValueError(f"Unknown method: {method}")


def _render_report(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    by_assumption = summary["metrics_by_assumption"]

    lines = [
        "# Kalman Assumption Stress-Test Slice",
        "",
        "This artifact is isolated from canonical benchmarking and is intended for mechanism-level hypothesis testing.",
        "",
        "## Guardrails",
        "",
        "- Do not use this slice for headline claims unless findings are replicated on broader canonical benchmarks.",
        "- Use this slice to falsify/support assumptions about reliability heterogeneity and expert redundancy.",
        "",
    ]

    for assumption_name, payload in by_assumption.items():
        lines.append(f"## Assumption type: {assumption_name}")
        lines.append("")
        lines.append(f"**Stress rationale:** {payload['rationale']}")
        lines.append("")
        lines.append("| Method | Recall@1 | Recall@5 | MRR | ΔMRR vs mean |")
        lines.append("| --- | ---: | ---: | ---: | ---: |")
        mean_mrr = payload["metrics"]["mean"]["mrr"]
        for method in methods:
            m = payload["metrics"][method]
            delta = m["mrr"] - mean_mrr
            lines.append(
                f"| {method} | {m['recall_at_1']:.4f} | {m['recall_at_5']:.4f} | {m['mrr']:.4f} | {delta:+.4f} |"
            )
        lines.append("")

    lines.extend(
        [
            "## Overall slice aggregate",
            "",
            "| Method | Recall@1 | Recall@5 | MRR |",
            "| --- | ---: | ---: | ---: |",
        ]
    )
    for method, vals in summary["overall_metrics"].items():
        lines.append(
            f"| {method} | {vals['recall_at_1']:.4f} | {vals['recall_at_5']:.4f} | {vals['mrr']:.4f} |"
        )

    return "\n".join(lines) + "\n"


def run_kalman_assumption_stress_test(
    output_dir: Path = Path("results/kalman_assumption_stress_test"),
    config: AssumptionStressConfig | None = None,
) -> dict[str, Any]:
    cfg = config or AssumptionStressConfig()
    rng = np.random.default_rng(cfg.random_seed)

    docs = _l2_normalize(rng.normal(size=(cfg.n_docs, cfg.dimension)))
    case_types = [
        "strong_specialist_asymmetry",
        "conflicting_specialists",
        "high_redundancy",
        "low_redundancy_complementary",
    ]
    model_names = [f"specialist_{i}" for i in range(cfg.n_specialists)]

    methods = ["mean", "hard_routing", "scalar_kalman", "correlation_aware_kalman"]
    structured_available = True
    methods.append("structured_kalman")

    all_case_tags: list[TaggedCase] = []
    metrics_by_assumption: dict[str, dict[str, Any]] = {}
    global_method_queries: dict[str, list[np.ndarray]] = {m: [] for m in methods}
    global_targets: list[np.ndarray] = []

    case_counter = 0
    for case_type in case_types:
        n = cfg.n_per_case_type
        val_true = _l2_normalize(rng.normal(size=(n, cfg.dimension)))
        test_true = _l2_normalize(rng.normal(size=(n, cfg.dimension)))
        val_targets = np.argmax(val_true @ docs.T, axis=1)
        test_targets = np.argmax(test_true @ docs.T, axis=1)

        val_obs, _, rationale = _sample_case_type(case_type=case_type, true_queries=val_true, rng=rng)
        test_obs, test_router_scores, _ = _sample_case_type(case_type=case_type, true_queries=test_true, rng=rng)

        sigma2, corr_matrix = _fit_validation_sigma_and_corr(
            val_true=val_true,
            val_observations=val_obs,
            model_names=model_names,
            variance_floor=cfg.variance_floor,
        )

        method_queries: dict[str, np.ndarray] = {}
        for method in methods:
            fused_rows = []
            for q_idx in range(n):
                fused_rows.append(
                    _fuse_query(
                        method=method,
                        embs=[obs[q_idx] for obs in test_obs],
                        sigma2=sigma2,
                        corr_matrix=corr_matrix,
                        inflation_alpha=cfg.inflation_alpha,
                        router_row=test_router_scores[q_idx],
                    )
                )
            method_queries[method] = np.asarray(fused_rows, dtype=np.float64)
            global_method_queries[method].append(method_queries[method])

        case_metrics = {
            method: _retrieval_metrics(method_queries[method], docs, test_targets)
            for method in methods
        }

        metrics_by_assumption[case_type] = {
            "rationale": rationale,
            "n_queries": n,
            "metrics": case_metrics,
            "estimated_sigma2": sigma2,
            "estimated_correlation": corr_matrix.tolist(),
        }

        for _ in range(n):
            all_case_tags.append(
                TaggedCase(
                    case_id=case_counter,
                    assumption_type=case_type,
                    rationale=rationale,
                )
            )
            case_counter += 1

        global_targets.append(test_targets)

    combined_targets = np.concatenate(global_targets, axis=0)
    overall_metrics = {
        method: _retrieval_metrics(
            np.vstack(global_method_queries[method]),
            docs,
            combined_targets,
        )
        for method in methods
    }

    summary: dict[str, Any] = {
        "config": asdict(cfg),
        "methods": methods,
        "structured_kalman_available": structured_available,
        "notes": {
            "slice_scope": "isolated_stress_test",
            "headline_claim_guardrail": "requires_replication_on_canonical",
            "intended_use": "mechanism_level_hypothesis_testing",
        },
        "metrics_by_assumption": metrics_by_assumption,
        "overall_metrics": overall_metrics,
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "case_tags.jsonl").write_text(
        "\n".join(json.dumps(asdict(case_tag)) for case_tag in all_case_tags) + "\n",
        encoding="utf-8",
    )
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary
