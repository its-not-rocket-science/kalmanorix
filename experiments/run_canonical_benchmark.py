#!/usr/bin/env python3
"""Run the canonical mixed-domain benchmark and emit publishable artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import numpy as np

from experiments.registry.config_schema import load_experiment_config
from experiments.registry.runner import DEFAULT_REAL_SPECIALISTS, run_experiment
from kalmanorix.benchmarks.canonical_benchmark import aggregate_strategy_metrics
from kalmanorix.benchmarks.report_generator import generate_guarded_findings_markdown
from kalmanorix.benchmarks.statistical_testing import generate_statistical_report


CANONICAL_METHOD_ALIASES = {
    "MeanFuser": "mean",
    "KalmanorixFuser": "kalman",
    "hard routing baseline": "router_only_top1",
    "all-routing + mean baseline": "uniform_mean_fusion",
    "LearnedGateFuser": "learned_gate_fuser",
}

CANONICAL_DECISION_RULES = {
    "primary_metric": "ndcg@10",
    "minimum_effect_size": 0.02,
    "adjusted_p_value_threshold": 0.05,
    "max_latency_ratio_vs_mean": 1.5,
    "max_flops_ratio_vs_mean": 1.1,
}
REPORT_METRICS = [
    "ndcg@5",
    "ndcg@10",
    "mrr@5",
    "mrr@10",
    "recall@1",
    "recall@5",
    "recall@10",
    "top1_success",
]
BUCKET_SIGNIFICANCE_MIN_PAIRS = 20
CALIBRATION_MIN_VALIDATION_QUERIES = 100
PAIRED_TEST_MIN_TEST_QUERIES = 50
PER_DOMAIN_MIN_QUERIES = 20
CONFIRMATORY_SLICE_MIN_PAIRS = PAIRED_TEST_MIN_TEST_QUERIES
CONFIRMATORY_SLICE_CHOICES = (
    "high_specialist_disagreement",
    "high_uncertainty_spread",
    "nontrivial_routing_case",
    "intersection_of_above",
)


def _quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    return float(np.quantile(np.asarray(values, dtype=float), q))


def _bucket_query_ids(
    *,
    query_ids: list[str],
    query_metadata: dict[str, dict[str, Any]],
    specialist_counts: dict[str, float],
    confidence_proxy: dict[str, float],
) -> tuple[dict[str, list[str]], dict[str, float]]:
    disagreement_values = [
        float(query_metadata.get(qid, {}).get("specialist_disagreement", 0.0))
        for qid in query_ids
    ]
    uncertainty_values = [
        float(query_metadata.get(qid, {}).get("uncertainty_spread", 0.0))
        for qid in query_ids
    ]
    router_conf_values = [
        float(
            query_metadata.get(qid, {}).get(
                "router_confidence", confidence_proxy.get(qid, 0.0)
            )
        )
        for qid in query_ids
    ]
    thresholds = {
        "specialist_disagreement_median": _quantile(disagreement_values, 0.5),
        "uncertainty_spread_median": _quantile(uncertainty_values, 0.5),
        "router_confidence_low": _quantile(router_conf_values, 0.33),
        "router_confidence_high": _quantile(router_conf_values, 0.67),
    }
    buckets = {
        "high_specialist_disagreement": [],
        "low_specialist_disagreement": [],
        "high_uncertainty_spread": [],
        "low_uncertainty_spread": [],
        "single_domain_clear_winner": [],
        "true_multi_domain_queries": [],
        "router_high_confidence": [],
        "router_low_confidence": [],
    }
    for qid in query_ids:
        meta = query_metadata.get(qid, {})
        disagreement = float(meta.get("specialist_disagreement", 0.0))
        uncertainty = float(meta.get("uncertainty_spread", 0.0))
        router_conf = float(meta.get("router_confidence", confidence_proxy.get(qid, 0.0)))
        is_multi = bool(meta.get("is_multi_domain", False))
        if not is_multi and float(specialist_counts.get(qid, 1.0)) > 1.0:
            is_multi = True

        if disagreement >= thresholds["specialist_disagreement_median"]:
            buckets["high_specialist_disagreement"].append(qid)
        else:
            buckets["low_specialist_disagreement"].append(qid)
        if uncertainty >= thresholds["uncertainty_spread_median"]:
            buckets["high_uncertainty_spread"].append(qid)
        else:
            buckets["low_uncertainty_spread"].append(qid)
        if is_multi:
            buckets["true_multi_domain_queries"].append(qid)
        if (not is_multi) and router_conf >= thresholds["router_confidence_high"]:
            buckets["single_domain_clear_winner"].append(qid)
        if router_conf >= thresholds["router_confidence_high"]:
            buckets["router_high_confidence"].append(qid)
        if router_conf <= thresholds["router_confidence_low"]:
            buckets["router_low_confidence"].append(qid)
    return buckets, thresholds


def _build_bucket_report(
    *,
    methods: dict[str, Any],
    query_ids: list[str],
    bucket_to_qids: dict[str, list[str]],
    seed: int,
    num_resamples: int,
) -> dict[str, Any]:
    method_keys = ["mean", "kalman", "router_only_top1", "router_only_topk_mean"]
    qid_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    bucket_payload: dict[str, Any] = {}
    consistent_gain_buckets: list[str] = []

    for bucket, qids in bucket_to_qids.items():
        idxs = [qid_to_idx[qid] for qid in qids if qid in qid_to_idx]
        per_method_metrics: dict[str, dict[str, float]] = {}
        for method in method_keys:
            metrics = methods[method]["query_level"]
            per_method_metrics[method] = {
                metric: (
                    float(np.mean([metrics[metric][idx] for idx in idxs])) if idxs else 0.0
                )
                for metric in REPORT_METRICS
            }

        n_pairs = len(idxs)
        comparisons: dict[str, Any] = {}
        for baseline in ["mean", "router_only_top1", "router_only_topk_mean"]:
            label = f"kalman_vs_{baseline}"
            if n_pairs < BUCKET_SIGNIFICANCE_MIN_PAIRS:
                comparisons[label] = {
                    "status": "exploratory_only",
                    "reason": f"n_pairs={n_pairs} < {BUCKET_SIGNIFICANCE_MIN_PAIRS}",
                }
                continue
            kalman_subset = {
                metric: [methods["kalman"]["query_level"][metric][idx] for idx in idxs]
                for metric in REPORT_METRICS
            }
            baseline_subset = {
                metric: [methods[baseline]["query_level"][metric][idx] for idx in idxs]
                for metric in REPORT_METRICS
            }
            stats = generate_statistical_report(
                reference_method="kalman",
                candidate_method=baseline,
                reference_metrics=kalman_subset,
                candidate_metrics=baseline_subset,
                seed=seed,
                num_resamples=num_resamples,
                config={"bucket": bucket},
            )
            ndcg10 = stats.comparisons["ndcg@10"]
            comparisons[label] = {
                "status": "inferential",
                "metric": "ndcg@10",
                "mean_difference": ndcg10.mean_difference,
                "ci95_low": ndcg10.confidence_interval.lower,
                "ci95_high": ndcg10.confidence_interval.upper,
                "p_value": ndcg10.p_value,
                "adjusted_p_value": ndcg10.adjusted_p_value,
                "n_pairs": ndcg10.n_pairs,
            }

        deltas = {
            baseline: per_method_metrics["kalman"]["ndcg@10"]
            - per_method_metrics[baseline]["ndcg@10"]
            for baseline in ["mean", "router_only_top1", "router_only_topk_mean"]
        }
        is_consistent_gain = all(delta > 0.0 for delta in deltas.values())
        if is_consistent_gain:
            inferential = [
                comparisons[f"kalman_vs_{baseline}"]
                for baseline in ["mean", "router_only_top1", "router_only_topk_mean"]
                if comparisons[f"kalman_vs_{baseline}"]["status"] == "inferential"
            ]
            if inferential and all(c["adjusted_p_value"] <= 0.05 for c in inferential):
                consistent_gain_buckets.append(bucket)

        bucket_payload[bucket] = {
            "n_queries": n_pairs,
            "method_metrics": per_method_metrics,
            "kalman_ndcg10_deltas": deltas,
            "comparisons": comparisons,
            "consistency_flag": (
                "consistent_gain"
                if is_consistent_gain
                else "mixed_or_no_gain"
            ),
            "exploratory_only": n_pairs < BUCKET_SIGNIFICANCE_MIN_PAIRS,
        }

    return {
        "minimum_pairs_for_significance": BUCKET_SIGNIFICANCE_MIN_PAIRS,
        "buckets": bucket_payload,
        "consistent_kalman_gain_buckets": consistent_gain_buckets,
    }


def _resolve_confirmatory_slice_ids(
    *,
    slice_name: str,
    query_ids: list[str],
    query_metadata: dict[str, dict[str, Any]],
    specialist_counts: dict[str, float],
    confidence_proxy: dict[str, float],
    bucket_to_qids: dict[str, list[str]],
    bucket_thresholds: dict[str, float],
) -> list[str]:
    if slice_name == "high_specialist_disagreement":
        return list(bucket_to_qids.get("high_specialist_disagreement", []))
    if slice_name == "high_uncertainty_spread":
        return list(bucket_to_qids.get("high_uncertainty_spread", []))

    nontrivial_routing_case: list[str] = []
    high_disagreement = set(bucket_to_qids.get("high_specialist_disagreement", []))
    high_uncertainty = set(bucket_to_qids.get("high_uncertainty_spread", []))
    multi_domain = set(bucket_to_qids.get("true_multi_domain_queries", []))
    router_low = set(bucket_to_qids.get("router_low_confidence", []))
    for qid in query_ids:
        meta = query_metadata.get(qid, {})
        router_conf = float(meta.get("router_confidence", confidence_proxy.get(qid, 0.0)))
        selected_count = float(specialist_counts.get(qid, 1.0))
        is_nontrivial = (
            qid in multi_domain
            or qid in router_low
            or qid in high_disagreement
            or qid in high_uncertainty
            or selected_count > 1.0
            or router_conf <= float(bucket_thresholds["router_confidence_low"])
        )
        if is_nontrivial:
            nontrivial_routing_case.append(qid)

    if slice_name == "nontrivial_routing_case":
        return nontrivial_routing_case
    if slice_name == "intersection_of_above":
        nontrivial = set(nontrivial_routing_case)
        return sorted(high_disagreement.intersection(high_uncertainty).intersection(nontrivial))
    raise ValueError(
        "Unknown confirmatory slice name: "
        f"{slice_name}. Expected one of {list(CONFIRMATORY_SLICE_CHOICES)}."
    )


def _build_confirmatory_slice_results(
    *,
    slice_name: str,
    methods: dict[str, Any],
    query_ids: list[str],
    selected_qids: list[str],
    seed: int,
    num_resamples: int,
) -> dict[str, Any]:
    qid_to_idx = {qid: idx for idx, qid in enumerate(query_ids)}
    idxs = [qid_to_idx[qid] for qid in selected_qids if qid in qid_to_idx]
    n_pairs = len(idxs)
    warnings: list[str] = []
    if n_pairs == 0:
        warnings.append(
            "Confirmatory slice contains zero paired queries; inferential testing is skipped."
        )
    elif n_pairs < CONFIRMATORY_SLICE_MIN_PAIRS:
        warnings.append(
            "Confirmatory slice is underpowered for inferential claims "
            f"(n_pairs={n_pairs} < {CONFIRMATORY_SLICE_MIN_PAIRS})."
        )

    method_keys = ["mean", "kalman", "router_only_top1", "router_only_topk_mean"]
    method_metrics: dict[str, dict[str, float]] = {}
    for method in method_keys:
        method_qlevel = methods[method]["query_level"]
        method_metrics[method] = {
            metric: (
                float(np.mean([method_qlevel[metric][idx] for idx in idxs])) if idxs else 0.0
            )
            for metric in REPORT_METRICS
        }

    paired_statistics: dict[str, Any] | None = None
    if n_pairs >= CONFIRMATORY_SLICE_MIN_PAIRS:
        kalman_subset = {
            metric: [methods["kalman"]["query_level"][metric][idx] for idx in idxs]
            for metric in REPORT_METRICS
        }
        mean_subset = {
            metric: [methods["mean"]["query_level"][metric][idx] for idx in idxs]
            for metric in REPORT_METRICS
        }
        stats = generate_statistical_report(
            reference_method="kalman",
            candidate_method="mean",
            reference_metrics=kalman_subset,
            candidate_metrics=mean_subset,
            seed=seed,
            num_resamples=num_resamples,
            config={"confirmatory_slice": slice_name},
        )
        paired_statistics = {
            metric: {
                "mean_difference": entry.mean_difference,
                "ci95_low": entry.confidence_interval.lower,
                "ci95_high": entry.confidence_interval.upper,
                "p_value": entry.p_value,
                "adjusted_p_value": entry.adjusted_p_value,
                "cohen_dz": entry.effect_size.cohen_dz,
                "rank_biserial": entry.effect_size.rank_biserial,
                "n_pairs": entry.n_pairs,
            }
            for metric, entry in stats.comparisons.items()
        }

    return {
        "slice_name": slice_name,
        "n_pairs": n_pairs,
        "minimum_pairs_for_inference": CONFIRMATORY_SLICE_MIN_PAIRS,
        "warning_count": len(warnings),
        "warnings": warnings,
        "method_metrics": method_metrics,
        "paired_statistics_kalman_vs_mean": paired_statistics,
    }


def _classify_kalman_vs_mean(summary: dict[str, Any]) -> dict[str, Any]:
    rules = CANONICAL_DECISION_RULES
    stats = summary["paired_statistics"]["kalman_vs_mean"]["overall"]
    methods = summary["methods"]
    primary_metric = rules["primary_metric"]
    primary_stats = stats[primary_metric]

    effect_delta = float(primary_stats["mean_difference"])
    adjusted_p_value = float(primary_stats["adjusted_p_value"])
    latency_ratio = float(
        methods["kalman"]["metrics"]["latency_ms"]["mean"]
        / methods["mean"]["metrics"]["latency_ms"]["mean"]
    )
    flops_ratio = float(
        methods["kalman"]["metrics"]["flops_proxy"]["mean"]
        / methods["mean"]["metrics"]["flops_proxy"]["mean"]
    )

    checks = {
        "effect_size_ok": effect_delta >= float(rules["minimum_effect_size"]),
        "adjusted_p_value_ok": adjusted_p_value
        <= float(rules["adjusted_p_value_threshold"]),
        "latency_ratio_ok": latency_ratio <= float(rules["max_latency_ratio_vs_mean"]),
        "flops_ratio_ok": flops_ratio <= float(rules["max_flops_ratio_vs_mean"]),
    }

    power_diag = summary["power_diagnostics"]["kalman_vs_mean"]
    sufficiently_powered = bool(power_diag["is_sufficiently_powered_for_target_effect"])

    if all(checks.values()):
        verdict = "supported"
    elif effect_delta <= 0.0 and adjusted_p_value <= float(
        rules["adjusted_p_value_threshold"]
    ):
        verdict = "unsupported"
    elif sufficiently_powered:
        verdict = "inconclusive_sufficiently_powered"
    else:
        verdict = "inconclusive_underpowered"

    return {
        "verdict": verdict,
        "rules": rules,
        "observed": {
            "primary_metric_delta": effect_delta,
            "primary_metric_adjusted_p_value": adjusted_p_value,
            "latency_ratio_vs_mean": latency_ratio,
            "flops_ratio_vs_mean": flops_ratio,
        },
        "checks": checks,
    }


def _load_split_counts(benchmark_path: Path) -> dict[str, int]:
    pyarrow_available = importlib.util.find_spec("pyarrow") is not None
    if pyarrow_available:
        import pyarrow.parquet as pq

        rows = pq.read_table(benchmark_path, columns=["split"]).to_pylist()
    else:
        rows = json.loads(benchmark_path.read_text(encoding="utf-8"))
    counts: dict[str, int] = {}
    for row in rows:
        split = str(row.get("split"))
        counts[split] = counts.get(split, 0) + 1
    required = {"train", "validation", "test"}
    missing = sorted(required.difference(counts))
    if missing:
        raise ValueError(
            f"Benchmark must contain train/validation/test splits; missing={missing}"
        )
    return counts


def _by_domain(
    query_ids: list[str], values: list[float], domains: dict[str, str]
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {}
    for qid, value in zip(query_ids, values, strict=True):
        out.setdefault(domains[qid], []).append(float(value))
    return out


def _render_report(summary: dict[str, Any]) -> str:
    methods = summary["methods"]
    stats = summary["paired_statistics"]["kalman_vs_mean"]
    decision = summary["decision"]["kalman_vs_mean"]
    rules = decision["rules"]
    observed = decision["observed"]
    checks = decision["checks"]
    verdict = decision["verdict"]
    power_diag = summary["power_diagnostics"]["kalman_vs_mean"]
    adequacy = summary["sample_size_adequacy"]

    lines = [
        "# Canonical Benchmark Report",
        "",
        "## Setup",
        f"- Benchmark: `{summary['benchmark']['path']}`",
        f"- Split evaluated: `{summary['benchmark']['evaluated_split']}`",
        f"- Available split counts: {summary['benchmark']['split_counts']}",
        f"- Specialists: {', '.join(summary['specialists'])}",
        f"- LearnedGateFuser included: `{summary['comparisons']['LearnedGateFuser']['included']}`",
        "",
        "## Aggregate Metrics (mean with 95% bootstrap CI)",
        "",
        "| Method | nDCG@5 | nDCG@10 | MRR@5 | MRR@10 | Recall@1 | Recall@5 | Recall@10 | Top-1 success | Latency (ms) | FLOPs proxy |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]

    ranking_by_ndcg10 = sorted(
        (
            (name, payload["metrics"]["ndcg@10"]["mean"])
            for name, payload in methods.items()
        ),
        key=lambda x: x[1],
        reverse=True,
    )

    for method_label, key in CANONICAL_METHOD_ALIASES.items():
        if key not in methods:
            continue
        payload = methods[key]["metrics"]
        lines.append(
            "| {label} | {ndcg5:.4f} [{ndcg5_l:.4f}, {ndcg5_h:.4f}] | {ndcg10:.4f} [{ndcg10_l:.4f}, {ndcg10_h:.4f}] | {mrr5:.4f} [{mrr5_l:.4f}, {mrr5_h:.4f}] | {mrr10:.4f} [{mrr10_l:.4f}, {mrr10_h:.4f}] | {rec1:.4f} [{rec1_l:.4f}, {rec1_h:.4f}] | {rec5:.4f} [{rec5_l:.4f}, {rec5_h:.4f}] | {rec10:.4f} [{rec10_l:.4f}, {rec10_h:.4f}] | {top1:.4f} [{top1_l:.4f}, {top1_h:.4f}] | {lat:.3f} [{lat_l:.3f}, {lat_h:.3f}] | {flops:.3f} [{flops_l:.3f}, {flops_h:.3f}] |".format(
                label=method_label,
                ndcg5=payload["ndcg@5"]["mean"],
                ndcg5_l=payload["ndcg@5"]["ci95_low"],
                ndcg5_h=payload["ndcg@5"]["ci95_high"],
                ndcg10=payload["ndcg@10"]["mean"],
                ndcg10_l=payload["ndcg@10"]["ci95_low"],
                ndcg10_h=payload["ndcg@10"]["ci95_high"],
                mrr5=payload["mrr@5"]["mean"],
                mrr5_l=payload["mrr@5"]["ci95_low"],
                mrr5_h=payload["mrr@5"]["ci95_high"],
                mrr10=payload["mrr@10"]["mean"],
                mrr10_l=payload["mrr@10"]["ci95_low"],
                mrr10_h=payload["mrr@10"]["ci95_high"],
                rec1=payload["recall@1"]["mean"],
                rec1_l=payload["recall@1"]["ci95_low"],
                rec1_h=payload["recall@1"]["ci95_high"],
                rec5=payload["recall@5"]["mean"],
                rec5_l=payload["recall@5"]["ci95_low"],
                rec5_h=payload["recall@5"]["ci95_high"],
                rec10=payload["recall@10"]["mean"],
                rec10_l=payload["recall@10"]["ci95_low"],
                rec10_h=payload["recall@10"]["ci95_high"],
                top1=payload["top1_success"]["mean"],
                top1_l=payload["top1_success"]["ci95_low"],
                top1_h=payload["top1_success"]["ci95_high"],
                lat=payload["latency_ms"]["mean"],
                lat_l=payload["latency_ms"]["ci95_low"],
                lat_h=payload["latency_ms"]["ci95_high"],
                flops=payload["flops_proxy"]["mean"],
                flops_l=payload["flops_proxy"]["ci95_low"],
                flops_h=payload["flops_proxy"]["ci95_high"],
            )
        )

    lines.extend(
        [
            "",
            "## Method Ranking Snapshot",
            "",
            "- Ranking by nDCG@10 (higher is better): "
            + " > ".join(f"`{name}` ({score:.4f})" for name, score in ranking_by_ndcg10),
            "",
            "## Decision Framework: KalmanorixFuser vs MeanFuser",
            "",
            "| Rule | Threshold | Observed | Pass |",
            "|---|---:|---:|---|",
            "| Primary metric (nDCG@10 Δ mean) | >= {threshold:.4f} | {observed_value:.6f} | {passed} |".format(
                threshold=rules["minimum_effect_size"],
                observed_value=observed["primary_metric_delta"],
                passed="yes" if checks["effect_size_ok"] else "no",
            ),
            "| Adjusted p-value (Holm) | <= {threshold:.4f} | {observed_value:.6f} | {passed} |".format(
                threshold=rules["adjusted_p_value_threshold"],
                observed_value=observed["primary_metric_adjusted_p_value"],
                passed="yes" if checks["adjusted_p_value_ok"] else "no",
            ),
            "| Latency ratio (Kalman/Mean) | <= {threshold:.3f} | {observed_value:.3f} | {passed} |".format(
                threshold=rules["max_latency_ratio_vs_mean"],
                observed_value=observed["latency_ratio_vs_mean"],
                passed="yes" if checks["latency_ratio_ok"] else "no",
            ),
            "| FLOPs ratio (Kalman/Mean) | <= {threshold:.3f} | {observed_value:.3f} | {passed} |".format(
                threshold=rules["max_flops_ratio_vs_mean"],
                observed_value=observed["flops_ratio_vs_mean"],
                passed="yes" if checks["flops_ratio_ok"] else "no",
            ),
            "",
            "## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)",
            "",
            f"- Number of evaluated test queries: **{power_diag['num_test_queries']}**",
            f"- Per-domain evaluated test counts: `{power_diag['per_domain_test_counts']}`",
            f"- Observed primary effect size (nDCG@10 Δ mean): `{power_diag['observed_effect_size']:.6f}`",
            f"- Detectable effect threshold estimate (80% power, α=0.05, paired-normal approximation): `{power_diag['detectable_effect_threshold_estimate']:.6f}`",
            f"- Target effect for decision rule: `{power_diag['target_effect_size']:.6f}`",
            f"- Sufficiently powered for target effect: `{power_diag['is_sufficiently_powered_for_target_effect']}`",
            "",
            "## Sample Size Adequacy Checks",
            "",
            "| Use case | Available | Minimum | Adequate | Notes |",
            "|---|---:|---:|---|---|",
            "| Uncertainty calibration (validation split) | {avail} | {minimum} | {ok} | {note} |".format(
                avail=adequacy["uncertainty_calibration"]["available_queries"],
                minimum=adequacy["uncertainty_calibration"]["minimum_required"],
                ok="yes" if adequacy["uncertainty_calibration"]["adequate"] else "no",
                note=adequacy["uncertainty_calibration"]["note"],
            ),
            "| Paired significance testing (test split) | {avail} | {minimum} | {ok} | {note} |".format(
                avail=adequacy["paired_significance_testing"]["available_queries"],
                minimum=adequacy["paired_significance_testing"]["minimum_required"],
                ok="yes" if adequacy["paired_significance_testing"]["adequate"] else "no",
                note=adequacy["paired_significance_testing"]["note"],
            ),
            "| Per-domain analysis (min test queries in any domain) | {avail} | {minimum} | {ok} | {note} |".format(
                avail=adequacy["per_domain_analysis"]["minimum_domain_count"],
                minimum=adequacy["per_domain_analysis"]["minimum_required_per_domain"],
                ok="yes" if adequacy["per_domain_analysis"]["adequate"] else "no",
                note=adequacy["per_domain_analysis"]["note"],
            ),
            "",
            "## Paired Statistical Test: KalmanorixFuser vs MeanFuser",
            "",
            "| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for metric in REPORT_METRICS:
        metric_payload = stats["overall"][metric]
        lines.append(
            "| {metric} | {delta:.6f} | [{low:.6f}, {high:.6f}] | {p:.6f} | {padj:.6f} |".format(
                metric=metric,
                delta=metric_payload["mean_difference"],
                low=metric_payload["ci95_low"],
                high=metric_payload["ci95_high"],
                p=metric_payload["p_value"],
                padj=metric_payload["adjusted_p_value"],
            )
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- **kalman_vs_mean:** `{verdict}`",
            "- Rule logic: `supported` if all checks pass; `unsupported` if nDCG@10 Δ <= 0 and Holm-adjusted p <= threshold; otherwise inconclusive is split into `inconclusive_underpowered` vs `inconclusive_sufficiently_powered` from the detectable-effect threshold estimate.",
        ]
    )
    confirmatory = summary.get("confirmatory_slice_results")
    if confirmatory is not None:
        lines.extend(
            [
                "",
                "## Confirmatory Slice (Kalman-vs-Mean)",
                "",
                f"- Slice name: `{confirmatory['slice_name']}`",
                f"- Paired query count: `{confirmatory['n_pairs']}`",
                f"- Minimum paired queries for inferential claims: `{confirmatory['minimum_pairs_for_inference']}`",
            ]
        )
        if confirmatory["warnings"]:
            lines.append("- Warnings:")
            lines.extend(f"  - {warning}" for warning in confirmatory["warnings"])
        lines.extend(
            [
                "",
                "| Method | nDCG@10 |",
                "|---|---:|",
            ]
        )
        for method in ["mean", "kalman", "router_only_top1", "router_only_topk_mean"]:
            value = confirmatory["method_metrics"][method]["ndcg@10"]
            lines.append(f"| {method} | {value:.4f} |")

        slice_stats = confirmatory.get("paired_statistics_kalman_vs_mean")
        if slice_stats is None:
            lines.append(
                "- Inferential paired testing was not run for the confirmatory slice due to insufficient paired queries."
            )
        else:
            lines.extend(
                [
                    "",
                    "### Confirmatory paired statistical test: KalmanorixFuser vs MeanFuser",
                    "",
                    "| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |",
                    "|---|---:|---|---:|---:|",
                ]
            )
            for metric in REPORT_METRICS:
                payload = slice_stats[metric]
                lines.append(
                    "| {metric} | {delta:.6f} | [{low:.6f}, {high:.6f}] | {p:.6f} | {padj:.6f} |".format(
                        metric=metric,
                        delta=payload["mean_difference"],
                        low=payload["ci95_low"],
                        high=payload["ci95_high"],
                        p=payload["p_value"],
                        padj=payload["adjusted_p_value"],
                    )
                )
    bucket_analysis = summary.get("bucket_analysis", {})
    if bucket_analysis:
        lines.extend(
            [
                "",
                "## Bucketed Analysis (Exploratory unless significance criteria are met)",
                "",
                "| Bucket | n | Mean | Kalman | Hard routing | Top-k mean | Δ(K-M) nDCG@10 | Δ(K-Hard) | Δ(K-TopK) | Significance status |",
                "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|",
            ]
        )
        for bucket, payload in bucket_analysis["buckets"].items():
            metrics = payload["method_metrics"]
            comps = payload["comparisons"]["kalman_vs_mean"]
            sig_status = comps["status"]
            if sig_status == "inferential":
                sig_status = f"inferential (adj p={comps['adjusted_p_value']:.4f})"
            lines.append(
                "| {bucket} | {n} | {mean:.4f} | {kalman:.4f} | {hard:.4f} | {topk:.4f} | {dkm:.4f} | {dkh:.4f} | {dkt:.4f} | {sig_status} |".format(
                    bucket=bucket,
                    n=payload["n_queries"],
                    mean=metrics["mean"]["ndcg@10"],
                    kalman=metrics["kalman"]["ndcg@10"],
                    hard=metrics["router_only_top1"]["ndcg@10"],
                    topk=metrics["router_only_topk_mean"]["ndcg@10"],
                    dkm=payload["kalman_ndcg10_deltas"]["mean"],
                    dkh=payload["kalman_ndcg10_deltas"]["router_only_top1"],
                    dkt=payload["kalman_ndcg10_deltas"]["router_only_topk_mean"],
                    sig_status=sig_status,
                )
            )
        lines.extend(["", "### Buckets with consistent Kalman gains"])
        buckets = bucket_analysis.get("consistent_kalman_gain_buckets", [])
        if buckets:
            lines.extend([f"- `{bucket}`" for bucket in buckets])
        else:
            lines.append("- None met the consistency + inferential significance criteria in this run.")
        lines.append(
            "- These subgroup findings are secondary and must not be promoted to headline claims without dedicated confirmatory evaluation."
        )
    if not summary["comparisons"]["LearnedGateFuser"]["included"]:
        lines.append(
            f"- LearnedGateFuser omitted: {summary['comparisons']['LearnedGateFuser']['reason']}"
        )
    lines.append(
        "- This report is descriptive for the configured setup and should not be generalized beyond it."
    )
    significance_rows = [
        {
            "reference": "kalman",
            "candidate": "mean",
            "metric": metric,
            "mean_diff": stats["overall"][metric]["mean_difference"],
            "adjusted_p_value": stats["overall"][metric]["adjusted_p_value"],
        }
        for metric in REPORT_METRICS
    ]
    lines.extend(
        [
            "",
            generate_guarded_findings_markdown(
                significance_rows=significance_rows,
                benchmark_limitations=[
                    "Evaluation is restricted to the selected benchmark split and may not cover broader deployment distributions.",
                    "Latency and FLOPs values are proxy measurements and should not be treated as universal throughput guarantees.",
                ],
            ),
        ]
    )
    return "\n".join(lines) + "\n"


def run_canonical_benchmark(
    *,
    benchmark_path: Path,
    output_dir: Path,
    split: str,
    max_queries: int | None,
    device: str,
    seed: int,
    num_resamples: int,
    confirmatory_slice: str | None = None,
) -> dict[str, Any]:
    split_counts = _load_split_counts(benchmark_path)

    config_payload = {
        "name": "canonical-benchmark",
        "experiment_type": "real_mixed",
        "seed": {"python": seed, "numpy": seed, "torch": seed},
        "artifacts": {
            "summary_json": str(output_dir / "runner_summary.json"),
            "details_json": str(output_dir / "runner_details.json"),
        },
        "dataset": {
            "kind": "mixed_parquet",
            "path": str(benchmark_path),
            "split": split,
            "max_queries": max_queries,
        },
        "models": {
            "kind": "hf_specialists",
            "device": device,
            "specialists": DEFAULT_REAL_SPECIALISTS,
        },
        "fusion": {
            "strategies": ["mean", "kalman"],
            "routing_mode": "all",
        },
        "evaluation": {"kind": "locked_protocol"},
        "reporting": {"print_stdout": False},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "canonical.json"
        cfg_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        cfg = load_experiment_config(cfg_path)
        details = run_experiment(cfg)

    (output_dir / "runner_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    query_level = details["query_level"]
    rankings = query_level["rankings"]
    ground_truth = {qid: set(ids) for qid, ids in query_level["ground_truth"].items()}
    domains = query_level["domains"]
    latencies = query_level["latency_ms"]
    flops_proxy = query_level["specialist_count_selected"]
    query_metadata = query_level.get("query_metadata", {})

    methods: dict[str, Any] = {}
    for method_key in sorted(rankings):
        methods[method_key] = aggregate_strategy_metrics(
            rankings=rankings[method_key],
            ground_truth=ground_truth,
            latency_ms=latencies[method_key],
            flops_proxy=flops_proxy[method_key],
            seed=seed,
            num_resamples=num_resamples,
        )

    required_methods = {
        "mean",
        "kalman",
        "router_only_top1",
        "router_only_topk_mean",
        "uniform_mean_fusion",
    }
    missing_required = sorted(required_methods.difference(methods))
    if missing_required:
        raise ValueError(
            "Canonical benchmark requires MeanFuser, KalmanorixFuser, hard-routing, "
            "top-k-mean routing baseline, and all-routing+mean baselines. "
            f"Missing strategies: {missing_required}"
        )

    ordered_qids = sorted(rankings["kalman"])
    per_domain_test_counts: dict[str, int] = {}
    for qid in ordered_qids:
        domain = str(domains[qid])
        per_domain_test_counts[domain] = per_domain_test_counts.get(domain, 0) + 1

    kalman_metrics = methods["kalman"]["query_level"]
    mean_metrics = methods["mean"]["query_level"]
    kalman_domain = {
        metric: _by_domain(ordered_qids, values, domains)
        for metric, values in kalman_metrics.items()
    }
    mean_domain = {
        metric: _by_domain(ordered_qids, values, domains)
        for metric, values in mean_metrics.items()
    }

    paired = generate_statistical_report(
        reference_method="kalman",
        candidate_method="mean",
        reference_metrics=kalman_metrics,
        candidate_metrics=mean_metrics,
        reference_metrics_by_domain=kalman_domain,
        candidate_metrics_by_domain=mean_domain,
        seed=seed,
        num_resamples=num_resamples,
        config={
            "benchmark": str(benchmark_path),
            "evaluated_split": split,
        },
    )

    paired_summary = {
        "overall": {
            metric: {
                "mean_difference": entry.mean_difference,
                "ci95_low": entry.confidence_interval.lower,
                "ci95_high": entry.confidence_interval.upper,
                "p_value": entry.p_value,
                "adjusted_p_value": entry.adjusted_p_value,
                "cohen_dz": entry.effect_size.cohen_dz,
                "rank_biserial": entry.effect_size.rank_biserial,
                "n_pairs": entry.n_pairs,
            }
            for metric, entry in paired.comparisons.items()
        },
        "domains": {
            domain: {
                metric: {
                    "mean_difference": entry.mean_difference,
                    "ci95_low": entry.confidence_interval.lower,
                    "ci95_high": entry.confidence_interval.upper,
                    "p_value": entry.p_value,
                    "adjusted_p_value": entry.adjusted_p_value,
                    "n_pairs": entry.n_pairs,
                }
                for metric, entry in domain_report.metrics.items()
            }
            for domain, domain_report in paired.domains.items()
            if domain != "overall"
        },
        "configuration_hash": paired.configuration_hash,
    }
    primary_metric = CANONICAL_DECISION_RULES["primary_metric"]
    primary_stats = paired_summary["overall"][primary_metric]
    primary_deltas = np.asarray(kalman_metrics[primary_metric], dtype=float) - np.asarray(
        mean_metrics[primary_metric], dtype=float
    )
    n_test = int(len(primary_deltas))
    std_delta = float(np.std(primary_deltas, ddof=1)) if n_test > 1 else 0.0
    detectable_threshold = (
        float((1.96 + 0.84) * (std_delta / np.sqrt(n_test)))
        if n_test > 1
        else float("inf")
    )
    power_diagnostics = {
        "kalman_vs_mean": {
            "num_test_queries": n_test,
            "per_domain_test_counts": dict(sorted(per_domain_test_counts.items())),
            "observed_effect_size": float(primary_stats["mean_difference"]),
            "target_effect_size": float(CANONICAL_DECISION_RULES["minimum_effect_size"]),
            "detectable_effect_threshold_estimate": detectable_threshold,
            "paired_delta_stddev": std_delta,
            "power_approximation": "detectable_effect ≈ (1.96+0.84)*std(delta)/sqrt(n)",
            "is_sufficiently_powered_for_target_effect": bool(
                detectable_threshold
                <= float(CANONICAL_DECISION_RULES["minimum_effect_size"])
            ),
        }
    }

    min_domain_count = min(per_domain_test_counts.values()) if per_domain_test_counts else 0
    sample_size_adequacy = {
        "uncertainty_calibration": {
            "available_queries": int(split_counts["validation"]),
            "minimum_required": CALIBRATION_MIN_VALIDATION_QUERIES,
            "adequate": bool(split_counts["validation"] >= CALIBRATION_MIN_VALIDATION_QUERIES),
            "note": "Validation split size governs stability of uncertainty calibration.",
        },
        "paired_significance_testing": {
            "available_queries": n_test,
            "minimum_required": PAIRED_TEST_MIN_TEST_QUERIES,
            "adequate": bool(n_test >= PAIRED_TEST_MIN_TEST_QUERIES),
            "note": "Test split paired query count governs inferential precision.",
        },
        "per_domain_analysis": {
            "minimum_domain_count": int(min_domain_count),
            "per_domain_counts": dict(sorted(per_domain_test_counts.items())),
            "minimum_required_per_domain": PER_DOMAIN_MIN_QUERIES,
            "adequate": bool(min_domain_count >= PER_DOMAIN_MIN_QUERIES),
            "note": "Lowest-count domain determines whether per-domain inference is stable.",
        },
    }

    bucket_to_qids, bucket_thresholds = _bucket_query_ids(
        query_ids=ordered_qids,
        query_metadata=query_metadata,
        specialist_counts=query_level.get("specialist_count_selected", {}).get(
            "router_only_top1", {}
        ),
        confidence_proxy=query_level.get("confidence_proxy", {}).get(
            "router_only_top1", {}
        ),
    )
    bucket_analysis = _build_bucket_report(
        methods=methods,
        query_ids=ordered_qids,
        bucket_to_qids=bucket_to_qids,
        seed=seed + 10_000,
        num_resamples=num_resamples,
    )
    bucket_analysis["definitions"] = {
        "high_specialist_disagreement": "specialist_disagreement >= median (entropy-normalized precision distribution across specialists)",
        "low_specialist_disagreement": "specialist_disagreement < median",
        "high_uncertainty_spread": "uncertainty_spread >= median (max sigma2 - min sigma2 across specialists)",
        "low_uncertainty_spread": "uncertainty_spread < median",
        "single_domain_clear_winner": "is_multi_domain == False and router_confidence >= high-confidence threshold",
        "true_multi_domain_queries": "is_multi_domain == True (secondary_domain exists and differs from dominant_domain, with fallback to selected specialist count > 1)",
        "router_high_confidence": "router_confidence >= 67th percentile",
        "router_low_confidence": "router_confidence <= 33rd percentile",
    }
    bucket_analysis["thresholds"] = bucket_thresholds

    confirmatory_slice_results: dict[str, Any] | None = None
    if confirmatory_slice is not None:
        selected_qids = _resolve_confirmatory_slice_ids(
            slice_name=confirmatory_slice,
            query_ids=ordered_qids,
            query_metadata=query_metadata,
            specialist_counts=query_level.get("specialist_count_selected", {}).get(
                "router_only_top1", {}
            ),
            confidence_proxy=query_level.get("confidence_proxy", {}).get(
                "router_only_top1", {}
            ),
            bucket_to_qids=bucket_to_qids,
            bucket_thresholds=bucket_thresholds,
        )
        confirmatory_slice_results = _build_confirmatory_slice_results(
            slice_name=confirmatory_slice,
            methods=methods,
            query_ids=ordered_qids,
            selected_qids=selected_qids,
            seed=seed + 20_000,
            num_resamples=num_resamples,
        )

    summary = {
        "benchmark": {
            "path": str(benchmark_path),
            "evaluated_split": split,
            "split_counts": split_counts,
            "max_queries": max_queries,
        },
        "seed": seed,
        "num_resamples": num_resamples,
        "specialists": [spec["name"] for spec in DEFAULT_REAL_SPECIALISTS],
        "comparisons": {
            name: {
                "strategy_key": key,
                "included": key in methods,
                "reason": (
                    "present"
                    if key in methods
                    else (
                        "LearnedGateFuser requires a two-specialist setup; current run uses "
                        f"{len(DEFAULT_REAL_SPECIALISTS)} specialists"
                        if name == "LearnedGateFuser"
                        else "strategy not emitted by benchmark runner"
                    )
                ),
            }
            for name, key in CANONICAL_METHOD_ALIASES.items()
        },
        "methods": methods,
        "paired_statistics": {"kalman_vs_mean": paired_summary},
        "power_diagnostics": power_diagnostics,
        "sample_size_adequacy": sample_size_adequacy,
        "bucket_analysis": bucket_analysis,
        "confirmatory_slice_results": confirmatory_slice_results,
    }
    summary["decision"] = {"kalman_vs_mean": _classify_kalman_vs_mean(summary)}

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(_render_report(summary), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical benchmark and generate report artifacts"
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.1.0/mixed_benchmark.json"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-queries", type=int, default=600)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-resamples", type=int, default=5000)
    parser.add_argument(
        "--confirmatory-slice",
        type=str,
        choices=CONFIRMATORY_SLICE_CHOICES,
        default=None,
        help="Named confirmatory evaluation slice for Kalman-vs-mean testing.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/canonical_benchmark_v2")
    )
    args = parser.parse_args()

    summary = run_canonical_benchmark(
        benchmark_path=args.benchmark_path,
        output_dir=args.output_dir,
        split=args.split,
        max_queries=args.max_queries,
        device=args.device,
        seed=args.seed,
        num_resamples=args.num_resamples,
        confirmatory_slice=args.confirmatory_slice,
    )
    print(
        json.dumps(
            {
                "paired_statistics": summary["paired_statistics"]["kalman_vs_mean"][
                    "overall"
                ],
                "decision": summary["decision"]["kalman_vs_mean"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
