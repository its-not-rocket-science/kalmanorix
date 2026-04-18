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
    "tuned weighted mean baseline": "tuned_weighted_mean_fusion",
    "learned linear combiner": "learned_linear_combiner",
    "LearnedGateFuser": "learned_gate_fuser",
}
CANONICAL_METHOD_KEY_ALIASES = {
    "fixed_weighted_mean_fusion": "tuned_weighted_mean_fusion",
}
CLAIM_READY_REQUIRED_BASELINES = {
    "uniform_mean_fusion",
    "tuned_weighted_mean_fusion",
    "learned_linear_combiner",
}
REQUIRED_SAMPLE_SIZE_BLOCKS = {
    "uncertainty_calibration",
    "paired_significance_testing",
    "per_domain_analysis",
}
REQUIRED_DECISION_KEYS = {
    "kalman_vs_mean",
    "kalman_vs_weighted_mean",
    "kalman_vs_learned_linear_combiner",
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
TOY_TEST_QUERY_THRESHOLD = max(10, PAIRED_TEST_MIN_TEST_QUERIES // 2)
TOY_PER_DOMAIN_THRESHOLD = max(5, PER_DOMAIN_MIN_QUERIES // 2)
TOY_VALIDATION_THRESHOLD = max(20, CALIBRATION_MIN_VALIDATION_QUERIES // 2)
CLAIM_READY_TEST_QUERY_THRESHOLD = PAIRED_TEST_MIN_TEST_QUERIES * 2
CLAIM_READY_PER_DOMAIN_THRESHOLD = PER_DOMAIN_MIN_QUERIES * 2
CLAIM_READY_VALIDATION_THRESHOLD = CALIBRATION_MIN_VALIDATION_QUERIES * 2
CLAIM_READY_DETECTABLE_EFFECT_RATIO = 0.75
CONFIRMATORY_SLICE_CHOICES = (
    "high_specialist_disagreement",
    "high_uncertainty_spread",
    "nontrivial_routing_case",
    "intersection_of_above",
)


def _canonical_method_key(method_key: str) -> str:
    return CANONICAL_METHOD_KEY_ALIASES.get(method_key, method_key)


def _merge_methods_with_canonical_keys(methods: dict[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    for method_key, payload in methods.items():
        merged[method_key] = payload
        canonical_key = _canonical_method_key(method_key)
        merged[canonical_key] = payload
    return merged


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
        router_conf = float(
            meta.get("router_confidence", confidence_proxy.get(qid, 0.0))
        )
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
                    float(np.mean([metrics[metric][idx] for idx in idxs]))
                    if idxs
                    else 0.0
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
                "consistent_gain" if is_consistent_gain else "mixed_or_no_gain"
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
        router_conf = float(
            meta.get("router_confidence", confidence_proxy.get(qid, 0.0))
        )
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
        return sorted(
            high_disagreement.intersection(high_uncertainty).intersection(nontrivial)
        )
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
                float(np.mean([method_qlevel[metric][idx] for idx in idxs]))
                if idxs
                else 0.0
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


def _classify_kalman_vs_baseline(
    summary: dict[str, Any], *, baseline_key: str
) -> dict[str, Any]:
    rules = CANONICAL_DECISION_RULES
    comparison_key = f"kalman_vs_{baseline_key}"
    stats = summary["paired_statistics"][comparison_key]["overall"]
    methods = summary["methods"]
    primary_metric = rules["primary_metric"]
    primary_stats = stats[primary_metric]

    effect_delta = float(primary_stats["mean_difference"])
    adjusted_p_value = float(primary_stats["adjusted_p_value"])
    latency_ratio = float(
        methods["kalman"]["metrics"]["latency_ms"]["mean"]
        / methods[baseline_key]["metrics"]["latency_ms"]["mean"]
    )
    flops_ratio = float(
        methods["kalman"]["metrics"]["flops_proxy"]["mean"]
        / methods[baseline_key]["metrics"]["flops_proxy"]["mean"]
    )

    checks = {
        "effect_size_ok": effect_delta >= float(rules["minimum_effect_size"]),
        "adjusted_p_value_ok": adjusted_p_value
        <= float(rules["adjusted_p_value_threshold"]),
        "latency_ratio_ok": latency_ratio <= float(rules["max_latency_ratio_vs_mean"]),
        "flops_ratio_ok": flops_ratio <= float(rules["max_flops_ratio_vs_mean"]),
    }

    power_diag = summary["power_diagnostics"][comparison_key]
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
            f"latency_ratio_vs_{baseline_key}": latency_ratio,
            f"flops_ratio_vs_{baseline_key}": flops_ratio,
        },
        "checks": checks,
    }


def _classify_benchmark_status(summary: dict[str, Any]) -> dict[str, Any]:
    adequacy = summary["sample_size_adequacy"]
    power_diag = summary["power_diagnostics"]["kalman_vs_mean"]
    test_count = int(power_diag["num_test_queries"])
    min_domain_count = int(adequacy["per_domain_analysis"]["minimum_domain_count"])
    validation_count = int(adequacy["uncertainty_calibration"]["available_queries"])
    detectable_effect = float(power_diag["detectable_effect_threshold_estimate"])
    target_effect = float(power_diag["target_effect_size"])
    detectable_ratio = (
        float(detectable_effect / target_effect)
        if np.isfinite(detectable_effect) and target_effect > 0.0
        else float("inf")
    )

    toy_reasons: list[str] = []
    if test_count < TOY_TEST_QUERY_THRESHOLD:
        toy_reasons.append(
            f"test_query_count={test_count} < {TOY_TEST_QUERY_THRESHOLD}"
        )
    if min_domain_count < TOY_PER_DOMAIN_THRESHOLD:
        toy_reasons.append(
            f"min_domain_test_count={min_domain_count} < {TOY_PER_DOMAIN_THRESHOLD}"
        )
    if validation_count < TOY_VALIDATION_THRESHOLD:
        toy_reasons.append(
            f"validation_query_count={validation_count} < {TOY_VALIDATION_THRESHOLD}"
        )

    if toy_reasons:
        status = "toy"
        status_note = "Sample is toy-scale for Kalman-vs-mean claims; treat outcomes as smoke-test signals only."
    else:
        minimally_powered_checks = {
            "test_query_count_ok": bool(test_count >= PAIRED_TEST_MIN_TEST_QUERIES),
            "per_domain_min_count_ok": bool(min_domain_count >= PER_DOMAIN_MIN_QUERIES),
            "validation_count_ok": bool(
                validation_count >= CALIBRATION_MIN_VALIDATION_QUERIES
            ),
            "detectable_effect_ok": bool(detectable_effect <= target_effect),
        }
        if all(minimally_powered_checks.values()):
            claim_ready_checks = {
                "test_query_count_claim_ready": bool(
                    test_count >= CLAIM_READY_TEST_QUERY_THRESHOLD
                ),
                "per_domain_min_count_claim_ready": bool(
                    min_domain_count >= CLAIM_READY_PER_DOMAIN_THRESHOLD
                ),
                "validation_count_claim_ready": bool(
                    validation_count >= CLAIM_READY_VALIDATION_THRESHOLD
                ),
                "detectable_effect_claim_ready": bool(
                    detectable_effect
                    <= (target_effect * CLAIM_READY_DETECTABLE_EFFECT_RATIO)
                ),
            }
            if all(claim_ready_checks.values()):
                status = "claim_ready"
                status_note = "Counts and detectable-effect headroom satisfy stricter claim-readiness thresholds."
            else:
                status = "minimally_powered"
                status_note = (
                    "Meets minimum power/coverage checks, but lacks claim-ready margin."
                )
        else:
            status = "underpowered"
            status_note = (
                "Above toy-scale, but misses at least one minimum power/coverage check."
            )

    return {
        "status": status,
        "status_note": status_note,
        "inputs": {
            "test_query_count": test_count,
            "minimum_per_domain_test_count": min_domain_count,
            "validation_query_count": validation_count,
            "detectable_effect_threshold_estimate": detectable_effect,
            "target_effect_size": target_effect,
            "detectable_effect_to_target_ratio": detectable_ratio,
        },
        "thresholds": {
            "toy": {
                "test_query_count_lt": TOY_TEST_QUERY_THRESHOLD,
                "minimum_per_domain_test_count_lt": TOY_PER_DOMAIN_THRESHOLD,
                "validation_query_count_lt": TOY_VALIDATION_THRESHOLD,
            },
            "minimally_powered": {
                "test_query_count_gte": PAIRED_TEST_MIN_TEST_QUERIES,
                "minimum_per_domain_test_count_gte": PER_DOMAIN_MIN_QUERIES,
                "validation_query_count_gte": CALIBRATION_MIN_VALIDATION_QUERIES,
                "detectable_effect_threshold_lte_target_effect": True,
            },
            "claim_ready": {
                "test_query_count_gte": CLAIM_READY_TEST_QUERY_THRESHOLD,
                "minimum_per_domain_test_count_gte": CLAIM_READY_PER_DOMAIN_THRESHOLD,
                "validation_query_count_gte": CLAIM_READY_VALIDATION_THRESHOLD,
                "detectable_effect_threshold_lte_target_effect_ratio": CLAIM_READY_DETECTABLE_EFFECT_RATIO,
            },
        },
    }


def _validate_summary_contract(summary: dict[str, Any]) -> None:
    if "benchmark_status" not in summary:
        raise ValueError("Canonical summary is missing required `benchmark_status`.")
    decision = summary.get("decision", {})
    missing_decisions = sorted(REQUIRED_DECISION_KEYS.difference(decision))
    if missing_decisions:
        raise ValueError(
            "Canonical summary is missing required decision blocks: "
            f"{missing_decisions}"
        )
    methods = summary.get("methods", {})
    missing_baselines = sorted(CLAIM_READY_REQUIRED_BASELINES.difference(methods))
    if missing_baselines:
        raise ValueError(
            "Canonical summary is missing required baselines for claim-readiness: "
            f"{missing_baselines}"
        )
    adequacy = summary.get("sample_size_adequacy", {})
    missing_adequacy = sorted(REQUIRED_SAMPLE_SIZE_BLOCKS.difference(adequacy))
    if missing_adequacy:
        raise ValueError(
            "Canonical summary is missing sample size adequacy blocks: "
            f"{missing_adequacy}"
        )


def _validate_report_contract(report_text: str) -> None:
    required_sections = (
        "## Power-Oriented Diagnostics (KalmanorixFuser vs MeanFuser)",
        "## Sample Size Adequacy Checks",
        "## Kalman vs simple and learned weighting baselines",
        "## Verdict",
    )
    missing_sections = [
        section for section in required_sections if section not in report_text
    ]
    if missing_sections:
        raise ValueError(
            f"Canonical report is missing required sections: {missing_sections}"
        )


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
    benchmark_status = summary["benchmark_status"]
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
        f"- **Benchmark status:** `{benchmark_status['status']}` — {benchmark_status['status_note']}",
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
            + " > ".join(
                f"`{name}` ({score:.4f})" for name, score in ranking_by_ndcg10
            ),
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
                ok="yes"
                if adequacy["paired_significance_testing"]["adequate"]
                else "no",
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
            "## Kalman vs simple and learned weighting baselines",
            "",
            "| Comparison | Δ nDCG@10 (Kalman-baseline) | 95% CI | Holm-adjusted p | Decision |",
            "|---|---:|---|---:|---|",
        ]
    )
    for comparison_key, decision_key in [
        ("kalman_vs_mean", "kalman_vs_mean"),
        ("kalman_vs_tuned_weighted_mean_fusion", "kalman_vs_weighted_mean"),
        (
            "kalman_vs_learned_linear_combiner",
            "kalman_vs_learned_linear_combiner",
        ),
    ]:
        if comparison_key not in summary["paired_statistics"]:
            continue
        entry = summary["paired_statistics"][comparison_key]["overall"]["ndcg@10"]
        lines.append(
            "| {label} | {delta:.6f} | [{low:.6f}, {high:.6f}] | {padj:.6f} | {decision} |".format(
                label=comparison_key,
                delta=entry["mean_difference"],
                low=entry["ci95_low"],
                high=entry["ci95_high"],
                padj=entry["adjusted_p_value"],
                decision=summary["decision"]
                .get(decision_key, {})
                .get("verdict", "not_evaluated"),
            )
        )

    lines.extend(
        [
            "",
            "## Verdict",
            "",
            f"- **benchmark_status:** `{benchmark_status['status']}`",
            f"- **kalman_vs_mean:** `{verdict}`",
            f"- **kalman_vs_weighted_mean:** `{summary['decision']['kalman_vs_weighted_mean']['verdict']}`",
            f"- **kalman_vs_learned_linear_combiner:** `{summary['decision']['kalman_vs_learned_linear_combiner']['verdict']}`",
            "- Interpretation: `benchmark_status` grades evidence readiness (`toy`, `underpowered`, `minimally_powered`, `claim_ready`) while verdict preserves the existing Kalman-vs-baseline decision rule.",
            "- Rule logic: `supported` if all checks pass; `unsupported` if nDCG@10 Δ <= 0 and Holm-adjusted p <= threshold; otherwise inconclusive is split into `inconclusive_underpowered` vs `inconclusive_sufficiently_powered` from the detectable-effect threshold estimate.",
        ]
    )
    replication = summary.get("replication")
    if replication is not None:
        lines.extend(
            [
                "",
                "## Replication Evidence",
                "",
                f"- Replication runs: `{replication['num_runs']}`",
                f"- Positive nDCG@10 deltas (Kalman-Mean): `{replication['fraction_positive_deltas']:.3f}`",
                f"- Statistically significant runs (Holm-adjusted p <= 0.05 on nDCG@10): `{replication['fraction_significant_runs']:.3f}`",
                f"- Median latency ratio (Kalman/Mean): `{replication['median_latency_ratio']:.3f}`",
                f"- Direction consistency: `{replication['direction_consistency']}`",
                "- Note: pooled summaries below are descriptive across replications and are not a formal meta-analytic significance test.",
                "",
                "| Run | Seed | Verdict | Δ nDCG@10 | Holm-adjusted p | Latency ratio |",
                "|---|---:|---|---:|---:|---:|",
            ]
        )
        for run in replication["per_run_verdicts"]:
            lines.append(
                "| {run_id} | {seed} | {verdict} | {delta:.6f} | {padj:.6f} | {latency:.3f} |".format(
                    run_id=run["run_id"],
                    seed=run["seed"],
                    verdict=run["verdict"],
                    delta=run["primary_delta_ndcg10"],
                    padj=run["primary_adjusted_p_value_ndcg10"],
                    latency=run["latency_ratio_vs_mean"],
                )
            )
        pooled = replication["pooled_effect_summaries"]
        lines.extend(
            [
                "",
                "| Pooled descriptor | Value |",
                "|---|---:|",
                f"| Query-count weighted mean Δ nDCG@10 | {pooled['weighted_mean_delta_ndcg10']:.6f} |",
                f"| Median run-level Δ nDCG@10 | {pooled['median_delta_ndcg10']:.6f} |",
                f"| Median run-level Holm-adjusted p | {pooled['median_adjusted_p_value_ndcg10']:.6f} |",
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
            lines.append(
                "- None met the consistency + inferential significance criteria in this run."
            )
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


def _build_replication_summary(run_summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not run_summaries:
        raise ValueError("Replication summary requires at least one run summary.")

    per_run_verdicts: list[dict[str, Any]] = []
    positive_count = 0
    significant_count = 0
    deltas: list[float] = []
    latency_ratios: list[float] = []
    weighted_delta_num = 0.0
    weighted_delta_den = 0

    for idx, summary in enumerate(run_summaries, start=1):
        comparison = summary["paired_statistics"]["kalman_vs_mean"]["overall"][
            "ndcg@10"
        ]
        delta = float(comparison["mean_difference"])
        padj = float(comparison["adjusted_p_value"])
        n_pairs = int(comparison["n_pairs"])
        latency_ratio = float(
            summary["decision"]["kalman_vs_mean"]["observed"]["latency_ratio_vs_mean"]
        )
        verdict = str(summary["decision"]["kalman_vs_mean"]["verdict"])
        seed = int(summary["seed"])
        run_id = f"run_{idx:03d}"

        if delta > 0.0:
            positive_count += 1
        if padj <= 0.05:
            significant_count += 1

        deltas.append(delta)
        latency_ratios.append(latency_ratio)
        weighted_delta_num += delta * n_pairs
        weighted_delta_den += n_pairs

        per_run_verdicts.append(
            {
                "run_id": run_id,
                "seed": seed,
                "verdict": verdict,
                "primary_delta_ndcg10": delta,
                "primary_adjusted_p_value_ndcg10": padj,
                "n_pairs": n_pairs,
                "latency_ratio_vs_mean": latency_ratio,
            }
        )

    num_runs = len(run_summaries)
    return {
        "num_runs": num_runs,
        "per_run_verdicts": per_run_verdicts,
        "fraction_positive_deltas": float(positive_count / num_runs),
        "fraction_significant_runs": float(significant_count / num_runs),
        "median_latency_ratio": float(
            np.median(np.asarray(latency_ratios, dtype=float))
        ),
        "direction_consistency": (
            "all_positive"
            if positive_count == num_runs
            else ("all_negative_or_zero" if positive_count == 0 else "mixed")
        ),
        "pooled_effect_summaries": {
            "weighted_mean_delta_ndcg10": (
                float(weighted_delta_num / weighted_delta_den)
                if weighted_delta_den > 0
                else 0.0
            ),
            "median_delta_ndcg10": float(np.median(np.asarray(deltas, dtype=float))),
            "median_adjusted_p_value_ndcg10": float(
                np.median(
                    np.asarray(
                        [
                            run["primary_adjusted_p_value_ndcg10"]
                            for run in per_run_verdicts
                        ],
                        dtype=float,
                    )
                )
            ),
            "note": "Descriptive pooling only; no formal pooled significance test is computed.",
        },
    }


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

    raw_methods: dict[str, Any] = {}
    for method_key in sorted(rankings):
        raw_methods[method_key] = aggregate_strategy_metrics(
            rankings=rankings[method_key],
            ground_truth=ground_truth,
            latency_ms=latencies[method_key],
            flops_proxy=flops_proxy[method_key],
            seed=seed,
            num_resamples=num_resamples,
        )
    methods = _merge_methods_with_canonical_keys(raw_methods)

    required_methods = {
        "mean",
        "kalman",
        "uniform_mean_fusion",
        *CLAIM_READY_REQUIRED_BASELINES,
    }
    structurally_applicable_gate = len(DEFAULT_REAL_SPECIALISTS) == 2
    if structurally_applicable_gate:
        required_methods.add("learned_gate_fuser")
    operational_required_methods = {"router_only_top1", "router_only_topk_mean"}
    missing_required = sorted(required_methods.difference(methods))
    if missing_required:
        raise ValueError(
            "Canonical benchmark requires MeanFuser, KalmanorixFuser, uniform mean "
            "fusion, tuned weighted mean fusion, learned linear combiner, and "
            "LearnedGateFuser when structurally applicable. "
            f"Missing strategies: {missing_required}"
        )
    missing_operational = sorted(operational_required_methods.difference(methods))
    if missing_operational:
        raise ValueError(
            "Canonical benchmark requires hard-routing and top-k-mean routing "
            f"baselines for subgroup diagnostics. Missing strategies: {missing_operational}"
        )

    ordered_qids = sorted(rankings["kalman"])
    per_domain_test_counts: dict[str, int] = {}
    for qid in ordered_qids:
        domain = str(domains[qid])
        per_domain_test_counts[domain] = per_domain_test_counts.get(domain, 0) + 1

    kalman_metrics = methods["kalman"]["query_level"]
    kalman_domain = {
        metric: _by_domain(ordered_qids, values, domains)
        for metric, values in kalman_metrics.items()
    }
    primary_metric = CANONICAL_DECISION_RULES["primary_metric"]
    paired_statistics: dict[str, Any] = {}
    power_diagnostics: dict[str, Any] = {}

    for baseline_key, include_domains in [
        ("mean", True),
        ("tuned_weighted_mean_fusion", False),
        ("learned_linear_combiner", False),
    ]:
        baseline_metrics = methods[baseline_key]["query_level"]
        baseline_domain = (
            {
                metric: _by_domain(ordered_qids, values, domains)
                for metric, values in baseline_metrics.items()
            }
            if include_domains
            else None
        )
        paired = generate_statistical_report(
            reference_method="kalman",
            candidate_method=baseline_key,
            reference_metrics=kalman_metrics,
            candidate_metrics=baseline_metrics,
            reference_metrics_by_domain=kalman_domain if include_domains else None,
            candidate_metrics_by_domain=baseline_domain,
            seed=seed,
            num_resamples=num_resamples,
            config={
                "benchmark": str(benchmark_path),
                "evaluated_split": split,
                "baseline": baseline_key,
            },
        )
        comparison_key = f"kalman_vs_{baseline_key}"
        paired_statistics[comparison_key] = {
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
            "domains": (
                {
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
                }
                if include_domains
                else {}
            ),
            "configuration_hash": paired.configuration_hash,
        }
        primary_stats = paired_statistics[comparison_key]["overall"][primary_metric]
        primary_deltas = np.asarray(
            kalman_metrics[primary_metric], dtype=float
        ) - np.asarray(baseline_metrics[primary_metric], dtype=float)
        n_test = int(len(primary_deltas))
        std_delta = float(np.std(primary_deltas, ddof=1)) if n_test > 1 else 0.0
        detectable_threshold = (
            float((1.96 + 0.84) * (std_delta / np.sqrt(n_test)))
            if n_test > 1
            else float("inf")
        )
        power_diagnostics[comparison_key] = {
            "num_test_queries": n_test,
            "per_domain_test_counts": dict(sorted(per_domain_test_counts.items())),
            "observed_effect_size": float(primary_stats["mean_difference"]),
            "target_effect_size": float(
                CANONICAL_DECISION_RULES["minimum_effect_size"]
            ),
            "detectable_effect_threshold_estimate": detectable_threshold,
            "paired_delta_stddev": std_delta,
            "power_approximation": "detectable_effect ≈ (1.96+0.84)*std(delta)/sqrt(n)",
            "is_sufficiently_powered_for_target_effect": bool(
                detectable_threshold
                <= float(CANONICAL_DECISION_RULES["minimum_effect_size"])
            ),
        }
    n_test = int(len(kalman_metrics[primary_metric]))

    min_domain_count = (
        min(per_domain_test_counts.values()) if per_domain_test_counts else 0
    )
    sample_size_adequacy = {
        "uncertainty_calibration": {
            "available_queries": int(split_counts["validation"]),
            "minimum_required": CALIBRATION_MIN_VALIDATION_QUERIES,
            "adequate": bool(
                split_counts["validation"] >= CALIBRATION_MIN_VALIDATION_QUERIES
            ),
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
        "paired_statistics": paired_statistics,
        "power_diagnostics": power_diagnostics,
        "sample_size_adequacy": sample_size_adequacy,
        "bucket_analysis": bucket_analysis,
        "confirmatory_slice_results": confirmatory_slice_results,
    }
    summary["decision"] = {
        "kalman_vs_mean": _classify_kalman_vs_baseline(summary, baseline_key="mean"),
        "kalman_vs_weighted_mean": _classify_kalman_vs_baseline(
            summary, baseline_key="tuned_weighted_mean_fusion"
        ),
        "kalman_vs_learned_linear_combiner": _classify_kalman_vs_baseline(
            summary, baseline_key="learned_linear_combiner"
        ),
    }
    summary["benchmark_status"] = _classify_benchmark_status(summary)
    _validate_summary_contract(summary)
    report_text = _render_report(summary)
    _validate_report_contract(report_text)

    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (output_dir / "report.md").write_text(report_text, encoding="utf-8")
    return summary


def _resolve_replication_seeds(
    *, seed: int, replication_seeds: str, replication_runs: int
) -> list[int]:
    if replication_seeds.strip():
        return [
            int(token.strip())
            for token in replication_seeds.split(",")
            if token.strip()
        ]
    if replication_runs <= 1:
        return [seed]
    return [seed + idx for idx in range(replication_runs)]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run canonical benchmark and generate report artifacts"
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet"),
        help="Benchmark path (canonical v3 default: benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet).",
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument(
        "--max-queries",
        type=int,
        default=1200,
        help="Maximum evaluated queries (canonical v3 default: 1200).",
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-resamples", type=int, default=5000)
    parser.add_argument(
        "--replication-runs",
        type=int,
        default=1,
        help="Number of repeated benchmark builds to run using derived seeds.",
    )
    parser.add_argument(
        "--replication-seeds",
        type=str,
        default="",
        help="Comma-separated explicit replication seeds. Overrides --replication-runs when provided.",
    )
    parser.add_argument(
        "--confirmatory-slice",
        type=str,
        choices=CONFIRMATORY_SLICE_CHOICES,
        default=None,
        help="Named confirmatory evaluation slice for Kalman-vs-mean testing.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/canonical_benchmark_v3"),
        help="Output artifact directory (canonical v3 default: results/canonical_benchmark_v3).",
    )
    args = parser.parse_args()

    seeds = _resolve_replication_seeds(
        seed=args.seed,
        replication_seeds=args.replication_seeds,
        replication_runs=args.replication_runs,
    )
    run_summaries: list[dict[str, Any]] = []
    if len(seeds) == 1:
        summary = run_canonical_benchmark(
            benchmark_path=args.benchmark_path,
            output_dir=args.output_dir,
            split=args.split,
            max_queries=args.max_queries,
            device=args.device,
            seed=seeds[0],
            num_resamples=args.num_resamples,
            confirmatory_slice=args.confirmatory_slice,
        )
    else:
        runs_dir = args.output_dir / "replication_runs"
        for idx, run_seed in enumerate(seeds, start=1):
            run_output_dir = runs_dir / f"run_{idx:03d}_seed_{run_seed}"
            run_summary = run_canonical_benchmark(
                benchmark_path=args.benchmark_path,
                output_dir=run_output_dir,
                split=args.split,
                max_queries=args.max_queries,
                device=args.device,
                seed=run_seed,
                num_resamples=args.num_resamples,
                confirmatory_slice=args.confirmatory_slice,
            )
            run_summaries.append(run_summary)
        summary = run_summaries[0]
        summary["replication"] = _build_replication_summary(run_summaries)
        args.output_dir.mkdir(parents=True, exist_ok=True)
        (args.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        (args.output_dir / "replication_summary.json").write_text(
            json.dumps(summary["replication"], indent=2), encoding="utf-8"
        )
        (args.output_dir / "report.md").write_text(
            _render_report(summary), encoding="utf-8"
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
