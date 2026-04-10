#!/usr/bin/env python3
"""Run the canonical mixed-domain benchmark and emit publishable artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


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
    "recall@10",
    "top1_success",
]


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

    if all(checks.values()):
        verdict = "supported"
    elif effect_delta <= 0.0 and adjusted_p_value <= float(
        rules["adjusted_p_value_threshold"]
    ):
        verdict = "unsupported"
    else:
        verdict = "inconclusive"

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
        "| Method | nDCG@5 | nDCG@10 | MRR@5 | MRR@10 | Recall@1 | Recall@10 | Top-1 success | Latency (ms) | FLOPs proxy |",
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
            "| {label} | {ndcg5:.4f} [{ndcg5_l:.4f}, {ndcg5_h:.4f}] | {ndcg10:.4f} [{ndcg10_l:.4f}, {ndcg10_h:.4f}] | {mrr5:.4f} [{mrr5_l:.4f}, {mrr5_h:.4f}] | {mrr10:.4f} [{mrr10_l:.4f}, {mrr10_h:.4f}] | {rec1:.4f} [{rec1_l:.4f}, {rec1_h:.4f}] | {rec10:.4f} [{rec10_l:.4f}, {rec10_h:.4f}] | {top1:.4f} [{top1_l:.4f}, {top1_h:.4f}] | {lat:.3f} [{lat_l:.3f}, {lat_h:.3f}] | {flops:.3f} [{flops_l:.3f}, {flops_h:.3f}] |".format(
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
            "- Rule logic: `supported` if all checks pass; `unsupported` if nDCG@10 Δ <= 0 and Holm-adjusted p <= threshold; otherwise `inconclusive`.",
        ]
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

    query_level = details["query_level"]
    rankings = query_level["rankings"]
    ground_truth = {qid: set(ids) for qid, ids in query_level["ground_truth"].items()}
    domains = query_level["domains"]
    latencies = query_level["latency_ms"]
    flops_proxy = query_level["specialist_count_selected"]

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

    required_methods = {"mean", "kalman", "router_only_top1", "uniform_mean_fusion"}
    missing_required = sorted(required_methods.difference(methods))
    if missing_required:
        raise ValueError(
            "Canonical benchmark requires MeanFuser, KalmanorixFuser, hard-routing, "
            f"and all-routing+mean baselines. Missing strategies: {missing_required}"
        )

    ordered_qids = sorted(rankings["kalman"])
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
