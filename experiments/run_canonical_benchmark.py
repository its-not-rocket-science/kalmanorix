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
    ndcg_delta = stats["overall"]["ndcg@10"]["mean_difference"]
    ndcg_padj = stats["overall"]["ndcg@10"]["adjusted_p_value"]
    kalman_wins = ndcg_delta > 0 and ndcg_padj < 0.05

    verdict = (
        "KalmanorixFuser outperforms MeanFuser on nDCG@10 with paired significance."
        if kalman_wins
        else "KalmanorixFuser does not show a statistically significant nDCG@10 improvement over MeanFuser in this run."
    )

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
        "| Method | nDCG@10 | Recall@10 | MRR@10 | Latency (ms) | FLOPs proxy |",
        "|---|---|---|---|---|---|",
    ]

    for method_label, key in CANONICAL_METHOD_ALIASES.items():
        if key not in methods:
            continue
        payload = methods[key]["metrics"]
        lines.append(
            "| {label} | {ndcg:.4f} [{ndcg_l:.4f}, {ndcg_h:.4f}] | {rec:.4f} [{rec_l:.4f}, {rec_h:.4f}] | {mrr:.4f} [{mrr_l:.4f}, {mrr_h:.4f}] | {lat:.3f} [{lat_l:.3f}, {lat_h:.3f}] | {flops:.3f} [{flops_l:.3f}, {flops_h:.3f}] |".format(
                label=method_label,
                ndcg=payload["ndcg@10"]["mean"],
                ndcg_l=payload["ndcg@10"]["ci95_low"],
                ndcg_h=payload["ndcg@10"]["ci95_high"],
                rec=payload["recall@10"]["mean"],
                rec_l=payload["recall@10"]["ci95_low"],
                rec_h=payload["recall@10"]["ci95_high"],
                mrr=payload["mrr@10"]["mean"],
                mrr_l=payload["mrr@10"]["ci95_low"],
                mrr_h=payload["mrr@10"]["ci95_high"],
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
            "## Paired Statistical Test: KalmanorixFuser vs MeanFuser",
            "",
            "| Metric | Δ mean (Kalman-Mean) | 95% CI | p | Holm-adjusted p |",
            "|---|---:|---|---:|---:|",
        ]
    )
    for metric in ["ndcg@10", "recall@10", "mrr@10"]:
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

    lines.extend(["", "## Interpretation", "", f"- {verdict}"])
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
        for metric in ["ndcg@10", "recall@10", "mrr@10"]
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
        default=Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-queries", type=int, default=150)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-resamples", type=int, default=5000)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/canonical_benchmark")
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
        json.dumps(summary["paired_statistics"]["kalman_vs_mean"]["overall"], indent=2)
    )


if __name__ == "__main__":
    main()
