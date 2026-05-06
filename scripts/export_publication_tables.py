from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def format_metric(value: float) -> str:
    return f"{value:.4f}"


def format_latency(value: float) -> str:
    return f"{value:.3f}"


def format_flops(value: float) -> str:
    return f"{value:.3f}"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def extract_method_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    label_map = {
        "kalman": "KalmanorixFuser",
        "mean": "MeanFuser",
        "fixed_weighted_mean_fusion": "fixed weighted mean baseline",
        "router_only_top1": "hard routing baseline",
        "learned_linear_combiner": "learned linear combiner",
        "single_generalist_model": "single generalist model",
        "uniform_mean_fusion": "all-routing + mean baseline",
    }

    order = [
        "kalman",
        "mean",
        "fixed_weighted_mean_fusion",
        "router_only_top1",
        "learned_linear_combiner",
        "single_generalist_model",
        "uniform_mean_fusion",
    ]

    methods = summary["methods"]
    rows: list[dict[str, str]] = []
    for key in order:
        if key not in methods:
            continue
        metrics = methods[key]["metrics"]
        rows.append(
            {
                "Method": label_map.get(key, key),
                "nDCG@10": format_metric(metrics["ndcg@10"]["mean"]),
                "MRR@10": format_metric(metrics["mrr@10"]["mean"]),
                "Recall@10": format_metric(metrics["recall@10"]["mean"]),
                "latency": format_latency(metrics["latency_ms"]["mean"]),
                "FLOPs proxy": format_flops(metrics["flops_proxy"]["mean"]),
            }
        )
    return rows


def extract_decision_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    decision = summary["decision"]["kalman_vs_mean"]
    return [
        {
            "Rule": "Primary metric (nDCG@10 Δ mean)",
            "Threshold": f">= {decision['rules']['minimum_effect_size']:.4f}",
            "Observed": f"{decision['observed']['primary_metric_delta']:.6f}",
            "Pass": "yes" if decision["checks"]["effect_size_ok"] else "no",
        },
        {
            "Rule": "Adjusted p-value (Holm)",
            "Threshold": f"<= {decision['rules']['adjusted_p_value_threshold']:.4f}",
            "Observed": f"{decision['observed']['primary_metric_adjusted_p_value']:.6f}",
            "Pass": "yes" if decision["checks"]["adjusted_p_value_ok"] else "no",
        },
        {
            "Rule": "Latency ratio (Kalman/Mean)",
            "Threshold": f"<= {decision['rules']['max_latency_ratio_vs_mean']:.3f}",
            "Observed": f"{decision['observed']['latency_ratio_vs_mean']:.3f}",
            "Pass": "yes" if decision["checks"]["latency_ratio_ok"] else "no",
        },
        {
            "Rule": "FLOPs ratio (Kalman/Mean)",
            "Threshold": f"<= {decision['rules']['max_flops_ratio_vs_mean']:.3f}",
            "Observed": f"{decision['observed']['flops_ratio_vs_mean']:.3f}",
            "Pass": "yes" if decision["checks"]["flops_ratio_ok"] else "no",
        },
    ]


def extract_baseline_rows(summary: dict[str, Any]) -> list[dict[str, str]]:
    comparisons = summary["paired_statistics"]
    lookup = {
        "kalman_vs_mean": "mean",
        "kalman_vs_fixed_weighted_mean_fusion": "fixed_weighted_mean_fusion",
        "kalman_vs_router_only_top1": "router_only_top1",
        "kalman_vs_learned_linear_combiner": "learned_linear_combiner",
    }

    methods = summary["methods"]
    rows: list[dict[str, str]] = []
    for comp, baseline_key in lookup.items():
        info = comparisons[comp]["overall"]["ndcg@10"]
        baseline_metrics = methods[baseline_key]["metrics"]
        rows.append(
            {
                "Comparison": comp,
                "Method": "KalmanorixFuser vs " + baseline_key,
                "nDCG@10": f"{info['mean_difference']:.6f}",
                "MRR@10": format_metric(
                    methods["kalman"]["metrics"]["mrr@10"]["mean"]
                    - baseline_metrics["mrr@10"]["mean"]
                ),
                "Recall@10": format_metric(
                    methods["kalman"]["metrics"]["recall@10"]["mean"]
                    - baseline_metrics["recall@10"]["mean"]
                ),
                "latency": format_latency(
                    methods["kalman"]["metrics"]["latency_ms"]["mean"]
                    - baseline_metrics["latency_ms"]["mean"]
                ),
                "FLOPs proxy": format_flops(
                    methods["kalman"]["metrics"]["flops_proxy"]["mean"]
                    - baseline_metrics["flops_proxy"]["mean"]
                ),
            }
        )
    return rows


def to_markdown(headers: list[str], rows: list[dict[str, str]]) -> str:
    table = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] * len(headers)) + "|",
    ]
    for row in rows:
        table.append("| " + " | ".join(row[h] for h in headers) + " |")
    return "\n".join(table)


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/canonical_benchmark_v3_fast_1200"),
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("docs/publication/tables"),
    )
    args = parser.parse_args()

    summary = read_json(args.artifact_dir / "summary.json")
    _ = (args.artifact_dir / "report.md").read_text(encoding="utf-8")

    note = "\n\n> Note: fast-local uses deterministic hash embeddings and is a CPU-feasible smoke/benchmark mode.\n"

    main_table = to_markdown(
        ["Method", "nDCG@10", "MRR@10", "Recall@10", "latency", "FLOPs proxy"],
        extract_method_rows(summary),
    )
    decision_table = to_markdown(
        ["Rule", "Threshold", "Observed", "Pass"],
        extract_decision_rows(summary),
    )
    baseline_table = to_markdown(
        [
            "Comparison",
            "Method",
            "nDCG@10",
            "MRR@10",
            "Recall@10",
            "latency",
            "FLOPs proxy",
        ],
        extract_baseline_rows(summary),
    )

    write(args.out_dir / "main_results_table.md", main_table + note)
    write(args.out_dir / "decision_rule_table.md", decision_table + note)
    write(args.out_dir / "baseline_comparison_table.md", baseline_table + note)


if __name__ == "__main__":
    main()
