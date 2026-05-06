from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def must_get(mapping: dict[str, Any], key: str, context: str) -> Any:
    if key not in mapping:
        raise ValueError(f"Missing required field '{key}' in {context}.")
    return mapping[key]


def must_metric(summary: dict[str, Any], method: str, metric: str) -> float:
    methods = must_get(summary, "methods", "summary")
    method_block = must_get(methods, method, "summary.methods")
    metrics = must_get(method_block, "metrics", f"summary.methods.{method}")
    metric_block = must_get(metrics, metric, f"summary.methods.{method}.metrics")
    return float(
        must_get(metric_block, "mean", f"summary.methods.{method}.metrics.{metric}")
    )


def fmt_metric(value: float) -> str:
    return f"{value:.4f}"


def fmt_latency(value: float) -> str:
    return f"{value:.3f}"


def fmt_flops(value: float) -> str:
    return f"{value:.3f}"


def fmt_delta(value: float) -> str:
    return f"{value:+.6f}"


def fmt_p(value: float) -> str:
    return f"{value:.4g}"


def latex_escape(text: str) -> str:
    return text.replace("_", "\\_")


def build_main_rows(
    summary: dict[str, Any],
) -> list[tuple[str, str, str, str, str, str]]:
    label_map = {
        "kalman": "KalmanorixFuser",
        "mean": "MeanFuser",
        "fixed_weighted_mean_fusion": "fixed weighted mean baseline",
        "router_only_top1": "hard routing baseline",
        "learned_linear_combiner": "learned linear combiner",
        "single_generalist_model": "single generalist model",
        "uniform_mean_fusion": "all-routing + mean baseline",
    }
    order = list(label_map)
    rows = []
    for key in order:
        rows.append(
            (
                label_map[key],
                fmt_metric(must_metric(summary, key, "ndcg@10")),
                fmt_metric(must_metric(summary, key, "mrr@10")),
                fmt_metric(must_metric(summary, key, "recall@10")),
                fmt_latency(must_metric(summary, key, "latency_ms")),
                fmt_flops(must_metric(summary, key, "flops_proxy")),
            )
        )
    return rows


def build_stat_rows(summary: dict[str, Any]) -> list[tuple[str, str, str, str, str]]:
    paired = must_get(summary, "paired_statistics", "summary")
    decisions = must_get(summary, "decision", "summary")
    items = [
        ("kalman_vs_mean", "mean", "kalman_vs_mean"),
        (
            "kalman_vs_fixed_weighted_mean_fusion",
            "fixed weighted mean",
            "kalman_vs_weighted_mean",
        ),
        ("kalman_vs_router_only_top1", "hard routing", "kalman_vs_router_only_top1"),
        (
            "kalman_vs_learned_linear_combiner",
            "learned linear combiner",
            "kalman_vs_learned_linear_combiner",
        ),
    ]
    rows = []
    for paired_key, label, decision_key in items:
        overall = must_get(
            must_get(paired, paired_key, "summary.paired_statistics"),
            "overall",
            f"summary.paired_statistics.{paired_key}",
        )
        ndcg = must_get(
            overall, "ndcg@10", f"summary.paired_statistics.{paired_key}.overall"
        )
        delta = float(
            must_get(ndcg, "mean_difference", f"{paired_key}.overall.ndcg@10")
        )
        ci_low = float(must_get(ndcg, "ci95_low", f"{paired_key}.overall.ndcg@10"))
        ci_high = float(must_get(ndcg, "ci95_high", f"{paired_key}.overall.ndcg@10"))
        p_adj = float(
            must_get(ndcg, "adjusted_p_value", f"{paired_key}.overall.ndcg@10")
        )
        verdict = str(
            must_get(
                must_get(decisions, decision_key, "summary.decision"),
                "verdict",
                f"summary.decision.{decision_key}",
            )
        )
        rows.append(
            (
                label,
                fmt_delta(delta),
                f"[{fmt_delta(ci_low)}, {fmt_delta(ci_high)}]",
                fmt_p(p_adj),
                verdict.replace("_", " "),
            )
        )
    return rows


def build_claim_gate_rows(summary: dict[str, Any]) -> list[tuple[str, str]]:
    benchmark_status = str(
        must_get(
            must_get(summary, "benchmark_status", "summary"),
            "status",
            "summary.benchmark_status",
        )
    )
    confirmatory = str(
        must_get(
            must_get(
                must_get(summary, "confirmatory_slice_results", "summary"),
                "decision",
                "summary.confirmatory_slice_results",
            ),
            "verdict",
            "summary.confirmatory_slice_results.decision",
        )
    )
    claim = must_get(summary, "claim_success_decision", "summary")
    required = must_get(claim, "required_decisions", "summary.claim_success_decision")
    final = str(must_get(claim, "status", "summary.claim_success_decision"))
    return [
        ("benchmark status", benchmark_status.replace("_", " ")),
        ("confirmatory slice status", confirmatory.replace("_", " ")),
        ("required baselines", ", ".join(str(x).replace("_", " ") for x in required)),
        ("final claim decision", final.replace("_", " ")),
    ]


def render_latex_main(rows):
    body = "\n".join(
        f"{latex_escape(m)} & {a} & {b} & {c} & {d} & {e} \\\\"
        for m, a, b, c, d, e in rows
    )
    return (
        "\\begin{tabular}{lccccc}\n"
        "\\toprule\n"
        "Method & nDCG@10 & MRR@10 & Recall@10 & latency (ms) & FLOPs proxy \\\\ \n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_latex_stats(rows):
    body = "\n".join(
        f"{latex_escape(b)} & {d} & {latex_escape(ci)} & {p} & {latex_escape(v)} \\\\"
        for b, d, ci, p, v in rows
    )
    return (
        "\\begin{tabular}{lcccc}\n"
        "\\toprule\n"
        "baseline & $\\Delta$ nDCG@10 & 95\\% CI & Holm-adjusted $p$ & verdict \\\\ \n"
        "\\midrule\n"
        f"{body}\n"
        "\\bottomrule\n"
        "\\end{tabular}\n"
    )


def render_claim_md(rows):
    lines = [
        "| claim gate item | status |",
        "|---|---|",
    ]
    for k, v in rows:
        lines.append(f"| {k} | {v} |")
    return "\n".join(lines) + "\n"


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-dir",
        type=Path,
        default=Path("results/canonical_benchmark_v3_fast_1200"),
    )
    args = parser.parse_args()

    summary_path = args.artifact_dir / "summary.json"
    report_path = args.artifact_dir / "report.md"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing required artefact: {summary_path}")
    if not report_path.exists():
        raise FileNotFoundError(f"Missing required artefact: {report_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))

    main_rows = build_main_rows(summary)
    stat_rows = build_stat_rows(summary)
    claim_rows = build_claim_gate_rows(summary)

    main_tex = render_latex_main(main_rows)
    stats_tex = render_latex_stats(stat_rows)
    claim_md = render_claim_md(claim_rows)

    write(Path("paper/arxiv/tables/main_results.tex"), main_tex)
    write(Path("paper/arxiv/tables/statistical_tests.tex"), stats_tex)
    write(Path("paper/tmlr/tables/main_results.tex"), main_tex)
    write(Path("paper/tmlr/tables/statistical_tests.tex"), stats_tex)
    write(Path("paper/joss/results_summary.md"), claim_md)


if __name__ == "__main__":
    main()
