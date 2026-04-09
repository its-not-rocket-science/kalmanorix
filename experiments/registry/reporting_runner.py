"""Paper-grade reporting runner for benchmark registry artifacts."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kalmanorix.benchmarks.statistical_testing import generate_statistical_report


PRIMARY_METRICS = ("recall@1", "recall@5", "recall@10", "mrr", "ndcg@10")


@dataclass(frozen=True)
class QueryEval:
    recall1: float
    recall5: float
    recall10: float
    mrr: float
    ndcg10: float


def _recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    if not relevant:
        return 0.0
    return float(len(set(ranked[:k]).intersection(relevant)) / len(relevant))


def _mrr(ranked: list[str], relevant: set[str]) -> float:
    for idx, doc_id in enumerate(ranked, start=1):
        if doc_id in relevant:
            return float(1.0 / idx)
    return 0.0


def _ndcg10(ranked: list[str], relevant: set[str]) -> float:
    if not relevant:
        return 0.0
    dcg = 0.0
    for idx, doc_id in enumerate(ranked[:10], start=1):
        if doc_id in relevant:
            dcg += 1.0 / np.log2(idx + 1)
    max_rank = min(10, len(relevant))
    idcg = sum(1.0 / np.log2(idx + 1) for idx in range(1, max_rank + 1))
    return float(dcg / idcg) if idcg > 0 else 0.0


def _evaluate_query(ranked: list[str], relevant: set[str]) -> QueryEval:
    return QueryEval(
        recall1=_recall_at_k(ranked, relevant, 1),
        recall5=_recall_at_k(ranked, relevant, 5),
        recall10=_recall_at_k(ranked, relevant, 10),
        mrr=_mrr(ranked, relevant),
        ndcg10=_ndcg10(ranked, relevant),
    )


def _compute_calibration(
    accuracies: np.ndarray, confidences: np.ndarray
) -> dict[str, float]:
    bins = np.linspace(0.0, 1.0, 11)
    bin_ids = np.clip(np.digitize(confidences, bins, right=True) - 1, 0, 9)
    ece = 0.0
    for b in range(10):
        mask = bin_ids == b
        if not np.any(mask):
            continue
        acc = float(np.mean(accuracies[mask]))
        conf = float(np.mean(confidences[mask]))
        weight = float(np.sum(mask) / len(accuracies))
        ece += weight * abs(acc - conf)
    brier = float(np.mean((accuracies - confidences) ** 2))
    return {
        "n_samples": float(len(accuracies)),
        "ece": float(ece),
        "brier_score": brier,
        "mean_confidence": float(np.mean(confidences)),
        "mean_accuracy": float(np.mean(accuracies)),
        "overconfidence_gap": float(np.mean(confidences) - np.mean(accuracies)),
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _render_markdown(
    *,
    template_dir: Path,
    overall_rows: list[dict[str, Any]],
    significance_rows: list[dict[str, Any]],
    calibration_rows: list[dict[str, Any]],
    figure_paths: dict[str, str],
) -> str:
    overall_tpl = (template_dir / "table_overall.md").read_text(encoding="utf-8")
    sig_tpl = (template_dir / "table_significance.md").read_text(encoding="utf-8")
    fig_tpl = (template_dir / "figure_captions.md").read_text(encoding="utf-8")

    def _as_table(rows: list[dict[str, Any]]) -> str:
        if not rows:
            return "_No rows_"
        header = "| " + " | ".join(rows[0].keys()) + " |"
        sep = "| " + " | ".join(["---"] * len(rows[0])) + " |"
        body = ["| " + " | ".join(str(v) for v in row.values()) + " |" for row in rows]
        return "\n".join([header, sep, *body])

    return "\n\n".join(
        [
            overall_tpl.format(table=_as_table(overall_rows)),
            sig_tpl.format(table=_as_table(significance_rows)),
            "## Calibration Summary\n\n" + _as_table(calibration_rows),
            fig_tpl.format(**figure_paths),
        ]
    )


def _plot_latency_memory(
    overall_rows: list[dict[str, Any]], output_dir: Path
) -> dict[str, str]:
    figure_dir = output_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    latency_png = figure_dir / "latency_memory_tradeoff.png"
    latency_pdf = figure_dir / "latency_memory_tradeoff.pdf"
    metric_png = figure_dir / "quality_latency_frontier.png"
    metric_pdf = figure_dir / "quality_latency_frontier.pdf"

    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return {
            "latency_memory_png": str(latency_png),
            "quality_latency_png": str(metric_png),
        }

    plt.style.use("seaborn-v0_8-whitegrid")
    x = [float(row["latency_ms_mean"]) for row in overall_rows]
    y = [float(row["memory_proxy_mean"]) for row in overall_rows]
    labels = [str(row["strategy"]) for row in overall_rows]
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(x, y, s=60)
    for xi, yi, label in zip(x, y, labels, strict=True):
        plt.annotate(
            label, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points"
        )
    plt.xlabel("Latency (ms, mean)")
    plt.ylabel("Memory proxy (mean selected specialists)")
    plt.title("Latency/Memory Tradeoff")
    plt.tight_layout()
    plt.savefig(latency_png)
    plt.savefig(latency_pdf)
    plt.close()

    qx = [float(row["latency_ms_mean"]) for row in overall_rows]
    qy = [float(row["mrr_mean"]) for row in overall_rows]
    plt.figure(figsize=(8, 5), dpi=200)
    plt.scatter(qx, qy, s=60)
    for xi, yi, label in zip(qx, qy, labels, strict=True):
        plt.annotate(
            label, (xi, yi), fontsize=8, xytext=(4, 4), textcoords="offset points"
        )
    plt.xlabel("Latency (ms, mean)")
    plt.ylabel("MRR (mean)")
    plt.title("Quality/Latency Frontier")
    plt.tight_layout()
    plt.savefig(metric_png)
    plt.savefig(metric_pdf)
    plt.close()
    return {
        "latency_memory_png": str(latency_png),
        "quality_latency_png": str(metric_png),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Paper-grade report generator")
    parser.add_argument("--summary-json", type=Path, required=True)
    parser.add_argument("--details-json", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--reference-strategy", type=str, default="kalman")
    args = parser.parse_args()

    summary = json.loads(args.summary_json.read_text(encoding="utf-8"))
    details_path = args.details_json
    if details_path is None:
        details_ref = summary.get("query_level", {}).get("details_json")
        if details_ref:
            details_path = Path(details_ref)
    if details_path is None or not details_path.exists():
        raise FileNotFoundError(
            "details json is required; provide --details-json or artifacts.details_json"
        )
    details = json.loads(details_path.read_text(encoding="utf-8"))

    query_level = details["query_level"]
    rankings_by_strategy = query_level["rankings"]
    domains = query_level["domains"]
    ground_truth = {qid: set(ids) for qid, ids in query_level["ground_truth"].items()}
    latencies = query_level["latency_ms"]
    confidences = query_level["confidence_proxy"]
    specialist_counts = query_level["specialist_count_selected"]

    overall_rows: list[dict[str, Any]] = []
    per_domain_rows: list[dict[str, Any]] = []
    calibration_rows: list[dict[str, Any]] = []
    metrics_by_strategy: dict[str, dict[str, list[float]]] = {}
    metrics_by_strategy_domain: dict[str, dict[str, dict[str, list[float]]]] = {}

    for strategy, rankings in rankings_by_strategy.items():
        metric_map = {name: [] for name in PRIMARY_METRICS}
        metric_by_domain: dict[str, dict[str, list[float]]] = {}
        ordered_qids = sorted(rankings)
        for qid in ordered_qids:
            qeval = _evaluate_query(rankings[qid], ground_truth[qid])
            domain = domains[qid]
            metric_by_domain.setdefault(domain, {name: [] for name in PRIMARY_METRICS})

            metric_map["recall@1"].append(qeval.recall1)
            metric_map["recall@5"].append(qeval.recall5)
            metric_map["recall@10"].append(qeval.recall10)
            metric_map["mrr"].append(qeval.mrr)
            metric_map["ndcg@10"].append(qeval.ndcg10)

            metric_by_domain[domain]["recall@1"].append(qeval.recall1)
            metric_by_domain[domain]["recall@5"].append(qeval.recall5)
            metric_by_domain[domain]["recall@10"].append(qeval.recall10)
            metric_by_domain[domain]["mrr"].append(qeval.mrr)
            metric_by_domain[domain]["ndcg@10"].append(qeval.ndcg10)

        metrics_by_strategy[strategy] = metric_map
        metrics_by_strategy_domain[strategy] = metric_by_domain

        overall_rows.append(
            {
                "strategy": strategy,
                "mrr_mean": round(float(np.mean(metric_map["mrr"])), 6),
                "recall@1_mean": round(float(np.mean(metric_map["recall@1"])), 6),
                "recall@5_mean": round(float(np.mean(metric_map["recall@5"])), 6),
                "ndcg@10_mean": round(float(np.mean(metric_map["ndcg@10"])), 6),
                "latency_ms_mean": round(
                    float(np.mean(list(latencies[strategy].values()))), 6
                ),
                "memory_proxy_mean": round(
                    float(np.mean(list(specialist_counts[strategy].values()))), 6
                ),
            }
        )
        for domain, values in metric_by_domain.items():
            per_domain_rows.append(
                {
                    "strategy": strategy,
                    "domain": domain,
                    "mrr_mean": round(float(np.mean(values["mrr"])), 6),
                    "recall@1_mean": round(float(np.mean(values["recall@1"])), 6),
                    "recall@5_mean": round(float(np.mean(values["recall@5"])), 6),
                    "ndcg@10_mean": round(float(np.mean(values["ndcg@10"])), 6),
                }
            )

        accuracy = np.asarray(metric_map["recall@1"], dtype=float)
        conf = np.asarray(
            [
                float(confidences[strategy][qid])
                for qid in sorted(confidences[strategy])
            ],
            dtype=float,
        )
        calibration = _compute_calibration(accuracy, conf)
        calibration_rows.append(
            {"strategy": strategy, **{k: round(v, 6) for k, v in calibration.items()}}
        )

    ref = (
        args.reference_strategy
        if args.reference_strategy in metrics_by_strategy
        else sorted(metrics_by_strategy)[0]
    )
    significance_rows: list[dict[str, Any]] = []
    for strategy in sorted(metrics_by_strategy):
        if strategy == ref:
            continue
        report = generate_statistical_report(
            reference_method=ref,
            candidate_method=strategy,
            reference_metrics=metrics_by_strategy[ref],
            candidate_metrics=metrics_by_strategy[strategy],
            reference_metrics_by_domain=metrics_by_strategy_domain[ref],
            candidate_metrics_by_domain=metrics_by_strategy_domain[strategy],
            seed=42,
        )
        for metric, entry in report.comparisons.items():
            significance_rows.append(
                {
                    "reference": ref,
                    "candidate": strategy,
                    "metric": metric,
                    "mean_diff": round(entry.mean_difference, 6),
                    "p_value": round(entry.p_value, 6),
                    "adjusted_p_value": round(entry.adjusted_p_value, 6),
                    "cohen_dz": round(entry.effect_size.cohen_dz, 6),
                    "rank_biserial": round(entry.effect_size.rank_biserial, 6),
                }
            )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(args.output_dir / "overall_metrics.csv", overall_rows)
    _write_csv(args.output_dir / "per_domain_metrics.csv", per_domain_rows)
    _write_csv(args.output_dir / "calibration_summary.csv", calibration_rows)
    _write_csv(args.output_dir / "statistical_significance.csv", significance_rows)

    figures = _plot_latency_memory(overall_rows, args.output_dir)
    markdown = _render_markdown(
        template_dir=Path(__file__).resolve().parent / "templates",
        overall_rows=overall_rows,
        significance_rows=significance_rows,
        calibration_rows=calibration_rows,
        figure_paths=figures,
    )
    (args.output_dir / "summary.md").write_text(markdown + "\n", encoding="utf-8")

    bundle = {
        "summary_source": str(args.summary_json),
        "details_source": str(details_path),
        "overall_metrics": overall_rows,
        "per_domain_metrics": per_domain_rows,
        "calibration_summary": calibration_rows,
        "statistical_significance": significance_rows,
        "figures": figures,
    }
    (args.output_dir / "results_bundle.json").write_text(
        json.dumps(bundle, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
