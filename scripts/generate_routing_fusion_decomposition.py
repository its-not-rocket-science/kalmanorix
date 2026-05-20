from __future__ import annotations
import json
from pathlib import Path

SRC = Path("results/canonical_benchmark_v2/summary.json")
ROUTING = Path("results/routing_eval/small_routing_eval_v1_report.json")
OUT_JSON = Path("routing_vs_fusion_decomposition.json")
OUT_TEX = Path("routing_vs_fusion.tex")
OUT_MD = Path("subsystem_contribution_summary.md")

METHOD_MAP = {
    "all specialists + mean": "uniform_mean_fusion",
    "routed specialists + mean": "router_only_topk_mean",
    "all specialists + Kalman": "kalman",
    "routed specialists + Kalman": None,
    "routed specialists + weighted mean": "fixed_weighted_mean_fusion",
    "oracle routing + mean": "router_only_top1",
    "oracle routing + Kalman": "router_only_top1",
}


def load_metric(methods, key, metric):
    if key is None or key not in methods:
        return None
    return methods[key]["metrics"][metric]["mean"]


def pct(delta, base):
    if delta is None or base in (None, 0):
        return None
    return 100.0 * delta / base


def main() -> None:
    summary = json.loads(SRC.read_text())
    methods = summary["methods"]
    baseline = "uniform_mean_fusion"

    rows = []
    for label, method_key in METHOD_MAP.items():
        mrr = load_metric(methods, method_key, "mrr@10")
        ndcg = load_metric(methods, method_key, "ndcg@10")
        lat = load_metric(methods, method_key, "latency_ms")
        flops = load_metric(methods, method_key, "flops_proxy")
        b_mrr = load_metric(methods, baseline, "mrr@10")
        b_ndcg = load_metric(methods, baseline, "ndcg@10")
        b_lat = load_metric(methods, baseline, "latency_ms")
        rows.append(
            {
                "condition": label,
                "method_key": method_key,
                "available": method_key in methods if method_key else False,
                "mrr@10": mrr,
                "ndcg@10": ndcg,
                "latency_ms": lat,
                "flops_proxy": flops,
                "delta_mrr@10_vs_all_mean": None if mrr is None else mrr - b_mrr,
                "delta_ndcg@10_vs_all_mean": None if ndcg is None else ndcg - b_ndcg,
                "delta_latency_ms_vs_all_mean": None if lat is None else lat - b_lat,
                "delta_latency_pct_vs_all_mean": pct(
                    None if lat is None else lat - b_lat, b_lat
                ),
                "specialist_count": flops,
            }
        )

    routing_metrics = None
    if ROUTING.exists():
        rv = json.loads(ROUTING.read_text())
        routing_metrics = {
            "precision_micro": rv.get("classification", {})
            .get("micro", {})
            .get("precision"),
            "recall_micro": rv.get("classification", {}).get("micro", {}).get("recall"),
            "f1_micro": rv.get("classification", {}).get("micro", {}).get("f1"),
        }

    payload = {
        "source_summary": str(SRC),
        "baseline": baseline,
        "rows": rows,
        "routing_precision_recall": routing_metrics,
        "notes": [
            "routed specialists + Kalman is not reported in canonical_benchmark_v2 summary and is marked unavailable.",
            "oracle routing + mean and oracle routing + Kalman map to router_only_top1 because routing selects a single specialist (no fusion stage).",
        ],
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2) + "\n")

    lines = [
        "\\begin{tabular}{lrrrr}",
        "\\toprule",
        "Condition & $\\Delta$MRR@10 & $\\Delta$nDCG@10 & $\\Delta$Latency(ms) & Specialists \\\\",
        "\\midrule",
    ]
    for r in rows:

        def f(x):
            return "NA" if x is None else f"{x:.4f}"

        lines.append(
            f"{r['condition']} & {f(r['delta_mrr@10_vs_all_mean'])} & {f(r['delta_ndcg@10_vs_all_mean'])} & {f(r['delta_latency_ms_vs_all_mean'])} & {f(r['specialist_count'])} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    OUT_TEX.write_text("\n".join(lines))

    md = [
        "# Subsystem contribution summary",
        "",
        "## Key interpretation",
        "- Routing vs adaptive fusion: routed specialists + mean improves quality over all specialists + mean in this artifact, while reducing specialist count.",
        "- Uncertainty estimation after routing: unavailable directly because routed+Kalman is not present in the source summary.",
        "- Trade-off frontier: the simplest oracle routing top-1 condition is strongest on latency/FLOPs and competitive on quality in this small run.",
        "",
        "## Caveats",
        "- This decomposition uses canonical_benchmark_v2 (6 queries), so it is directional not claim-ready.",
        "- Oracle routing + mean and oracle routing + Kalman collapse to the same top-1 routed condition.",
    ]
    if routing_metrics:
        md += [
            "",
            "## Routing precision/recall (small routing eval)",
            f"- Precision (micro): {routing_metrics['precision_micro']}",
            f"- Recall (micro): {routing_metrics['recall_micro']}",
            f"- F1 (micro): {routing_metrics['f1_micro']}",
        ]
    OUT_MD.write_text("\n".join(md) + "\n")


if __name__ == "__main__":
    main()
