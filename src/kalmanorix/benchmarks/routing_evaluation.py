"""Stable routing evaluation utilities for semantic and confidence routing.

This module intentionally keeps evaluation logic small, deterministic, and decoupled
from heavyweight model-loading code so benchmark evidence is easier to maintain.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import fmean
from typing import Any


@dataclass(frozen=True)
class RoutingSample:
    """Single labeled query for routing evaluation."""

    query_id: str
    relevant_domains: tuple[str, ...]
    semantic_scores: dict[str, float]
    confidence_scores: dict[str, float] | None = None
    routing_overhead_ms: float = 0.0
    domain_flops: dict[str, float] | None = None
    domain_latency_ms: dict[str, float] | None = None
    quality_delta: float = 0.0


@dataclass(frozen=True)
class RoutingRunConfig:
    """Configuration for a single routing pass."""

    mode: str = "semantic"
    semantic_threshold: float = 0.7
    confidence_threshold: float = 0.8
    fallback: str = "top1"
    quality_tolerance: float = 0.0


@dataclass(frozen=True)
class ThresholdSweepConfig:
    """Configuration for threshold-robustness analysis."""

    mode: str = "semantic"
    semantic_thresholds: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8)
    confidence_threshold: float = 0.8
    fallback: str = "top1"
    quality_tolerance: float = 0.0


def _safe_div(num: float, den: float) -> float:
    return 0.0 if den == 0 else num / den


def _top1(score_map: dict[str, float]) -> list[str]:
    if not score_map:
        return []
    return [max(score_map, key=score_map.get)]


def _select_domains(sample: RoutingSample, config: RoutingRunConfig) -> list[str]:
    if not sample.semantic_scores:
        return []

    semantic_selected = [
        d
        for d, score in sample.semantic_scores.items()
        if score >= config.semantic_threshold
    ]

    if config.mode == "semantic":
        if semantic_selected:
            return sorted(semantic_selected)
        return _top1(sample.semantic_scores) if config.fallback == "top1" else []

    if config.mode != "confidence":
        raise ValueError(f"Unsupported routing mode: {config.mode}")

    confidence = sample.confidence_scores or sample.semantic_scores
    top_domain = _top1(confidence)
    if not top_domain:
        return []

    top_conf = confidence[top_domain[0]]
    if top_conf >= config.confidence_threshold:
        return top_domain

    if semantic_selected:
        return sorted(semantic_selected)
    return _top1(sample.semantic_scores) if config.fallback == "top1" else []


def _costs(
    sample: RoutingSample, selected: list[str]
) -> tuple[float, float, float, float]:
    default_flops = 1.0
    default_latency = 1.0

    domains = sorted(sample.semantic_scores)
    flops_map = sample.domain_flops or {d: default_flops for d in domains}
    latency_map = sample.domain_latency_ms or {d: default_latency for d in domains}

    all_flops = float(sum(flops_map.values()))
    routed_flops = float(sum(flops_map.get(d, default_flops) for d in selected))

    all_latency = float(sum(latency_map.values()))
    routed_latency = float(
        sample.routing_overhead_ms
        + sum(latency_map.get(d, default_latency) for d in selected)
    )
    return all_flops, routed_flops, all_latency, routed_latency


def evaluate_routing(
    samples: list[RoutingSample], config: RoutingRunConfig
) -> dict[str, Any]:
    """Evaluate routing quality and efficiency for a fixed threshold setup."""

    per_query: list[dict[str, Any]] = []

    for sample in samples:
        selected = _select_domains(sample, config)
        relevant = set(sample.relevant_domains)
        selected_set = set(selected)

        tp = len(relevant & selected_set)
        precision = _safe_div(tp, len(selected_set))
        recall = _safe_div(tp, len(relevant))
        f1 = _safe_div(2 * precision * recall, precision + recall)

        all_flops, routed_flops, all_latency, routed_latency = _costs(sample, selected)
        flops_savings = _safe_div(all_flops - routed_flops, all_flops)
        latency_delta_ms = all_latency - routed_latency

        quality_preserved = sample.quality_delta >= -config.quality_tolerance
        compute_win = flops_savings > 0 or latency_delta_ms > 0

        if quality_preserved and compute_win:
            category = "quality_preserving_win"
        elif (not quality_preserved) and compute_win:
            category = "compute_only_win"
        elif not quality_preserved:
            category = "failure_quality_loss"
        elif recall == 0:
            category = "failure_zero_recall"
        else:
            category = "neutral"

        per_query.append(
            {
                "query_id": sample.query_id,
                "selected_domains": selected,
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "flops": {
                    "all": all_flops,
                    "routed": routed_flops,
                    "savings_fraction": flops_savings,
                },
                "latency_ms": {
                    "all": all_latency,
                    "routed": routed_latency,
                    "delta": latency_delta_ms,
                },
                "quality_delta": sample.quality_delta,
                "quality_preserved": quality_preserved,
                "category": category,
            }
        )

    def _avg(field: str) -> float:
        vals = [float(row[field]) for row in per_query]
        return float(fmean(vals)) if vals else 0.0

    summary = {
        "routing_precision": _avg("precision"),
        "routing_recall": _avg("recall"),
        "routing_f1": _avg("f1"),
        "avg_flops_savings_fraction": float(
            fmean([row["flops"]["savings_fraction"] for row in per_query])
            if per_query
            else 0.0
        ),
        "avg_latency_delta_ms": float(
            fmean([row["latency_ms"]["delta"] for row in per_query])
            if per_query
            else 0.0
        ),
    }

    quality_wins = [
        row for row in per_query if row["category"] == "quality_preserving_win"
    ]
    compute_only = [row for row in per_query if row["category"] == "compute_only_win"]
    failures = [
        row
        for row in per_query
        if row["category"] in {"failure_quality_loss", "failure_zero_recall"}
    ]

    return {
        "config": asdict(config),
        "summary": summary,
        "report": {
            "quality_preserving_routing_wins": {
                "count": len(quality_wins),
                "queries": [row["query_id"] for row in quality_wins],
            },
            "compute_only_wins": {
                "count": len(compute_only),
                "queries": [row["query_id"] for row in compute_only],
            },
            "failure_modes": {
                "count": len(failures),
                "queries": [row["query_id"] for row in failures],
                "breakdown": {
                    "quality_loss": sum(
                        row["category"] == "failure_quality_loss" for row in per_query
                    ),
                    "zero_recall": sum(
                        row["category"] == "failure_zero_recall" for row in per_query
                    ),
                },
            },
        },
        "per_query": per_query,
    }


def evaluate_threshold_robustness(
    samples: list[RoutingSample],
    config: ThresholdSweepConfig,
) -> dict[str, Any]:
    """Run threshold sweep and summarize robustness of routing behavior."""

    runs = []
    for threshold in config.semantic_thresholds:
        run_cfg = RoutingRunConfig(
            mode=config.mode,
            semantic_threshold=float(threshold),
            confidence_threshold=config.confidence_threshold,
            fallback=config.fallback,
            quality_tolerance=config.quality_tolerance,
        )
        run = evaluate_routing(samples, run_cfg)
        runs.append(
            {
                "semantic_threshold": float(threshold),
                "summary": run["summary"],
            }
        )

    def _range(key: str) -> float:
        values = [float(r["summary"][key]) for r in runs]
        return (max(values) - min(values)) if values else 0.0

    best = max(runs, key=lambda r: r["summary"]["routing_f1"]) if runs else None

    return {
        "config": asdict(config),
        "threshold_runs": runs,
        "robustness": {
            "best_semantic_threshold_by_f1": None
            if best is None
            else best["semantic_threshold"],
            "f1_range": _range("routing_f1"),
            "precision_range": _range("routing_precision"),
            "recall_range": _range("routing_recall"),
            "flops_savings_range": _range("avg_flops_savings_fraction"),
            "latency_delta_range_ms": _range("avg_latency_delta_ms"),
        },
    }


def render_routing_eval_markdown(report: dict[str, Any]) -> str:
    """Render a compact markdown summary for a routing evaluation artifact."""

    single_run = report["single_run"]
    summary = single_run["summary"]
    split = single_run["report"]
    robustness = report["threshold_robustness"]["robustness"]
    threshold_runs = report["threshold_robustness"]["threshold_runs"]

    lines = [
        "# Routing Evaluation Report",
        "",
        "## Single-run summary",
        f"- Routing precision: **{summary['routing_precision']:.3f}**",
        f"- Routing recall: **{summary['routing_recall']:.3f}**",
        f"- Routing F1: **{summary['routing_f1']:.3f}**",
        f"- Avg FLOPs savings fraction: **{summary['avg_flops_savings_fraction']:.3f}**",
        f"- Avg latency delta (all - routed, ms): **{summary['avg_latency_delta_ms']:.3f}**",
        "",
        "## Outcome split (wins and failures)",
        (
            f"- Quality-preserving routing wins: **{split['quality_preserving_routing_wins']['count']}** "
            f"({', '.join(split['quality_preserving_routing_wins']['queries']) or 'none'})"
        ),
        (
            f"- Compute-only wins (quality loss tolerated by config): "
            f"**{split['compute_only_wins']['count']}** "
            f"({', '.join(split['compute_only_wins']['queries']) or 'none'})"
        ),
        (
            f"- Failure modes: **{split['failure_modes']['count']}** "
            f"({', '.join(split['failure_modes']['queries']) or 'none'})"
        ),
        (
            "- Failure breakdown: "
            f"quality_loss={split['failure_modes']['breakdown']['quality_loss']}, "
            f"zero_recall={split['failure_modes']['breakdown']['zero_recall']}"
        ),
        "",
        "## Threshold robustness",
        (
            "- Best semantic threshold by F1: "
            f"**{robustness['best_semantic_threshold_by_f1']}**"
        ),
        f"- F1 range across sweep: **{robustness['f1_range']:.3f}**",
        f"- Precision range across sweep: **{robustness['precision_range']:.3f}**",
        f"- Recall range across sweep: **{robustness['recall_range']:.3f}**",
        (
            "- FLOPs savings range across sweep: "
            f"**{robustness['flops_savings_range']:.3f}**"
        ),
        "",
        "### Sweep table",
        "| Threshold | Precision | Recall | F1 | FLOPs savings | Latency delta ms |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for run in threshold_runs:
        run_summary = run["summary"]
        lines.append(
            "| "
            f"{run['semantic_threshold']:.2f} | "
            f"{run_summary['routing_precision']:.3f} | "
            f"{run_summary['routing_recall']:.3f} | "
            f"{run_summary['routing_f1']:.3f} | "
            f"{run_summary['avg_flops_savings_fraction']:.3f} | "
            f"{run_summary['avg_latency_delta_ms']:.3f} |"
        )

    lines.extend(
        [
            "",
            "### Per-query outcomes",
            "| Query | Selected domains | Precision | Recall | F1 | FLOPs savings | Latency delta ms | Quality delta | Category |",
            "|---|---|---:|---:|---:|---:|---:|---:|---|",
        ]
    )
    for row in single_run["per_query"]:
        lines.append(
            "| "
            f"{row['query_id']} | "
            f"{', '.join(row['selected_domains']) or '(none)'} | "
            f"{row['precision']:.3f} | "
            f"{row['recall']:.3f} | "
            f"{row['f1']:.3f} | "
            f"{row['flops']['savings_fraction']:.3f} | "
            f"{row['latency_ms']['delta']:.3f} | "
            f"{row['quality_delta']:.3f} | "
            f"{row['category']} |"
        )

    lines.append("")
    return "\n".join(lines)


def _load_samples(path: Path) -> list[RoutingSample]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload["samples"] if isinstance(payload, dict) else payload

    samples = []
    for row in rows:
        samples.append(
            RoutingSample(
                query_id=str(row["query_id"]),
                relevant_domains=tuple(row.get("relevant_domains", [])),
                semantic_scores={
                    k: float(v) for k, v in row["semantic_scores"].items()
                },
                confidence_scores=(
                    None
                    if row.get("confidence_scores") is None
                    else {k: float(v) for k, v in row["confidence_scores"].items()}
                ),
                routing_overhead_ms=float(row.get("routing_overhead_ms", 0.0)),
                domain_flops=(
                    None
                    if row.get("domain_flops") is None
                    else {k: float(v) for k, v in row["domain_flops"].items()}
                ),
                domain_latency_ms=(
                    None
                    if row.get("domain_latency_ms") is None
                    else {k: float(v) for k, v in row["domain_latency_ms"].items()}
                ),
                quality_delta=float(row.get("quality_delta", 0.0)),
            )
        )
    return samples


def _parse_thresholds(raw: str) -> tuple[float, ...]:
    values = [float(chunk.strip()) for chunk in raw.split(",") if chunk.strip()]
    if not values:
        raise ValueError("At least one threshold is required")
    return tuple(values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate semantic/confidence routing."
    )
    parser.add_argument(
        "--dataset", type=Path, required=True, help="JSON file with routing samples"
    )
    parser.add_argument(
        "--output", type=Path, required=True, help="Where to write JSON report"
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=None,
        help="Optional path for compact markdown report",
    )
    parser.add_argument(
        "--mode", choices=["semantic", "confidence"], default="semantic"
    )
    parser.add_argument("--semantic-threshold", type=float, default=0.7)
    parser.add_argument("--semantic-thresholds", type=str, default="0.5,0.6,0.7,0.8")
    parser.add_argument("--confidence-threshold", type=float, default=0.8)
    parser.add_argument("--quality-tolerance", type=float, default=0.0)

    args = parser.parse_args()
    samples = _load_samples(args.dataset)

    single = evaluate_routing(
        samples,
        RoutingRunConfig(
            mode=args.mode,
            semantic_threshold=args.semantic_threshold,
            confidence_threshold=args.confidence_threshold,
            quality_tolerance=args.quality_tolerance,
        ),
    )
    robustness = evaluate_threshold_robustness(
        samples,
        ThresholdSweepConfig(
            mode=args.mode,
            semantic_thresholds=_parse_thresholds(args.semantic_thresholds),
            confidence_threshold=args.confidence_threshold,
            quality_tolerance=args.quality_tolerance,
        ),
    )

    report = {
        "schema_version": "routing_eval.v1",
        "single_run": single,
        "threshold_robustness": robustness,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if args.markdown_output is not None:
        args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_output.write_text(
            render_routing_eval_markdown(report), encoding="utf-8"
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
