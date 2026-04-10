#!/usr/bin/env python3
"""Benchmark adaptive route-or-fuse policy against fixed baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from experiments.registry.config_schema import load_experiment_config
from experiments.registry.runner import DEFAULT_REAL_SPECIALISTS, run_experiment


def _mode_outcomes(
    *,
    query_ids: list[str],
    policy_usage: dict[str, dict[str, Any]],
    strategy_rankings: dict[str, list[str]],
    ground_truth: dict[str, set[str]],
) -> dict[str, Any]:
    mode_to_hits: dict[str, list[float]] = {}
    for qid in query_ids:
        mode = policy_usage[qid]["mode"]
        ranked = strategy_rankings[qid]
        hit = 1.0 if ranked and ranked[0] in ground_truth[qid] else 0.0
        mode_to_hits.setdefault(mode, []).append(hit)

    return {
        mode: {
            "num_queries": len(values),
            "top1_success": (sum(values) / len(values)) if values else 0.0,
        }
        for mode, values in sorted(mode_to_hits.items())
    }


def run_adaptive_route_or_fuse(
    *,
    benchmark_path: Path,
    output_dir: Path,
    split: str,
    max_queries: int | None,
    device: str,
    seed: int,
    selector_type: str,
) -> dict[str, Any]:
    config_payload = {
        "name": "adaptive-route-or-fuse",
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
            "strategies": ["kalman"],
            "routing_mode": "all",
            "options": {
                "router_top_k": 2,
                "adaptive_selector_type": selector_type,
            },
        },
        "evaluation": {"kind": "locked_protocol"},
        "reporting": {"print_stdout": False},
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "adaptive_route_or_fuse.json"
        cfg_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        cfg = load_experiment_config(cfg_path)
        details = run_experiment(cfg)

    (output_dir / "runner_details.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )

    query_level = details["query_level"]
    rankings = query_level["rankings"]
    ground_truth = {
        qid: set(ids) for qid, ids in query_level["ground_truth"].items()
    }
    adaptive_usage = query_level["policy_usage"].get("adaptive_route_or_fuse", {})

    if not adaptive_usage:
        raise ValueError("adaptive_route_or_fuse policy telemetry missing from runner")

    selected_strategies = [
        "router_only_top1",
        "uniform_mean_fusion",
        "kalman",
        "adaptive_route_or_fuse",
    ]
    query_ids = sorted(rankings["adaptive_route_or_fuse"])
    frequencies: dict[str, int] = {}
    for qid in query_ids:
        frequencies[adaptive_usage[qid]["mode"]] = frequencies.get(adaptive_usage[qid]["mode"], 0) + 1

    summary = {
        "benchmark": {
            "path": str(benchmark_path),
            "split": split,
            "max_queries": max_queries,
            "seed": seed,
        },
        "selector_type": selector_type,
        "selection_frequencies": frequencies,
        "per_mode_outcomes": _mode_outcomes(
            query_ids=query_ids,
            policy_usage=adaptive_usage,
            strategy_rankings=rankings["adaptive_route_or_fuse"],
            ground_truth=ground_truth,
        ),
        "strategy_metrics": {
            name: details["results"][name]["global_primary"]
            for name in selected_strategies
        },
    }

    report_lines = [
        "# Adaptive Route-or-Fuse Benchmark",
        "",
        f"- Selector type: `{selector_type}`",
        f"- Benchmark: `{benchmark_path}` ({split})",
        f"- Max queries: `{max_queries}`",
        "",
        "## Selection Frequencies",
    ]
    total = max(len(query_ids), 1)
    for mode, count in sorted(frequencies.items()):
        report_lines.append(f"- {mode}: {count} ({count / total:.1%})")

    report_lines.extend(["", "## Per-mode outcomes (adaptive)"])
    for mode, payload in summary["per_mode_outcomes"].items():
        report_lines.append(
            f"- {mode}: n={payload['num_queries']}, top1_success={payload['top1_success']:.4f}"
        )

    report_lines.extend(["", "## Strategy comparison (global_primary)"])
    for name in selected_strategies:
        metrics = summary["strategy_metrics"][name]
        report_lines.append(
            f"- {name}: mrr={metrics['mrr']['mean']:.4f}, recall@1={metrics['recall@1']['mean']:.4f}, recall@5={metrics['recall@5']['mean']:.4f}"
        )

    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "report.md").write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run adaptive route-or-fuse benchmark")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.1.0/mixed_benchmark.json"),
    )
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--max-queries", type=int, default=600)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--selector-type", type=str, default="rule", choices=["rule", "learned"])
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/adaptive_route_or_fuse")
    )
    args = parser.parse_args()

    summary = run_adaptive_route_or_fuse(
        benchmark_path=args.benchmark_path,
        output_dir=args.output_dir,
        split=args.split,
        max_queries=args.max_queries,
        device=args.device,
        seed=args.seed,
        selector_type=args.selector_type,
    )
    print(json.dumps(summary["selection_frequencies"], indent=2))


if __name__ == "__main__":
    main()
