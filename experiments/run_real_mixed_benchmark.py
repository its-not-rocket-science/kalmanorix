#!/usr/bin/env python3
"""Run real mixed-domain benchmark via the benchmark registry."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any


from experiments.registry.config_schema import load_experiment_config
from experiments.registry.runner import DEFAULT_REAL_SPECIALISTS, run_experiment


def run_real_benchmark(
    benchmark_path: Path,
    split: str,
    max_queries: int | None,
    output_path: Path,
    device: str,
) -> dict[str, Any]:
    """Backwards-compatible wrapper for real mixed benchmark."""
    config_payload = {
        "name": "real-mixed-compat",
        "experiment_type": "real_mixed",
        "seed": {"python": 42, "numpy": 42, "torch": 42},
        "artifacts": {"summary_json": str(output_path)},
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
        "fusion": {"strategies": ["mean", "kalman"], "routing_mode": "all"},
        "evaluation": {"kind": "locked_protocol"},
        "reporting": {"print_stdout": False},
    }

    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "real_mixed_compat.json"
        cfg_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        cfg = load_experiment_config(cfg_path)
        summary = run_experiment(cfg)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run real mixed-domain retrieval benchmark"
    )
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_mixed_benchmark/real_benchmark_summary.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    summary = run_real_benchmark(
        benchmark_path=args.benchmark_path,
        split=args.split,
        max_queries=args.max_queries,
        output_path=args.output,
        device=args.device,
    )
    print(json.dumps(summary["delta_last_minus_first"], indent=2))


if __name__ == "__main__":
    main()
