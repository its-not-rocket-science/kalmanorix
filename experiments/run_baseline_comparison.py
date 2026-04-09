#!/usr/bin/env python3
"""Run real mixed benchmark with mandatory Kalman + rigorous baseline comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from experiments.registry.config_schema import load_experiment_config
from experiments.registry.runner import DEFAULT_REAL_SPECIALISTS, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-path", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=150)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--generalist-model-name", default="general_qa")
    parser.add_argument("--router-top-k", type=int, default=2)
    args = parser.parse_args()

    payload = {
        "name": "real-mixed-baseline-comparison",
        "experiment_type": "real_mixed",
        "seed": {"python": 42, "numpy": 42, "torch": 42},
        "artifacts": {"summary_json": str(args.output)},
        "dataset": {
            "kind": "mixed_parquet",
            "path": str(args.benchmark_path),
            "split": args.split,
            "max_queries": args.max_queries,
        },
        "models": {
            "kind": "hf_specialists",
            "device": args.device,
            "specialists": DEFAULT_REAL_SPECIALISTS,
        },
        "fusion": {
            "strategies": ["mean", "kalman"],
            "routing_mode": "all",
            "options": {
                "generalist_model_name": args.generalist_model_name,
                "router_top_k": args.router_top_k,
                "fixed_weights": {"general_qa": 0.5, "biomedical": 0.25, "finance": 0.25},
            },
        },
        "evaluation": {"kind": "locked_protocol"},
        "reporting": {"print_stdout": False},
    }

    with TemporaryDirectory() as tmpdir:
        config_path = Path(tmpdir) / "baseline_comparison.json"
        config_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        config = load_experiment_config(config_path)
        summary = run_experiment(config)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["kalman_guardrail"], indent=2))


if __name__ == "__main__":
    main()
