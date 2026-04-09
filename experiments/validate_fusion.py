"""Fusion validation entrypoint backed by benchmark registry configs."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory


sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from experiments.registry.config_schema import load_experiment_config
from experiments.registry.runner import DEFAULT_REAL_SPECIALISTS, run_experiment


def run_debug_synthetic_smoke(seed: int = 42) -> dict[str, float]:
    """Quick synthetic sanity check (debug only, non-headline)."""
    config_payload = {
        "name": "synthetic-debug-smoke",
        "experiment_type": "synthetic_smoke",
        "seed": {"python": seed, "numpy": seed, "torch": seed},
        "artifacts": {"summary_json": "results/tmp/synthetic_smoke.json"},
        "dataset": {"kind": "synthetic_toy", "split": "test"},
        "models": {"kind": "debug_keyword", "device": "cpu"},
        "fusion": {"strategies": ["kalman", "mean"], "routing_mode": "all"},
        "evaluation": {"kind": "synthetic_recall"},
        "reporting": {"print_stdout": False},
    }
    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "synthetic.json"
        cfg_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        cfg = load_experiment_config(cfg_path)
        summary = run_experiment(cfg)
    return summary["metrics"]


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-synthetic", action="store_true")
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

    if args.debug_synthetic:
        metrics = run_debug_synthetic_smoke()
        print("Debug synthetic smoke:")
        print(metrics)
        return

    config_payload = {
        "name": "validate-fusion-real-mixed",
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
        "fusion": {"strategies": ["mean", "kalman"], "routing_mode": "all"},
        "evaluation": {"kind": "locked_protocol"},
    }

    with TemporaryDirectory() as tmpdir:
        cfg_path = Path(tmpdir) / "real_mixed.json"
        cfg_path.write_text(json.dumps(config_payload, indent=2), encoding="utf-8")
        cfg = load_experiment_config(cfg_path)
        summary = run_experiment(cfg)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print("Primary (real-data) benchmark complete.")
    print(summary["delta_last_minus_first"])


if __name__ == "__main__":
    main()
