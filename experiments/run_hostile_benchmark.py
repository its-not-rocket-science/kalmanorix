#!/usr/bin/env python3
"""Run a hostile canonical benchmark configuration as a credibility layer."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.run_canonical_benchmark import run_canonical_benchmark


def _load_hostile_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = {"benchmark_path", "split", "max_queries", "seed", "num_resamples"}
    missing = sorted(required.difference(payload))
    if missing:
        raise ValueError(f"Missing required hostile config keys: {missing}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run hostile benchmark credibility layer using canonical runner"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("experiments/configs/kalman_hostile_benchmark_v1.json"),
        help="Hostile benchmark JSON config.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional artifact output override.",
    )
    args = parser.parse_args()

    cfg = _load_hostile_config(args.config)
    output_dir = Path(cfg.get("output_dir", "results/kalman_hostile_benchmark_v1"))
    if args.output_dir is not None:
        output_dir = args.output_dir

    run_canonical_benchmark(
        benchmark_path=Path(cfg["benchmark_path"]),
        output_dir=output_dir,
        split=str(cfg["split"]),
        max_queries=int(cfg["max_queries"]),
        device=str(cfg.get("device", "cpu")),
        seed=int(cfg["seed"]),
        num_resamples=int(cfg["num_resamples"]),
        confirmatory_slice=str(
            cfg.get(
                "confirmatory_slice",
                "preregistered_high_disagreement_high_uncertainty_multi_domain_low_router_confidence",
            )
        ),
    )


if __name__ == "__main__":
    main()
