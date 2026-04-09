#!/usr/bin/env python3
"""Run fusion baseline suite from JSON or YAML config.

Legacy notice:
This script emits legacy ad-hoc metrics (recall@1, recall@5, mrr) for backward
compatibility. New experiments should use `kalmanorix.benchmarks.evaluation_protocol`
and persist the preregistered metric suite.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any
import warnings

from kalmanorix.benchmarks.fusion_baselines import run_experiment


def _load_config(path: Path) -> dict[str, Any]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("PyYAML is required for YAML configs") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def main() -> None:
    warnings.warn(
        "experiments/run_fusion_baseline_suite.py is legacy; use preregistered "
        "evaluation_protocol metrics for new experiments.",
        DeprecationWarning,
        stacklevel=2,
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", required=True, type=Path, help="Path to config file"
    )
    parser.add_argument(
        "--output", required=True, type=Path, help="Path to output JSON"
    )
    args = parser.parse_args()

    config = _load_config(args.config)
    results = run_experiment(config)

    payload = [
        {
            "strategy": result.strategy_name,
            "recall@1": result.recall_at_1,
            "recall@5": result.recall_at_5,
            "mrr": result.mrr,
        }
        for result in results
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
