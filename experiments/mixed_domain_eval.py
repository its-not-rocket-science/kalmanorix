#!/usr/bin/env python3
"""Legacy mixed-domain evaluation entrypoint.

This script previously ran a synthetic toy-corpus benchmark with handcrafted
embedders. It is now a compatibility wrapper so the default execution path uses
real mixed-domain retrieval data and real specialist models.

Default behavior (primary validation):
    python experiments/mixed_domain_eval.py
        -> runs experiments/run_real_mixed_benchmark.py

Debug-only behavior (synthetic smoke):
    python experiments/mixed_domain_eval.py --debug-synthetic
        -> runs experiments/validate_fusion.py --debug-synthetic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.validate_fusion import run_debug_synthetic_smoke


def main() -> None:
    """Dispatch to real benchmark (default) or synthetic debug smoke path."""
    parser = argparse.ArgumentParser(
        description=(
            "Compatibility wrapper: real benchmark by default; synthetic path is "
            "debug-only"
        )
    )
    parser.add_argument(
        "--debug-synthetic",
        action="store_true",
        help="Run synthetic toy smoke checks only (non-headline validation)",
    )
    args = parser.parse_args()

    if args.debug_synthetic:
        metrics = run_debug_synthetic_smoke()
        print("[DEBUG ONLY] synthetic smoke metrics")
        print(metrics)
        return

    from experiments.run_real_mixed_benchmark import main as run_real_main

    run_real_main()


if __name__ == "__main__":
    main()
