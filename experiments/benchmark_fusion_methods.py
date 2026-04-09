#!/usr/bin/env python3
"""Deprecated synthetic benchmark.

This script is retained only for smoke/debug checks.
For primary validation, use:
    python experiments/run_real_mixed_benchmark.py

If emitting JSON in downstream wrappers, preserve the real benchmark schema,
including a top-level "p_values" object.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from experiments.validate_fusion import run_debug_synthetic_smoke


def main() -> None:
    """Run debug synthetic smoke benchmark only."""
    metrics = run_debug_synthetic_smoke()
    print("[DEBUG ONLY] synthetic smoke metrics")
    print(metrics)


if __name__ == "__main__":
    main()
