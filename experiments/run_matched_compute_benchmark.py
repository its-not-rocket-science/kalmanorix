"""Run matched-compute specialists-vs-monolith benchmark."""

from __future__ import annotations

import argparse
from pathlib import Path

from kalmanorix.benchmarks.matched_compute import run_matched_compute_benchmark


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/matched_compute"),
        help="Directory where summary.json and report.md are written.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--samples-per-domain", type=int, default=1200)
    parser.add_argument("--test-size", type=int, default=600)
    parser.add_argument("--semantic-top-k", type=int, default=2)
    args = parser.parse_args()

    run_matched_compute_benchmark(
        output_dir=args.output_dir,
        seed=args.seed,
        samples_per_domain=args.samples_per_domain,
        test_size=args.test_size,
        semantic_top_k=args.semantic_top_k,
    )


if __name__ == "__main__":
    main()
