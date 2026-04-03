#!/usr/bin/env python3
"""Visualize Kalman vs Mean retrieval performance across domains."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "matplotlib is required. Install with `pip install matplotlib`."
    ) from exc


def load_rows(csv_path: Path):
    with csv_path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-csv",
        type=Path,
        default=Path("results/milestone_1_3_kalman_vs_mean.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/kalman_vs_mean_plot.png"),
    )
    args = parser.parse_args()

    rows = load_rows(args.results_csv)
    if not rows:
        raise ValueError(f"No rows found in {args.results_csv}")

    by_domain = defaultdict(lambda: {"k_h1": [], "m_h1": [], "k_mrr": [], "m_mrr": []})
    for row in rows:
        domain = row["domain"]
        by_domain[domain]["k_h1"].append(float(row["kalman_hit1"]))
        by_domain[domain]["m_h1"].append(float(row["mean_hit1"]))
        by_domain[domain]["k_mrr"].append(float(row["kalman_mrr"]))
        by_domain[domain]["m_mrr"].append(float(row["mean_mrr"]))

    domains = sorted(by_domain.keys())
    kalman_hit1 = [np.mean(by_domain[d]["k_h1"]) for d in domains]
    mean_hit1 = [np.mean(by_domain[d]["m_h1"]) for d in domains]
    kalman_mrr = [np.mean(by_domain[d]["k_mrr"]) for d in domains]
    mean_mrr = [np.mean(by_domain[d]["m_mrr"]) for d in domains]

    x = np.arange(len(domains))
    w = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    axes[0].bar(x - w / 2, kalman_hit1, width=w, label="Kalman", color="#2E7D32")
    axes[0].bar(x + w / 2, mean_hit1, width=w, label="Mean", color="#1565C0")
    axes[0].set_title("Hit@1 by Domain")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(domains)
    axes[0].set_ylim(0.0, 1.0)
    axes[0].grid(axis="y", alpha=0.2)
    axes[0].legend()

    axes[1].bar(x - w / 2, kalman_mrr, width=w, label="Kalman", color="#2E7D32")
    axes[1].bar(x + w / 2, mean_mrr, width=w, label="Mean", color="#1565C0")
    axes[1].set_title("MRR by Domain")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(domains)
    axes[1].set_ylim(0.0, 1.0)
    axes[1].grid(axis="y", alpha=0.2)
    axes[1].legend()

    fig.suptitle("Milestone 1.3: Kalman Fusion vs Simple Mean")
    fig.tight_layout()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, bbox_inches="tight")
    print(f"Saved plot: {args.output}")


if __name__ == "__main__":
    main()
