# Milestone 2.3 — Efficiency Benchmarking (Completed)

This folder stores the concrete artifacts for the efficiency milestone.

## What was proven

- Semantic routing reduced FLOPs by selecting only relevant specialists.
- Efficiency gains scale with specialist count by avoiding unnecessary model calls.
- The benchmark package here is intended to preserve reproducible evidence for the
  roadmap claim that modular routing yields a substantial compute advantage.

## Artifacts in this folder

- `semantic_routing_benchmark.csv` — tabular benchmark snapshot.
- `flops_reduction_analysis.ipynb` — companion analysis notebook.

If additional reruns are added, include timestamped variants and keep this README
updated with provenance information (commit SHA, environment, dataset/config).
