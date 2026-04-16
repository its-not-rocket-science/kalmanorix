# Benchmark Results Registry

This directory tracks **verifiable benchmark artifacts** for Kalmanorix milestones.

## Transparency policy

Core scientific validation results are still pending for several roadmap milestones.
We intentionally include explicit placeholders for unfinished benchmarks rather than
publishing speculative outcomes.

- Milestone 1.3 (Kalman vs averaging): **pending final statistical report**
- Milestone 2.1 (specialists vs monolith): **completed matched-compute artifact with mixed/inconclusive verdict**
- Milestone 2.2 (OOD robustness): **completed guarded artifact with inconclusive verdict**
- Milestone 2.3 (efficiency): **completed artifact directory present**

## Release-quality directory contract

For each major `results/<track>/` directory, keep:

1. `README.md` or `report.md` (human-readable context)
2. a reproducibility command documented in `README.md` or `report.md`
3. `summary.json` with a machine-readable index

Newly normalized track summaries use `schema_version = "phase_eval.v1"` for indexing,
while legacy run-level summaries are preserved verbatim for provenance.

## Naming consistency map (directory ↔ runner module)

| Results directory | Primary generator |
|---|---|
| `results/canonical_benchmark/` | `experiments/run_canonical_benchmark.py` |
| `results/canonical_benchmark_v2/` | `experiments/run_canonical_benchmark.py` |
| `results/canonical_benchmark_v3/` | `experiments/run_canonical_benchmark.py` |
| `results/matched_compute/` | `experiments/run_matched_compute_benchmark.py` |
| `results/ood_robustness/` | `experiments/run_kalman_assumption_stress_test.py` |
| `results/routing_eval/` | `kalmanorix-eval-routing` CLI |

Use snake_case naming in both result folders and experiment entrypoints.
When introducing a new benchmark, prefer `results/<benchmark_track>/` and `experiments/run_<benchmark_track>.py`.

## Provenance note

Do not delete older milestone directories or historical artifacts. Add new versioned folders
(e.g., `canonical_benchmark_v4`) when protocol changes are substantial.

For overall status and target dates, see:
- Project roadmap: `docs/contributing/roadmap.md`
- Top-level project status: `README.md`
