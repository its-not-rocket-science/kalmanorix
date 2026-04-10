# Benchmark Results Registry

This directory tracks **verifiable benchmark artifacts** for Kalmanorix milestones.

## Transparency policy

Core scientific validation results are still pending for several roadmap milestones.
We intentionally include explicit placeholders for unfinished benchmarks rather than
publishing speculative outcomes.

- Milestone 1.3 (Kalman vs averaging): **pending final statistical report**
- Milestone 2.1 (specialists vs monolith): **pending final statistical report**
- Milestone 2.2 (OOD robustness): **pending final statistical report**
- Milestone 2.3 (efficiency): **completed artifact directory present**

Post-canonical closure scaffolds (structure only, not completed studies):
- `results/matched_compute/{summary_template.json, report_template.md}`
- `results/uncertainty_ablation/{summary_template.json, report_template.md}`
- `results/ood_robustness/{summary_template.json, report_template.md}`

For overall status and target dates, see:
- Project roadmap: `docs/contributing/roadmap.md`
- Top-level project status: `README.md`

## Conventions

- Each milestone folder contains reproducibility instructions and output locations.
- Any completed claim should be backed by committed raw outputs (CSV/JSON) and an
  analysis artifact (notebook/script/report).
- Placeholders (`NOT_COMPLETED_YET.md`) must be replaced only after running the
  benchmark and recording statistically valid results.
