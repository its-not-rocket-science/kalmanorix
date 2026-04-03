# Milestone 1.3 Reproduction Instructions

## Goal

Compare Kalman fusion against averaging on mixed-domain retrieval and generate
artifacts suitable for statistical hypothesis validation.

## Prerequisites

```bash
pip install -e ".[dev]"
```

## Run sequence

```bash
python experiments/benchmark_fusion_methods.py
python experiments/compare_fusion_strategies.py
```

## Save outputs into milestone folder

When benchmark scripts complete, copy/export their final artifacts to:

- `results/milestone_1_3/fusion_benchmark_raw.csv`
- `results/milestone_1_3/fusion_benchmark_summary.json`
- `results/milestone_1_3/statistical_test_report.md`

## Validation checklist

- [ ] Kalman and averaging both evaluated on the same test split.
- [ ] Statistical test reported with explicit p-value and alpha threshold.
- [ ] Output JSON validates against `expected_output_schema.json`.
- [ ] Commit SHA and run date recorded in summary output.
