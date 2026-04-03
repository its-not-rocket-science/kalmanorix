# Milestone 2.1 — NOT COMPLETED YET

## Status

The final, report-grade benchmark artifact for Specialists vs Monolith has not
been recorded in this folder yet.

## Target completion date (from roadmap)

- **July 2026** (Milestone 2.1, within Q3 2026 roadmap window).

## Success criteria

- Fused specialists outperform monolithic model on mixed-domain evaluation.
- Compute parity is maintained (comparable total training FLOPs).
- Improvement is statistically significant (**p < 0.05**).

## Exact benchmark commands

```bash
pip install -e ".[train]"
python experiments/create_configs.py
python experiments/run_milestone_2_1.py --config experiments/configs/milestone_2_1.yaml
python experiments/evaluate_milestone_2_1.py --config experiments/configs/milestone_2_1.yaml
```

## Expected output file locations

- `results/milestone_2_1/specialists_vs_monolith_metrics.csv`
- `results/milestone_2_1/compute_parity_report.json`
- `results/milestone_2_1/statistical_significance.md`

## Actual results (to be filled later)

- Date run:
- Commit SHA:
- Config path:
- Specialist metric(s):
- Monolith metric(s):
- Compute parity evidence:
- Statistical test + p-value:
- Decision (met success criteria?):
- Notes:
