# Milestone 2.2 — NOT COMPLETED YET

## Status

The final, report-grade OOD robustness benchmark artifact has not been recorded
in this folder yet.

## Target completion date (from roadmap)

- **August 2026** (Milestone 2.2, within Q3 2026 roadmap window).

## Success criteria

- Kalman fusion shows smaller OOD performance drop than naive averaging.
- Target effect from roadmap: approximately 20% smaller drop.
- Difference is statistically significant (**p < 0.05**).

## Exact benchmark commands

```bash
pip install -e ".[train]"
python experiments/run_milestone_2_2.py --config experiments/configs/milestone_2_2.yaml
python experiments/validate_covariance.py
python experiments/validate_fusion.py
```

## Expected output file locations

- `results/milestone_2_2/ood_robustness_metrics.csv`
- `results/milestone_2_2/covariance_ablation_results.json`
- `results/milestone_2_2/statistical_significance.md`

## Actual results (to be filled later)

- Date run:
- Commit SHA:
- Config path:
- OOD drop (Kalman):
- OOD drop (Averaging):
- Relative drop reduction (%):
- Statistical test + p-value:
- Decision (met success criteria?):
- Notes:
