# Example Calibration Report

This is an example of the standard calibration report emitted by `experiments/run_milestone_2_2.py`.

| Model | N | ECE | Brier | Mean confidence | Mean accuracy | Overconfidence gap |
|---|---:|---:|---:|---:|---:|---:|
| specialist_medical | 500 | 0.0621 | 0.1840 | 0.6412 | 0.6033 | 0.0379 |
| specialist_legal | 500 | 0.0742 | 0.2014 | 0.6557 | 0.5890 | 0.0667 |
| monolith | 500 | 0.0553 | 0.1761 | 0.6244 | 0.6120 | 0.0124 |
| fusion_kalman | 500 | 0.0408 | 0.1597 | 0.6115 | 0.6019 | 0.0096 |
| fusion_mean | 500 | 0.0699 | 0.1926 | 0.6468 | 0.5857 | 0.0611 |
| ablation_constant_variance | 500 | 0.0810 | 0.2148 | 0.6682 | 0.5755 | 0.0927 |

## Reliability plots

- Specialists: `reliability_specialist_<domain>.png`
- Fusion: `reliability_kalman.png`, `reliability_mean.png`
- Baselines: `reliability_monolith.png`, `reliability_ablation_constant.png`
