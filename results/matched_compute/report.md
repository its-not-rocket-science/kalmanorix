# Matched Compute: Specialists vs Monolith

## Assumptions

- Training budget proxy uses `6 * params_proxy * tokens * epochs`.
- Specialists: 4 models, 1 epoch each.
- Monolith: 1 epochs (chosen to match training compute).
- Inference FLOPs proxy includes specialist invocation cost and routing overhead.

## Results

| Strategy | Acc@1 | MRR | Train Budget Proxy | Inference FLOPs Proxy (mean) | Active Specialists (mean) |
|---|---:|---:|---:|---:|---:|
| monolith_baseline | 1.0000 | 1.0000 | 4246732800 | 442368.0 | 1.00 |
| specialists_all_routing | 1.0000 | 1.0000 | 4246732800 | 1822556.0 | 4.00 |
| specialists_semantic_routing | 1.0000 | 1.0000 | 4246732800 | 973209.0 | 2.00 |
| specialists_kalman_fusion | 1.0000 | 1.0000 | 4246732800 | 1017445.0 | 2.00 |

## Fairness checks

- Training compute parity achieved: **True** 
(ratio=1.0000, tolerance=0.0100).
- Validation checks passed: **True** (missing/inconsistent assumptions raise errors).
- Inference FLOPs ratios vs monolith:
  - `monolith_baseline`: 1.000x
  - `specialists_all_routing`: 4.120x
  - `specialists_semantic_routing`: 2.200x
  - `specialists_kalman_fusion`: 2.300x

## Conclusion quality

Training parity was enforced, but inference-cost asymmetry remains material; this run does not support a strong conclusion about overall efficiency superiority.
