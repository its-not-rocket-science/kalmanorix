# Uncertainty calibration report

- Selected objective: `rank_error_proxy`
- Selection rule: Select objective maximizing validation diagnostic gain + validation Kalman-vs-Mean delta change.
- Powered for calibration: `True`
- Minimum support threshold: `6`
- Per-specialist support: `{'tech': 15, 'cook': 11}`
- Fallback reason: `None`

## Split diagnostics

- Split counts: `{'train': 46, 'validation': 22, 'test': 22}`
- Validation domains: `{'cook': 7, 'mixed': 4, 'tech': 11}`
- Validation query buckets: `{'cross_domain_compositional': 8, 'direct': 2, 'long_form': 12}`

## Selected calibrators

- `cook`: calibrator=`isotonic`, fallback=`False`, mse=`0.004345`, support=`11`
- `tech`: calibrator=`isotonic`, fallback=`False`, mse=`0.001228`, support=`15`

## Kalman-vs-Mean (calibrated sigma2)

- Validation delta change: `0.000000`
- Test delta change: `0.000000`

If delta change is non-positive, calibration did not improve the downstream benchmark under this powered regime.
