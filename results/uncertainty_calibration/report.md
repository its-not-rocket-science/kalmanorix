# Uncertainty calibration report

- Selected objective: `distance_to_relevant_doc_centroid`
- Selection rule: Select objective maximizing validation diagnostic gain + validation Kalman-vs-Mean delta change.
- Powered for calibration: `True`
- Minimum support threshold: `24`
- Per-specialist support: `{'tech': 43, 'cook': 31}`
- Fallback reason: `None`

## Split diagnostics

- Split counts: `{'train': 119, 'validation': 62, 'test': 59}`
- Validation domains: `{'cook': 19, 'mixed': 12, 'tech': 31}`
- Validation query buckets: `{'cross_domain_compositional': 22, 'direct': 8, 'long_form': 32}`

## Selected calibrators

- `cook`: calibrator=`isotonic`, fallback=`False`, mse=`0.125773`, support=`31`
- `tech`: calibrator=`isotonic`, fallback=`False`, mse=`0.088322`, support=`43`

## Kalman-vs-Mean (calibrated sigma2)

- Validation delta change: `0.000000`
- Test delta change: `0.000000`

If delta change is non-positive, calibration did not improve the downstream benchmark under this powered regime.
