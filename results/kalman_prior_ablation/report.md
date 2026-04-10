# Kalman Prior Ablation

Does a stronger prior make Kalman fusion materially more useful?

**Answer:** Yes in this benchmark: kalman_generalist_prior achieved the best latency-normalized error and should be preferred over current Kalman.

## Metrics

| Method | Mean Error | Latency (ms) | Latency-Normalized Error | Calibration ECE |
| --- | ---: | ---: | ---: | ---: |
| mean_fusion | 0.12699 | 2.400 | 0.30477 | 0.10704 |
| kalman_current | 0.09369 | 2.600 | 0.24358 | 0.08687 |
| kalman_generalist_prior | 0.08645 | 2.600 | 0.22477 | 0.08069 |
| kalman_learned_linear_prior | 0.09270 | 2.600 | 0.24103 | 0.08638 |
| kalman_residuals | 0.09369 | 2.600 | 0.24358 | 0.08687 |

## Learned Prior Fit

- fit split: `train`
- n_fit: 220
- weights: [0.5391442309856102, 0.20632303087624604, 0.15615839214370839, 0.09837434599443551]
