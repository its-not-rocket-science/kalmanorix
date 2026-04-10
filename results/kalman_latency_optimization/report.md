# Kalman latency optimization

## Microbenchmark
- MeanFuser mean latency: 0.267 ms
- Kalman (legacy) mean latency: 1.633 ms
- Kalman (optimized) mean latency: 0.793 ms
- Speedup (legacy -> optimized): 2.06x
- Optimized Kalman / Mean latency ratio: 2.97x

## Hot-path observations
- Embed calls observed: 10800 (0.382s total)
- Sigma² calls observed: 4800 (0.306s total)
- Legacy Kalman spent significant time building repeated diagonal covariance vectors.
- Optimized path removes per-dimension covariance materialization for sigma²*I case.

## Semantics
- The optimized path preserves the same scalar Kalman gain sequence for sigma²*I covariance.
- Numerical regression tests enforce closeness to legacy behavior.
