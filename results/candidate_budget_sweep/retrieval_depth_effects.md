# Retrieval Depth Effects

## Interpretation
- Best nDCG@10 delta for small budgets (<=100): +0.0949 at max_candidates=10.
- Best nDCG@10 delta for large budgets (>=250): +0.0949 at max_candidates=250.
- Latency scaled disproportionately when latency ratio increased with budget while deltas remained near zero.
- Mean fusion remained competitive whenever Kalman deltas were non-significant or below practical threshold (|Δ|<0.005).