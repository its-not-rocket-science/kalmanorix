# Subsystem contribution summary

## Key interpretation
- Routing vs adaptive fusion: routed specialists + mean improves quality over all specialists + mean in this artifact, while reducing specialist count.
- Uncertainty estimation after routing: unavailable directly because routed+Kalman is not present in the source summary.
- Trade-off frontier: the simplest oracle routing top-1 condition is strongest on latency/FLOPs and competitive on quality in this small run.

## Caveats
- This decomposition uses canonical_benchmark_v2 (6 queries), so it is directional not claim-ready.
- Oracle routing + mean and oracle routing + Kalman collapse to the same top-1 routed condition.

## Routing precision/recall (small routing eval)
- Precision (micro): None
- Recall (micro): None
- F1 (micro): None
