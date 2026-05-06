| Comparison | Method | nDCG@10 | MRR@10 | Recall@10 | latency | FLOPs proxy |
|---|---|---|---|---|---|---|
| kalman_vs_mean | KalmanorixFuser vs mean | -0.000203 | 0.0003 | -0.0005 | 4.300 | 0.000 |
| kalman_vs_fixed_weighted_mean_fusion | KalmanorixFuser vs fixed_weighted_mean_fusion | 0.000105 | 0.0007 | -0.0005 | 200.855 | 0.000 |
| kalman_vs_router_only_top1 | KalmanorixFuser vs router_only_top1 | -0.003747 | -0.0122 | 0.0059 | 196.425 | 2.000 |
| kalman_vs_learned_linear_combiner | KalmanorixFuser vs learned_linear_combiner | 0.000661 | -0.0033 | 0.0024 | 200.917 | 1.000 |

> Note: fast-local uses deterministic hash embeddings and is a CPU-feasible smoke/benchmark mode.

