| Method | nDCG@10 | MRR@10 | Recall@10 | latency | FLOPs proxy |
|---|---|---|---|---|---|
| KalmanorixFuser | 0.0941 | 0.1025 | 0.1358 | 202.858 | 3.000 |
| MeanFuser | 0.0943 | 0.1022 | 0.1362 | 198.558 | 3.000 |
| fixed weighted mean baseline | 0.0940 | 0.1018 | 0.1363 | 2.004 | 3.000 |
| hard routing baseline | 0.0979 | 0.1147 | 0.1299 | 6.433 | 1.000 |
| learned linear combiner | 0.0934 | 0.1058 | 0.1334 | 1.942 | 2.000 |
| single generalist model | 0.0952 | 0.1090 | 0.1287 | 2.108 | 1.000 |
| all-routing + mean baseline | 0.0940 | 0.1018 | 0.1363 | 2.043 | 3.000 |

> Note: fast-local uses deterministic hash embeddings and is a CPU-feasible smoke/benchmark mode.

