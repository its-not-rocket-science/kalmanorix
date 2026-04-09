# Canonical Benchmark Report

This is a placeholder artifact.

To generate the full report (NDCG@10, Recall@10, MRR@10, latency, FLOPs proxy, confidence intervals, and paired Kalman-vs-Mean statistics), run:

```bash
kalmanorix-run-canonical-benchmark \
  --benchmark-path benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet \
  --split test \
  --output-dir results/canonical_benchmark
```

Interpretation policy:
- Do not claim KalmanorixFuser improves over MeanFuser unless the paired statistical section indicates a statistically significant improvement for the configured run.
- If no significance is observed, report that clearly.
