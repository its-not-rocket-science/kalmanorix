# Canonical Benchmark v3 (v1.2.0 hard benchmark)

This folder is reserved for canonical benchmark outputs on `benchmarks/mixed_beir_v1.2.0`.

## Reproducible commands

```bash
PYTHONPATH=src python scripts/build_mixed_benchmark.py \
  --output-dir benchmarks/mixed_beir_v1.2.0 \
  --seed 1337 \
  --max-candidates 80 \
  --cross-domain-negative-ratio 0.60 \
  --max-queries-per-domain 1800 \
  --max-test-queries-per-domain 360 \
  --hard-queries-per-category-per-domain 20

PYTHONPATH=src python experiments/run_canonical_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet \
  --split test \
  --max-queries 1200 \
  --output-dir results/canonical_benchmark_v3
```

## Comparison target

Compare against the older canonical artifact:

- `results/canonical_benchmark/summary.json` (v1.0.0)

Focus ranking shifts for:

- hard routing baseline (`router_only_top1`)
- mean fusion (`mean`)
- Kalman fusion (`kalman`)
- additional strong baselines included in summary outputs
