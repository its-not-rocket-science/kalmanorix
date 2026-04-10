# Canonical Benchmark v2 (Stronger Evaluation Protocol)

This folder is reserved for the stronger canonical benchmark artifacts and preserves
provenance by leaving the prior canonical run untouched in `results/canonical_benchmark/`.

## Reproducible command

```bash
PYTHONPATH=src python scripts/build_mixed_benchmark.py \
  --output-dir benchmarks/mixed_beir_v1.1.0 \
  --seed 1337 \
  --max-candidates 80 \
  --cross-domain-negative-ratio 0.45 \
  --max-queries-per-domain 900 \
  --max-test-queries-per-domain 180

PYTHONPATH=src python experiments/run_canonical_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.1.0/mixed_benchmark.json \
  --split test \
  --max-queries 600 \
  --output-dir results/canonical_benchmark_v2
```

## Delta from v1

- Benchmark version bump: `mixed_beir_v1.0.0` → `mixed_beir_v1.1.0`.
- Domain set expanded from 3 domains to 6 domains.
- Harder candidate pools via cross-domain hard negatives.
- Additional decision metrics: `nDCG@5`, `MRR@5`, `Recall@1`, and `top1_success`.
- Canonical report now includes a direct method ranking snapshot.

## Expected artifacts

- `summary.json`
- `report.md`
- `runner_summary.json`
- `runner_details.json`
