# Real Mixed-Domain Benchmark Migration (April 2026)

## 1) Concrete benchmark plan

Primary evaluation now runs on **real retrieval data with qrels** assembled from BEIR:

- `BeIR/nq` → `general_qa`
- `BeIR/scifact` → `biomedical`
- `BeIR/fiqa` → `finance`

Benchmark artifact path:

- `benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet`

Primary specialist set (real HF models, three distinct domains):

- `sentence-transformers/all-mpnet-base-v2` (general QA specialist)
- `emilyalsentzer/Bio_ClinicalBERT` (biomedical specialist)
- `ProsusAI/finbert` (finance specialist)

Protocol + metrics:

- Locked protocol from `kalmanorix.benchmarks.evaluation_protocol`
- Primary metrics: Recall@1/5/10, MRR, nDCG@10
- Secondary metrics: latency_ms
- Comparisons: Kalman fusion vs mean fusion

## 2) Code changes required

- Add a real-data benchmark runner:
  - `experiments/run_real_mixed_benchmark.py`
- Rewire `experiments/validate_fusion.py`:
  - default path = real-data benchmark
  - `--debug-synthetic` flag = toy smoke path only
- Demote synthetic benchmark script:
  - `experiments/benchmark_fusion_methods.py` now runs only debug synthetic smoke

## 3) New experiment entry points

Primary:

```bash
python experiments/run_real_mixed_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet \
  --split test \
  --max-queries 150 \
  --output results/real_mixed_benchmark/real_benchmark_summary.json
```

Compatibility entrypoint (now real by default):

```bash
python experiments/validate_fusion.py
```

Debug/smoke only:

```bash
python experiments/validate_fusion.py --debug-synthetic
python experiments/benchmark_fusion_methods.py
```

## 4) Migration map (old scripts → debug only)

- `experiments/benchmark_fusion_methods.py`: **debug only** (synthetic smoke)
- `experiments/validate_fusion.py --debug-synthetic`: **debug only** (toy corpus + keyword embedder)
- Primary headline validation moved to:
  - `experiments/run_real_mixed_benchmark.py`
  - `experiments/validate_fusion.py` (default mode)

## Operational note

Use `scripts/build_mixed_benchmark.py` to regenerate benchmark artifacts when needed.
