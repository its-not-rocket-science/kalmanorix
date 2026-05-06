# Reproducibility guide for publication artifact

This document captures a **reproducibility target** for the publication artifact. It is intended to make reruns easier, but it does **not** claim exact runtime portability across machines.

## 1) Environment

Record your local runtime environment before running:

- Python: `<PYTHON_VERSION_PLACEHOLDER>`
- OS: `<OS_PLACEHOLDER>`
- Hardware notes:
  - CPU: `<CPU_PLACEHOLDER>`
  - GPU: `<GPU_PLACEHOLDER or NONE>`

Benchmark manifest dependency pins (from `benchmarks/mixed_beir_v1.2.0/benchmark_manifest.json`):

- `datasets==2.19.1`
- `pyarrow==14.0.2`

## 2) Benchmark generation command

Use the benchmark builder to regenerate the publication benchmark artifacts:

```bash
PYTHONPATH=src python scripts/build_mixed_benchmark.py \
  --output-dir benchmarks/mixed_beir_v1.2.0 \
  --seed 1337 \
  --max-candidates 80 \
  --cross-domain-negative-ratio 0.60 \
  --max-queries-per-domain 1800 \
  --max-test-queries-per-domain 360 \
  --hard-queries-per-category-per-domain 20
```

PowerShell variant:

```powershell
$env:PYTHONPATH="src"; python scripts/build_mixed_benchmark.py --output-dir benchmarks/mixed_beir_v1.2.0 --seed 1337 --max-candidates 80 --cross-domain-negative-ratio 0.60 --max-queries-per-domain 1800 --max-test-queries-per-domain 360 --hard-queries-per-category-per-domain 20
```

## 3) Fast-local canonical v3 command

Run the fast-local canonical benchmark v3 pass:

```bash
PYTHONPATH=src python experiments/run_canonical_benchmark.py \
  --benchmark-path benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet \
  --output-dir results/canonical_benchmark_v3_fast_1200 \
  --max-queries 1200
```

PowerShell variant:

```powershell
$env:PYTHONPATH="src"; python experiments/run_canonical_benchmark.py --benchmark-path benchmarks/mixed_beir_v1.2.0/mixed_benchmark.parquet --output-dir results/canonical_benchmark_v3_fast_1200 --max-queries 1200
```

## 4) Output directories to verify

After a successful run, verify these directories exist and contain manifest/report outputs:

- `benchmarks/mixed_beir_v1.2.0/`
- `results/canonical_benchmark_v3_fast_1200/`

## 5) Expected key result (fast-local canonical v3)

For the reference fast-local canonical v3 artifact, the key values are:

- `benchmark_status`: `claim_ready`
- `n_pairs`: `1193`
- `delta_ndcg@10`: `-0.000203`
- `adjusted_p_value`: `1.0`

Treat these as reproducibility anchors for this artifact revision.

## 6) Notes on large artifacts

- Do **not** commit the full corpus parquet into git history.
- If publishing outside the repository, archive large artifacts via one of:
  - Zenodo
  - OSF
  - Hugging Face Datasets

This keeps the repository lightweight while preserving public reproducibility of publication assets.
