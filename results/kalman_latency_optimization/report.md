# Kalman latency optimization

## Environment assumptions
- Queries: 200, specialists: 6, embedding dim: 768, batch size: 20
- Benchmark file: `benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet` (split=test, max_queries=600)
- Timing: `time.perf_counter`; memory proxy: `tracemalloc` peak allocated KiB

## Single-query latency (Panoramix.brew)
- MeanFuser: mean=0.355 ms, p50=0.345 ms, p95=0.445 ms, mem_proxy=1515.5 KiB
- Kalman legacy: mean=1.720 ms, p50=1.673 ms, p95=2.102 ms, mem_proxy=2857.5 KiB
- Kalman optimized scalar-σ²: mean=0.840 ms, p50=0.794 ms, p95=1.118 ms, mem_proxy=2761.5 KiB
- Speedup (legacy -> optimized): 2.05x
- Optimized Kalman / Mean latency ratio: 2.37x

## Batch latency (fuser.fuse_batch)
- MeanFuser batch: mean=14.446 ms, p50=15.443 ms, p95=16.946 ms, mem_proxy=2749.6 KiB
- Kalman legacy batch: mean=18.022 ms, p50=17.069 ms, p95=22.848 ms, mem_proxy=5264.3 KiB
- Kalman optimized batch: mean=14.754 ms, p50=14.169 ms, p95=17.895 ms, mem_proxy=3950.3 KiB
- Batch speedup (legacy -> optimized): 1.22x

## Numerical deviation vs legacy
- Single-query: vector_max_abs=4.507e-02, vector_rms=9.017e-03, weight_max_abs=6.366e-03, covariance_max_abs=2.589e-03
- Batch: vector_max_abs=4.103e-02, vector_rms=9.054e-03, weight_max_abs=1.901e-02, covariance_max_abs=3.148e-03

## Canonical benchmark rerun with optimized Kalman path
- Verdict: `inconclusive_underpowered`
- Decision-framework latency ratio (kalman/mean): 0.924 (threshold <= 1.500)
- Latency check passed: `True`
- Primary metric delta (ndcg@10): 0.0949 (Holm-adjusted p=1.0000)
- Canonical artifacts: `results/kalman_latency_optimization/canonical/summary.json` and `results/kalman_latency_optimization/canonical/report.md`
- Canonical v3 latency-ratio basis: **optimized Kalman path = yes** (ratio comes from the canonical rerun above, executed with `KalmanorixFuser` fast scalar-σ² path).

## Hot-path proxy
- Embed calls observed: 12000 (0.438s cumulative)
- Sigma² calls observed: 4800 (0.282s cumulative)
