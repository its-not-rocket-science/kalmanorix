# Kalman latency optimization

## Environment assumptions
- Queries: 200, specialists: 6, embedding dim: 768, batch size: 20
- Benchmark file: `benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet` (split=test, max_queries=600)
- Timing: `time.perf_counter`; memory proxy: `tracemalloc` peak allocated KiB

## Single-query latency (Panoramix.brew)
- MeanFuser: mean=0.318 ms, p50=0.307 ms, p95=0.382 ms, mem_proxy=1514.2 KiB
- Kalman legacy: mean=1.652 ms, p50=1.600 ms, p95=1.874 ms, mem_proxy=2855.3 KiB
- Kalman optimized scalar-σ²: mean=0.803 ms, p50=0.789 ms, p95=0.929 ms, mem_proxy=2760.9 KiB
- Speedup (legacy -> optimized): 2.06x
- Optimized Kalman / Mean latency ratio: 2.53x

## Batch latency (fuser.fuse_batch)
- MeanFuser batch: mean=6.085 ms, p50=4.958 ms, p95=10.597 ms, mem_proxy=2932.6 KiB
- Kalman legacy batch: mean=17.615 ms, p50=16.993 ms, p95=21.125 ms, mem_proxy=6107.1 KiB
- Kalman optimized batch: mean=14.296 ms, p50=13.341 ms, p95=17.667 ms, mem_proxy=4673.9 KiB
- Batch speedup (legacy -> optimized): 1.23x

## Numerical deviation vs legacy
- Single-query: vector_max_abs=4.295e-02, vector_rms=9.030e-03, weight_max_abs=7.384e-03, covariance_max_abs=3.211e-03
- Batch: vector_max_abs=4.325e-02, vector_rms=9.063e-03, weight_max_abs=1.731e-02, covariance_max_abs=2.746e-03

## Canonical benchmark rerun with optimized Kalman path
- Verdict: `inconclusive_underpowered`
- Decision-framework latency ratio (kalman/mean): 2.008 (threshold <= 1.500)
- Latency check passed: `False`
- Primary metric delta (ndcg@10): 0.0949 (Holm-adjusted p=1.0000)
- Canonical artifacts: `results/kalman_latency_optimization/canonical/summary.json` and `results/kalman_latency_optimization/canonical/report.md`

## Hot-path proxy
- Embed calls observed: 12000 (0.394s cumulative)
- Sigma² calls observed: 4800 (0.279s cumulative)
