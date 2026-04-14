#!/usr/bin/env python3
"""Generate reproducible latency optimization evidence for Kalman fusion."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
import tracemalloc

import numpy as np

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village


@dataclass
class TimedEmbedder:
    base: callable
    calls: int = 0
    seconds: float = 0.0

    def __call__(self, query: str) -> np.ndarray:
        t0 = time.perf_counter()
        out = self.base(query)
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out


@dataclass
class TimedSigma2:
    base: callable
    calls: int = 0
    seconds: float = 0.0

    def __call__(self, query: str) -> float:
        t0 = time.perf_counter()
        out = float(self.base(query))
        self.seconds += time.perf_counter() - t0
        self.calls += 1
        return out


def _make_village(n_specialists: int, d: int) -> tuple[Village, list[TimedEmbedder], list[TimedSigma2]]:
    rng = np.random.default_rng(42)
    modules: list[SEF] = []
    embeds: list[TimedEmbedder] = []
    sigmas: list[TimedSigma2] = []
    for i in range(n_specialists):
        direction = rng.normal(size=(d,))
        direction /= np.linalg.norm(direction) + 1e-12
        centroid = rng.normal(size=(d,))
        centroid /= np.linalg.norm(centroid) + 1e-12

        def make_embed(vec: np.ndarray):
            def _embed(query: str) -> np.ndarray:
                q = len(query.split()) / 8.0
                noise = rng.normal(scale=0.01, size=(d,))
                z = vec * q + noise
                return z.astype(np.float64)

            return _embed

        base_embed = make_embed(direction)
        timed_embed = TimedEmbedder(base_embed)

        def make_sigma2(embedder: callable, center: np.ndarray):
            def _sigma2(query: str) -> float:
                z = embedder(query)
                z = z / (np.linalg.norm(z) + 1e-12)
                sim = float(z @ center)
                return max(1e-4, 0.1 + (1.0 - sim) * 0.5)

            return _sigma2

        timed_sigma2 = TimedSigma2(make_sigma2(timed_embed, centroid))
        embeds.append(timed_embed)
        sigmas.append(timed_sigma2)
        modules.append(SEF(name=f"s{i}", embed=timed_embed, sigma2=timed_sigma2, domain_centroid=centroid))

    return Village(modules=modules), embeds, sigmas


def _benchmark_strategy(
    queries: list[str],
    village: Village,
    strategy: str,
) -> tuple[dict[str, float], list[np.ndarray], list[dict[str, float]], list[dict[str, object] | None]]:
    router = ScoutRouter(mode="all")
    if strategy == "mean":
        fuser = MeanFuser()
    elif strategy == "kalman_legacy":
        fuser = KalmanorixFuser(use_fast_scalar_path=False)
    elif strategy == "kalman_optimized":
        fuser = KalmanorixFuser(use_fast_scalar_path=True)
    else:
        raise ValueError(strategy)

    panoramix = Panoramix(fuser=fuser)
    latencies_ms = []
    fused_vectors: list[np.ndarray] = []
    weights_list: list[dict[str, float]] = []
    metadata_list: list[dict[str, object] | None] = []

    tracemalloc.start()
    peak_memory_bytes = 0
    for query in queries:
        t0 = time.perf_counter()
        potion = panoramix.brew(query, village=village, scout=router)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        fused_vectors.append(np.asarray(potion.vector, dtype=np.float64))
        weights_list.append(potion.weights)
        metadata_list.append(potion.meta if isinstance(potion.meta, dict) else None)
        _, peak = tracemalloc.get_traced_memory()
        peak_memory_bytes = max(peak_memory_bytes, int(peak))
    tracemalloc.stop()

    metrics = {
        "mean_ms": float(np.mean(latencies_ms)),
        "p50_ms": float(np.percentile(latencies_ms, 50)),
        "p95_ms": float(np.percentile(latencies_ms, 95)),
        "memory_peak_kib": float(peak_memory_bytes / 1024.0),
    }
    return metrics, fused_vectors, weights_list, metadata_list


def _benchmark_batch_strategy(
    queries: list[str],
    modules: list[SEF],
    strategy: str,
    batch_size: int,
) -> tuple[dict[str, float], list[np.ndarray], list[dict[str, float]], list[dict[str, object]]]:
    if strategy == "mean":
        fuser = MeanFuser()
    elif strategy == "kalman_legacy":
        fuser = KalmanorixFuser(use_fast_scalar_path=False)
    elif strategy == "kalman_optimized":
        fuser = KalmanorixFuser(use_fast_scalar_path=True)
    else:
        raise ValueError(strategy)

    if len(queries) % batch_size != 0:
        raise ValueError("n_queries must be divisible by batch_size")

    latencies_ms = []
    tracemalloc.start()
    peak_memory_bytes = 0
    fused_all: list[np.ndarray] = []
    weights_all: list[dict[str, float]] = []
    meta_all: list[dict[str, object]] = []

    for start in range(0, len(queries), batch_size):
        q_batch = queries[start : start + batch_size]
        t0 = time.perf_counter()
        fused, weights, meta = fuser.fuse_batch(q_batch, modules)
        latencies_ms.append((time.perf_counter() - t0) * 1000.0)
        fused_all.extend(np.asarray(x, dtype=np.float64) for x in fused)
        weights_all.extend(weights)
        if meta is None:
            meta_all.extend({} for _ in q_batch)
        else:
            meta_all.extend(meta)
        _, peak = tracemalloc.get_traced_memory()
        peak_memory_bytes = max(peak_memory_bytes, int(peak))
    tracemalloc.stop()
    return (
        {
            "mean_ms": float(np.mean(latencies_ms)),
            "p50_ms": float(np.percentile(latencies_ms, 50)),
            "p95_ms": float(np.percentile(latencies_ms, 95)),
            "memory_peak_kib": float(peak_memory_bytes / 1024.0),
            "batch_size": float(batch_size),
            "queries_per_batch": float(batch_size),
        },
        fused_all,
        weights_all,
        meta_all,
    )


def _numerical_deviation(
    reference_vectors: list[np.ndarray],
    candidate_vectors: list[np.ndarray],
    reference_weights: list[dict[str, float]],
    candidate_weights: list[dict[str, float]],
    reference_meta: list[dict[str, object] | None],
    candidate_meta: list[dict[str, object] | None],
) -> dict[str, float]:
    if len(reference_vectors) != len(candidate_vectors):
        raise ValueError("Mismatched vector counts")

    max_abs = 0.0
    rms_acc = 0.0
    n_total = 0
    weight_abs = 0.0
    cov_abs = 0.0
    for ref, cand, w_ref, w_cand, m_ref, m_cand in zip(
        reference_vectors,
        candidate_vectors,
        reference_weights,
        candidate_weights,
        reference_meta,
        candidate_meta,
    ):
        delta = np.abs(cand - ref)
        max_abs = max(max_abs, float(np.max(delta)))
        rms_acc += float(np.sum((cand - ref) ** 2))
        n_total += int(ref.size)
        for key, v_ref in w_ref.items():
            weight_abs = max(weight_abs, abs(float(w_cand[key]) - float(v_ref)))
        if m_ref and m_cand and "fused_covariance" in m_ref and "fused_covariance" in m_cand:
            ref_cov = np.asarray(m_ref["fused_covariance"], dtype=np.float64)
            cand_cov = np.asarray(m_cand["fused_covariance"], dtype=np.float64)
            cov_abs = max(cov_abs, float(np.max(np.abs(cand_cov - ref_cov))))
    return {
        "vector_max_abs": float(max_abs),
        "vector_rms": float(np.sqrt(rms_acc / max(n_total, 1))),
        "weight_max_abs": float(weight_abs),
        "covariance_max_abs": float(cov_abs),
    }


def _run_canonical_benchmark(out_dir: Path, benchmark_path: Path, max_queries: int) -> dict[str, object]:
    canonical_out = out_dir / "canonical"
    cmd = [
        "python",
        "experiments/run_canonical_benchmark.py",
        "--benchmark-path",
        str(benchmark_path),
        "--split",
        "test",
        "--max-queries",
        str(max_queries),
        "--output-dir",
        str(canonical_out),
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = ".:src"
    subprocess.run(cmd, check=True, env=env)
    canonical_summary = json.loads((canonical_out / "summary.json").read_text(encoding="utf-8"))
    decision = canonical_summary["decision"]["kalman_vs_mean"]
    observed = decision["observed"]
    return {
        "output_dir": str(canonical_out),
        "decision_verdict": decision["verdict"],
        "latency_ratio_vs_mean": float(observed["latency_ratio_vs_mean"]),
        "latency_ratio_threshold": float(decision["rules"]["max_latency_ratio_vs_mean"]),
        "latency_ratio_ok": bool(decision["checks"]["latency_ratio_ok"]),
        "primary_metric_delta": float(observed["primary_metric_delta"]),
        "primary_metric_adjusted_p_value": float(observed["primary_metric_adjusted_p_value"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/kalman_latency_optimization"))
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--n-specialists", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument(
        "--canonical-benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet"),
    )
    parser.add_argument("--canonical-max-queries", type=int, default=600)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    village, embeds, sigmas = _make_village(args.n_specialists, args.dim)
    queries = [f"query {i} mixed domain latency benchmark" for i in range(args.n_queries)]

    strategy_metrics = {}
    strategy_outputs = {}
    for name in ("mean", "kalman_legacy", "kalman_optimized"):
        metrics, vectors, weights, meta = _benchmark_strategy(queries, village, name)
        strategy_metrics[name] = metrics
        strategy_outputs[name] = {
            "vectors": vectors,
            "weights": weights,
            "meta": meta,
        }

    batch_metrics = {}
    batch_outputs = {}
    for name in ("mean", "kalman_legacy", "kalman_optimized"):
        metrics, vectors, weights, meta = _benchmark_batch_strategy(
            queries=queries,
            modules=village.modules,
            strategy=name,
            batch_size=args.batch_size,
        )
        batch_metrics[name] = metrics
        batch_outputs[name] = {
            "vectors": vectors,
            "weights": weights,
            "meta": meta,
        }

    embed_calls = sum(e.calls for e in embeds)
    embed_seconds = sum(e.seconds for e in embeds)
    sigma_calls = sum(s.calls for s in sigmas)
    sigma_seconds = sum(s.seconds for s in sigmas)

    speedup = strategy_metrics["kalman_legacy"]["mean_ms"] / strategy_metrics["kalman_optimized"]["mean_ms"]
    ratio_vs_mean = strategy_metrics["kalman_optimized"]["mean_ms"] / strategy_metrics["mean"]["mean_ms"]
    batch_speedup = batch_metrics["kalman_legacy"]["mean_ms"] / batch_metrics["kalman_optimized"]["mean_ms"]

    deviation_scalar = _numerical_deviation(
        strategy_outputs["kalman_legacy"]["vectors"],
        strategy_outputs["kalman_optimized"]["vectors"],
        strategy_outputs["kalman_legacy"]["weights"],
        strategy_outputs["kalman_optimized"]["weights"],
        strategy_outputs["kalman_legacy"]["meta"],
        strategy_outputs["kalman_optimized"]["meta"],
    )
    deviation_batch = _numerical_deviation(
        batch_outputs["kalman_legacy"]["vectors"],
        batch_outputs["kalman_optimized"]["vectors"],
        batch_outputs["kalman_legacy"]["weights"],
        batch_outputs["kalman_optimized"]["weights"],
        batch_outputs["kalman_legacy"]["meta"],
        batch_outputs["kalman_optimized"]["meta"],
    )

    canonical = _run_canonical_benchmark(
        out_dir=out,
        benchmark_path=args.canonical_benchmark_path,
        max_queries=args.canonical_max_queries,
    )

    summary = {
        "assumptions": {
            "python": "python3",
            "dtype": "float64",
            "n_queries": args.n_queries,
            "n_specialists": args.n_specialists,
            "embedding_dim": args.dim,
            "batch_size": args.batch_size,
            "canonical_benchmark_path": str(args.canonical_benchmark_path),
            "canonical_max_queries": args.canonical_max_queries,
            "timer": "time.perf_counter",
            "memory_proxy": "tracemalloc peak resident Python allocation (KiB)",
        },
        "single_query_strategies": strategy_metrics,
        "batch_strategies": batch_metrics,
        "speedup_legacy_to_optimized": speedup,
        "optimized_vs_mean_latency_ratio": ratio_vs_mean,
        "batch_speedup_legacy_to_optimized": batch_speedup,
        "numerical_deviation_vs_legacy": {
            "single_query": deviation_scalar,
            "batch": deviation_batch,
        },
        "profiling": {
            "embed_calls": embed_calls,
            "embed_seconds": embed_seconds,
            "sigma2_calls": sigma_calls,
            "sigma2_seconds": sigma_seconds,
        },
        "canonical_rerun": canonical,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    report = [
        "# Kalman latency optimization",
        "",
        "## Environment assumptions",
        f"- Queries: {args.n_queries}, specialists: {args.n_specialists}, embedding dim: {args.dim}, batch size: {args.batch_size}",
        f"- Benchmark file: `{args.canonical_benchmark_path}` (split=test, max_queries={args.canonical_max_queries})",
        "- Timing: `time.perf_counter`; memory proxy: `tracemalloc` peak allocated KiB",
        "",
        "## Single-query latency (Panoramix.brew)",
        f"- MeanFuser: mean={strategy_metrics['mean']['mean_ms']:.3f} ms, p50={strategy_metrics['mean']['p50_ms']:.3f} ms, p95={strategy_metrics['mean']['p95_ms']:.3f} ms, mem_proxy={strategy_metrics['mean']['memory_peak_kib']:.1f} KiB",
        f"- Kalman legacy: mean={strategy_metrics['kalman_legacy']['mean_ms']:.3f} ms, p50={strategy_metrics['kalman_legacy']['p50_ms']:.3f} ms, p95={strategy_metrics['kalman_legacy']['p95_ms']:.3f} ms, mem_proxy={strategy_metrics['kalman_legacy']['memory_peak_kib']:.1f} KiB",
        f"- Kalman optimized scalar-σ²: mean={strategy_metrics['kalman_optimized']['mean_ms']:.3f} ms, p50={strategy_metrics['kalman_optimized']['p50_ms']:.3f} ms, p95={strategy_metrics['kalman_optimized']['p95_ms']:.3f} ms, mem_proxy={strategy_metrics['kalman_optimized']['memory_peak_kib']:.1f} KiB",
        f"- Speedup (legacy -> optimized): {speedup:.2f}x",
        f"- Optimized Kalman / Mean latency ratio: {ratio_vs_mean:.2f}x",
        "",
        "## Batch latency (fuser.fuse_batch)",
        f"- MeanFuser batch: mean={batch_metrics['mean']['mean_ms']:.3f} ms, p50={batch_metrics['mean']['p50_ms']:.3f} ms, p95={batch_metrics['mean']['p95_ms']:.3f} ms, mem_proxy={batch_metrics['mean']['memory_peak_kib']:.1f} KiB",
        f"- Kalman legacy batch: mean={batch_metrics['kalman_legacy']['mean_ms']:.3f} ms, p50={batch_metrics['kalman_legacy']['p50_ms']:.3f} ms, p95={batch_metrics['kalman_legacy']['p95_ms']:.3f} ms, mem_proxy={batch_metrics['kalman_legacy']['memory_peak_kib']:.1f} KiB",
        f"- Kalman optimized batch: mean={batch_metrics['kalman_optimized']['mean_ms']:.3f} ms, p50={batch_metrics['kalman_optimized']['p50_ms']:.3f} ms, p95={batch_metrics['kalman_optimized']['p95_ms']:.3f} ms, mem_proxy={batch_metrics['kalman_optimized']['memory_peak_kib']:.1f} KiB",
        f"- Batch speedup (legacy -> optimized): {batch_speedup:.2f}x",
        "",
        "## Numerical deviation vs legacy",
        f"- Single-query: vector_max_abs={deviation_scalar['vector_max_abs']:.3e}, vector_rms={deviation_scalar['vector_rms']:.3e}, weight_max_abs={deviation_scalar['weight_max_abs']:.3e}, covariance_max_abs={deviation_scalar['covariance_max_abs']:.3e}",
        f"- Batch: vector_max_abs={deviation_batch['vector_max_abs']:.3e}, vector_rms={deviation_batch['vector_rms']:.3e}, weight_max_abs={deviation_batch['weight_max_abs']:.3e}, covariance_max_abs={deviation_batch['covariance_max_abs']:.3e}",
        "",
        "## Canonical benchmark rerun with optimized Kalman path",
        f"- Verdict: `{canonical['decision_verdict']}`",
        f"- Decision-framework latency ratio (kalman/mean): {canonical['latency_ratio_vs_mean']:.3f} (threshold <= {canonical['latency_ratio_threshold']:.3f})",
        f"- Latency check passed: `{canonical['latency_ratio_ok']}`",
        f"- Primary metric delta (ndcg@10): {canonical['primary_metric_delta']:.4f} (Holm-adjusted p={canonical['primary_metric_adjusted_p_value']:.4f})",
        f"- Canonical artifacts: `{canonical['output_dir']}/summary.json` and `{canonical['output_dir']}/report.md`",
        "",
        "## Hot-path proxy",
        f"- Embed calls observed: {embed_calls} ({embed_seconds:.3f}s cumulative)",
        f"- Sigma² calls observed: {sigma_calls} ({sigma_seconds:.3f}s cumulative)",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
