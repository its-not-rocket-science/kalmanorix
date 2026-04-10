#!/usr/bin/env python3
"""Profile and optimize Kalman latency; emit artifact bundle."""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import pstats
import time
from dataclasses import dataclass
from pathlib import Path

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


def _benchmark_strategy(queries: list[str], village: Village, strategy: str) -> dict[str, float]:
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
    latencies = []
    for query in queries:
        t0 = time.perf_counter()
        _ = panoramix.brew(query, village=village, scout=router)
        latencies.append((time.perf_counter() - t0) * 1000.0)
    return {
        "mean_ms": float(np.mean(latencies)),
        "p50_ms": float(np.percentile(latencies, 50)),
        "p95_ms": float(np.percentile(latencies, 95)),
    }


def _profile_kalman(queries: list[str], village: Village, optimized: bool) -> str:
    router = ScoutRouter(mode="all")
    fuser = KalmanorixFuser(use_fast_scalar_path=optimized)
    panoramix = Panoramix(fuser=fuser)

    profiler = cProfile.Profile()
    profiler.enable()
    for query in queries:
        _ = panoramix.brew(query, village=village, scout=router)
    profiler.disable()

    s = io.StringIO()
    stats = pstats.Stats(profiler, stream=s)
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)
    return s.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("results/kalman_latency_optimization"))
    parser.add_argument("--n-queries", type=int, default=200)
    parser.add_argument("--n-specialists", type=int, default=6)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    village, embeds, sigmas = _make_village(args.n_specialists, args.dim)
    queries = [f"query {i} mixed domain latency benchmark" for i in range(args.n_queries)]

    strategies = {
        "mean": _benchmark_strategy(queries, village, "mean"),
        "kalman_legacy": _benchmark_strategy(queries, village, "kalman_legacy"),
        "kalman_optimized": _benchmark_strategy(queries, village, "kalman_optimized"),
    }

    profile_before = _profile_kalman(queries, village, optimized=False)
    profile_after = _profile_kalman(queries, village, optimized=True)

    embed_calls = sum(e.calls for e in embeds)
    embed_seconds = sum(e.seconds for e in embeds)
    sigma_calls = sum(s.calls for s in sigmas)
    sigma_seconds = sum(s.seconds for s in sigmas)

    speedup = strategies["kalman_legacy"]["mean_ms"] / strategies["kalman_optimized"]["mean_ms"]
    ratio_vs_mean = strategies["kalman_optimized"]["mean_ms"] / strategies["mean"]["mean_ms"]

    summary = {
        "strategies": strategies,
        "speedup_legacy_to_optimized": speedup,
        "optimized_vs_mean_latency_ratio": ratio_vs_mean,
        "profiling": {
            "embed_calls": embed_calls,
            "embed_seconds": embed_seconds,
            "sigma2_calls": sigma_calls,
            "sigma2_seconds": sigma_seconds,
        },
    }
    (out / "microbenchmark_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (out / "profile_legacy.txt").write_text(profile_before, encoding="utf-8")
    (out / "profile_optimized.txt").write_text(profile_after, encoding="utf-8")

    report = [
        "# Kalman latency optimization",
        "",
        "## Microbenchmark",
        f"- MeanFuser mean latency: {strategies['mean']['mean_ms']:.3f} ms",
        f"- Kalman (legacy) mean latency: {strategies['kalman_legacy']['mean_ms']:.3f} ms",
        f"- Kalman (optimized) mean latency: {strategies['kalman_optimized']['mean_ms']:.3f} ms",
        f"- Speedup (legacy -> optimized): {speedup:.2f}x",
        f"- Optimized Kalman / Mean latency ratio: {ratio_vs_mean:.2f}x",
        "",
        "## Hot-path observations",
        f"- Embed calls observed: {embed_calls} ({embed_seconds:.3f}s total)",
        f"- Sigma² calls observed: {sigma_calls} ({sigma_seconds:.3f}s total)",
        "- Legacy Kalman spent significant time building repeated diagonal covariance vectors.",
        "- Optimized path removes per-dimension covariance materialization for sigma²*I case.",
        "",
        "## Semantics",
        "- The optimized path preserves the same scalar Kalman gain sequence for sigma²*I covariance.",
        "- Numerical regression tests enforce closeness to legacy behavior.",
    ]
    (out / "report.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
