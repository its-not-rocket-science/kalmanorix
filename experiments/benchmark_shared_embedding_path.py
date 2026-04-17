#!/usr/bin/env python3
"""Microbenchmarks for shared-embedding-path latency improvements.

Compares old path (fuser computes embeddings internally) vs new path
(Panoramix precomputes aligned specialist embeddings once per query).
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village


@dataclass
class BenchmarkCase:
    name: str
    fuser: object
    uncertainty_heavy: bool = False


class CountingEmbedder:
    def __init__(self, direction: np.ndarray):
        self.direction = np.asarray(direction, dtype=np.float64)
        self.calls = 0

    def __call__(self, query: str) -> np.ndarray:
        self.calls += 1
        scale = (len(query.split()) + 1) / 8.0
        return (self.direction * scale).astype(np.float64)


class EmbeddingAwareSigma2:
    def __init__(self, base: float = 0.1):
        self.base = float(base)

    def __call__(self, query: str) -> float:
        # Legacy fallback path: intentionally computes from query only.
        return self.base + (len(query) % 7) * 0.01

    def estimate_with_embedding(self, query: str, embedding: np.ndarray) -> float:
        return float(self.base + 0.05 * np.linalg.norm(embedding))


def _build_village(
    n_specialists: int, dim: int, uncertainty_heavy: bool
) -> tuple[Village, list[CountingEmbedder]]:
    rng = np.random.default_rng(123)
    modules: list[SEF] = []
    embedders: list[CountingEmbedder] = []
    for i in range(n_specialists):
        direction = rng.normal(size=(dim,))
        direction = direction / (np.linalg.norm(direction) + 1e-12)
        embedder = CountingEmbedder(direction)
        embedders.append(embedder)

        if uncertainty_heavy:
            sigma2 = EmbeddingAwareSigma2(base=0.05 + 0.02 * i)
        else:
            sigma2 = 0.1 + 0.03 * i

        modules.append(SEF(name=f"m{i}", embed=embedder, sigma2=sigma2))

    return Village(modules=modules), embedders


def _run_case(
    case: BenchmarkCase,
    queries: list[str],
    n_specialists: int,
    dim: int,
) -> dict[str, float | int]:
    village_old, old_embedders = _build_village(
        n_specialists=n_specialists,
        dim=dim,
        uncertainty_heavy=case.uncertainty_heavy,
    )
    village_new, new_embedders = _build_village(
        n_specialists=n_specialists,
        dim=dim,
        uncertainty_heavy=case.uncertainty_heavy,
    )
    scout = ScoutRouter(mode="all")

    old_path = Panoramix(fuser=case.fuser, use_shared_embedding_path=False)
    new_path = Panoramix(fuser=case.fuser, use_shared_embedding_path=True)

    t0 = time.perf_counter()
    for query in queries:
        _ = old_path.brew(query, village_old, scout)
    old_ms = (time.perf_counter() - t0) * 1000.0

    t1 = time.perf_counter()
    for query in queries:
        _ = new_path.brew(query, village_new, scout)
    new_ms = (time.perf_counter() - t1) * 1000.0

    return {
        "old_total_ms": float(old_ms),
        "new_total_ms": float(new_ms),
        "speedup_x": float(old_ms / max(new_ms, 1e-12)),
        "old_embed_calls": int(sum(e.calls for e in old_embedders)),
        "new_embed_calls": int(sum(e.calls for e in new_embedders)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/shared_embedding_microbench.json")
    )
    parser.add_argument("--n-queries", type=int, default=400)
    parser.add_argument("--n-specialists", type=int, default=8)
    parser.add_argument("--dim", type=int, default=768)
    args = parser.parse_args()

    queries = [
        f"query {i} for shared embedding microbenchmark" for i in range(args.n_queries)
    ]

    cases = [
        BenchmarkCase(name="mean_fusion", fuser=MeanFuser()),
        BenchmarkCase(
            name="kalman_fusion", fuser=KalmanorixFuser(use_fast_scalar_path=True)
        ),
        BenchmarkCase(
            name="uncertainty_heavy_kalman",
            fuser=KalmanorixFuser(use_fast_scalar_path=True),
            uncertainty_heavy=True,
        ),
    ]

    results: dict[str, object] = {
        "n_queries": args.n_queries,
        "n_specialists": args.n_specialists,
        "dimension": args.dim,
        "cases": {},
    }
    for case in cases:
        results["cases"][case.name] = _run_case(
            case=case,
            queries=queries,
            n_specialists=args.n_specialists,
            dim=args.dim,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Wrote microbenchmark results to {args.output}")


if __name__ == "__main__":
    main()
