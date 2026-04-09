#!/usr/bin/env python3
"""Deployment-oriented efficiency benchmark for Kalmanorix.

This benchmark redesign focuses on real deployment tradeoffs:
- Uses genuinely distinct specialists (no synthetic duplicates).
- Separates router embedding cost, routing decision cost, fusion cost, and end-to-end cost.
- Reports deployment scenarios:
  1) all specialists loaded,
  2) lazy specialist loading,
  3) cached routing,
  4) batched inference.
- Performs memory accounting with shared vs scenario-specific deltas.

Usage:
    python experiments/benchmark_efficiency.py \
      --output results/efficiency/deployment_efficiency.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import numpy as np

from kalmanorix import KalmanorixFuser, MeanFuser, SEF
from kalmanorix.panoramix import Panoramix
from kalmanorix.sef_io import SEFArtefact

try:
    import psutil
except Exception:  # pragma: no cover - optional dependency
    psutil = None


Embedder = Callable[[str], np.ndarray]


@dataclass
class BenchmarkConfig:
    """Configuration for deployment-oriented efficiency benchmarking."""

    sefs_dir: Path = Path("artefacts/sefs")
    models_dir: Path = Path("models")
    output: Path = Path("results/efficiency/deployment_efficiency.json")
    repeats: int = 20
    similarity_threshold: float = 0.70
    routing_mode: str = "semantic"
    batch_size: int = 8


def rss_mb() -> Optional[float]:
    """Return process resident set size in MiB if available."""
    if psutil is None:
        return None
    return psutil.Process().memory_info().rss / (1024 * 1024)


class LazyEmbedderPool:
    """Lazily load embedders so memory/load accounting mirrors deployment."""

    def __init__(self, embedder_paths: Dict[str, Path]) -> None:
        self.embedder_paths = dict(embedder_paths)
        self._models: Dict[str, Any] = {}
        self._embedders: Dict[str, Embedder] = {}

    def get(self, embedder_id: str) -> Embedder:
        if embedder_id in self._embedders:
            return self._embedders[embedder_id]

        if embedder_id not in self.embedder_paths:
            raise KeyError(f"Missing checkpoint for embedder_id={embedder_id}")

        from sentence_transformers import SentenceTransformer  # pylint: disable=import-outside-toplevel

        model = SentenceTransformer(str(self.embedder_paths[embedder_id]))

        def embed(text: str) -> np.ndarray:
            vec = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
            return vec.astype(np.float64)

        self._models[embedder_id] = model
        self._embedders[embedder_id] = embed
        return embed

    @property
    def loaded_embedder_ids(self) -> List[str]:
        return sorted(self._embedders.keys())


@dataclass
class SpecialistDefinition:
    """Lightweight specialist metadata for routing + on-demand loading."""

    name: str
    path: Path
    embedder_id: str
    centroid: np.ndarray


def load_specialist_definitions(
    sefs_dir: Path,
    pool: LazyEmbedderPool,
    fast_embedder_id: Optional[str] = None,
) -> List[SpecialistDefinition]:
    """Load distinct specialists with routing centroids in one shared embedding space."""
    paths = sorted(sefs_dir.glob("*.json"))
    if not paths:
        raise ValueError(f"No SEF artefacts found in {sefs_dir}")

    definitions: List[SpecialistDefinition] = []
    seen_names: set[str] = set()

    for path in paths:
        art = SEFArtefact.load(path)
        if art.name in seen_names:
            raise ValueError(f"Duplicate specialist name detected: {art.name}")
        seen_names.add(art.name)

        calibration_texts: List[str] = []
        if isinstance(art.sigma2_params, dict):
            calibration_texts = list(art.sigma2_params.get("calibration_texts", []))

        if not calibration_texts:
            raise ValueError(
                f"Specialist '{art.name}' has no calibration_texts for semantic routing"
            )

        centroid_embedder_id = fast_embedder_id or art.embedder_id
        centroid_embedder = pool.get(centroid_embedder_id)
        vectors = np.vstack([centroid_embedder(text) for text in calibration_texts])
        centroid = np.mean(vectors, axis=0)
        norm = np.linalg.norm(centroid)
        if norm == 0:
            raise ValueError(f"Centroid norm is zero for specialist '{art.name}'")

        definitions.append(
            SpecialistDefinition(
                name=art.name,
                path=path,
                embedder_id=art.embedder_id,
                centroid=centroid / norm,
            )
        )

    return definitions


def semantic_route(
    query: str,
    specialists: List[SpecialistDefinition],
    fast_embedder: Embedder,
    threshold: float,
    embedding_cache: Optional[Dict[str, np.ndarray]] = None,
) -> tuple[List[SpecialistDefinition], float, float, bool]:
    """Semantic routing with separated embedding vs routing decision timings.

    Returns:
        selected specialists,
        embedding_ms,
        routing_ms,
        cache_hit flag.
    """
    cache_hit = False
    embed_start = time.perf_counter()
    if embedding_cache is not None and query in embedding_cache:
        qvec = embedding_cache[query]
        cache_hit = True
    else:
        qvec = fast_embedder(query)
        qnorm = np.linalg.norm(qvec)
        if qnorm > 0:
            qvec = qvec / qnorm
        if embedding_cache is not None:
            embedding_cache[query] = qvec
    embedding_ms = (time.perf_counter() - embed_start) * 1000.0

    route_start = time.perf_counter()
    scored = [(spec, float(np.dot(qvec, spec.centroid))) for spec in specialists]
    selected = [spec for spec, sim in scored if sim >= threshold]
    if not selected:
        selected = [max(scored, key=lambda item: item[1])[0]]
    routing_ms = (time.perf_counter() - route_start) * 1000.0
    return selected, embedding_ms, routing_ms, cache_hit


def load_specialist_runtime(
    definition: SpecialistDefinition,
    pool: LazyEmbedderPool,
    loaded: Dict[str, SEF],
) -> tuple[SEF, float, float]:
    """Load a specialist on demand and return (module, load_ms, memory_delta_mb)."""
    if definition.name in loaded:
        return loaded[definition.name], 0.0, 0.0

    before = rss_mb()
    t0 = time.perf_counter()
    art = SEFArtefact.load(definition.path)
    embed = pool.get(art.embedder_id)
    sigma2 = art.build_sigma2()
    module = SEF(
        name=art.name,
        embed=embed,
        sigma2=sigma2,
        meta=art.meta,
        domain_centroid=definition.centroid,
    )
    load_ms = (time.perf_counter() - t0) * 1000.0
    after = rss_mb()
    memory_delta = 0.0 if before is None or after is None else max(0.0, after - before)
    loaded[definition.name] = module
    return module, load_ms, memory_delta


def summarize(values: Iterable[float]) -> Dict[str, float]:
    arr = list(values)
    if not arr:
        return {"mean": 0.0, "p50": 0.0, "p95": 0.0}
    p95_idx = max(0, min(len(arr) - 1, int(round(0.95 * (len(arr) - 1)))))
    sorted_arr = sorted(arr)
    return {
        "mean": float(statistics.fmean(arr)),
        "p50": float(statistics.median(arr)),
        "p95": float(sorted_arr[p95_idx]),
    }


def build_queries(repeats: int) -> List[str]:
    """Create mixed-domain workload used for all scenarios."""
    base = [
        "best laptop battery life for travel",
        "how to season cast iron for stew",
        "ev charging speed at home vs public",
        "ultrabook thermal throttling tips",
        "slow cooker chili without beans",
        "battery degradation after fast charging",
    ]
    queries = []
    while len(queries) < repeats:
        queries.extend(base)
    return queries[:repeats]


def run_single_inference(
    query: str,
    selected_modules: List[SEF],
    strategy: str,
) -> float:
    """Run fusion and return fusion latency (includes specialist embedding work)."""
    fuser = MeanFuser() if strategy == "mean" else KalmanorixFuser()
    pan = Panoramix(fuser=fuser)
    t0 = time.perf_counter()
    _ = pan.fuser.fuse(query, selected_modules)
    return (time.perf_counter() - t0) * 1000.0


def run_all_loaded(
    definitions: List[SpecialistDefinition],
    pool: LazyEmbedderPool,
    queries: List[str],
    strategy: str,
    threshold: float,
) -> Dict[str, Any]:
    loaded: Dict[str, SEF] = {}
    shared_before = rss_mb()
    load_ms = []
    load_mem = []
    for spec in definitions:
        module, lms, dmem = load_specialist_runtime(spec, pool, loaded)
        _ = module
        load_ms.append(lms)
        load_mem.append(dmem)
    shared_after = rss_mb()

    fast_embedder = pool.get(definitions[0].embedder_id)

    embed_cost, route_cost, fusion_cost, e2e_cost = [], [], [], []
    selected_sizes = []
    for q in queries:
        t0 = time.perf_counter()
        selected_defs, ems, rms, _ = semantic_route(q, definitions, fast_embedder, threshold)
        selected_modules = [loaded[d.name] for d in selected_defs]
        fms = run_single_inference(q, selected_modules, strategy)
        total = (time.perf_counter() - t0) * 1000.0

        embed_cost.append(ems)
        route_cost.append(rms)
        fusion_cost.append(fms)
        e2e_cost.append(total)
        selected_sizes.append(len(selected_modules))

    return {
        "scenario": "all_specialists_loaded",
        "routing_cache": "disabled",
        "batch_size": 1,
        "specialists_loaded": len(loaded),
        "specialists_available": len(definitions),
        "selected_specialists": summarize(selected_sizes),
        "load_cost_ms": summarize(load_ms),
        "memory": {
            "shared_process_rss_mb_before": shared_before,
            "shared_process_rss_mb_after": shared_after,
            "specialist_incremental_rss_mb": float(sum(load_mem)),
        },
        "cost_ms": {
            "embedding": summarize(embed_cost),
            "routing": summarize(route_cost),
            "fusion": summarize(fusion_cost),
            "end_to_end": summarize(e2e_cost),
        },
    }


def run_lazy_loading(
    definitions: List[SpecialistDefinition],
    pool: LazyEmbedderPool,
    queries: List[str],
    strategy: str,
    threshold: float,
) -> Dict[str, Any]:
    loaded: Dict[str, SEF] = {}
    fast_embedder = pool.get(definitions[0].embedder_id)

    embed_cost, route_cost, fusion_cost, load_cost, e2e_cost = [], [], [], [], []
    selected_sizes = []
    load_mem = []

    for q in queries:
        t0 = time.perf_counter()
        selected_defs, ems, rms, _ = semantic_route(q, definitions, fast_embedder, threshold)

        selected_modules: List[SEF] = []
        per_query_load_ms = 0.0
        for sd in selected_defs:
            module, lms, dmem = load_specialist_runtime(sd, pool, loaded)
            selected_modules.append(module)
            per_query_load_ms += lms
            load_mem.append(dmem)

        fms = run_single_inference(q, selected_modules, strategy)
        total = (time.perf_counter() - t0) * 1000.0

        embed_cost.append(ems)
        route_cost.append(rms)
        fusion_cost.append(fms)
        load_cost.append(per_query_load_ms)
        e2e_cost.append(total)
        selected_sizes.append(len(selected_modules))

    return {
        "scenario": "lazy_loading",
        "routing_cache": "disabled",
        "batch_size": 1,
        "specialists_loaded": len(loaded),
        "specialists_available": len(definitions),
        "selected_specialists": summarize(selected_sizes),
        "load_cost_ms": summarize(load_cost),
        "memory": {
            "specialist_incremental_rss_mb": float(sum(load_mem)),
            "loaded_embedder_ids": pool.loaded_embedder_ids,
        },
        "cost_ms": {
            "embedding": summarize(embed_cost),
            "routing": summarize(route_cost),
            "fusion": summarize(fusion_cost),
            "end_to_end": summarize(e2e_cost),
        },
    }


def run_cached_routing(
    definitions: List[SpecialistDefinition],
    pool: LazyEmbedderPool,
    queries: List[str],
    strategy: str,
    threshold: float,
) -> Dict[str, Any]:
    loaded: Dict[str, SEF] = {}
    for spec in definitions:
        load_specialist_runtime(spec, pool, loaded)

    fast_embedder = pool.get(definitions[0].embedder_id)
    cache: Dict[str, np.ndarray] = {}

    # warm cache once
    for q in set(queries):
        semantic_route(q, definitions, fast_embedder, threshold, embedding_cache=cache)

    embed_cost, route_cost, fusion_cost, e2e_cost = [], [], [], []
    cache_hits = 0
    selected_sizes = []

    for q in queries:
        t0 = time.perf_counter()
        selected_defs, ems, rms, hit = semantic_route(
            q,
            definitions,
            fast_embedder,
            threshold,
            embedding_cache=cache,
        )
        if hit:
            cache_hits += 1
        selected_modules = [loaded[d.name] for d in selected_defs]
        fms = run_single_inference(q, selected_modules, strategy)
        total = (time.perf_counter() - t0) * 1000.0

        embed_cost.append(ems)
        route_cost.append(rms)
        fusion_cost.append(fms)
        e2e_cost.append(total)
        selected_sizes.append(len(selected_modules))

    return {
        "scenario": "cached_routing",
        "routing_cache": "warmed_query_embedding_cache",
        "cache_hit_rate": float(cache_hits / len(queries)) if queries else 0.0,
        "batch_size": 1,
        "specialists_loaded": len(loaded),
        "specialists_available": len(definitions),
        "selected_specialists": summarize(selected_sizes),
        "load_cost_ms": summarize([0.0 for _ in queries]),
        "memory": {
            "routing_cache_entries": len(cache),
            "loaded_embedder_ids": pool.loaded_embedder_ids,
        },
        "cost_ms": {
            "embedding": summarize(embed_cost),
            "routing": summarize(route_cost),
            "fusion": summarize(fusion_cost),
            "end_to_end": summarize(e2e_cost),
        },
    }


def run_batched(
    definitions: List[SpecialistDefinition],
    pool: LazyEmbedderPool,
    queries: List[str],
    strategy: str,
    threshold: float,
    batch_size: int,
) -> Dict[str, Any]:
    loaded: Dict[str, SEF] = {}
    for spec in definitions:
        load_specialist_runtime(spec, pool, loaded)

    fast_embedder = pool.get(definitions[0].embedder_id)
    fuser = MeanFuser() if strategy == "mean" else KalmanorixFuser()

    embed_cost, route_cost, fusion_cost, e2e_cost = [], [], [], []
    selected_sizes = []

    for i in range(0, len(queries), batch_size):
        batch = queries[i : i + batch_size]
        if not batch:
            continue

        selected_for_batch: List[List[SEF]] = []
        batch_embed_ms = 0.0
        batch_route_ms = 0.0

        t0 = time.perf_counter()
        for q in batch:
            selected_defs, ems, rms, _ = semantic_route(q, definitions, fast_embedder, threshold)
            selected_for_batch.append([loaded[d.name] for d in selected_defs])
            selected_sizes.append(len(selected_defs))
            batch_embed_ms += ems
            batch_route_ms += rms

        # If selections match, use fuse_batch directly.
        first = selected_for_batch[0]
        same_selection = all(
            len(sel) == len(first) and all(a.name == b.name for a, b in zip(sel, first))
            for sel in selected_for_batch[1:]
        )

        fuse_start = time.perf_counter()
        if same_selection:
            _ = fuser.fuse_batch(batch, first)
        else:
            for q, sel in zip(batch, selected_for_batch):
                _ = fuser.fuse(q, sel)
        batch_fusion_ms = (time.perf_counter() - fuse_start) * 1000.0
        batch_total_ms = (time.perf_counter() - t0) * 1000.0

        per_q = max(1, len(batch))
        embed_cost.extend([batch_embed_ms / per_q] * per_q)
        route_cost.extend([batch_route_ms / per_q] * per_q)
        fusion_cost.extend([batch_fusion_ms / per_q] * per_q)
        e2e_cost.extend([batch_total_ms / per_q] * per_q)

    return {
        "scenario": "batched_inference",
        "routing_cache": "disabled",
        "batch_size": batch_size,
        "specialists_loaded": len(loaded),
        "specialists_available": len(definitions),
        "selected_specialists": summarize(selected_sizes),
        "load_cost_ms": summarize([0.0 for _ in e2e_cost]),
        "memory": {
            "loaded_embedder_ids": pool.loaded_embedder_ids,
            "batch_size": batch_size,
        },
        "cost_ms": {
            "embedding": summarize(embed_cost),
            "routing": summarize(route_cost),
            "fusion": summarize(fusion_cost),
            "end_to_end": summarize(e2e_cost),
        },
    }


def run_benchmark(config: BenchmarkConfig) -> Dict[str, Any]:
    embedder_paths = {
        "tech-minilm": config.models_dir / "tech-minilm",
        "cook-minilm": config.models_dir / "cook-minilm",
        "charge-minilm": config.models_dir / "charge-minilm",
    }
    pool = LazyEmbedderPool(embedder_paths=embedder_paths)
    definitions = load_specialist_definitions(config.sefs_dir, pool)

    queries = build_queries(config.repeats)

    scenarios = [
        run_all_loaded(definitions, pool, queries, "kalman", config.similarity_threshold),
        run_lazy_loading(definitions, pool, queries, "kalman", config.similarity_threshold),
        run_cached_routing(definitions, pool, queries, "kalman", config.similarity_threshold),
        run_batched(
            definitions,
            pool,
            queries,
            "kalman",
            config.similarity_threshold,
            config.batch_size,
        ),
    ]

    return {
        "schema_version": "efficiency_report.v2",
        "benchmark": {
            "name": "deployment_efficiency",
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "routing_mode": config.routing_mode,
            "similarity_threshold": config.similarity_threshold,
            "query_count": len(queries),
            "distinct_specialists": [d.name for d in definitions],
            "distinct_specialist_count": len(definitions),
        },
        "scenarios": scenarios,
    }


def parse_args() -> BenchmarkConfig:
    parser = argparse.ArgumentParser(description="Deployment-oriented efficiency benchmark")
    parser.add_argument("--sefs-dir", type=Path, default=Path("artefacts/sefs"))
    parser.add_argument("--models-dir", type=Path, default=Path("models"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/efficiency/deployment_efficiency.json"),
    )
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--similarity-threshold", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=8)

    args = parser.parse_args()
    return BenchmarkConfig(
        sefs_dir=args.sefs_dir,
        models_dir=args.models_dir,
        output=args.output,
        repeats=args.repeats,
        similarity_threshold=args.similarity_threshold,
        batch_size=args.batch_size,
    )


def main() -> None:
    cfg = parse_args()
    report = run_benchmark(cfg)
    cfg.output.parent.mkdir(parents=True, exist_ok=True)
    cfg.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
