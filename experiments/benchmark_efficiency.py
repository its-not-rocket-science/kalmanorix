#!/usr/bin/env python3
"""
Efficiency benchmarking for Kalmanorix fusion (Milestone 2.3).

Measures inference FLOPs, memory usage, and latency for fusion vs single large model.
Uses the enhanced compute tracker with memory tracking and inference mode.

Key metrics:
- Memory usage for multiple loaded specialists (CPU RAM, GPU VRAM)
- Inference latency per query (wall time)
- Estimated FLOPs per token (inference mode)
- Comparison to hypothetical monolithic model (sum of specialist parameters)

Usage:
    python experiments/benchmark_efficiency.py [--output results/efficiency.json]
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

# Kalmanorix imports
from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village
from kalmanorix.compute_tracker import ComputeTracker
from kalmanorix.registry import EmbedderRegistry
from kalmanorix.sef_io import SEFArtefact
from kalmanorix.types import Embedder


@dataclass
class BenchmarkConfig:
    """Configuration for efficiency benchmarking."""

    # Specialist loading
    sefs_dir: Path = Path("artefacts/sefs")
    models_dir: Path = Path("models")
    registry_json: Optional[Path] = None
    repo_root: Path = Path(".")

    # Benchmark parameters
    query_text: str = "Test query about battery life and cooking stew"
    num_repeats: int = 10  # For timing stability
    specialist_counts: List[int] = field(
        default_factory=lambda: [1, 2, 3, 5, 10, 20]
    )  # For full benchmark

    # Compute tracking
    track_gpu: bool = False
    track_memory: bool = True
    output_path: Optional[Path] = None

    # Fusion strategies to benchmark
    strategies: List[str] = field(default_factory=lambda: ["mean", "kalman"])


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""

    # Configuration
    specialist_count: int
    strategy: str
    num_modules_loaded: int

    # Timing metrics
    latency_mean_ms: float
    latency_std_ms: float

    # Compute metrics
    total_flops: int
    flops_per_token: float
    tokens_per_second: float

    # Memory metrics
    cpu_memory_mb: Optional[int] = None
    peak_cpu_memory_mb: Optional[int] = None
    peak_gpu_memory_mb: Optional[int] = None

    # GPU metrics (if available)
    gpu_energy_joules: Optional[float] = None
    gpu_memory_mb: Optional[int] = None
    gpu_utilization_percent: Optional[float] = None

    # Derived efficiency ratios
    flops_vs_monolith: float = 1.0  # FLOPs relative to monolithic model


def make_st_embedder(checkpoint_path: str) -> Embedder:
    """Create a SentenceTransformer embedder (returns np.float64 unit vectors)."""
    # pylint: disable=import-outside-toplevel
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(checkpoint_path)

    def embed(text: str) -> np.ndarray:
        v = model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float64)

    return embed


def load_embedder_registry(
    models_dir: Path,
    registry_json: Optional[Path],
) -> EmbedderRegistry:
    """Build an EmbedderRegistry mapping embedder_id -> callable."""
    mapping: Dict[str, str] = {}

    if registry_json is not None and registry_json.exists():
        raw = json.loads(registry_json.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError("registry JSON must be an object mapping id -> path")
        mapping = {str(k): str(v) for k, v in raw.items()}
    else:
        # Default mapping based on directory structure
        mapping = {
            "tech-minilm": str(models_dir / "tech-minilm"),
            "cook-minilm": str(models_dir / "cook-minilm"),
            "charge-minilm": str(models_dir / "charge-minilm"),
        }

    embedders = {eid: make_st_embedder(path) for eid, path in mapping.items()}
    return EmbedderRegistry(embedders=embedders)


def load_base_sefs(
    sefs_dir: Path,
    registry: EmbedderRegistry,
    repo_root: Path,
) -> List[SEF]:
    """Load base SEFs from artefacts."""
    modules: List[SEF] = []

    for p in sorted(sefs_dir.glob("*.json")):
        art = SEFArtefact.load(p)
        embed = registry.get(art.embedder_id)
        sigma2 = art.build_sigma2(registry=registry, base_dir=repo_root)

        modules.append(SEF(name=art.name, embed=embed, sigma2=sigma2, meta=art.meta))

    return modules


def create_scaled_village(
    base_modules: List[SEF],
    target_count: int,
) -> Village:
    """
    Create a village with target_count modules by duplicating base modules.

    For scaling benchmarks, we need more specialists than we have trained.
    We create duplicates with unique names and slightly modified sigma2.
    """
    if target_count <= len(base_modules):
        # Use subset of base modules
        modules = base_modules[:target_count]
    else:
        # Duplicate base modules with modifications
        modules = []
        for i in range(target_count):
            base = base_modules[i % len(base_modules)]
            # Create a copy with unique name
            new_name = f"{base.name}_{i}"
            # Optionally modify sigma2 slightly for diversity
            # For now, keep same sigma2 function
            modules.append(
                SEF(
                    name=new_name,
                    embed=base.embed,
                    sigma2=base.sigma2,
                    meta=dict(base.meta) if base.meta else None,
                )
            )

    return Village(modules=modules)


def benchmark_single_run(
    village: Village,
    strategy: str,
    query: str,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    """
    Run a single benchmark for a given village and fusion strategy.

    Uses compute tracker to measure FLOPs, memory, and timing.
    """
    # Create fusion strategy
    if strategy == "mean":
        fuser = MeanFuser()
    elif strategy == "kalman":
        fuser = KalmanorixFuser()
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    pan = Panoramix(fuser=fuser)
    scout = ScoutRouter(mode="all")

    # Warmup
    for _ in range(3):
        _ = pan.brew(query, village=village, scout=scout)

    # Time measurement with repetitions
    latencies_ms = []

    # Use compute tracker for FLOPs and memory measurement
    # Estimate parameters: sum of specialist parameters (each ~22M for MiniLM-L6)
    # This is approximate; real parameter count would require model inspection
    estimated_params_per_specialist = (
        22_000_000  # From compute_tracker.estimate_parameters
    )
    total_params = estimated_params_per_specialist * len(village.modules)

    # Approximate tokens in query (rough estimate: words * 1.3)
    tokens_per_query = max(1, len(query.split()) * 1.3)

    # Create compute tracker directly with total parameters
    tracker = ComputeTracker(
        model_parameters=total_params,
        track_gpu=config.track_gpu,
        mode="inference",
        track_memory=config.track_memory,
    )

    with tracker:
        # Process the query multiple times for stable timing
        for _ in range(config.num_repeats):
            start = time.perf_counter()
            _ = pan.brew(query, village=village, scout=scout)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

            # Record tokens processed by all specialists
            # Each specialist processes the query tokens once
            tracker.add_tokens(int(tokens_per_query))

    metrics = tracker.get_metrics()

    # Calculate statistics
    latency_mean = float(np.mean(latencies_ms))
    latency_std = float(np.std(latencies_ms))

    # Compute FLOPs metrics
    flops_per_query = metrics.total_flops / config.num_repeats
    flops_per_token = flops_per_query / tokens_per_query

    # FLOPs ratio relative to single specialist (baseline)
    # Expected ratio ~ number of specialists (each processes query)
    flops_single_specialist = 2 * estimated_params_per_specialist * tokens_per_query
    flops_ratio = (
        flops_per_query / flops_single_specialist
        if flops_single_specialist > 0
        else 1.0
    )

    return BenchmarkResult(
        specialist_count=len(village.modules),
        strategy=strategy,
        num_modules_loaded=len(village.modules),
        latency_mean_ms=latency_mean,
        latency_std_ms=latency_std,
        total_flops=metrics.total_flops,
        flops_per_token=flops_per_token,
        tokens_per_second=metrics.tokens_per_second,
        cpu_memory_mb=metrics.cpu_memory_mb,
        peak_cpu_memory_mb=metrics.peak_cpu_memory_mb,
        peak_gpu_memory_mb=metrics.peak_gpu_memory_mb,
        gpu_energy_joules=metrics.gpu_energy_joules,
        gpu_memory_mb=metrics.gpu_memory_mb,
        gpu_utilization_percent=metrics.gpu_utilization_percent,
        flops_vs_monolith=flops_ratio,
    )


def run_benchmarks(config: BenchmarkConfig) -> List[BenchmarkResult]:
    """Run all benchmark configurations."""
    print("Loading embedder registry...")
    registry = load_embedder_registry(
        models_dir=config.models_dir,
        registry_json=config.registry_json,
    )

    print("Loading base SEFs...")
    base_sefs = load_base_sefs(
        sefs_dir=config.sefs_dir,
        registry=registry,
        repo_root=config.repo_root,
    )

    print(f"Loaded {len(base_sefs)} base SEFs")

    results = []

    for count in config.specialist_counts:
        print(f"\n--- Benchmarking with {count} specialists ---")

        village = create_scaled_village(base_sefs, count)

        for strategy in config.strategies:
            print(f"  Strategy: {strategy}")

            result = benchmark_single_run(
                village=village,
                strategy=strategy,
                query=config.query_text,
                config=config,
            )

            results.append(result)

            print(
                f"    Latency: {result.latency_mean_ms:.2f} ± {result.latency_std_ms:.2f} ms"
            )
            print(f"    FLOPs ratio vs monolith: {result.flops_vs_monolith:.3f}")
            if result.cpu_memory_mb:
                print(f"    CPU memory: {result.cpu_memory_mb} MB")
            if result.peak_cpu_memory_mb:
                print(f"    Peak CPU memory: {result.peak_cpu_memory_mb} MB")

    return results


def save_results(results: List[BenchmarkResult], output_path: Path) -> None:
    """Save benchmark results to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    serializable = []
    for r in results:
        data = {
            "specialist_count": r.specialist_count,
            "strategy": r.strategy,
            "num_modules_loaded": r.num_modules_loaded,
            "latency_mean_ms": r.latency_mean_ms,
            "latency_std_ms": r.latency_std_ms,
            "total_flops": r.total_flops,
            "flops_per_token": r.flops_per_token,
            "tokens_per_second": r.tokens_per_second,
            "cpu_memory_mb": r.cpu_memory_mb,
            "peak_cpu_memory_mb": r.peak_cpu_memory_mb,
            "peak_gpu_memory_mb": r.peak_gpu_memory_mb,
            "gpu_energy_joules": r.gpu_energy_joules,
            "gpu_memory_mb": r.gpu_memory_mb,
            "gpu_utilization_percent": r.gpu_utilization_percent,
            "flops_vs_monolith": r.flops_vs_monolith,
        }
        # Remove None values
        data = {k: v for k, v in data.items() if v is not None}
        serializable.append(data)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"\nResults saved to {output_path}")


def print_summary(results: List[BenchmarkResult]) -> None:
    """Print a summary table of benchmark results."""
    print("\n" + "=" * 80)
    print("EFFICIENCY BENCHMARK SUMMARY")
    print("=" * 80)

    # Group by specialist count
    by_count: Dict[int, Dict[str, BenchmarkResult]] = {}
    for r in results:
        if r.specialist_count not in by_count:
            by_count[r.specialist_count] = {}
        by_count[r.specialist_count][r.strategy] = r

    print("\nSpecialists | Strategy | Latency (ms) | FLOPs ratio | CPU Mem (MB)")
    print("-" * 80)

    for count in sorted(by_count.keys()):
        for strategy in ["mean", "kalman"]:
            if strategy in by_count[count]:
                r = by_count[count][strategy]
                mem = r.cpu_memory_mb or 0
                print(
                    f"{count:11d} | {strategy:8s} | {r.latency_mean_ms:6.2f} ± {r.latency_std_ms:4.2f} | {r.flops_vs_monolith:11.3f} | {mem:12d}"
                )

    print("\nNotes:")
    print("- FLOPs ratio: 1.0 means same FLOPs as monolithic model")
    print("- Ratio > 1.0 means fusion uses more FLOPs (due to duplicate processing)")
    print(
        "- Ideal modular deployment would have ratio close to 1.0 with low latency overhead"
    )


def main() -> None:
    """Main benchmarking routine."""
    parser = argparse.ArgumentParser(description="Kalmanorix efficiency benchmarking")
    parser.add_argument(
        "--sefs-dir",
        type=str,
        default="artefacts/sefs",
        help="Directory containing SEF JSON artefacts",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--registry-json",
        type=str,
        default="",
        help="Optional JSON mapping embedder_id -> checkpoint path",
    )
    parser.add_argument(
        "--repo-root",
        type=str,
        default=".",
        help="Base directory for resolving relative paths",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="results/efficiency_benchmark.json",
        help="Output JSON file for results",
    )
    parser.add_argument(
        "--track-gpu",
        action="store_true",
        help="Track GPU energy consumption (requires pynvml)",
    )
    parser.add_argument(
        "--no-memory", action="store_true", help="Disable memory tracking"
    )
    parser.add_argument(
        "--repeats", type=int, default=10, help="Number of repetitions for timing"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Test query about battery life and cooking stew",
        help="Query text to use for benchmarking",
    )

    args = parser.parse_args()

    config = BenchmarkConfig(
        sefs_dir=Path(args.sefs_dir),
        models_dir=Path(args.models_dir),
        registry_json=Path(args.registry_json) if args.registry_json else None,
        repo_root=Path(args.repo_root),
        query_text=args.query,
        num_repeats=args.repeats,
        track_gpu=args.track_gpu,
        track_memory=not args.no_memory,
        output_path=Path(args.output),
    )

    print("Starting efficiency benchmarking (Milestone 2.3)")
    print(f"Query: {config.query_text}")
    print(f"Specialist counts: {config.specialist_counts}")
    print(f"Strategies: {config.strategies}")
    print(f"Track GPU: {config.track_gpu}")
    print(f"Track memory: {config.track_memory}")
    print(f"Output: {config.output_path}")

    results = run_benchmarks(config)

    if config.output_path:
        save_results(results, config.output_path)

    print_summary(results)


if __name__ == "__main__":
    main()
