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
from typing import Dict, List, Optional, Union, Callable, Any

import numpy as np

# Kalmanorix imports
from kalmanorix import (
    KalmanorixFuser,
    MeanFuser,
    Panoramix,
    ScoutRouter,
    SEF,
    Village,
)
from kalmanorix.compute_tracker import ComputeTracker
from kalmanorix.registry import EmbedderRegistry
from kalmanorix.sef_io import SEFArtefact
from kalmanorix.uncertainty import CentroidDistanceSigma2
from kalmanorix.threshold_heuristics import (
    threshold_top_k,
    threshold_relative_to_max,
    threshold_adaptive_spread,
    threshold_query_length_adaptive,
)
from kalmanorix.types import Embedder, Vec


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

    # Routing configuration
    routing_modes: List[str] = field(default_factory=lambda: ["all", "semantic"])
    fast_embedder_checkpoint: Optional[str] = None
    fast_embedder: Optional[Embedder] = None
    similarity_threshold: Union[float, str] = 0.7
    threshold_kwargs: Dict[str, Any] = field(default_factory=dict)
    fallback_mode: str = "all"


@dataclass
class BenchmarkResult:
    """Results for a single benchmark run."""

    # Configuration
    specialist_count: int
    strategy: str
    routing_mode: str
    num_modules_loaded: int

    # Selection metrics (semantic routing)
    specialists_selected_count: int
    selection_efficiency: float  # selected/loaded

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
    fast_embedder: Optional[Embedder] = None,
) -> List[SEF]:
    """Load base SEFs from artefacts.

    If fast_embedder is provided, compute domain centroids using fast_embedder
    from calibration texts (instead of using sigma2.centroid which is in the
    specialist's embedding space). This ensures semantic routing works correctly.
    """
    modules: List[SEF] = []

    for p in sorted(sefs_dir.glob("*.json")):
        art = SEFArtefact.load(p)
        embed = registry.get(art.embedder_id)
        sigma2 = art.build_sigma2(registry=registry, base_dir=repo_root)

        domain_centroid = None
        calibration_texts = None

        # Try to get calibration texts from sigma2_params
        if hasattr(art, "sigma2_params") and isinstance(art.sigma2_params, dict):
            calibration_texts = art.sigma2_params.get("calibration_texts")

        if fast_embedder is not None and calibration_texts is not None:
            # Compute centroid using fast embedder (shared embedding space)
            embs = [fast_embedder(t) for t in calibration_texts]
            centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            domain_centroid = centroid
            # Store calibration texts in meta for potential reuse
            if art.meta is None:
                art.meta = {}
            art.meta["calibration_texts"] = calibration_texts
            print(
                f"    Computed centroid for {art.name} with fast embedder, norm={norm:.6f}"
            )
        elif isinstance(sigma2, CentroidDistanceSigma2):
            # Fallback to centroid from sigma2 (specialist's embedding space)
            domain_centroid = sigma2.centroid
            # Note: this centroid may be in different space than fast_embedder
            print(
                f"    Using sigma2 centroid for {art.name} (specialist embedder space)"
            )

        modules.append(
            SEF(
                name=art.name,
                embed=embed,
                sigma2=sigma2,
                meta=art.meta,
                domain_centroid=domain_centroid,
            )
        )

    return modules


def load_fast_embedder(checkpoint_path: Optional[str]) -> Optional[Embedder]:
    """Load fast embedder for semantic routing.

    If checkpoint_path is None, returns None (use default embedder).
    """
    if checkpoint_path is None:
        return None
    return make_st_embedder(checkpoint_path)


def ensure_centroids_computed(village: Village, fast_embedder: Embedder) -> None:
    """Ensure each module in village has a domain_centroid computed using fast_embedder.

    If a module's domain_centroid is None but has calibration_texts in meta,
    compute centroid and set it. This ensures semantic routing works with
    consistent embedding space.
    """
    for module in village.modules:
        if module.domain_centroid is not None:
            continue
        if module.meta and "calibration_texts" in module.meta:
            texts = module.meta["calibration_texts"]
            if not texts:
                continue
            embs = [fast_embedder(t) for t in texts]
            centroid = np.mean(embs, axis=0)
            norm = np.linalg.norm(centroid)
            if norm > 0:
                centroid = centroid / norm
            module.domain_centroid = centroid
            # Optionally update meta to indicate centroid computed
            if module.meta is None:
                module.meta = {}
            module.meta["centroid_computed_with_fast_embedder"] = True


def get_similarity_threshold(
    threshold: Union[float, str],
    threshold_kwargs: Dict[str, Any],
) -> Union[float, Callable[[str, Vec, List[float]], float]]:
    """Convert threshold specification to float or callable for ScoutRouter.

    Args:
        threshold: Either float (fixed threshold) or string naming a heuristic.
        threshold_kwargs: Keyword arguments for the heuristic function.

    Returns:
        Either float threshold or callable that computes threshold from similarities.
    """
    if isinstance(threshold, (int, float)):
        return float(threshold)

    # Map string to heuristic function
    mapping = {
        "top_k": threshold_top_k,
        "relative_to_max": threshold_relative_to_max,
        "adaptive_spread": threshold_adaptive_spread,
        "query_length_adaptive": threshold_query_length_adaptive,
    }
    if threshold not in mapping:
        raise ValueError(
            f"Unknown threshold heuristic: {threshold}. "
            f"Available: {list(mapping.keys())}"
        )
    func = mapping[threshold]

    # Create partial function with kwargs
    from functools import partial

    return partial(func, **threshold_kwargs)


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
                    domain_centroid=base.domain_centroid,
                )
            )

    return Village(modules=modules)


def benchmark_single_run(
    village: Village,
    strategy: str,
    query: str,
    config: BenchmarkConfig,
    routing_mode: str = "all",
) -> BenchmarkResult:
    """
    Run a single benchmark for a given village, fusion strategy, and routing mode.

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

    # Create appropriate ScoutRouter based on routing mode
    if routing_mode == "all":
        scout = ScoutRouter(mode="all")
    elif routing_mode == "semantic":
        # Load fast embedder (use preloaded from config if available)
        fast_embedder = config.fast_embedder
        if fast_embedder is None and village.modules:
            # Fallback to first specialist's embedder
            fast_embedder = village.modules[0].embed

        # Get threshold function
        threshold = get_similarity_threshold(
            config.similarity_threshold,
            config.threshold_kwargs,
        )

        scout = ScoutRouter(
            mode="semantic",
            fast_embedder=fast_embedder,
            similarity_threshold=threshold,
            fallback_mode=config.fallback_mode,
        )
    else:
        raise ValueError(f"Unknown routing mode: {routing_mode}")

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

    selected_counts = []
    with tracker:
        # Process the query multiple times for stable timing
        for _ in range(config.num_repeats):
            start = time.perf_counter()
            potion = pan.brew(query, village=village, scout=scout)
            end = time.perf_counter()
            latencies_ms.append((end - start) * 1000)

            # Record which specialists were selected
            selected_modules = None
            if potion.meta and "selected_modules" in potion.meta:
                selected_modules = potion.meta["selected_modules"]
            if selected_modules is not None:
                selected_count = len(selected_modules)
            else:
                selected_count = len(village.modules)
            selected_counts.append(selected_count)

            # Record tokens processed (tokens per query * selected specialists)
            # FLOPs will be calculated manually based on selected_count
            tracker.add_tokens(int(tokens_per_query * selected_count))

    metrics = tracker.get_metrics()

    # Calculate statistics
    latency_mean = float(np.mean(latencies_ms))
    latency_std = float(np.std(latencies_ms))
    avg_selected_count = float(np.mean(selected_counts))

    # Compute FLOPs metrics based on selected specialists
    # FLOPs per query = 2 * params * tokens * selected_count
    flops_single_specialist = 2 * estimated_params_per_specialist * tokens_per_query
    flops_per_query = flops_single_specialist * avg_selected_count
    flops_per_token = flops_per_query / tokens_per_query
    total_flops_estimated = int(flops_per_query * config.num_repeats)

    # FLOPs ratio relative to single specialist (baseline)
    # Should equal avg_selected_count (number of specialists processing query)
    flops_ratio = avg_selected_count

    # Compute selection efficiency
    selection_efficiency = (
        avg_selected_count / len(village.modules) if village.modules else 0.0
    )
    # Compute tokens per second based on actual processed tokens
    tokens_processed_per_query = tokens_per_query * avg_selected_count
    tokens_per_second_actual = (
        tokens_processed_per_query / (latency_mean / 1000) if latency_mean > 0 else 0.0
    )

    return BenchmarkResult(
        specialist_count=len(village.modules),
        strategy=strategy,
        routing_mode=routing_mode,
        num_modules_loaded=len(village.modules),
        specialists_selected_count=int(round(avg_selected_count)),
        selection_efficiency=selection_efficiency,
        latency_mean_ms=latency_mean,
        latency_std_ms=latency_std,
        total_flops=total_flops_estimated,
        flops_per_token=flops_per_token,
        tokens_per_second=tokens_per_second_actual,
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
    # Determine fast embedder checkpoint
    if config.fast_embedder_checkpoint:
        fast_embedder_checkpoint = config.fast_embedder_checkpoint
    else:
        # Default to tech-minilm (general domain)
        fast_embedder_checkpoint = str(config.models_dir / "tech-minilm")
    fast_embedder = load_fast_embedder(fast_embedder_checkpoint)
    config.fast_embedder = fast_embedder
    print(f"Using fast embedder: {fast_embedder_checkpoint}")

    base_sefs = load_base_sefs(
        sefs_dir=config.sefs_dir,
        registry=registry,
        repo_root=config.repo_root,
        fast_embedder=fast_embedder,
    )

    print(f"Loaded {len(base_sefs)} base SEFs")

    # Ensure base SEFs have centroids computed with fast embedder
    if fast_embedder is not None:
        base_village = Village(modules=base_sefs)
        ensure_centroids_computed(base_village, fast_embedder)

    results = []

    for count in config.specialist_counts:
        print(f"\n--- Benchmarking with {count} specialists ---")

        village = create_scaled_village(base_sefs, count)
        # Ensure centroids computed for scaled village (in case new modules missing centroids)
        if fast_embedder is not None:
            ensure_centroids_computed(village, fast_embedder)

        for strategy in config.strategies:
            for routing_mode in config.routing_modes:
                print(f"  Strategy: {strategy}, Routing: {routing_mode}")

                result = benchmark_single_run(
                    village=village,
                    strategy=strategy,
                    query=config.query_text,
                    routing_mode=routing_mode,
                    config=config,
                )

                results.append(result)

                print(
                    f"    Latency: {result.latency_mean_ms:.2f} ± {result.latency_std_ms:.2f} ms"
                )
                print(f"    FLOPs ratio vs monolith: {result.flops_vs_monolith:.3f}")
                print(
                    f"    Specialists selected: {result.specialists_selected_count}/{result.specialist_count} (efficiency: {result.selection_efficiency:.2f})"
                )
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
            "routing_mode": r.routing_mode,
            "num_modules_loaded": r.num_modules_loaded,
            "specialists_selected_count": r.specialists_selected_count,
            "selection_efficiency": r.selection_efficiency,
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

    # Group by specialist count, strategy, and routing mode
    # Dict[count, Dict[strategy, Dict[routing_mode, result]]]
    by_count: Dict[int, Dict[str, Dict[str, BenchmarkResult]]] = {}
    for r in results:
        if r.specialist_count not in by_count:
            by_count[r.specialist_count] = {}
        if r.strategy not in by_count[r.specialist_count]:
            by_count[r.specialist_count][r.strategy] = {}
        by_count[r.specialist_count][r.strategy][r.routing_mode] = r

    print(
        "\nSpecialists | Strategy | Routing   | Latency (ms) | FLOPs ratio | Selected | Efficiency | CPU Mem (MB)"
    )
    print("-" * 100)

    for count in sorted(by_count.keys()):
        for strategy in ["mean", "kalman"]:
            if strategy in by_count[count]:
                for routing_mode in ["all", "semantic"]:
                    if routing_mode in by_count[count][strategy]:
                        r = by_count[count][strategy][routing_mode]
                        mem = r.cpu_memory_mb or 0
                        selected = (
                            f"{r.specialists_selected_count}/{r.specialist_count}"
                        )
                        efficiency = f"{r.selection_efficiency:.2f}"
                        print(
                            f"{count:11d} | {strategy:8s} | {routing_mode:9s} | {r.latency_mean_ms:6.2f} ± {r.latency_std_ms:4.2f} | {r.flops_vs_monolith:11.3f} | {selected:9s} | {efficiency:10s} | {mem:12d}"
                        )

    print("\nNotes:")
    print("- FLOPs ratio: 1.0 means same FLOPs as monolithic model")
    print("- Ratio > 1.0 means fusion uses more FLOPs (due to duplicate processing)")
    print("- Selected: specialists selected / total specialists (semantic routing)")
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
        "--specialist-counts",
        type=str,
        default="1,2,3,5,10,20",
        help="Comma-separated list of specialist counts to benchmark",
    )
    parser.add_argument(
        "--strategies",
        type=str,
        default="mean,kalman",
        help="Comma-separated list of fusion strategies to benchmark (mean, kalman)",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="Test query about battery life and cooking stew",
        help="Query text to use for benchmarking",
    )
    parser.add_argument(
        "--routing-modes",
        type=str,
        default="all,semantic",
        help="Comma-separated routing modes to benchmark (all, semantic)",
    )
    parser.add_argument(
        "--fast-embedder-checkpoint",
        type=str,
        default="",
        help="Path to fast embedder model for semantic routing (optional)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=str,
        default="0.7",
        help="Similarity threshold for semantic routing: float or heuristic name (top_k, relative_to_max, adaptive_spread, query_length_adaptive)",
    )
    parser.add_argument(
        "--threshold-kwargs",
        type=str,
        default="{}",
        help="JSON string of keyword arguments for threshold heuristic",
    )
    parser.add_argument(
        "--fallback-mode",
        type=str,
        default="all",
        help="Fallback mode when no specialists meet threshold (all, none)",
    )

    args = parser.parse_args()

    # Parse routing modes
    routing_modes = [m.strip() for m in args.routing_modes.split(",") if m.strip()]
    # Parse specialist counts
    specialist_counts = [
        int(c.strip()) for c in args.specialist_counts.split(",") if c.strip()
    ]
    # Parse strategies
    strategies = [s.strip() for s in args.strategies.split(",") if s.strip()]
    # Parse similarity threshold: try float, otherwise keep string
    try:
        similarity_threshold = float(args.similarity_threshold)
    except ValueError:
        similarity_threshold = args.similarity_threshold
    # Parse threshold kwargs JSON
    threshold_kwargs = json.loads(args.threshold_kwargs)
    # Fast embedder checkpoint: empty string -> None
    fast_embedder_checkpoint = (
        args.fast_embedder_checkpoint if args.fast_embedder_checkpoint else None
    )

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
        routing_modes=routing_modes,
        specialist_counts=specialist_counts,
        strategies=strategies,
        fast_embedder_checkpoint=fast_embedder_checkpoint,
        similarity_threshold=similarity_threshold,
        threshold_kwargs=threshold_kwargs,
        fallback_mode=args.fallback_mode,
    )

    print("Starting efficiency benchmarking (Milestone 2.3)")
    print(f"Query: {config.query_text}")
    print(f"Specialist counts: {config.specialist_counts}")
    print(f"Strategies: {config.strategies}")
    print(f"Routing modes: {config.routing_modes}")
    print(f"Track GPU: {config.track_gpu}")
    print(f"Track memory: {config.track_memory}")
    print(f"Output: {config.output_path}")

    results = run_benchmarks(config)

    if config.output_path:
        save_results(results, config.output_path)

    print_summary(results)


if __name__ == "__main__":
    main()
