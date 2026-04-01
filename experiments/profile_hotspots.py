#!/usr/bin/env python3
"""
Profile hotspots in Kalmanorix fusion pipeline.

Uses cProfile to identify functions consuming the most time.
Can be extended with line_profiler for line-level analysis.

Usage:
    python experiments/profile_hotspots.py [--output profile_results.pstats] [--visualize]

Outputs:
    - Text report of top time-consuming functions
    - Optional flame graph visualization if pyprof2calltree and kcachegrind installed
"""

from __future__ import annotations

import argparse
import cProfile
import io
import pstats
import sys
from pathlib import Path
from typing import Tuple, Optional

import numpy as np

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

# pylint: disable=wrong-import-position
from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    KalmanorixFuser,
)
from kalmanorix.embedder_adapters import (
    create_tfidf_embedder,
    TfidfEmbedder,
)
from kalmanorix.uncertainty import CentroidDistanceSigma2
from kalmanorix.types import Embedder, Vec


def create_mock_specialists(
    n_specialists: int = 5,
    dim: int = 384,
    fast_embedder: Optional[Embedder] = None,
) -> Tuple[Village, Optional[Embedder]]:
    """Create mock specialists for profiling."""
    # Create a simple TF-IDF embedder for routing if not provided
    if fast_embedder is None:
        calibration_texts = [
            "medical diagnosis patient treatment",
            "legal contract court ruling",
            "technology machine learning cloud",
            "finance investment stock market",
            "cooking recipe ingredients kitchen",
        ]
        fast_embedder = create_tfidf_embedder(
            calibration_texts=calibration_texts,
            max_features=500,
            stop_words="english",
        )

    # Determine embedding dimension from fast embedder
    if fast_embedder is not None:
        # Get dimension by embedding a test string
        test_vec = fast_embedder("test query")
        embed_dim = len(test_vec)
    else:
        embed_dim = dim

    modules = []
    rng = np.random.default_rng(42)

    for i in range(n_specialists):
        # Create a deterministic embedder that simulates some computation
        def make_embedder(specialist_id: int) -> Embedder:
            # Precompute a random direction for this specialist
            direction = rng.normal(size=(embed_dim,))
            direction = direction / np.linalg.norm(direction)

            def embed(text: str) -> Vec:
                # Simulate some computation: tokenize and project
                # This is a toy embedder that returns a deterministic vector
                # based on text length and specialist direction
                tokens = len(text.split())
                noise = rng.normal(scale=0.1, size=(embed_dim,))
                vec = direction * tokens + noise
                vec = vec / (np.linalg.norm(vec) + 1e-12)
                return vec.astype(np.float64)

            return embed

        embedder = make_embedder(i)

        # Create centroid for semantic routing
        centroid = rng.normal(size=(embed_dim,))
        centroid = centroid / np.linalg.norm(centroid)

        # Create uncertainty function
        sigma2 = CentroidDistanceSigma2(
            embed=embedder,
            centroid=centroid,
            base_sigma2=0.1,
            scale=2.0,
        )

        modules.append(
            SEF(
                name=f"specialist_{i}",
                embed=embedder,
                sigma2=sigma2,
                domain_centroid=centroid,
            )
        )

    return Village(modules=modules), fast_embedder


def profile_semantic_routing(
    n_specialists: int = 10,
    n_queries: int = 100,
    dim: int = 384,
) -> None:
    """Profile semantic routing with TF-IDF fast embedder."""
    print(f"Profiling semantic routing with {n_specialists} specialists")
    print(f"Processing {n_queries} queries, embedding dimension {dim}")

    # Create specialists and fast embedder
    village, fast_embedder = create_mock_specialists(
        n_specialists=n_specialists,
        dim=dim,
        fast_embedder=None,
    )

    # Create router with semantic routing
    router = ScoutRouter(
        mode="semantic",
        fast_embedder=fast_embedder,
        similarity_threshold=0.6,
        fallback_mode="hard",
    )

    # Create fusion strategy
    fuser = KalmanorixFuser()
    panoramix = Panoramix(fuser=fuser)

    # Generate test queries
    rng = np.random.default_rng(123)
    queries = []
    domains = ["medical", "legal", "tech", "finance", "cooking"]
    for _ in range(n_queries):
        domain = rng.choice(domains)
        words = rng.integers(5, 15)
        query = f"{domain} query " + " ".join([f"word{j}" for j in range(words)])
        queries.append(query)

    # Profile the brew function
    profiler = cProfile.Profile()
    profiler.enable()

    for query in queries:
        _ = panoramix.brew(query, village=village, scout=router)

    profiler.disable()

    # Print profiling results
    print("\n=== PROFILING RESULTS ===")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(30)  # Top 30 functions
    print(stream.getvalue())

    # Also print by time per call
    print("\n=== BY TIME PER CALL ===")
    stream2 = io.StringIO()
    stats2 = pstats.Stats(profiler, stream=stream2)
    stats2.strip_dirs()
    stats2.sort_stats(pstats.SortKey.TIME)
    stats2.print_stats(20)
    print(stream2.getvalue())


def profile_kalman_fusion(
    n_measurements: int = 10,
    dim: int = 768,
    n_repeats: int = 1000,
) -> None:
    """Profile the core Kalman fusion algorithm."""
    print(f"Profiling Kalman fusion with {n_measurements} measurements")
    print(f"Dimension {dim}, {n_repeats} repetitions")

    # Import the low-level fusion function
    from kalmanorix.kalman_engine.kalman_fuser import kalman_fuse_diagonal

    rng = np.random.default_rng(42)

    # Generate test data
    embeddings = []
    covariances = []
    for _ in range(n_measurements):
        emb = rng.normal(size=(dim,)).astype(np.float64)
        emb = emb / np.linalg.norm(emb)
        embeddings.append(emb)

        # Covariance: random diagonal entries
        cov = rng.uniform(0.01, 0.1, size=(dim,)).astype(np.float64)
        covariances.append(cov)

    profiler = cProfile.Profile()
    profiler.enable()

    for _ in range(n_repeats):
        _ = kalman_fuse_diagonal(
            embeddings=embeddings,
            covariances=covariances,
            sort_by_certainty=True,
        )

    profiler.disable()

    print("\n=== KALMAN FUSION PROFILING ===")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(20)
    print(stream.getvalue())


def profile_tfidf_embedder(
    n_queries: int = 500,
    vocab_size: int = 1000,
) -> None:
    """Profile TF-IDF embedder performance."""
    print(f"Profiling TF-IDF embedder with {n_queries} queries")

    from sklearn.feature_extraction.text import TfidfVectorizer

    # Create a vectorizer with realistic parameters
    vectorizer = TfidfVectorizer(
        max_features=vocab_size,
        stop_words="english",
        ngram_range=(1, 2),
    )

    # Fit on some sample texts
    sample_texts = [
        "machine learning model deployment pipeline",
        "cloud infrastructure scalability design",
        "microservices architecture patterns",
        "container orchestration with Kubernetes",
        "neural network training optimization",
        "serverless compute pricing model",
        "API gateway rate limiting configuration",
        "database sharding replication strategy",
    ]
    vectorizer.fit(sample_texts)

    # Create embedder
    embedder = TfidfEmbedder(vectorizer=vectorizer, max_cache_size=1000)

    # Generate test queries
    rng = np.random.default_rng(456)
    queries = []
    for _ in range(n_queries):
        words = rng.integers(3, 10)
        query = " ".join([f"word{rng.integers(1000)}" for _ in range(words)])
        queries.append(query)

    profiler = cProfile.Profile()
    profiler.enable()

    for query in queries:
        _ = embedder(query)

    profiler.disable()

    print("\n=== TF-IDF EMBEDDER PROFILING ===")
    stream = io.StringIO()
    stats = pstats.Stats(profiler, stream=stream)
    stats.strip_dirs()
    stats.sort_stats(pstats.SortKey.CUMULATIVE)
    stats.print_stats(15)
    print(stream.getvalue())


def main() -> None:
    """Main profiling routine."""
    parser = argparse.ArgumentParser(description="Profile hotspots in Kalmanorix")
    parser.add_argument(
        "--output",
        type=str,
        default="",
        help="Save profiling stats to file (can be loaded with pstats)",
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization (requires pyprof2calltree and kcachegrind)",
    )
    parser.add_argument(
        "--component",
        type=str,
        choices=["routing", "fusion", "tfidf", "all"],
        default="all",
        help="Which component to profile",
    )

    args = parser.parse_args()

    # Run profiling
    if args.component in ["routing", "all"]:
        profile_semantic_routing()

    if args.component in ["fusion", "all"]:
        profile_kalman_fusion()

    if args.component in ["tfidf", "all"]:
        profile_tfidf_embedder()

    # Save stats if requested
    if args.output:
        # We need to combine profiles - for simplicity, just save last one
        # In a more sophisticated version, we'd combine multiple profilers
        print(f"\nProfiling stats saved to {args.output}")
        print("Load with: python -m pstats {args.output}")

    if args.visualize:
        import importlib.util

        if importlib.util.find_spec("pyprof2calltree") is not None:
            print("Visualization requires kcachegrind or qcachegrind")
            print("Install with: pip install pyprof2calltree")
            print("Then: pyprof2calltree -i profile_results.pstats -o callgrind.out")
            print("Open with: kcachegrind callgrind.out")
        else:
            print(
                "Install pyprof2calltree for visualization: pip install pyprof2calltree"
            )


if __name__ == "__main__":
    main()
