"""Fusion validation entrypoint.

Primary validation path uses real mixed-domain retrieval data and real specialist
models (see ``experiments/run_real_mixed_benchmark.py``).

Synthetic/toy validation remains available strictly as a smoke/debug path via
``--debug-synthetic``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village
from kalmanorix.toy_corpus import build_toy_corpus
from kalmanorix.uncertainty import KeywordSigma2



class DebugKeywordEmbedder:
    """Deterministic toy embedder for smoke tests only."""

    def __init__(self, dim: int, keywords: list[str], seed: int) -> None:
        self.dim = dim
        self.keywords = keywords
        rng = np.random.default_rng(seed)
        self.base = rng.normal(size=(dim,))
        self.base /= np.linalg.norm(self.base) + 1e-12
        self.dir = rng.normal(size=(dim,))
        self.dir /= np.linalg.norm(self.dir) + 1e-12

    def __call__(self, text: str) -> np.ndarray:
        vec = self.base.copy()
        t = text.lower()
        if any(kw in t for kw in self.keywords):
            vec += 2.0 * self.dir
        vec /= np.linalg.norm(vec) + 1e-12
        return vec.astype(np.float64)


def run_debug_synthetic_smoke(seed: int = 42) -> dict[str, float]:
    """Quick synthetic sanity check (debug only, non-headline)."""
    np.random.seed(seed)
    corpus = build_toy_corpus(british_spelling=True)

    tech = SEF(
        name="tech",
        embed=DebugKeywordEmbedder(96, ["battery", "cpu", "gpu", "camera"], seed=7),
        sigma2=KeywordSigma2({"battery", "cpu", "gpu", "camera"}, 0.1, 0.5),
    )
    cook = SEF(
        name="cook",
        embed=DebugKeywordEmbedder(96, ["braise", "simmer", "sauce", "oven"], seed=11),
        sigma2=KeywordSigma2({"braise", "simmer", "sauce", "oven"}, 0.1, 0.5),
    )
    village = Village([tech, cook])
    scout = ScoutRouter(mode="all")

    def recall_at_1(fuser) -> float:
        pan = Panoramix(fuser=fuser)
        doc_mat = np.stack([pan.brew(d, village=village, scout=scout).vector for d in corpus.docs])
        hits = []
        for q, true_id in corpus.queries:
            qv = pan.brew(q, village=village, scout=scout).vector
            pred = int(np.argmax(doc_mat @ qv))
            hits.append(float(pred == true_id))
        return float(np.mean(hits))

    return {
        "kalman_recall@1": recall_at_1(KalmanorixFuser()),
        "mean_recall@1": recall_at_1(MeanFuser()),
    }


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug-synthetic", action="store_true")
    parser.add_argument(
        "--benchmark-path",
        type=Path,
        default=Path("benchmarks/mixed_beir_v1.0.0/mixed_benchmark.parquet"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-queries", type=int, default=150)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/real_mixed_benchmark/real_benchmark_summary.json"),
    )
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    if args.debug_synthetic:
        metrics = run_debug_synthetic_smoke()
        print("Debug synthetic smoke:")
        print(metrics)
        return

    from experiments.run_real_mixed_benchmark import run_real_benchmark

    summary = run_real_benchmark(
        benchmark_path=args.benchmark_path,
        split=args.split,
        max_queries=args.max_queries,
        output_path=args.output,
        device=args.device,
    )
    print("Primary (real-data) benchmark complete.")
    print(summary["delta_kalman_minus_mean"])


if __name__ == "__main__":
    main()
