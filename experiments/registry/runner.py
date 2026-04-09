"""Benchmark registry runner with configuration-driven experiment execution."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from kalmanorix.benchmarks import QueryRanking

from experiments.registry.config_schema import BenchmarkExperimentConfig, load_experiment_config
from experiments.registry.datasets import load_dataset
from experiments.registry.evaluation import evaluate_locked, evaluate_synthetic_recall
from experiments.registry.fusion import build_strategy, rank_query
from experiments.registry.models import build_village
from experiments.registry.reporting import write_json


DEFAULT_REAL_SPECIALISTS = [
    {
        "name": "general_qa",
        "domain": "general_qa",
        "model_name": "sentence-transformers/all-mpnet-base-v2",
    },
    {
        "name": "biomedical",
        "domain": "biomedical",
        "model_name": "emilyalsentzer/Bio_ClinicalBERT",
    },
    {
        "name": "finance",
        "domain": "finance",
        "model_name": "ProsusAI/finbert",
    },
]


def set_global_seed(seed_python: int, seed_numpy: int, seed_torch: int) -> None:
    """Set reproducibility seeds across libraries."""
    random.seed(seed_python)
    np.random.seed(seed_numpy)
    try:
        import torch

        torch.manual_seed(seed_torch)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed_torch)
    except ImportError:
        return


def _run_synthetic(config: BenchmarkExperimentConfig) -> dict[str, Any]:
    corpus = load_dataset(
        kind=config.dataset.kind,
        path=config.dataset.path,
        split=config.dataset.split,
        max_queries=config.dataset.max_queries,
    )
    village = build_village(
        kind=config.models.kind,
        payload=corpus,
        specialists=config.models.specialists,
        device=config.models.device,
    )
    strategies = {
        name: build_strategy(name=name, routing_mode=config.fusion.routing_mode)
        for name in config.fusion.strategies
    }
    metrics = evaluate_synthetic_recall(corpus=corpus, village=village, strategies=strategies)
    return {
        "name": config.name,
        "experiment_type": config.experiment_type,
        "seed": config.seed.__dict__,
        "metrics": metrics,
    }


def _run_real_mixed(config: BenchmarkExperimentConfig) -> dict[str, Any]:
    rows = load_dataset(
        kind=config.dataset.kind,
        path=config.dataset.path,
        split=config.dataset.split,
        max_queries=config.dataset.max_queries,
    )
    specialists = config.models.specialists or DEFAULT_REAL_SPECIALISTS
    village = build_village(
        kind=config.models.kind,
        payload=rows,
        specialists=specialists,
        device=config.models.device,
    )

    rankings_by_strategy: dict[str, dict[str, QueryRanking]] = {}
    latencies_by_strategy: dict[str, dict[str, float]] = {}
    for strategy_name in config.fusion.strategies:
        scout, pan = build_strategy(name=strategy_name, routing_mode=config.fusion.routing_mode)
        rankings: dict[str, QueryRanking] = {}
        latencies: dict[str, float] = {}
        for row in rows:
            ranked_ids, latency = rank_query(
                query_text=row["query_text"],
                candidates=row["candidate_documents"],
                village=village,
                scout=scout,
                pan=pan,
            )
            rankings[row["query_id"]] = QueryRanking(doc_ids=tuple(ranked_ids))
            latencies[row["query_id"]] = latency
        rankings_by_strategy[strategy_name] = rankings
        latencies_by_strategy[strategy_name] = latencies

    reports = evaluate_locked(
        rows=rows,
        rankings_by_strategy=rankings_by_strategy,
        latencies_by_strategy=latencies_by_strategy,
    )

    baseline = config.fusion.strategies[0]
    target = config.fusion.strategies[-1]
    delta = {
        metric: reports[target]["global_primary"][metric]["mean"]
        - reports[baseline]["global_primary"][metric]["mean"]
        for metric in reports[baseline]["global_primary"]
    }

    return {
        "name": config.name,
        "experiment_type": config.experiment_type,
        "seed": config.seed.__dict__,
        "dataset": {
            "path": str(config.dataset.path),
            "split": config.dataset.split,
            "max_queries": config.dataset.max_queries,
        },
        "specialists": specialists,
        "results": reports,
        "delta_last_minus_first": delta,
    }


def _run_efficiency(config: BenchmarkExperimentConfig) -> dict[str, Any]:
    from experiments.benchmark_efficiency import BenchmarkConfig, run_benchmarks

    options = config.evaluation.options
    bench_config = BenchmarkConfig(
        sefs_dir=Path(options.get("sefs_dir", "artefacts/sefs")),
        models_dir=Path(options.get("models_dir", "models")),
        registry_json=Path(options["registry_json"]) if options.get("registry_json") else None,
        repo_root=Path(options.get("repo_root", ".")),
        query_text=options.get("query_text", "Test query about battery life and cooking stew"),
        num_repeats=int(options.get("num_repeats", 10)),
        specialist_counts=[int(v) for v in options.get("specialist_counts", [1, 2, 3, 5, 10, 20])],
        track_gpu=bool(options.get("track_gpu", False)),
        track_memory=bool(options.get("track_memory", True)),
        strategies=config.fusion.strategies,
        routing_modes=options.get("routing_modes", [config.fusion.routing_mode]),
        fast_embedder_checkpoint=options.get("fast_embedder_checkpoint"),
        similarity_threshold=options.get("similarity_threshold", 0.7),
        threshold_kwargs=options.get("threshold_kwargs", {}),
        fallback_mode=options.get("fallback_mode", "all"),
    )
    results = run_benchmarks(bench_config)
    return {
        "name": config.name,
        "experiment_type": config.experiment_type,
        "seed": config.seed.__dict__,
        "results": [result.__dict__ for result in results],
    }


def run_experiment(config: BenchmarkExperimentConfig) -> dict[str, Any]:
    """Execute experiment by type."""
    set_global_seed(config.seed.python, config.seed.numpy, config.seed.torch)
    if config.experiment_type == "synthetic_smoke":
        return _run_synthetic(config)
    if config.experiment_type == "real_mixed":
        return _run_real_mixed(config)
    if config.experiment_type == "efficiency":
        return _run_efficiency(config)
    raise ValueError(f"Unsupported experiment_type: {config.experiment_type}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark registry runner")
    parser.add_argument("--config", type=Path, required=True, help="YAML config file")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    summary = run_experiment(config)
    write_json(config.artifacts.summary_json, summary)

    if config.reporting.print_stdout:
        print(summary)


if __name__ == "__main__":
    main()
