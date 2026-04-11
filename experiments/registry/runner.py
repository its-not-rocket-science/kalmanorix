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

from experiments.registry.config_schema import (
    BenchmarkExperimentConfig,
    load_experiment_config,
)
from experiments.registry.datasets import load_dataset
from experiments.registry.evaluation import evaluate_locked, evaluate_synthetic_recall
from experiments.registry.fusion import (
    RetrievalFusionStrategy,
    build_retrieval_baselines,
    build_strategy,
    rank_query,
    rank_query_with_baseline,
)
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
    metrics = evaluate_synthetic_recall(
        corpus=corpus, village=village, strategies=strategies
    )
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
    confidence_by_strategy: dict[str, dict[str, float]] = {}
    specialist_count_by_strategy: dict[str, dict[str, float]] = {}
    policy_usage_by_strategy: dict[str, dict[str, Any]] = {}
    query_metadata: dict[str, dict[str, Any]] = {}

    def _confidence_from_selected(
        query_text: str, selected_modules: list[Any]
    ) -> float:
        if not selected_modules:
            return 0.0
        inv_variances = np.asarray(
            [
                1.0 / (1.0 + float(module.sigma2_for(query_text)))
                for module in selected_modules
            ],
            dtype=float,
        )
        return float(np.mean(inv_variances))

    def _confidence_from_baseline(
        *,
        query_text: str,
        strategy: RetrievalFusionStrategy,
        village_obj: Any,
    ) -> tuple[float, float]:
        modules = village_obj.modules
        weights = strategy.weights_for_query(query_text=query_text, modules=modules)
        if len(weights) != len(modules):
            weights = np.full(len(modules), 1.0 / len(modules), dtype=float)
        weights = np.clip(weights, 0.0, None)
        total = float(np.sum(weights))
        if total <= 0.0:
            weights = np.full(len(modules), 1.0 / len(modules), dtype=float)
        else:
            weights = weights / total
        inv_var = np.asarray(
            [1.0 / (1.0 + float(module.sigma2_for(query_text))) for module in modules],
            dtype=float,
        )
        return float(np.sum(weights * inv_var)), float(np.count_nonzero(weights > 0.0))

    for strategy_name in config.fusion.strategies:
        scout, pan = build_strategy(
            name=strategy_name, routing_mode=config.fusion.routing_mode
        )
        rankings: dict[str, QueryRanking] = {}
        latencies: dict[str, float] = {}
        confidences: dict[str, float] = {}
        specialist_counts: dict[str, float] = {}
        for row in rows:
            selected_modules = scout.select(query=row["query_text"], village=village)
            ranked_ids, latency = rank_query(
                query_text=row["query_text"],
                candidates=row["candidate_documents"],
                village=village,
                scout=scout,
                pan=pan,
            )
            rankings[row["query_id"]] = QueryRanking(doc_ids=tuple(ranked_ids))
            latencies[row["query_id"]] = latency
            confidences[row["query_id"]] = _confidence_from_selected(
                query_text=row["query_text"],
                selected_modules=selected_modules,
            )
            specialist_counts[row["query_id"]] = float(len(selected_modules))
        rankings_by_strategy[strategy_name] = rankings
        latencies_by_strategy[strategy_name] = latencies
        confidence_by_strategy[strategy_name] = confidences
        specialist_count_by_strategy[strategy_name] = specialist_counts

    for row in rows:
        query_text = row["query_text"]
        sigma2_by_specialist = {
            module.name: float(module.sigma2_for(query_text)) for module in village.modules
        }
        precision_by_specialist = {
            name: float(1.0 / max(value, 1e-8))
            for name, value in sigma2_by_specialist.items()
        }
        precision_values = np.asarray(list(precision_by_specialist.values()), dtype=float)
        sorted_precision = np.sort(precision_values)[::-1]
        top_precision = float(sorted_precision[0]) if len(sorted_precision) else 0.0
        second_precision = float(sorted_precision[1]) if len(sorted_precision) > 1 else 0.0
        precision_sum = float(np.sum(precision_values))
        if precision_sum > 0.0:
            precision_probs = precision_values / precision_sum
            entropy = -np.sum(precision_probs * np.log(precision_probs + 1e-12))
            max_entropy = np.log(max(len(precision_values), 1))
            disagreement = float(entropy / max(max_entropy, 1e-8))
        else:
            disagreement = 0.0

        secondary_domain = row.get("secondary_domain")
        is_multi_domain = bool(
            secondary_domain
            and str(secondary_domain).strip()
            and str(secondary_domain) != str(row.get("dominant_domain", ""))
        )
        query_metadata[row["query_id"]] = {
            "dominant_domain": row.get("dominant_domain", row.get("domain_label")),
            "secondary_domain": secondary_domain,
            "query_category": row.get("query_category"),
            "ambiguity_category": row.get("ambiguity_category"),
            "ambiguity_score": row.get("ambiguity_score"),
            "fusion_usefulness_bucket": row.get("fusion_usefulness_bucket"),
            "is_multi_domain": is_multi_domain,
            "sigma2_by_specialist": sigma2_by_specialist,
            "precision_by_specialist": precision_by_specialist,
            "uncertainty_spread": float(
                max(sigma2_by_specialist.values()) - min(sigma2_by_specialist.values())
            )
            if sigma2_by_specialist
            else 0.0,
            "router_top1_precision": top_precision,
            "router_top2_precision": second_precision,
            "router_confidence": float(top_precision / max(precision_sum, 1e-8)),
            "router_margin": float(
                (top_precision - second_precision) / max(top_precision, 1e-8)
            ),
            "specialist_disagreement": disagreement,
        }

    baseline_strategies = build_retrieval_baselines(config.fusion.options)
    for strategy in baseline_strategies:
        strategy.fit(rows=rows, village=village, options=config.fusion.options)
        rankings = {}
        latencies = {}
        confidences = {}
        specialist_counts = {}
        policy_usage: dict[str, Any] = {}
        for row in rows:
            ranked_ids, latency = rank_query_with_baseline(
                query_text=row["query_text"],
                candidates=row["candidate_documents"],
                village=village,
                strategy=strategy,
            )
            rankings[row["query_id"]] = QueryRanking(doc_ids=tuple(ranked_ids))
            latencies[row["query_id"]] = latency
            confidence, count = _confidence_from_baseline(
                query_text=row["query_text"],
                strategy=strategy,
                village_obj=village,
            )
            confidences[row["query_id"]] = confidence
            specialist_counts[row["query_id"]] = count
            diagnostics = strategy.diagnostics_for_query(
                query_text=row["query_text"], modules=village.modules
            )
            if diagnostics is not None:
                policy_usage[row["query_id"]] = diagnostics
        rankings_by_strategy[strategy.name] = rankings
        latencies_by_strategy[strategy.name] = latencies
        confidence_by_strategy[strategy.name] = confidences
        specialist_count_by_strategy[strategy.name] = specialist_counts
        if policy_usage:
            policy_usage_by_strategy[strategy.name] = policy_usage

    reports = evaluate_locked(
        rows=rows,
        rankings_by_strategy=rankings_by_strategy,
        latencies_by_strategy=latencies_by_strategy,
        specialist_count_by_strategy=specialist_count_by_strategy,
    )

    primary_metric = "mrr"
    kalman_key = "kalman"
    ranked_methods = sorted(
        reports.items(),
        key=lambda item: item[1]["global_primary"][primary_metric]["mean"],
        reverse=True,
    )

    kalman_score = (
        reports[kalman_key]["global_primary"][primary_metric]["mean"]
        if kalman_key in reports
        else None
    )
    best_baseline = None
    best_baseline_score = None
    for name, rep in ranked_methods:
        if name == kalman_key:
            continue
        best_baseline = name
        best_baseline_score = rep["global_primary"][primary_metric]["mean"]
        break

    baseline = config.fusion.strategies[0]
    target = config.fusion.strategies[-1]
    delta = {
        metric: reports[target]["global_primary"][metric]["mean"]
        - reports[baseline]["global_primary"][metric]["mean"]
        for metric in reports[baseline]["global_primary"]
    }

    comparison_table = [
        {
            "strategy": name,
            "mrr_mean": rep["global_primary"]["mrr"]["mean"],
            "recall@1_mean": rep["global_primary"]["recall@1"]["mean"],
            "recall@5_mean": rep["global_primary"]["recall@5"]["mean"],
            "delta_vs_kalman_mrr": (
                None
                if kalman_score is None
                else rep["global_primary"]["mrr"]["mean"] - kalman_score
            ),
        }
        for name, rep in ranked_methods
    ]

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
        "query_level": {
            "domains": {row["query_id"]: row["domain_label"] for row in rows},
            "ground_truth": {
                row["query_id"]: list(row["ground_truth_relevant_ids"]) for row in rows
            },
            "rankings": {
                strategy_name: {
                    query_id: list(ranking.doc_ids)
                    for query_id, ranking in rankings.items()
                }
                for strategy_name, rankings in rankings_by_strategy.items()
            },
            "latency_ms": latencies_by_strategy,
            "confidence_proxy": confidence_by_strategy,
            "specialist_count_selected": specialist_count_by_strategy,
            "policy_usage": policy_usage_by_strategy,
            "query_metadata": query_metadata,
        },
        "comparison_table": comparison_table,
        "kalman_guardrail": {
            "primary_metric": primary_metric,
            "kalman_score": kalman_score,
            "best_non_kalman_strategy": best_baseline,
            "best_non_kalman_score": best_baseline_score,
            "kalman_minus_best_non_kalman": (
                None
                if kalman_score is None or best_baseline_score is None
                else kalman_score - best_baseline_score
            ),
        },
        "delta_last_minus_first": delta,
    }


def _run_efficiency(config: BenchmarkExperimentConfig) -> dict[str, Any]:
    from experiments.benchmark_efficiency import BenchmarkConfig, run_benchmarks

    options = config.evaluation.options
    bench_config = BenchmarkConfig(
        sefs_dir=Path(options.get("sefs_dir", "artefacts/sefs")),
        models_dir=Path(options.get("models_dir", "models")),
        registry_json=Path(options["registry_json"])
        if options.get("registry_json")
        else None,
        repo_root=Path(options.get("repo_root", ".")),
        query_text=options.get(
            "query_text", "Test query about battery life and cooking stew"
        ),
        num_repeats=int(options.get("num_repeats", 10)),
        specialist_counts=[
            int(v) for v in options.get("specialist_counts", [1, 2, 3, 5, 10, 20])
        ],
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
    details = summary
    if config.artifacts.details_json is not None:
        write_json(config.artifacts.details_json, details)

    lightweight_summary = dict(summary)
    if "query_level" in lightweight_summary:
        lightweight_summary["query_level"] = {
            "details_json": str(config.artifacts.details_json)
            if config.artifacts.details_json is not None
            else None
        }
    write_json(config.artifacts.summary_json, lightweight_summary)

    if config.reporting.print_stdout:
        print(summary)


if __name__ == "__main__":
    main()
