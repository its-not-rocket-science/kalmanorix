"""Benchmark building utilities."""

from .fusion_baselines import (
    EmbeddingDataset,
    EvaluationResult,
    FusionStrategy,
    build_strategies,
    evaluate_strategy,
    run_experiment,
)
from .mixed_domain import build_mixed_domain_benchmark, main
from .evaluation_protocol import (
    EvaluationReport,
    QueryRanking,
    evaluate_locked_protocol,
)

__all__ = [
    "EmbeddingDataset",
    "EvaluationResult",
    "FusionStrategy",
    "build_strategies",
    "evaluate_strategy",
    "run_experiment",
    "build_mixed_domain_benchmark",
    "main",
    "QueryRanking",
    "EvaluationReport",
    "evaluate_locked_protocol",
]
