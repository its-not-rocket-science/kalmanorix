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
    MetricDefinition,
    PRIMARY_METRICS,
    QueryRanking,
    SECONDARY_METRICS,
    evaluate_locked_protocol,
)
from .statistical_testing import (
    BootstrapConfidenceInterval,
    MetricComparisonReport,
    PairedSignificanceResult,
    StatisticalComparisonReport,
    bootstrap_confidence_interval,
    generate_statistical_report,
    paired_effect_size,
    paired_significance_test,
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
    "MetricDefinition",
    "PRIMARY_METRICS",
    "SECONDARY_METRICS",
    "EvaluationReport",
    "evaluate_locked_protocol",
    "BootstrapConfidenceInterval",
    "PairedSignificanceResult",
    "MetricComparisonReport",
    "StatisticalComparisonReport",
    "bootstrap_confidence_interval",
    "paired_significance_test",
    "paired_effect_size",
    "generate_statistical_report",
]
