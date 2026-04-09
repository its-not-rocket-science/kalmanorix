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
from .report_generator import generate_appendix_markdown
from .statistical_testing import (
    BootstrapConfidenceInterval,
    DomainComparisonReport,
    EffectSizeResult,
    ExperimentLogEntry,
    MetricComparisonReport,
    PairedSignificanceResult,
    StatisticalComparisonReport,
    bootstrap_confidence_interval,
    configuration_hash,
    generate_statistical_report,
    paired_effect_size,
    paired_significance_test,
    render_appendix_table,
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
    "EffectSizeResult",
    "MetricComparisonReport",
    "DomainComparisonReport",
    "ExperimentLogEntry",
    "StatisticalComparisonReport",
    "bootstrap_confidence_interval",
    "paired_significance_test",
    "paired_effect_size",
    "configuration_hash",
    "generate_statistical_report",
    "render_appendix_table",
    "generate_appendix_markdown",
]
