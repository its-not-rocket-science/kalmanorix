"""Statistical testing utilities for paired benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np
from scipy.stats import wilcoxon


@dataclass(frozen=True)
class BootstrapConfidenceInterval:
    """Paired bootstrap confidence interval for mean differences."""

    lower: float
    upper: float
    confidence_level: float
    num_resamples: int
    seed: int


@dataclass(frozen=True)
class PairedSignificanceResult:
    """Result from paired significance testing."""

    statistic: float
    p_value: float
    method: str
    estimable: bool


@dataclass(frozen=True)
class MetricComparisonReport:
    """Statistical summary for one metric comparison."""

    metric: str
    mean_difference: float
    confidence_interval: BootstrapConfidenceInterval
    p_value: float
    effect_size: float
    n_pairs: int
    test_statistic: float


@dataclass(frozen=True)
class StatisticalComparisonReport:
    """Report for a paired benchmark comparison."""

    reference_method: str
    candidate_method: str
    seed: int
    comparisons: Mapping[str, MetricComparisonReport]


def _to_1d_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    return array


def bootstrap_confidence_interval(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    confidence_level: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> BootstrapConfidenceInterval:
    """Compute a paired bootstrap CI for mean(sample_a - sample_b)."""

    if not (0.0 < confidence_level < 1.0):
        raise ValueError("confidence_level must be in (0, 1)")
    if num_resamples <= 0:
        raise ValueError("num_resamples must be positive")

    a = _to_1d_float_array(sample_a, name="sample_a")
    b = _to_1d_float_array(sample_b, name="sample_b")
    if len(a) != len(b):
        raise ValueError("sample_a and sample_b must have the same length")
    if len(a) == 0:
        raise ValueError("samples must not be empty")

    deltas = a - b
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(deltas), size=(num_resamples, len(deltas)))
    means = np.mean(deltas[indices], axis=1)

    alpha = 1.0 - confidence_level
    lower, upper = np.quantile(means, [alpha / 2.0, 1.0 - alpha / 2.0])
    return BootstrapConfidenceInterval(
        lower=float(lower),
        upper=float(upper),
        confidence_level=float(confidence_level),
        num_resamples=int(num_resamples),
        seed=int(seed),
    )


def paired_significance_test(
    sample_a: Sequence[float],
    sample_b: Sequence[float],
    *,
    method: str = "wilcoxon",
) -> PairedSignificanceResult:
    """Run paired significance testing on query-level values."""

    if method != "wilcoxon":
        raise ValueError("only 'wilcoxon' is currently supported")

    a = _to_1d_float_array(sample_a, name="sample_a")
    b = _to_1d_float_array(sample_b, name="sample_b")
    if len(a) != len(b):
        raise ValueError("sample_a and sample_b must have the same length")
    if len(a) == 0:
        raise ValueError("samples must not be empty")

    deltas = a - b
    if np.allclose(deltas, 0.0):
        return PairedSignificanceResult(statistic=0.0, p_value=1.0, method=method, estimable=False)

    result = wilcoxon(a, b, alternative="two-sided", zero_method="wilcox", method="auto")
    return PairedSignificanceResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method=method,
        estimable=True,
    )


def paired_effect_size(sample_a: Sequence[float], sample_b: Sequence[float]) -> float:
    """Compute Cohen's dz for paired samples."""

    a = _to_1d_float_array(sample_a, name="sample_a")
    b = _to_1d_float_array(sample_b, name="sample_b")
    if len(a) != len(b):
        raise ValueError("sample_a and sample_b must have the same length")
    if len(a) == 0:
        raise ValueError("samples must not be empty")

    deltas = a - b
    std = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
    if std <= 0.0:
        return 0.0
    return float(np.mean(deltas) / std)


def generate_statistical_report(
    *,
    reference_method: str,
    candidate_method: str,
    reference_metrics: Mapping[str, Sequence[float]],
    candidate_metrics: Mapping[str, Sequence[float]],
    confidence_level: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
) -> StatisticalComparisonReport:
    """Generate a deterministic statistical report across metrics."""

    metrics = sorted(set(reference_metrics).intersection(candidate_metrics))
    if not metrics:
        raise ValueError("reference_metrics and candidate_metrics must share at least one metric")

    comparisons: dict[str, MetricComparisonReport] = {}
    for offset, metric in enumerate(metrics):
        ref = reference_metrics[metric]
        cand = candidate_metrics[metric]
        ref_array = _to_1d_float_array(ref, name=f"reference_metrics[{metric}]")
        cand_array = _to_1d_float_array(cand, name=f"candidate_metrics[{metric}]")
        if len(ref_array) != len(cand_array):
            raise ValueError(f"metric '{metric}' has mismatched paired sample lengths")

        ci = bootstrap_confidence_interval(
            ref_array,
            cand_array,
            confidence_level=confidence_level,
            num_resamples=num_resamples,
            seed=seed + offset,
        )
        significance = paired_significance_test(ref_array, cand_array)
        effect_size = paired_effect_size(ref_array, cand_array)
        mean_difference = float(np.mean(ref_array - cand_array))

        comparisons[metric] = MetricComparisonReport(
            metric=metric,
            mean_difference=mean_difference,
            confidence_interval=ci,
            p_value=significance.p_value,
            effect_size=effect_size,
            n_pairs=int(len(ref_array)),
            test_statistic=significance.statistic,
        )

    return StatisticalComparisonReport(
        reference_method=reference_method,
        candidate_method=candidate_method,
        seed=seed,
        comparisons=MappingProxyType(comparisons),
    )
