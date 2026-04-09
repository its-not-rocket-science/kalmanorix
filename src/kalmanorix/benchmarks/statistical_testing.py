"""Auditable statistical testing utilities for paired benchmark comparisons.

Design goals:
- deterministic execution for reproducibility
- paired, non-parametric inference at query level
- multiple-comparison correction over a pre-declared hypothesis family
- auditable experiment logs with explicit seeds and configuration hash
"""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from math import sqrt
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy.stats import rankdata, wilcoxon


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
    adjusted_p_value: float | None = None


@dataclass(frozen=True)
class EffectSizeResult:
    """Effect sizes for paired comparisons."""

    cohen_dz: float
    rank_biserial: float


@dataclass(frozen=True)
class MetricComparisonReport:
    """Statistical summary for one metric comparison."""

    metric: str
    mean_difference: float
    confidence_interval: BootstrapConfidenceInterval
    p_value: float
    adjusted_p_value: float
    effect_size: EffectSizeResult
    n_pairs: int
    test_statistic: float


@dataclass(frozen=True)
class DomainComparisonReport:
    """Per-domain metric comparison report."""

    domain: str
    metrics: Mapping[str, MetricComparisonReport]


@dataclass(frozen=True)
class ExperimentLogEntry:
    """Immutable audit trail entry for one hypothesis test."""

    reference_method: str
    candidate_method: str
    domain: str
    metric: str
    bootstrap_seed: int
    configuration_hash: str


@dataclass(frozen=True)
class StatisticalComparisonReport:
    """Auditable report for paired benchmark comparisons."""

    reference_method: str
    candidate_method: str
    seed: int
    correction_method: str
    configuration_hash: str
    domains: Mapping[str, DomainComparisonReport]
    comparisons: Mapping[str, MetricComparisonReport]
    experiment_log: tuple[ExperimentLogEntry, ...]


def _to_1d_float_array(values: Sequence[float], *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a 1D sequence")
    return array


def configuration_hash(config: Mapping[str, Any]) -> str:
    """Return a stable SHA-256 hash for experiment configuration dictionaries."""

    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return sha256(canonical.encode("utf-8")).hexdigest()


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
        return PairedSignificanceResult(
            statistic=0.0,
            p_value=1.0,
            method=method,
            estimable=False,
            adjusted_p_value=1.0,
        )

    result = wilcoxon(
        a, b, alternative="two-sided", zero_method="wilcox", method="auto"
    )
    return PairedSignificanceResult(
        statistic=float(result.statistic),
        p_value=float(result.pvalue),
        method=method,
        estimable=True,
    )


def paired_effect_size(
    sample_a: Sequence[float], sample_b: Sequence[float]
) -> EffectSizeResult:
    """Compute paired effect sizes (Cohen's dz and rank-biserial correlation)."""

    a = _to_1d_float_array(sample_a, name="sample_a")
    b = _to_1d_float_array(sample_b, name="sample_b")
    if len(a) != len(b):
        raise ValueError("sample_a and sample_b must have the same length")
    if len(a) == 0:
        raise ValueError("samples must not be empty")

    deltas = a - b
    std = np.std(deltas, ddof=1) if len(deltas) > 1 else 0.0
    dz = 0.0 if std <= 0.0 else float(np.mean(deltas) / std)

    non_zero = deltas[np.abs(deltas) > 0.0]
    if len(non_zero) == 0:
        return EffectSizeResult(cohen_dz=dz, rank_biserial=0.0)

    ranks = rankdata(np.abs(non_zero), method="average")
    w_plus = float(np.sum(ranks[non_zero > 0.0]))
    w_minus = float(np.sum(ranks[non_zero < 0.0]))
    denom = w_plus + w_minus
    rank_biserial = 0.0 if denom <= 0.0 else (w_plus - w_minus) / denom
    return EffectSizeResult(cohen_dz=dz, rank_biserial=float(rank_biserial))


def _holm_bonferroni_adjust(p_values: Sequence[float]) -> np.ndarray:
    p_array = np.asarray(p_values, dtype=float)
    m = len(p_array)
    if m == 0:
        return p_array

    order = np.argsort(p_array)
    adjusted_sorted = np.empty(m, dtype=float)
    for i, idx in enumerate(order):
        adjusted_sorted[i] = (m - i) * p_array[idx]
    adjusted_sorted = np.maximum.accumulate(adjusted_sorted)
    adjusted_sorted = np.clip(adjusted_sorted, 0.0, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def generate_statistical_report(
    *,
    reference_method: str,
    candidate_method: str,
    reference_metrics: Mapping[str, Sequence[float]],
    candidate_metrics: Mapping[str, Sequence[float]],
    reference_metrics_by_domain: Mapping[str, Mapping[str, Sequence[float]]]
    | None = None,
    candidate_metrics_by_domain: Mapping[str, Mapping[str, Sequence[float]]]
    | None = None,
    confidence_level: float = 0.95,
    num_resamples: int = 10_000,
    seed: int = 0,
    correction_method: str = "holm",
    config: Mapping[str, Any] | None = None,
) -> StatisticalComparisonReport:
    """Generate an auditable statistical report.

    The correction is applied once across all reported hypotheses (all domains + overall,
    all shared metrics), preventing post-hoc selective testing.
    """

    if correction_method != "holm":
        raise ValueError("only 'holm' correction is currently supported")

    metrics = sorted(set(reference_metrics).intersection(candidate_metrics))
    if not metrics:
        raise ValueError(
            "reference_metrics and candidate_metrics must share at least one metric"
        )

    effective_config: dict[str, Any] = {
        "confidence_level": confidence_level,
        "num_resamples": num_resamples,
        "seed": seed,
        "correction_method": correction_method,
        "metrics": metrics,
    }
    if config is not None:
        effective_config["user_config"] = dict(config)
    cfg_hash = configuration_hash(effective_config)

    domain_payloads: dict[
        str, tuple[Mapping[str, Sequence[float]], Mapping[str, Sequence[float]]]
    ] = {}
    if (
        reference_metrics_by_domain is not None
        or candidate_metrics_by_domain is not None
    ):
        if reference_metrics_by_domain is None or candidate_metrics_by_domain is None:
            raise ValueError(
                "both reference_metrics_by_domain and candidate_metrics_by_domain must be provided"
            )
        shared_domains = sorted(
            set(reference_metrics_by_domain).intersection(candidate_metrics_by_domain)
        )
        if not shared_domains:
            raise ValueError("domain maps must share at least one domain")
        for domain in shared_domains:
            domain_payloads[domain] = (
                reference_metrics_by_domain[domain],
                candidate_metrics_by_domain[domain],
            )

    hypotheses: list[tuple[str, str]] = []
    raw_results: dict[tuple[str, str], tuple[MetricComparisonReport, int]] = {}
    experiment_log: list[ExperimentLogEntry] = []
    seed_counter = 0

    analysis_domains = ["overall", *sorted(domain_payloads)]
    for domain in analysis_domains:
        if domain == "overall":
            ref_metrics = reference_metrics
            cand_metrics = candidate_metrics
        else:
            ref_metrics, cand_metrics = domain_payloads[domain]

        shared = sorted(
            set(ref_metrics).intersection(cand_metrics).intersection(metrics)
        )
        for metric in shared:
            ref_array = _to_1d_float_array(
                ref_metrics[metric], name=f"reference[{domain}][{metric}]"
            )
            cand_array = _to_1d_float_array(
                cand_metrics[metric], name=f"candidate[{domain}][{metric}]"
            )
            if len(ref_array) != len(cand_array):
                raise ValueError(
                    f"domain '{domain}' metric '{metric}' has mismatched paired sample lengths"
                )
            if len(ref_array) == 0:
                raise ValueError(f"domain '{domain}' metric '{metric}' has no samples")

            bootstrap_seed = seed + seed_counter
            seed_counter += 1

            ci = bootstrap_confidence_interval(
                ref_array,
                cand_array,
                confidence_level=confidence_level,
                num_resamples=num_resamples,
                seed=bootstrap_seed,
            )
            significance = paired_significance_test(ref_array, cand_array)
            effect_size = paired_effect_size(ref_array, cand_array)
            mean_difference = float(np.mean(ref_array - cand_array))

            comparison = MetricComparisonReport(
                metric=metric,
                mean_difference=mean_difference,
                confidence_interval=ci,
                p_value=significance.p_value,
                adjusted_p_value=significance.p_value,
                effect_size=effect_size,
                n_pairs=int(len(ref_array)),
                test_statistic=significance.statistic,
            )
            key = (domain, metric)
            hypotheses.append(key)
            raw_results[key] = (comparison, bootstrap_seed)
            experiment_log.append(
                ExperimentLogEntry(
                    reference_method=reference_method,
                    candidate_method=candidate_method,
                    domain=domain,
                    metric=metric,
                    bootstrap_seed=bootstrap_seed,
                    configuration_hash=cfg_hash,
                )
            )

    adjusted = _holm_bonferroni_adjust([raw_results[h][0].p_value for h in hypotheses])

    comparisons: dict[str, MetricComparisonReport] = {}
    domain_reports: dict[str, DomainComparisonReport] = {}
    by_domain_metrics: dict[str, dict[str, MetricComparisonReport]] = {}

    for idx, key in enumerate(hypotheses):
        domain, metric = key
        base, _ = raw_results[key]
        updated = MetricComparisonReport(
            metric=base.metric,
            mean_difference=base.mean_difference,
            confidence_interval=base.confidence_interval,
            p_value=base.p_value,
            adjusted_p_value=float(adjusted[idx]),
            effect_size=base.effect_size,
            n_pairs=base.n_pairs,
            test_statistic=base.test_statistic,
        )
        if domain == "overall":
            comparisons[metric] = updated
        by_domain_metrics.setdefault(domain, {})[metric] = updated

    for domain, metrics_payload in by_domain_metrics.items():
        domain_reports[domain] = DomainComparisonReport(
            domain=domain,
            metrics=MappingProxyType(dict(sorted(metrics_payload.items()))),
        )

    return StatisticalComparisonReport(
        reference_method=reference_method,
        candidate_method=candidate_method,
        seed=seed,
        correction_method=correction_method,
        configuration_hash=cfg_hash,
        domains=MappingProxyType(dict(sorted(domain_reports.items()))),
        comparisons=MappingProxyType(comparisons),
        experiment_log=tuple(experiment_log),
    )


def render_appendix_table(report: StatisticalComparisonReport) -> str:
    """Render a markdown table suitable for a paper appendix."""

    header = (
        "| Domain | Metric | Δ Mean | 95% CI | p | p_adj (Holm) | "
        "Cohen's dz | Rank-biserial | n |"
    )
    sep = "|---|---:|---:|---|---:|---:|---:|---:|---:|"
    lines = [header, sep]

    for domain, domain_report in sorted(report.domains.items()):
        for metric, result in sorted(domain_report.metrics.items()):
            ci = result.confidence_interval
            lines.append(
                "| {domain} | {metric} | {mean:.4f} | [{low:.4f}, {high:.4f}] | {p:.4g} | {padj:.4g} | {dz:.4f} | {rb:.4f} | {n} |".format(
                    domain=domain,
                    metric=metric,
                    mean=result.mean_difference,
                    low=ci.lower,
                    high=ci.upper,
                    p=result.p_value,
                    padj=result.adjusted_p_value,
                    dz=result.effect_size.cohen_dz,
                    rb=result.effect_size.rank_biserial,
                    n=result.n_pairs,
                )
            )
    lines.append("")
    lines.append(f"Configuration hash: `{report.configuration_hash}`")
    lines.append(
        f"Seed root: `{report.seed}`; correction: `{report.correction_method}`; hypotheses: `{len(report.experiment_log)}`"
    )
    return "\n".join(lines)
