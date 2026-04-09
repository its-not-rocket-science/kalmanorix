"""Tests for benchmark statistical testing utilities."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import wilcoxon

from kalmanorix.benchmarks.statistical_testing import (
    bootstrap_confidence_interval,
    generate_statistical_report,
    paired_effect_size,
    paired_significance_test,
    render_appendix_table,
)


def test_bootstrap_confidence_interval_is_reproducible() -> None:
    a = np.array([0.7, 0.6, 0.5, 0.8, 0.65], dtype=float)
    b = np.array([0.6, 0.55, 0.4, 0.7, 0.5], dtype=float)

    ci_1 = bootstrap_confidence_interval(a, b, num_resamples=2_000, seed=123)
    ci_2 = bootstrap_confidence_interval(a, b, num_resamples=2_000, seed=123)

    assert ci_1.lower == pytest.approx(ci_2.lower)
    assert ci_1.upper == pytest.approx(ci_2.upper)
    observed = float(np.mean(a - b))
    assert ci_1.lower <= observed <= ci_1.upper


def test_paired_significance_matches_scipy_and_handles_degenerate_case() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(loc=0.0, scale=1.0, size=50)
    improved = baseline + 0.4

    significant = paired_significance_test(improved, baseline)
    expected = wilcoxon(
        improved, baseline, alternative="two-sided", zero_method="wilcox", method="auto"
    )
    assert significant.estimable is True
    assert significant.statistic == pytest.approx(float(expected.statistic))
    assert significant.p_value == pytest.approx(float(expected.pvalue))

    degenerate = paired_significance_test(baseline, baseline)
    assert degenerate.estimable is False
    assert degenerate.p_value == pytest.approx(1.0)


def test_report_generator_outputs_per_domain_overall_and_holm_correction() -> None:
    reference_metrics = {
        "ndcg@10": [0.42, 0.35, 0.51, 0.49],
        "mrr": [0.56, 0.44, 0.67, 0.62],
    }
    candidate_metrics = {
        "ndcg@10": [0.38, 0.30, 0.47, 0.45],
        "mrr": [0.51, 0.40, 0.63, 0.59],
    }
    report = generate_statistical_report(
        reference_method="kalman",
        candidate_method="mean_fuser",
        reference_metrics=reference_metrics,
        candidate_metrics=candidate_metrics,
        reference_metrics_by_domain={
            "finance": {"ndcg@10": [0.5, 0.45], "mrr": [0.62, 0.59]},
            "biomed": {"ndcg@10": [0.34, 0.28], "mrr": [0.50, 0.44]},
        },
        candidate_metrics_by_domain={
            "finance": {"ndcg@10": [0.46, 0.40], "mrr": [0.56, 0.55]},
            "biomed": {"ndcg@10": [0.29, 0.25], "mrr": [0.45, 0.40]},
        },
        num_resamples=2_000,
        seed=11,
        config={"benchmark": "mixed-domain-v1"},
    )

    ndcg_overall = report.comparisons["ndcg@10"]
    assert ndcg_overall.mean_difference == pytest.approx(
        np.mean(
            np.array(reference_metrics["ndcg@10"])
            - np.array(candidate_metrics["ndcg@10"])
        )
    )
    assert 0.0 <= ndcg_overall.p_value <= 1.0
    assert 0.0 <= ndcg_overall.adjusted_p_value <= 1.0

    assert "overall" in report.domains
    assert "finance" in report.domains
    assert "biomed" in report.domains
    # 3 domains x 2 metrics = 6 hypotheses in one Holm family.
    assert len(report.experiment_log) == 6
    assert len({entry.bootstrap_seed for entry in report.experiment_log}) == 6
    assert all(
        entry.configuration_hash == report.configuration_hash
        for entry in report.experiment_log
    )


def test_effect_size_has_expected_sign() -> None:
    reference = [0.6, 0.7, 0.55, 0.61]
    candidate = [0.4, 0.5, 0.45, 0.5]
    effect = paired_effect_size(reference, candidate)
    assert effect.cohen_dz > 0.0
    assert effect.rank_biserial > 0.0


def test_appendix_table_contains_required_columns() -> None:
    report = generate_statistical_report(
        reference_method="kalman",
        candidate_method="mean_fuser",
        reference_metrics={"mrr": [0.6, 0.5, 0.7]},
        candidate_metrics={"mrr": [0.4, 0.45, 0.5]},
        num_resamples=500,
        seed=3,
    )
    table = render_appendix_table(report)
    assert "| Domain | Metric | Δ Mean |" in table
    assert "p_adj (Holm)" in table
    assert "Configuration hash" in table


def test_holm_adjusted_p_values_are_not_smaller_than_raw_p_values() -> None:
    report = generate_statistical_report(
        reference_method="kalman",
        candidate_method="mean",
        reference_metrics={
            "ndcg@10": [0.6, 0.6, 0.7, 0.55, 0.52],
            "recall@10": [1.0, 1.0, 0.8, 0.8, 0.6],
            "mrr@10": [0.7, 0.8, 0.6, 0.5, 0.5],
        },
        candidate_metrics={
            "ndcg@10": [0.5, 0.55, 0.6, 0.5, 0.5],
            "recall@10": [0.8, 0.8, 0.8, 0.6, 0.6],
            "mrr@10": [0.65, 0.7, 0.55, 0.5, 0.45],
        },
        num_resamples=600,
        seed=19,
    )
    for metric in ["ndcg@10", "recall@10", "mrr@10"]:
        assert (
            report.comparisons[metric].adjusted_p_value
            >= report.comparisons[metric].p_value
        )


def test_input_validation_errors() -> None:
    with pytest.raises(ValueError, match="same length"):
        bootstrap_confidence_interval([1.0], [1.0, 2.0])

    with pytest.raises(ValueError, match="share at least one metric"):
        generate_statistical_report(
            reference_method="a",
            candidate_method="b",
            reference_metrics={"mrr": [0.1]},
            candidate_metrics={"ndcg@10": [0.2]},
        )
