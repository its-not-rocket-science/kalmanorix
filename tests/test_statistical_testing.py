"""Tests for benchmark statistical testing utilities."""

from __future__ import annotations

import numpy as np
import pytest

from kalmanorix.benchmarks.statistical_testing import (
    bootstrap_confidence_interval,
    generate_statistical_report,
    paired_effect_size,
    paired_significance_test,
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


def test_paired_significance_detects_shift_and_handles_degenerate_case() -> None:
    rng = np.random.default_rng(7)
    baseline = rng.normal(loc=0.0, scale=1.0, size=50)
    improved = baseline + 0.4

    significant = paired_significance_test(improved, baseline)
    assert significant.estimable is True
    assert significant.p_value < 0.05

    degenerate = paired_significance_test(baseline, baseline)
    assert degenerate.estimable is False
    assert degenerate.p_value == pytest.approx(1.0)


def test_report_generator_outputs_required_statistics() -> None:
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
        num_resamples=2_000,
        seed=11,
    )

    ndcg = report.comparisons["ndcg@10"]
    assert ndcg.mean_difference == pytest.approx(np.mean(np.array(reference_metrics["ndcg@10"]) - np.array(candidate_metrics["ndcg@10"])))
    assert 0.0 <= ndcg.p_value <= 1.0
    expected_seed = 11 + sorted(set(reference_metrics).intersection(candidate_metrics)).index("ndcg@10")
    assert ndcg.confidence_interval.seed == expected_seed
    assert ndcg.effect_size == pytest.approx(
        paired_effect_size(reference_metrics["ndcg@10"], candidate_metrics["ndcg@10"])
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
