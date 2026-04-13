from __future__ import annotations

import json

import numpy as np

from kalmanorix.benchmarks.uncertainty_calibration import (
    CALIBRATION_OBJECTIVES,
    ValidationPowerConfig,
    run_uncertainty_calibration,
    run_uncertainty_calibration_objective_study,
)
from kalmanorix.uncertainty_calibration import fit_scalar_calibrator


def test_no_train_validation_leakage(tmp_path) -> None:
    summary = run_uncertainty_calibration(tmp_path)
    for row in summary["leakage_checks"]:
        assert row["intersection"] == []


def test_calibrator_serialization_is_deterministic(tmp_path) -> None:
    x = np.linspace(0.1, 2.0, 12)
    y = x * 0.5 + 0.1
    fit = fit_scalar_calibrator(x, y, method="affine")

    p1 = tmp_path / "a.json"
    p2 = tmp_path / "b.json"
    fit.calibrator.to_json(p1)
    fit.calibrator.to_json(p2)

    assert p1.read_text(encoding="utf-8") == p2.read_text(encoding="utf-8")
    payload = json.loads(p1.read_text(encoding="utf-8"))
    assert payload["name"] == "affine"


def test_monotonic_calibrators_are_monotone_non_decreasing() -> None:
    x = np.linspace(0.1, 3.0, 40)
    y = np.sqrt(x)
    for method in ("temperature", "isotonic", "piecewise_monotonic"):
        fit = fit_scalar_calibrator(x, y, method=method)
        preds = fit.calibrator.transform(x)
        diffs = np.diff(preds)
        assert np.all(diffs >= -1e-9)


def test_fallback_for_too_small_calibration_data() -> None:
    x = np.array([0.2, 0.4, 0.8], dtype=np.float64)
    y = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    fit = fit_scalar_calibrator(x, y, method="isotonic", min_samples=8)
    assert fit.used_fallback is True
    assert fit.calibrator.name == "identity"


def test_validation_powered_split_and_status(tmp_path) -> None:
    summary = run_uncertainty_calibration(tmp_path)
    assert summary["status"] == "sufficient"
    assert summary["powered_for_calibration"] is True
    assert summary["validation_power"]["validation_count"] >= 8
    assert summary["validation_power"]["specialist_effective_support"]["tech"] >= 6
    assert summary["validation_power"]["specialist_effective_support"]["cook"] >= 6
    assert "validation_by_query_bucket" in summary["validation_power"]
    assert summary["minimum_support_threshold"] == 6
    assert summary["per_specialist_support_counts"]["tech"] >= 6


def test_underpowered_validation_emits_explicit_status(tmp_path) -> None:
    summary = run_uncertainty_calibration(
        tmp_path,
        power_config=ValidationPowerConfig(
            min_validation_total=500,
            min_validation_per_domain=220,
            min_effective_support_per_specialist=260,
            min_validation_per_query_bucket=260,
            calibrator_min_samples=260,
        ),
    )
    assert summary["status"] == "underpowered_validation"
    assert summary["powered_for_calibration"] is False
    assert summary["validation_power"]["failures"]
    assert summary["selected_calibrators"]["tech"]["fallback"] is True
    assert summary["selected_calibrators"]["tech"]["sufficiently_powered"] is False
    assert summary["fallback_reason"] == "underpowered_validation"


def test_objective_study_covers_required_objectives_and_uses_validation_selection(tmp_path) -> None:
    study = run_uncertainty_calibration_objective_study(tmp_path)
    assert study["selection_is_validation_only"] is True
    assert set(CALIBRATION_OBJECTIVES).issubset(study["objective_reports"])
    assert study["selected_objective"] in CALIBRATION_OBJECTIVES
    assert study["selected_objective"] in study["validation_transfer_scores"]


def test_report_contains_bucket_outcomes_and_validation_test_deltas(tmp_path) -> None:
    study = run_uncertainty_calibration_objective_study(tmp_path)
    report = study["selected_report"]
    assert "validation" in report["benchmark_delta"]
    assert "delta_change" in report["benchmark_delta"]["validation"]
    assert report["per_bucket_outcomes"]
    assert (tmp_path / "report.md").exists()
