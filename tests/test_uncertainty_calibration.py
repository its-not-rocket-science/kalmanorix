from __future__ import annotations

import json

import numpy as np

from kalmanorix.benchmarks.uncertainty_calibration import run_uncertainty_calibration
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
