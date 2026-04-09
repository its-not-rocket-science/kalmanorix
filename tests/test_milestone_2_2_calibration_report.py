from __future__ import annotations

from pathlib import Path

import numpy as np

from kalmanorix.experiment_reporting import write_calibration_report
from kalmanorix.calibration import CalibrationResult


def _mock_result(ece: float) -> CalibrationResult:
    edges = np.linspace(0.0, 1.0, 3)
    centers = (edges[:-1] + edges[1:]) / 2
    return CalibrationResult(
        ece=ece,
        brier_score=0.1 + ece,
        n_samples=10,
        bin_edges=edges,
        bin_centers=centers,
        bin_accuracies=np.array([0.4, 0.8]),
        bin_confidences=np.array([0.5, 0.7]),
        bin_counts=np.array([5, 5]),
        mean_confidence=0.6,
        mean_accuracy=0.55,
    )


def test_write_calibration_report_creates_markdown_and_json(tmp_path: Path) -> None:
    write_calibration_report(
        experiment_dir=tmp_path,
        specialist_calibration={"medical": _mock_result(0.05)},
        monolith_calibration=_mock_result(0.06),
        kalman_calibration=_mock_result(0.03),
        mean_calibration=_mock_result(0.07),
        ablation_calibration=_mock_result(0.08),
    )

    report = tmp_path / "calibration_report.md"
    summary = tmp_path / "calibration_summary.json"

    assert report.exists()
    assert summary.exists()

    txt = report.read_text(encoding="utf-8")
    assert "Calibration Report" in txt
    assert "fusion_kalman" in txt
