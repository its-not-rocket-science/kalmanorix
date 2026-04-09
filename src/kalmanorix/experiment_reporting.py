"""Standard experiment reporting helpers."""

from __future__ import annotations

import json
from pathlib import Path

from .calibration import CalibrationResult, calibration_summary


def write_calibration_report(
    *,
    experiment_dir: Path,
    specialist_calibration: dict[str, CalibrationResult],
    monolith_calibration: CalibrationResult,
    kalman_calibration: CalibrationResult,
    mean_calibration: CalibrationResult,
    ablation_calibration: CalibrationResult,
) -> None:
    """Write markdown + JSON calibration summaries for experiment reports."""
    rows = {
        **{
            f"specialist_{name}": calibration_summary(res)
            for name, res in specialist_calibration.items()
        },
        "monolith": calibration_summary(monolith_calibration),
        "fusion_kalman": calibration_summary(kalman_calibration),
        "fusion_mean": calibration_summary(mean_calibration),
        "ablation_constant_variance": calibration_summary(ablation_calibration),
    }

    payload = {
        name: {k: float(v) for k, v in stats.items()} for name, stats in rows.items()
    }
    (experiment_dir / "calibration_summary.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )

    lines = [
        "# Calibration Report",
        "",
        "| Model | N | ECE | Brier | Mean confidence | Mean accuracy | Overconfidence gap |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, stats in payload.items():
        lines.append(
            f"| {name} | {int(stats['n_samples'])} | {stats['ece']:.4f} | {stats['brier_score']:.4f} "
            f"| {stats['mean_confidence']:.4f} | {stats['mean_accuracy']:.4f} "
            f"| {stats['overconfidence_gap']:.4f} |"
        )

    lines.extend(
        [
            "",
            "## Reliability plots",
            "",
            "- Specialists: `reliability_specialist_<domain>.png`",
            "- Fusion: `reliability_kalman.png`, `reliability_mean.png`",
            "- Baselines: `reliability_monolith.png`, `reliability_ablation_constant.png`",
        ]
    )
    (experiment_dir / "calibration_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
