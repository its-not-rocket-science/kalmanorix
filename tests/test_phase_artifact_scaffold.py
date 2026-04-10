from __future__ import annotations

import json
from pathlib import Path

import pytest

from kalmanorix.benchmarks.phase_artifacts import (
    OUTCOME_LABELS,
    PhaseTrackSpec,
    build_summary_template,
    validate_phase_summary_structure,
)


def test_summary_template_contains_all_outcome_slots() -> None:
    payload = build_summary_template(
        PhaseTrackSpec(
            slug="matched_compute",
            title="Matched Compute",
            objective="Objective text",
        )
    )

    validate_phase_summary_structure(payload)
    assert set(payload["outcome_slots"].keys()) == set(OUTCOME_LABELS)


@pytest.mark.parametrize(
    "track_path",
    [
        Path("results/matched_compute/summary_template.json"),
        Path("results/uncertainty_ablation/summary_template.json"),
        Path("results/ood_robustness/summary_template.json"),
    ],
)
def test_phase_summary_templates_validate(track_path: Path) -> None:
    payload = json.loads(track_path.read_text(encoding="utf-8"))
    validate_phase_summary_structure(payload)


@pytest.mark.parametrize(
    "report_path",
    [
        Path("results/matched_compute/report_template.md"),
        Path("results/uncertainty_ablation/report_template.md"),
        Path("results/ood_robustness/report_template.md"),
    ],
)
def test_report_templates_cover_all_outcomes(report_path: Path) -> None:
    text = report_path.read_text(encoding="utf-8")
    for label in OUTCOME_LABELS:
        assert f"**{label}:" in text
