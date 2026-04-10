"""Scaffolding helpers for post-canonical empirical artifact tracks.

These helpers intentionally create conservative templates that separate
placeholder structure from demonstrated evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from .report_generator import generate_guarded_findings_markdown

OUTCOME_LABELS: tuple[str, str, str, str] = (
    "positive",
    "null",
    "inconclusive",
    "regression",
)


@dataclass(frozen=True)
class PhaseTrackSpec:
    slug: str
    title: str
    objective: str


def build_summary_template(spec: PhaseTrackSpec) -> dict[str, Any]:
    """Build a pending-state summary template for a phase track."""

    return {
        "schema_version": "phase_eval.v1",
        "track": spec.slug,
        "title": spec.title,
        "status": "pending",
        "objective": spec.objective,
        "outcome_slots": {
            label: {
                "supported": False,
                "evidence": [],
                "notes": "",
            }
            for label in OUTCOME_LABELS
        },
        "artifacts": {
            "summary": "summary.json",
            "report": "report.md",
        },
        "notes": [
            "Scaffold only: populate this file after running the preregistered evaluation.",
            "Do not promote planned outcomes to demonstrated findings.",
        ],
    }


def render_report_template(spec: PhaseTrackSpec) -> str:
    """Render a conservative report template for a phase track."""

    guarded_block = generate_guarded_findings_markdown(significance_rows=[])
    return "\n".join(
        [
            f"# {spec.title} Report",
            "",
            "## Current Evidence State",
            f"- Artifact path: `results/{spec.slug}/`.",
            "- State: **Scaffold only (pending run)**.",
            "- Interpretation boundary: no outcome is demonstrated until real artifacts are generated.",
            "",
            "## Objective",
            f"- {spec.objective}",
            "",
            "## Outcome slots (to be completed with evidence)",
            "- **positive:** [pending evidence]",
            "- **null:** [pending evidence]",
            "- **inconclusive:** [pending evidence]",
            "- **regression:** [pending evidence]",
            "",
            "## Guarded findings scaffold",
            "",
            guarded_block,
            "",
            "## Artifact checklist",
            "- [ ] `summary.json` updated with real metrics and statistical outputs.",
            "- [ ] `report.md` updated with benchmark-specific findings.",
            "- [ ] Synthetic/debug runs labeled and separated from headline evidence.",
        ]
    )


def validate_phase_summary_structure(payload: dict[str, Any]) -> None:
    """Validate minimal structure required for phase summary artifacts."""

    required_top_level = {
        "schema_version",
        "track",
        "title",
        "status",
        "objective",
        "outcome_slots",
        "artifacts",
    }
    missing = required_top_level.difference(payload.keys())
    if missing:
        raise ValueError(f"Missing top-level keys: {sorted(missing)}")

    if payload["schema_version"] != "phase_eval.v1":
        raise ValueError("schema_version must be 'phase_eval.v1'")

    outcome_slots = payload["outcome_slots"]
    if not isinstance(outcome_slots, dict):
        raise ValueError("outcome_slots must be a mapping")

    for label in OUTCOME_LABELS:
        if label not in outcome_slots:
            raise ValueError(f"Missing outcome slot: {label}")
        slot = outcome_slots[label]
        if not isinstance(slot, dict):
            raise ValueError(f"Outcome slot '{label}' must be an object")
        for key in ("supported", "evidence", "notes"):
            if key not in slot:
                raise ValueError(f"Outcome slot '{label}' missing key: {key}")


def write_phase_track_scaffold(base_dir: Path, spec: PhaseTrackSpec) -> None:
    """Create/refresh scaffold files for one phase track."""

    output_dir = base_dir / spec.slug
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary_template.json").write_text(
        json.dumps(build_summary_template(spec), indent=2),
        encoding="utf-8",
    )
    (output_dir / "report_template.md").write_text(
        render_report_template(spec), encoding="utf-8"
    )
