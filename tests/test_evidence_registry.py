from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
REGISTRY = REPO / "results/evidence_registry.json"

MANUSCRIPT_FILES = [
    "README.md",
    "ROADMAP.md",
    "docs/research/results.md",
    "paper/paper_draft_deprecated.md",
    "paper/joss/paper.md",
    "paper/arxiv/README.md",
    "paper/tmlr/README.md",
    "paper/joss/README.md",
    "paper/joss/results_summary.md",
]


def test_registry_schema_and_required_claims() -> None:
    payload = json.loads(REGISTRY.read_text(encoding="utf-8"))
    claims = payload["claims"]
    required = {
        "routing_efficiency",
        "kalman_vs_mean_quality",
        "matched_compute_specialists_vs_monolith",
        "ood_uncertainty_weighting",
        "uncertainty_calibration_downstream",
        "uncertainty_ablation_downstream",
        "covariance_ablation_value",
        "correlation_aware_fusion",
        "kalman_latency_optimization",
    }
    seen = {c["claim_id"] for c in claims}
    assert required.issubset(seen)
    for claim in claims:
        assert claim["evidence_status"] in {
            "supported",
            "unresolved",
            "inconclusive",
            "null",
            "regression",
            "exploratory",
        }
        assert claim["artifact_paths_used"]
        assert claim["headline_safe_sentence"]
        assert claim["regenerate_command"]


def test_manuscripts_reference_evidence_registry() -> None:
    for rel in MANUSCRIPT_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        assert "results/evidence_registry.json" in text, (
            f"{rel} must reference the evidence registry."
        )


def test_no_stale_headline_numbers_conflicting_with_registry() -> None:
    registry_text = REGISTRY.read_text(encoding="utf-8")
    forbidden_literals = ["+0.0037", "~2.06x", "65% average FLOPs reduction"]
    for rel in MANUSCRIPT_FILES:
        text = (REPO / rel).read_text(encoding="utf-8")
        for lit in forbidden_literals:
            if lit in text:
                assert lit in registry_text, (
                    f"{rel} contains stale literal {lit} not backed by registry."
                )
