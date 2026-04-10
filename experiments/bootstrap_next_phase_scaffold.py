"""Bootstrap artifact scaffolding for post-canonical empirical tracks."""

from __future__ import annotations

from pathlib import Path

from kalmanorix.benchmarks.phase_artifacts import (
    PhaseTrackSpec,
    write_phase_track_scaffold,
)


def main() -> None:
    base_dir = Path("results")
    tracks = [
        PhaseTrackSpec(
            slug="matched_compute",
            title="Matched Compute",
            objective=(
                "Evaluate specialists vs monolith under explicit training/inference "
                "compute parity constraints with guarded interpretation."
            ),
        ),
        PhaseTrackSpec(
            slug="uncertainty_ablation",
            title="Uncertainty Ablation",
            objective=(
                "Quantify how uncertainty estimation choices affect retrieval, "
                "calibration, and sensitivity to variance mis-specification."
            ),
        ),
        PhaseTrackSpec(
            slug="ood_robustness",
            title="OOD Robustness",
            objective=(
                "Evaluate robustness under distribution shift and distinguish "
                "supported effects from null/inconclusive/regression outcomes."
            ),
        ),
    ]

    for spec in tracks:
        write_phase_track_scaffold(base_dir, spec)
        print(f"Scaffolded results/{spec.slug}/")


if __name__ == "__main__":
    main()
