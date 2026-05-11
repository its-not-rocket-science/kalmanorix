from __future__ import annotations

import json
import subprocess
from pathlib import Path


def test_validate_publication_claims_blocks_prohibited_text(tmp_path: Path) -> None:
    gate = {
        "final_verdict": "blocked",
        "allowed_headline_sentence": "Kalman fusion did not clear the canonical claim gate against mean fusion.",
        "prohibited_claims": ["Kalman fusion beats mean fusion."],
    }
    gate_path = tmp_path / "claim_gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    manuscript = tmp_path / "paper.md"
    manuscript.write_text("Results: Kalman fusion beats mean fusion.", encoding="utf-8")

    proc = subprocess.run(
        [
            "python",
            "scripts/validate_publication_claims.py",
            "--claim-gate",
            str(gate_path),
            str(manuscript),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode != 0
    assert "prohibited claim found" in (proc.stderr + proc.stdout)
