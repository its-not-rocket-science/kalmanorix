"""
Golden-output style test for examples/minimal_fusion_demo.py.

This test does NOT assert exact numeric values.
Instead, it verifies that:
- the demo runs without error
- key sections are printed
- all fusion strategies appear in the output

This keeps the demo stable and user-facing without making it brittle.
"""

import subprocess
import sys
from pathlib import Path


def test_minimal_fusion_demo_runs_and_prints_sections():
    """
    Ensure the minimal fusion demo executes successfully and
    prints the expected high-level sections.
    """
    repo_root = Path(__file__).resolve().parents[1]
    demo_path = repo_root / "examples" / "minimal_fusion_demo.py"

    result = subprocess.run(
        [sys.executable, str(demo_path)],
        capture_output=True,
        text=True,
        check=True,
    )

    out = result.stdout

    # Structural assertions (golden-style, but robust)
    assert "Query:" in out
    assert "Hard routing" in out
    assert "Mean fusion" in out
    assert "KalmanorixFuser" in out
    assert "LearnedGateFuser" in out
    assert "Cosine similarities" in out
