from __future__ import annotations

import json
import subprocess
import sys


def test_reporting_runner_generates_artifacts(tmp_path) -> None:
    summary_path = tmp_path / "summary.json"
    details_path = tmp_path / "details.json"
    out_dir = tmp_path / "report"

    payload = {
        "query_level": {
            "domains": {"q1": "tech", "q2": "cook"},
            "ground_truth": {"q1": ["d1"], "q2": ["d4"]},
            "rankings": {
                "kalman": {"q1": ["d1", "d2"], "q2": ["d4", "d3"]},
                "mean": {"q1": ["d2", "d1"], "q2": ["d3", "d4"]},
            },
            "latency_ms": {
                "kalman": {"q1": 5.0, "q2": 6.0},
                "mean": {"q1": 4.0, "q2": 4.5},
            },
            "confidence_proxy": {
                "kalman": {"q1": 0.9, "q2": 0.8},
                "mean": {"q1": 0.7, "q2": 0.6},
            },
            "specialist_count_selected": {
                "kalman": {"q1": 2.0, "q2": 2.0},
                "mean": {"q1": 2.0, "q2": 2.0},
            },
        }
    }
    summary_path.write_text(
        json.dumps({"query_level": {"details_json": str(details_path)}}),
        encoding="utf-8",
    )
    details_path.write_text(json.dumps(payload), encoding="utf-8")

    subprocess.run(
        [
            sys.executable,
            "-m",
            "experiments.registry.reporting_runner",
            "--summary-json",
            str(summary_path),
            "--details-json",
            str(details_path),
            "--output-dir",
            str(out_dir),
        ],
        check=True,
    )

    assert (out_dir / "overall_metrics.csv").exists()
    assert (out_dir / "per_domain_metrics.csv").exists()
    assert (out_dir / "calibration_summary.csv").exists()
    assert (out_dir / "statistical_significance.csv").exists()
    assert (out_dir / "summary.md").exists()
    assert (out_dir / "results_bundle.json").exists()
