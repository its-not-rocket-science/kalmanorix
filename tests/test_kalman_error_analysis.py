from __future__ import annotations

from pathlib import Path

from kalmanorix.benchmarks.kalman_error_analysis import (
    build_query_records,
    generate_bucket_summaries,
    generate_kalman_error_analysis_report,
)


def _details() -> dict[str, object]:
    return {
        "query_level": {
            "ground_truth": {
                "q1": ["d1"],
                "q2": ["d2"],
                "q3": ["d3"],
                "q4": ["d4"],
            },
            "rankings": {
                "mean": {
                    "q1": ["d1", "x"],
                    "q2": ["x", "d2"],
                    "q3": ["x", "y", "d3"],
                    "q4": ["x", "y", "d4"],
                },
                "kalman": {
                    "q1": ["d1", "x"],
                    "q2": ["d2", "x"],
                    "q3": ["d3", "x"],
                    "q4": ["x", "d4"],
                },
                "router_only_top1": {
                    "q1": ["d1", "x"],
                    "q2": ["x", "d2"],
                    "q3": ["x", "d3"],
                    "q4": ["d4", "x"],
                },
            },
            "confidence_proxy": {
                "mean": {"q1": 0.8, "q2": 0.7, "q3": 0.4, "q4": 0.2},
                "kalman": {"q1": 0.82, "q2": 0.71, "q3": 0.41, "q4": 0.21},
                "router_only_top1": {"q1": 0.9, "q2": 0.65, "q3": 0.3, "q4": 0.1},
            },
            "specialist_count_selected": {
                "router_only_top1": {"q1": 1.0, "q2": 1.0, "q3": 2.0, "q4": 2.0}
            },
        }
    }


def test_build_records_and_buckets() -> None:
    records = build_query_records(_details())
    assert len(records) == 4
    summaries = generate_bucket_summaries(records)
    assert any(row["bucket"] == "single-domain" for row in summaries)
    assert any(row["bucket"] == "router confidence: high" for row in summaries)


def test_report_written(tmp_path: Path) -> None:
    out = tmp_path / "report.md"
    report = generate_kalman_error_analysis_report(_details(), out)
    assert out.exists()
    assert "## Bucket metrics" in report
    assert "Actionable hypotheses" in report
