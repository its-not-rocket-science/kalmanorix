from __future__ import annotations

from pathlib import Path

from kalmanorix.benchmarks.report_generator import generate_guarded_findings_markdown


def test_guarded_findings_markdown_snapshot() -> None:
    markdown = generate_guarded_findings_markdown(
        significance_rows=[
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "ndcg@10",
                "mean_diff": 0.021,
                "adjusted_p_value": 0.01,
            },
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "recall@10",
                "mean_diff": -0.031,
                "adjusted_p_value": 0.02,
            },
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "mrr",
                "mean_diff": 0.0,
                "adjusted_p_value": 0.4,
            },
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "recall@1",
                "mean_diff": 0.002,
                "adjusted_p_value": 0.6,
            },
        ],
    )
    expected = (
        Path(__file__).parent
        / "snapshots"
        / "guarded_findings_markdown_expected.md"
    ).read_text(encoding="utf-8")
    assert markdown.strip() == expected.strip()
