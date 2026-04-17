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
        Path(__file__).parent / "snapshots" / "guarded_findings_markdown_expected.md"
    ).read_text(encoding="utf-8")
    assert markdown.strip() == expected.strip()


def test_guarded_findings_positive_result_wording() -> None:
    markdown = generate_guarded_findings_markdown(
        significance_rows=[
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "ndcg@10",
                "mean_diff": 0.04,
                "adjusted_p_value": 0.01,
            }
        ]
    )
    assert "Positive result: kalman exceeds mean on ndcg@10" in markdown
    assert "demonstrated only for this benchmark configuration" in markdown


def test_guarded_findings_null_result_wording() -> None:
    markdown = generate_guarded_findings_markdown(
        significance_rows=[
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "mrr@10",
                "mean_diff": 0.0,
                "adjusted_p_value": 0.8,
            }
        ]
    )
    assert (
        "Null result: no statistically reliable difference is demonstrated" in markdown
    )
    assert "mrr@10" in markdown


def test_guarded_findings_inconclusive_result_wording() -> None:
    markdown = generate_guarded_findings_markdown(
        significance_rows=[
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "recall@10",
                "mean_diff": 0.002,
                "adjusted_p_value": 0.4,
            }
        ]
    )
    assert "Inconclusive result: kalman versus mean on recall@10" in markdown
    assert "Additional power or tighter controls are required" in markdown


def test_guarded_findings_regression_wording() -> None:
    markdown = generate_guarded_findings_markdown(
        significance_rows=[
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "ndcg@10",
                "mean_diff": -0.02,
                "adjusted_p_value": 0.01,
            }
        ]
    )
    assert "Regression: kalman underperforms mean on ndcg@10" in markdown
    assert "demonstrated risk until mitigated" in markdown


def test_guarded_findings_no_overclaim_for_unresolved_findings() -> None:
    markdown = generate_guarded_findings_markdown(
        significance_rows=[
            {
                "reference": "kalman",
                "candidate": "mean",
                "metric": "ndcg@10",
                "mean_diff": 0.005,
                "adjusted_p_value": 0.3,
            }
        ]
    )
    assert "No demonstrated directional effect is established" in markdown
    assert "Positive result:" not in markdown
    assert "outperforms" not in markdown
