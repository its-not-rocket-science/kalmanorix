"""Report rendering helpers for statistical comparison outputs."""

from __future__ import annotations

from .statistical_testing import StatisticalComparisonReport, render_appendix_table


def generate_appendix_markdown(report: StatisticalComparisonReport) -> str:
    """Generate a paper-appendix-ready markdown report table."""

    return render_appendix_table(report)
