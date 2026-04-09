"""Report rendering helpers for statistical comparison outputs.

This module intentionally enforces conservative language so generated markdown
does not claim effects that are unsupported by the supplied evidence.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from .statistical_testing import StatisticalComparisonReport, render_appendix_table


def generate_appendix_markdown(report: StatisticalComparisonReport) -> str:
    """Generate a paper-appendix-ready markdown report table."""

    return render_appendix_table(report)


ResultLabel = Literal["positive", "null", "inconclusive", "regression"]


@dataclass(frozen=True)
class Finding:
    """One guarded finding rendered in markdown."""

    label: ResultLabel
    message: str


def template_positive_result(
    *,
    reference: str,
    candidate: str,
    metric: str,
    mean_diff: float,
    adjusted_p_value: float,
) -> str:
    """Template for a positive result with explicit statistical guardrails."""

    return (
        f"Positive result: {reference} exceeds {candidate} on {metric} "
        f"(Δ={mean_diff:.6f}, Holm-adjusted p={adjusted_p_value:.6f}). "
        "This is demonstrated only for this benchmark configuration."
    )


def template_null_result(
    *,
    reference: str,
    candidate: str,
    metric: str,
    mean_diff: float,
    adjusted_p_value: float,
) -> str:
    """Template for a null result that avoids overclaiming."""

    return (
        f"Null result: no statistically reliable difference is demonstrated for "
        f"{reference} versus {candidate} on {metric} "
        f"(Δ={mean_diff:.6f}, Holm-adjusted p={adjusted_p_value:.6f})."
    )


def template_inconclusive_result(
    *,
    reference: str,
    candidate: str,
    metric: str,
    mean_diff: float,
    adjusted_p_value: float,
) -> str:
    """Template for an inconclusive result with next-step framing."""

    return (
        f"Inconclusive result: {reference} versus {candidate} on {metric} "
        f"shows ambiguous evidence (Δ={mean_diff:.6f}, Holm-adjusted p={adjusted_p_value:.6f}). "
        "Additional power or tighter controls are required before drawing conclusions."
    )


def template_regression(
    *,
    reference: str,
    candidate: str,
    metric: str,
    mean_diff: float,
    adjusted_p_value: float,
) -> str:
    """Template for a statistically supported regression."""

    return (
        f"Regression: {reference} underperforms {candidate} on {metric} "
        f"(Δ={mean_diff:.6f}, Holm-adjusted p={adjusted_p_value:.6f}). "
        "Treat this as a demonstrated risk until mitigated by follow-up experiments."
    )


def _classify_row(row: dict[str, Any], *, alpha: float) -> Finding:
    reference = str(row["reference"])
    candidate = str(row["candidate"])
    metric = str(row["metric"])
    mean_diff = float(row["mean_diff"])
    padj = float(row["adjusted_p_value"])

    if not isfinite(mean_diff) or not isfinite(padj):
        return Finding(
            label="inconclusive",
            message=(
                f"Inconclusive result: unable to evaluate {reference} versus "
                f"{candidate} on {metric} due to non-finite statistics."
            ),
        )

    if padj <= alpha:
        if mean_diff > 0.0:
            return Finding(
                label="positive",
                message=template_positive_result(
                    reference=reference,
                    candidate=candidate,
                    metric=metric,
                    mean_diff=mean_diff,
                    adjusted_p_value=padj,
                ),
            )
        if mean_diff < 0.0:
            return Finding(
                label="regression",
                message=template_regression(
                    reference=reference,
                    candidate=candidate,
                    metric=metric,
                    mean_diff=mean_diff,
                    adjusted_p_value=padj,
                ),
            )
        return Finding(
            label="inconclusive",
            message=template_inconclusive_result(
                reference=reference,
                candidate=candidate,
                metric=metric,
                mean_diff=mean_diff,
                adjusted_p_value=padj,
            ),
        )
    if abs(mean_diff) < 1e-12:
        return Finding(
            label="null",
            message=template_null_result(
                reference=reference,
                candidate=candidate,
                metric=metric,
                mean_diff=mean_diff,
                adjusted_p_value=padj,
            ),
        )
    return Finding(
        label="inconclusive",
        message=template_inconclusive_result(
            reference=reference,
            candidate=candidate,
            metric=metric,
            mean_diff=mean_diff,
            adjusted_p_value=padj,
        ),
    )


def generate_guarded_findings_markdown(
    *,
    significance_rows: list[dict[str, Any]],
    alpha: float = 0.05,
    benchmark_limitations: list[str] | None = None,
) -> str:
    """Generate mandatory scientific-credibility sections for benchmark reports."""

    findings = [_classify_row(row, alpha=alpha) for row in significance_rows]
    demonstrated = [f for f in findings if f.label in ("positive", "regression")]
    unresolved = [f for f in findings if f.label in ("null", "inconclusive")]

    lines = ["## Demonstrated findings", ""]
    if demonstrated:
        lines.extend(f"- {finding.message}" for finding in demonstrated)
    else:
        lines.append(
            "- No demonstrated directional effect is established by the current statistical evidence."
        )

    lines.extend(["", "## Unresolved findings", ""])
    if unresolved:
        lines.extend(f"- {finding.message}" for finding in unresolved)
    else:
        lines.append("- No unresolved pairwise findings were detected in this output.")

    lines.extend(
        [
            "",
            "## Threats to validity",
            "",
            "- Query sets may under-represent long-tail or adversarial cases.",
            "- Metric families are correlated; adjusted p-values reduce but do not eliminate interpretability risk.",
            "- The analysis is paired and benchmark-specific; external generalization is not demonstrated by default.",
            "",
            "## Benchmark limitations",
            "",
        ]
    )

    if benchmark_limitations:
        lines.extend(f"- {item}" for item in benchmark_limitations)
    else:
        lines.extend(
            [
                "- Results depend on the provided benchmark artifacts and should be treated as conditional evidence.",
                "- Latency and memory proxies are environment-sensitive and may shift under different hardware/runtime settings.",
            ]
        )

    lines.extend(
        [
            "",
            "## Recommended next experiments",
            "",
            "- Increase held-out query count and rebalance domains before promoting unresolved findings.",
            "- Add stress tests for distribution shift and low-resource domains to challenge demonstrated effects.",
            "- Replicate on an independent benchmark slice with pre-registered metrics and hypotheses.",
        ]
    )
    return "\n".join(lines)
