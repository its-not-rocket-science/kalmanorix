"""Bucketed per-query error analysis for Kalman fusion behavior."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any, Iterable, Mapping


@dataclass(frozen=True)
class QueryRecord:
    query_id: str
    mean_score: float
    kalman_score: float
    hard_score: float
    specialist_count: float
    router_confidence: float
    agreement_proxy: float
    uncertainty_spread: float

    @property
    def delta_vs_mean(self) -> float:
        return self.kalman_score - self.mean_score

    @property
    def delta_vs_hard(self) -> float:
        return self.kalman_score - self.hard_score


def _mrr_at_10(ranking: list[str], relevant: set[str]) -> float:
    for idx, doc_id in enumerate(ranking[:10], start=1):
        if doc_id in relevant:
            return float(1.0 / idx)
    return 0.0


def _jaccard_top10(a: list[str], b: list[str]) -> float:
    sa, sb = set(a[:10]), set(b[:10])
    union = sa.union(sb)
    if not union:
        return 1.0
    return float(len(sa.intersection(sb)) / len(union))


def _quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return 0.0
    idx = int(round((len(sorted_values) - 1) * q))
    return float(sorted_values[max(0, min(idx, len(sorted_values) - 1))])


def build_query_records(details: Mapping[str, Any]) -> list[QueryRecord]:
    ql = details["query_level"]
    rankings = ql["rankings"]
    ground_truth = {qid: set(docs) for qid, docs in ql["ground_truth"].items()}
    mean_rankings = rankings["mean"]
    kalman_rankings = rankings["kalman"]
    hard_rankings = rankings["router_only_top1"]

    confidence = ql.get("confidence_proxy", {})
    hard_conf = confidence.get("router_only_top1", {})
    mean_conf = confidence.get("mean", {})
    kalman_conf = confidence.get("kalman", {})

    specialist_counts = ql.get("specialist_count_selected", {}).get("router_only_top1", {})

    records: list[QueryRecord] = []
    for qid in sorted(ground_truth):
        mean_rank = list(mean_rankings[qid])
        kalman_rank = list(kalman_rankings[qid])
        hard_rank = list(hard_rankings[qid])
        spread_values = [
            float(mean_conf.get(qid, 0.0)),
            float(kalman_conf.get(qid, 0.0)),
            float(hard_conf.get(qid, 0.0)),
        ]
        records.append(
            QueryRecord(
                query_id=qid,
                mean_score=_mrr_at_10(mean_rank, ground_truth[qid]),
                kalman_score=_mrr_at_10(kalman_rank, ground_truth[qid]),
                hard_score=_mrr_at_10(hard_rank, ground_truth[qid]),
                specialist_count=float(specialist_counts.get(qid, 1.0)),
                router_confidence=float(hard_conf.get(qid, 0.0)),
                agreement_proxy=_jaccard_top10(mean_rank, hard_rank),
                uncertainty_spread=max(spread_values) - min(spread_values),
            )
        )
    return records


def _summarize_bucket(name: str, rows: list[QueryRecord]) -> dict[str, Any]:
    if not rows:
        return {
            "bucket": name,
            "n": 0,
            "mean": 0.0,
            "kalman": 0.0,
            "hard": 0.0,
            "delta_kalman_vs_mean": 0.0,
            "delta_kalman_vs_hard": 0.0,
            "consistency": "insufficient",
        }
    delta_mean = [row.delta_vs_mean for row in rows]
    abs_delta = [abs(v) for v in delta_mean]
    avg_delta = mean(delta_mean)
    if all(v > 0 for v in delta_mean):
        consistency = "helps"
    elif all(v < 0 for v in delta_mean):
        consistency = "hurts"
    elif mean(abs_delta) <= 0.01:
        consistency = "redundant"
    else:
        consistency = "mixed"
    return {
        "bucket": name,
        "n": len(rows),
        "mean": mean(row.mean_score for row in rows),
        "kalman": mean(row.kalman_score for row in rows),
        "hard": mean(row.hard_score for row in rows),
        "delta_kalman_vs_mean": avg_delta,
        "delta_kalman_vs_hard": mean(row.delta_vs_hard for row in rows),
        "mean_abs_delta": mean(abs_delta),
        "consistency": consistency,
    }


def generate_bucket_summaries(records: list[QueryRecord]) -> list[dict[str, Any]]:
    conf_values = sorted(row.router_confidence for row in records)
    uncertainty_values = sorted(row.uncertainty_spread for row in records)
    agreement_values = sorted(row.agreement_proxy for row in records)
    low_conf = _quantile(conf_values, 0.33)
    high_conf = _quantile(conf_values, 0.66)
    high_uncertainty = _quantile(uncertainty_values, 0.5)
    high_agreement = _quantile(agreement_values, 0.5)

    buckets: list[tuple[str, Iterable[QueryRecord]]] = [
        ("single-domain", [r for r in records if r.specialist_count <= 1.0]),
        ("multi-domain", [r for r in records if r.specialist_count > 1.0]),
        ("high specialist agreement", [r for r in records if r.agreement_proxy >= high_agreement]),
        ("specialist disagreement", [r for r in records if r.agreement_proxy < high_agreement]),
        ("high uncertainty spread", [r for r in records if r.uncertainty_spread >= high_uncertainty]),
        ("low uncertainty spread", [r for r in records if r.uncertainty_spread < high_uncertainty]),
        ("router confidence: low", [r for r in records if r.router_confidence <= low_conf]),
        ("router confidence: mid", [r for r in records if low_conf < r.router_confidence <= high_conf]),
        ("router confidence: high", [r for r in records if r.router_confidence > high_conf]),
        (
            "in-domain (proxy)",
            [r for r in records if r.specialist_count <= 1.0 and r.router_confidence > high_conf],
        ),
        (
            "ambiguous (proxy)",
            [r for r in records if r.specialist_count > 1.0 or r.router_confidence <= low_conf],
        ),
    ]
    return [_summarize_bucket(name, list(rows)) for name, rows in buckets]


def render_markdown_report(*, bucket_summaries: list[dict[str, Any]], total_queries: int) -> str:
    lines = [
        "# Kalman Bucketed Error Analysis",
        "",
        "This report is query-bucketed and descriptive. It does **not** promote exploratory subgroups into global claims.",
        "",
        f"- Total queries analyzed: {total_queries}",
        "- Metric used for per-query comparison: MRR@10",
        "- Compared methods in every bucket: mean fusion, Kalman fusion, hard routing",
        "",
        "## Bucket metrics",
        "",
        "| Bucket | n | Mean fusion | Kalman fusion | Hard routing | Kalman-Mean | Kalman-Hard | Pattern |",
        "|---|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in bucket_summaries:
        lines.append(
            "| {bucket} | {n} | {mean:.4f} | {kalman:.4f} | {hard:.4f} | {d1:.4f} | {d2:.4f} | {pattern} |".format(
                bucket=row["bucket"],
                n=row["n"],
                mean=row["mean"],
                kalman=row["kalman"],
                hard=row["hard"],
                d1=row["delta_kalman_vs_mean"],
                d2=row["delta_kalman_vs_hard"],
                pattern=row["consistency"],
            )
        )

    helps = [r for r in bucket_summaries if r["consistency"] == "helps" and r["n"] >= 3]
    hurts = [r for r in bucket_summaries if r["consistency"] == "hurts" and r["n"] >= 3]
    redundant = [r for r in bucket_summaries if r["consistency"] == "redundant" and r["n"] >= 3]

    lines.extend(["", "## Empirical patterns", ""])
    lines.append("### Buckets where Kalman consistently helps")
    lines.extend([f"- {r['bucket']} (n={r['n']}, Δ={r['delta_kalman_vs_mean']:.4f})" for r in helps] or ["- None met the consistency and minimum-size filter in this run."])
    lines.append("")
    lines.append("### Buckets where Kalman hurts")
    lines.extend([f"- {r['bucket']} (n={r['n']}, Δ={r['delta_kalman_vs_mean']:.4f})" for r in hurts] or ["- None met the consistency and minimum-size filter in this run."])
    lines.append("")
    lines.append("### Buckets where Kalman appears redundant")
    lines.extend([f"- {r['bucket']} (n={r['n']}, mean |Δ|={r['mean_abs_delta']:.4f})" for r in redundant] or ["- None met the redundancy and minimum-size filter in this run."])

    lines.extend(
        [
            "",
            "## Actionable hypotheses for next fusion revision",
            "",
            "- If gains cluster in ambiguous/disagreement buckets, make Kalman conditional on disagreement and keep mean fusion for high-agreement cases.",
            "- If regressions cluster in low-confidence buckets, add a router-confidence gate that falls back to hard routing or conservative mean.",
            "- If low-uncertainty-spread buckets are redundant, skip covariance-heavy updates there to cut latency.",
            "- Re-run with larger held-out query counts per bucket before elevating any subgroup into product-level policy.",
        ]
    )
    return "\n".join(lines) + "\n"


def generate_kalman_error_analysis_report(details: Mapping[str, Any], output_path: Path) -> str:
    records = build_query_records(details)
    bucket_summaries = generate_bucket_summaries(records)
    report = render_markdown_report(
        bucket_summaries=bucket_summaries,
        total_queries=len(records),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(report, encoding="utf-8")
    return report
