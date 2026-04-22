from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "kalman_evidence_dashboard"


@dataclass(frozen=True)
class SourcedValue:
    value: Any
    source_path: str
    source_json_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "source": {
                "path": self.source_path,
                "json_path": self.source_json_path,
            },
        }


def _load_json(relative_path: str) -> dict[str, Any]:
    with (ROOT / relative_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _traffic_light_from_verdict(verdict: str) -> str:
    if verdict in {"supported", "pass"}:
        return "green"
    if verdict in {
        "unsupported",
        "not_supported",
        "regression",
        "failed",
        "not_ready_for_default_rollout",
    }:
        return "red"
    return "yellow"


def _classify_replication_status(
    *, canonical_verdict: str, replication: dict[str, Any] | None
) -> str:
    if replication is None:
        return "not_replicated"
    run_verdicts = [
        str(run.get("verdict", "unknown"))
        for run in replication.get("per_run_verdicts", [])
    ]
    if not run_verdicts:
        return "not_replicated"
    unique_verdicts = set(run_verdicts)
    if unique_verdicts == {"supported"}:
        sign_consistency = replication.get(
            "sign_consistency_delta_kalman_minus_mean", "mixed"
        )
        latency_consistency = replication.get("latency_ratio_consistency", "mixed")
        if (
            canonical_verdict == "supported"
            and sign_consistency == "all_positive"
            and latency_consistency == "all_within_threshold"
        ):
            return "replicated_supported"
    if len(unique_verdicts) == 1 and canonical_verdict in unique_verdicts:
        return "replicated_same_verdict"
    return "replicated_mixed_verdict"


def _build_summary() -> dict[str, Any]:
    canonical = _load_json("results/canonical_benchmark_v2/summary.json")
    uncertainty_ablation = _load_json("results/uncertainty_ablation/summary.json")
    replication = canonical.get("replication")
    replication_sources = ["results/canonical_benchmark_v2/summary.json"]
    if replication is None:
        latency_rerun = _load_json(
            "results/kalman_latency_optimization/canonical/summary.json"
        )
        replication = {
            "per_run_verdicts": [
                {"verdict": latency_rerun["decision"]["kalman_vs_mean"]["verdict"]}
            ]
        }
        replication_sources.append(
            "results/kalman_latency_optimization/canonical/summary.json"
        )

    canonical_decision = canonical["decision"]["kalman_vs_mean"]
    canonical_verdict = canonical_decision["verdict"]
    canonical_light = _traffic_light_from_verdict(canonical_verdict)

    confirmatory = canonical.get("confirmatory_slice_results")
    if confirmatory is None:
        confirmatory_verdict = "not_run_in_committed_canonical_artifact"
        confirmatory_light = "yellow"
        confirmatory_source_path = "results/canonical_benchmark_v2/summary.json"
        confirmatory_source_json_path = "$.confirmatory_slice_results"
    else:
        confirmatory_verdict = confirmatory.get("decision", {}).get(
            "verdict", "unresolved"
        )
        confirmatory_light = _traffic_light_from_verdict(confirmatory_verdict)
        confirmatory_source_path = "results/canonical_benchmark_v2/summary.json"
        confirmatory_source_json_path = "$.confirmatory_slice_results.decision.verdict"

    kalman_ndcg10 = canonical["methods"]["kalman"]["metrics"]["ndcg@10"]["mean"]
    mean_ndcg10 = canonical["methods"]["mean"]["metrics"]["ndcg@10"]["mean"]
    router_ndcg10 = canonical["methods"]["router_only_top1"]["metrics"]["ndcg@10"][
        "mean"
    ]
    delta_ndcg10 = canonical["paired_statistics"]["kalman_vs_mean"]["overall"][
        "ndcg@10"
    ]["mean_difference"]
    pvalue_ndcg10 = canonical["paired_statistics"]["kalman_vs_mean"]["overall"][
        "ndcg@10"
    ]["adjusted_p_value"]
    baseline_light = "yellow" if pvalue_ndcg10 > 0.05 else "green"

    uncertainty_answer = uncertainty_ablation["answer"]
    uncertainty_light = "yellow" if "limited" in uncertainty_answer.lower() else "green"

    latency_ratio = canonical_decision["observed"]["latency_ratio_vs_mean"]
    latency_threshold = canonical_decision["rules"]["max_latency_ratio_vs_mean"]
    latency_ok = canonical_decision["checks"]["latency_ratio_ok"]
    latency_light = "green" if latency_ok else "red"

    replication_value = _classify_replication_status(
        canonical_verdict=canonical_verdict, replication=replication
    )
    replication_light = (
        "green"
        if replication_value == "replicated_supported"
        else (
            "yellow"
            if replication_value in {"not_replicated", "replicated_same_verdict"}
            else "red"
        )
    )

    overall = "yellow"
    if (
        canonical_light == "green"
        and confirmatory_light == "green"
        and latency_light == "green"
    ):
        overall = "green"
    elif canonical_light == "red":
        overall = "red"

    return {
        "artifact": "kalman_evidence_dashboard.v2",
        "generated_from": [
            "results/canonical_benchmark_v2/summary.json",
            "results/uncertainty_ablation/summary.json",
            *replication_sources[1:],
        ],
        "traffic_light_legend": {
            "green": "supported",
            "yellow": "unresolved",
            "red": "unsupported in tested regime",
        },
        "overall_hypothesis_status": {
            "traffic_light": overall,
            "reason": "Canonical verdict is inconclusive_underpowered with failed latency gate; confirmatory slice was not run in committed canonical artifacts.",
        },
        "canonical_benchmark_verdict": {
            "traffic_light": canonical_light,
            "verdict": SourcedValue(
                value=canonical_verdict,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.decision.kalman_vs_mean.verdict",
            ).as_dict(),
        },
        "confirmatory_slice_verdict": {
            "traffic_light": confirmatory_light,
            "verdict": SourcedValue(
                value=confirmatory_verdict,
                source_path=confirmatory_source_path,
                source_json_path=confirmatory_source_json_path,
            ).as_dict(),
        },
        "baseline_comparisons": {
            "traffic_light": baseline_light,
            "kalman_ndcg10": SourcedValue(
                value=kalman_ndcg10,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.methods.kalman.metrics.ndcg@10.mean",
            ).as_dict(),
            "mean_ndcg10": SourcedValue(
                value=mean_ndcg10,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.methods.mean.metrics.ndcg@10.mean",
            ).as_dict(),
            "router_top1_ndcg10": SourcedValue(
                value=router_ndcg10,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.methods.router_only_top1.metrics.ndcg@10.mean",
            ).as_dict(),
            "kalman_minus_mean_ndcg10": SourcedValue(
                value=delta_ndcg10,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.paired_statistics.kalman_vs_mean.overall.ndcg@10.mean_difference",
            ).as_dict(),
            "kalman_vs_mean_adjusted_p_value_ndcg10": SourcedValue(
                value=pvalue_ndcg10,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.paired_statistics.kalman_vs_mean.overall.ndcg@10.adjusted_p_value",
            ).as_dict(),
        },
        "uncertainty_ablation_result": {
            "traffic_light": uncertainty_light,
            "answer": SourcedValue(
                value=uncertainty_answer,
                source_path="results/uncertainty_ablation/summary.json",
                source_json_path="$.answer",
            ).as_dict(),
        },
        "latency_ratio": {
            "traffic_light": latency_light,
            "ratio_vs_mean": SourcedValue(
                value=latency_ratio,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.decision.kalman_vs_mean.observed.latency_ratio_vs_mean",
            ).as_dict(),
            "max_allowed_ratio": SourcedValue(
                value=latency_threshold,
                source_path="results/canonical_benchmark_v2/summary.json",
                source_json_path="$.decision.kalman_vs_mean.rules.max_latency_ratio_vs_mean",
            ).as_dict(),
        },
        "replication_status": {
            "traffic_light": replication_light,
            "status": {
                "value": replication_value,
                "sources": [
                    {
                        "path": source_path,
                        "json_path": (
                            "$.replication.per_run_verdicts[*].verdict"
                            if source_path
                            == "results/canonical_benchmark_v2/summary.json"
                            else "$.decision.kalman_vs_mean.verdict"
                        ),
                    }
                    for source_path in replication_sources
                ],
            },
            "canonical_verdict": canonical_verdict,
            "replication_run_verdicts": [
                str(run.get("verdict", "unknown"))
                for run in replication.get("per_run_verdicts", [])
            ],
        },
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    legend = summary["traffic_light_legend"]

    def row(label: str, section: dict[str, Any], detail: str) -> str:
        return f"| {label} | {section['traffic_light']} | {detail} |"

    lines = [
        "# Kalman-vs-Mean Evidence Dashboard",
        "",
        f"**Overall status:** `{summary['overall_hypothesis_status']['traffic_light']}`",
        "",
        "## Traffic-light legend",
        f"- `green` = {legend['green']}",
        f"- `yellow` = {legend['yellow']}",
        f"- `red` = {legend['red']}",
        "",
        "## Compact summary",
        "",
        "| Field | Traffic light | Evidence (artifact-sourced) |",
        "|---|---|---|",
        row(
            "Canonical benchmark verdict",
            summary["canonical_benchmark_verdict"],
            (
                "`"
                + summary["canonical_benchmark_verdict"]["verdict"]["value"]
                + "` from "
                + f"`{summary['canonical_benchmark_verdict']['verdict']['source']['path']}`"
            ),
        ),
        row(
            "Confirmatory slice verdict",
            summary["confirmatory_slice_verdict"],
            (
                "`"
                + summary["confirmatory_slice_verdict"]["verdict"]["value"]
                + "` from "
                + f"`{summary['confirmatory_slice_verdict']['verdict']['source']['path']}`"
            ),
        ),
        row(
            "Baseline comparisons",
            summary["baseline_comparisons"],
            (
                "Kalman nDCG@10 "
                + f"{summary['baseline_comparisons']['kalman_ndcg10']['value']:.4f}"
                + ", Mean "
                + f"{summary['baseline_comparisons']['mean_ndcg10']['value']:.4f}"
                + ", Δ "
                + f"{summary['baseline_comparisons']['kalman_minus_mean_ndcg10']['value']:.4f}"
                + ", adjusted p="
                + f"{summary['baseline_comparisons']['kalman_vs_mean_adjusted_p_value_ndcg10']['value']:.4f}"
            ),
        ),
        row(
            "Uncertainty ablation result",
            summary["uncertainty_ablation_result"],
            summary["uncertainty_ablation_result"]["answer"]["value"],
        ),
        row(
            "Latency ratio",
            summary["latency_ratio"],
            (
                "Kalman/Mean="
                + f"{summary['latency_ratio']['ratio_vs_mean']['value']:.3f}"
                + " vs threshold="
                + f"{summary['latency_ratio']['max_allowed_ratio']['value']:.3f}"
            ),
        ),
        row(
            "Replication status",
            summary["replication_status"],
            (
                "`"
                + summary["replication_status"]["status"]["value"]
                + "` (canonical verdict="
                + f"`{summary['replication_status']['canonical_verdict']}`"
                + ", replication verdicts="
                + f"`{summary['replication_status']['replication_run_verdicts']}`"
                + ")"
            ),
        ),
        "",
        "## Source artifacts",
    ]

    for path in summary["generated_from"]:
        lines.append(f"- `{path}`")

    lines.extend(
        [
            "",
            "All fields above are extracted from committed JSON artifacts via `scripts/build_kalman_evidence_dashboard.py`.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary = _build_summary()
    markdown = _build_markdown(summary)

    with (OUTPUT_DIR / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")

    with (OUTPUT_DIR / "report.md").open("w", encoding="utf-8") as handle:
        handle.write(markdown)


if __name__ == "__main__":
    main()
