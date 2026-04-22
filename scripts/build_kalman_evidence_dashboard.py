from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "results" / "kalman_evidence_dashboard"
CANONICAL_V3_PATH = "results/canonical_benchmark_v3/summary.json"
CLAIM_DECISION_PATH = "results/kalman_latency_optimization/canonical/summary.json"


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


def _derive_claim_ready_support(
    *,
    canonical_v3_status: str,
    canonical_v3_verdict: str,
    confirmatory_verdict: str,
    kalman_vs_mean_verdict: str,
    kalman_vs_weighted_mean_verdict: str,
    kalman_vs_router_only_top1_verdict: str,
    latency_gate_ok: bool,
    replication_status: str,
) -> str:
    confirmatory_present = confirmatory_verdict != "missing_confirmatory_evidence"
    if not confirmatory_present:
        return "no"
    all_supported = (
        canonical_v3_verdict == "supported"
        and confirmatory_verdict == "supported"
        and kalman_vs_mean_verdict == "supported"
        and kalman_vs_weighted_mean_verdict == "supported"
        and kalman_vs_router_only_top1_verdict == "supported"
        and latency_gate_ok
        and replication_status == "replicated_supported"
    )
    if all_supported:
        return "yes"
    if canonical_v3_status == "placeholder_pending_run":
        return "not_yet"
    return "no"


def _build_summary() -> dict[str, Any]:
    canonical_v3 = _load_json(CANONICAL_V3_PATH)
    canonical = _load_json(CLAIM_DECISION_PATH)
    uncertainty_ablation = _load_json("results/uncertainty_ablation/summary.json")
    replication = canonical.get("replication")
    replication_sources = [CLAIM_DECISION_PATH]
    if replication is None:
        latency_rerun = _load_json(
            "results/kalman_latency_optimization/canonical/summary.json"
        )
        replication = {
            "per_run_verdicts": [
                {"verdict": latency_rerun["decision"]["kalman_vs_mean"]["verdict"]}
            ]
        }
        replication_sources.append(CLAIM_DECISION_PATH)

    canonical_v3_status = str(canonical_v3.get("status", "unknown"))
    canonical_v3_verdict = str(
        canonical_v3.get("decision", {})
        .get("kalman_vs_mean", {})
        .get("verdict", "not_available_placeholder_pending_run")
    )

    canonical_decision = canonical["decision"]["kalman_vs_mean"]
    canonical_verdict = canonical_decision["verdict"]
    canonical_light = _traffic_light_from_verdict(canonical_verdict)

    confirmatory = canonical_v3.get("confirmatory_slice_results")
    if confirmatory is None:
        confirmatory_verdict = "missing_confirmatory_evidence"
        confirmatory_light = "red"
        confirmatory_source_path = CANONICAL_V3_PATH
        confirmatory_source_json_path = "$.confirmatory_slice_results"
    else:
        confirmatory_verdict = confirmatory.get("decision", {}).get(
            "verdict", "unresolved"
        )
        confirmatory_light = _traffic_light_from_verdict(confirmatory_verdict)
        confirmatory_source_path = CANONICAL_V3_PATH
        confirmatory_source_json_path = "$.confirmatory_slice_results.decision.verdict"

    kalman_vs_mean_verdict = canonical["decision"]["kalman_vs_mean"]["verdict"]
    kalman_vs_weighted_mean_verdict = canonical["decision"]["kalman_vs_weighted_mean"][
        "verdict"
    ]
    kalman_vs_router_only_top1_verdict = canonical["decision"][
        "kalman_vs_router_only_top1"
    ]["verdict"]

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

    claim_ready_support = _derive_claim_ready_support(
        canonical_v3_status=canonical_v3_status,
        canonical_v3_verdict=canonical_v3_verdict,
        confirmatory_verdict=confirmatory_verdict,
        kalman_vs_mean_verdict=kalman_vs_mean_verdict,
        kalman_vs_weighted_mean_verdict=kalman_vs_weighted_mean_verdict,
        kalman_vs_router_only_top1_verdict=kalman_vs_router_only_top1_verdict,
        latency_gate_ok=latency_ok,
        replication_status=replication_value,
    )
    claim_ready_light = (
        "green"
        if claim_ready_support == "yes"
        else "red"
        if claim_ready_support == "no"
        else "yellow"
    )

    claim_reasons: list[str] = []
    if canonical_v3_status != "completed":
        claim_reasons.append(
            f"canonical v3 benchmark status is `{canonical_v3_status}`."
        )
    if canonical_v3_verdict != "supported":
        claim_reasons.append(f"canonical v3 verdict is `{canonical_v3_verdict}`.")
    if confirmatory_verdict != "supported":
        claim_reasons.append(f"confirmatory slice verdict is `{confirmatory_verdict}`.")
    if kalman_vs_mean_verdict != "supported":
        claim_reasons.append(f"kalman_vs_mean verdict is `{kalman_vs_mean_verdict}`.")
    if kalman_vs_weighted_mean_verdict != "supported":
        claim_reasons.append(
            f"kalman_vs_weighted_mean verdict is `{kalman_vs_weighted_mean_verdict}`."
        )
    if kalman_vs_router_only_top1_verdict != "supported":
        claim_reasons.append(
            "kalman_vs_router_only_top1 verdict is "
            f"`{kalman_vs_router_only_top1_verdict}`."
        )
    if not latency_ok:
        claim_reasons.append(
            "latency gate failed with "
            f"ratio `{latency_ratio:.3f}` over threshold `{latency_threshold:.3f}`."
        )
    if replication_value != "replicated_supported":
        claim_reasons.append(f"replication status is `{replication_value}`.")
    if not claim_reasons:
        claim_reasons.append("all required evidence gates are satisfied.")

    return {
        "artifact": "kalman_evidence_dashboard.v3",
        "generated_from": list(
            dict.fromkeys(
                [
                    CANONICAL_V3_PATH,
                    CLAIM_DECISION_PATH,
                    "results/uncertainty_ablation/summary.json",
                    *replication_sources[1:],
                ]
            )
        ),
        "traffic_light_legend": {
            "green": "supported",
            "yellow": "unresolved",
            "red": "unsupported in tested regime",
        },
        "claim_ready_support": {
            "traffic_light": claim_ready_light,
            "status": claim_ready_support,
            "required_confirmatory_evidence_present": (
                confirmatory_verdict != "missing_confirmatory_evidence"
            ),
        },
        "canonical_v3_benchmark_status": {
            "traffic_light": (
                "green" if canonical_v3_status == "completed" else "yellow"
            ),
            "status": SourcedValue(
                value=canonical_v3_status,
                source_path=CANONICAL_V3_PATH,
                source_json_path="$.status",
            ).as_dict(),
        },
        "canonical_v3_verdict": {
            "traffic_light": _traffic_light_from_verdict(canonical_v3_verdict),
            "verdict": SourcedValue(
                value=canonical_v3_verdict,
                source_path=CANONICAL_V3_PATH,
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
        "kalman_vs_mean": {
            "traffic_light": _traffic_light_from_verdict(kalman_vs_mean_verdict),
            "verdict": SourcedValue(
                value=kalman_vs_mean_verdict,
                source_path=CLAIM_DECISION_PATH,
                source_json_path="$.decision.kalman_vs_mean.verdict",
            ).as_dict(),
        },
        "kalman_vs_weighted_mean": {
            "traffic_light": _traffic_light_from_verdict(
                kalman_vs_weighted_mean_verdict
            ),
            "verdict": SourcedValue(
                value=kalman_vs_weighted_mean_verdict,
                source_path=CLAIM_DECISION_PATH,
                source_json_path="$.decision.kalman_vs_weighted_mean.verdict",
            ).as_dict(),
        },
        "kalman_vs_router_only_top1": {
            "traffic_light": _traffic_light_from_verdict(
                kalman_vs_router_only_top1_verdict
            ),
            "verdict": SourcedValue(
                value=kalman_vs_router_only_top1_verdict,
                source_path=CLAIM_DECISION_PATH,
                source_json_path="$.decision.kalman_vs_router_only_top1.verdict",
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
        "latency_gate_status": {
            "traffic_light": latency_light,
            "ratio_vs_mean": SourcedValue(
                value=latency_ratio,
                source_path=CLAIM_DECISION_PATH,
                source_json_path="$.decision.kalman_vs_mean.observed.latency_ratio_vs_mean",
            ).as_dict(),
            "max_allowed_ratio": SourcedValue(
                value=latency_threshold,
                source_path=CLAIM_DECISION_PATH,
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
        "why_claim_ready_or_not": {
            "title": "Why the repo can / cannot currently claim Kalman beats mean",
            "claim_ready_support": claim_ready_support,
            "reasons": claim_reasons,
        },
    }


def _build_markdown(summary: dict[str, Any]) -> str:
    legend = summary["traffic_light_legend"]

    def row(label: str, section: dict[str, Any], detail: str) -> str:
        return f"| {label} | {section['traffic_light']} | {detail} |"

    lines = [
        "# Kalman-vs-Mean Evidence Dashboard",
        "",
        ("**Claim-ready support:** " + f"`{summary['claim_ready_support']['status']}`"),
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
            "Canonical v3 benchmark status",
            summary["canonical_v3_benchmark_status"],
            (
                "`"
                + summary["canonical_v3_benchmark_status"]["status"]["value"]
                + "` from "
                + f"`{summary['canonical_v3_benchmark_status']['status']['source']['path']}`"
            ),
        ),
        row(
            "Canonical v3 verdict",
            summary["canonical_v3_verdict"],
            (
                "`"
                + summary["canonical_v3_verdict"]["verdict"]["value"]
                + "` from "
                + f"`{summary['canonical_v3_verdict']['verdict']['source']['path']}`"
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
            "Kalman vs mean",
            summary["kalman_vs_mean"],
            (
                "`"
                + summary["kalman_vs_mean"]["verdict"]["value"]
                + "` from "
                + f"`{summary['kalman_vs_mean']['verdict']['source']['path']}`"
            ),
        ),
        row(
            "Kalman vs weighted mean",
            summary["kalman_vs_weighted_mean"],
            (
                "`"
                + summary["kalman_vs_weighted_mean"]["verdict"]["value"]
                + "` from "
                + f"`{summary['kalman_vs_weighted_mean']['verdict']['source']['path']}`"
            ),
        ),
        row(
            "Kalman vs router only top1",
            summary["kalman_vs_router_only_top1"],
            (
                "`"
                + summary["kalman_vs_router_only_top1"]["verdict"]["value"]
                + "` from "
                + f"`{summary['kalman_vs_router_only_top1']['verdict']['source']['path']}`"
            ),
        ),
        row(
            "Uncertainty ablation result",
            summary["uncertainty_ablation_result"],
            summary["uncertainty_ablation_result"]["answer"]["value"],
        ),
        row(
            "Latency gate status",
            summary["latency_gate_status"],
            (
                "Kalman/Mean="
                + f"{summary['latency_gate_status']['ratio_vs_mean']['value']:.3f}"
                + " vs threshold="
                + f"{summary['latency_gate_status']['max_allowed_ratio']['value']:.3f}"
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
        "## Why the repo can / cannot currently claim Kalman beats mean",
    ]
    for reason in summary["why_claim_ready_or_not"]["reasons"]:
        lines.append(f"- {reason}")

    lines.extend(
        [
            "",
            "## Source artifacts",
        ]
    )

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
