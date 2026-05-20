from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "results"

OUTPUT_MD = RESULTS_DIR / "failure_analysis_report.md"
OUTPUT_PDF = RESULTS_DIR / "failure_analysis_report.pdf"
OUTPUT_JSON = RESULTS_DIR / "failure_analysis_summary.json"

SOURCES = {
    "baseline_breadth": "results/canonical_benchmark_v2/summary.json",
    "uncertainty_ablations": "results/uncertainty_ablation/summary.json",
    "routing_vs_fusion": "routing_vs_fusion_decomposition.json",
    "specialist_diversity": "results/correlation_aware_fusion/summary.json",
    "robustness": "results/ood_robustness/summary.json",
    "candidate_budget": "results/candidate_budget_sweep/candidate_budget_sweep.json",
    "embedding_regimes": "results/negative_result_diagnostics/summary.json",
}


def _load_json(relative_path: str) -> dict[str, Any] | None:
    path = ROOT / relative_path
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(value: Any, digits: int = 4) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def _build_summary() -> dict[str, Any]:
    baseline = _load_json(SOURCES["baseline_breadth"]) or {}
    uncertainty = _load_json(SOURCES["uncertainty_ablations"]) or {}
    routing = _load_json(SOURCES["routing_vs_fusion"]) or {}
    diversity = _load_json(SOURCES["specialist_diversity"]) or {}
    robustness = _load_json(SOURCES["robustness"]) or {}
    budget = _load_json(SOURCES["candidate_budget"]) or {}
    embed = _load_json(SOURCES["embedding_regimes"]) or {}

    methods = baseline.get("methods", {})
    kalman_ndcg = (
        methods.get("kalman", {}).get("metrics", {}).get("ndcg@10", {}).get("mean")
    )
    mean_ndcg = (
        methods.get("uniform_mean_fusion", {})
        .get("metrics", {})
        .get("ndcg@10", {})
        .get("mean")
    )

    uncertainty_answer = uncertainty.get("answer", "Unavailable")
    corr_answer = diversity.get("answer", "Unavailable")
    robustness_verdict = robustness.get("rule_based_verdict", {})
    budget_rows = budget.get("results", [])
    budget_deltas = [
        r.get("delta_ndcg@10")
        for r in budget_rows
        if r.get("delta_ndcg@10") is not None
    ]
    budget_latency = [
        r.get("latency_ratio_kalman_over_mean")
        for r in budget_rows
        if r.get("latency_ratio_kalman_over_mean") is not None
    ]

    diagnostics = embed.get("diagnostics", {})
    reversal_requirements = embed.get("reversal_requirements", [])

    failed_statement = (
        "The claim that Kalman-style adaptive fusion should replace simpler fusion as default "
        "is unsupported in confirmatory evidence; observed effect sizes were near zero while "
        "latency overhead remained positive."
    )
    likely_why = [
        diagnostics.get(
            "effect_size_detectability", "No detectability note available."
        ),
        diagnostics.get(
            "baseline_competitiveness", "No baseline competitiveness note available."
        ),
        diagnostics.get(
            "uncertainty_calibration_impact",
            "No uncertainty calibration note available.",
        ),
    ]
    unsupported_assumptions = [
        "Assumption that uncertainty-aware weighting reliably increases retrieval quality.",
        "Assumption that additional adaptive fusion complexity is justified by measurable gains.",
        "Assumption that gains persist under confirmatory and robustness-oriented protocols.",
    ]
    promising_components = [
        "Candidate-budget sweep shows directional quality gains on tiny runs, suggesting some signal under limited conditions.",
        "Routing-oriented decompositions and specialist diversity analyses remain methodologically useful for diagnostics.",
        "Robustness artifact structure is in place, enabling reproducible follow-up falsification tests.",
    ]

    summary = {
        "generated_on": str(date.today()),
        "sources": SOURCES,
        "aggregates": {
            "baseline": {
                "num_methods": len(methods),
                "kalman_ndcg_at_10": kalman_ndcg,
                "uniform_mean_ndcg_at_10": mean_ndcg,
                "delta_ndcg_at_10": None
                if None in (kalman_ndcg, mean_ndcg)
                else kalman_ndcg - mean_ndcg,
            },
            "uncertainty_ablation_answer": uncertainty_answer,
            "routing_vs_fusion_rows": len(routing.get("rows", [])),
            "specialist_diversity_answer": corr_answer,
            "robustness_rule_based_verdict": robustness_verdict,
            "candidate_budget": {
                "num_budgets": len(budget_rows),
                "mean_delta_ndcg_at_10": sum(budget_deltas) / len(budget_deltas)
                if budget_deltas
                else None,
                "mean_latency_ratio_kalman_over_mean": sum(budget_latency)
                / len(budget_latency)
                if budget_latency
                else None,
            },
            "embedding_regime_diagnostics": diagnostics,
        },
        "answers": {
            "what_failed": failed_statement,
            "why_likely_failed": likely_why,
            "unsupported_assumptions": unsupported_assumptions,
            "promising_components": promising_components,
            "overturn_evidence_required": reversal_requirements,
        },
    }
    return summary


def _write_markdown(summary: dict[str, Any]) -> None:
    agg = summary["aggregates"]
    ans = summary["answers"]
    lines = [
        "# Failure Analysis Report",
        "",
        f"Generated on: {summary['generated_on']}",
        "",
        "Tone: empirical and restrained. This report aggregates available artifacts and avoids claim inflation.",
        "",
        "## Aggregated evidence coverage",
        f"- Baseline breadth: `{agg['baseline']['num_methods']}` methods in canonical benchmark v2.",
        f"- Uncertainty ablations: {agg['uncertainty_ablation_answer']}",
        f"- Routing-vs-fusion decomposition rows: `{agg['routing_vs_fusion_rows']}`.",
        f"- Specialist diversity diagnostics: {agg['specialist_diversity_answer']}",
        f"- Robustness studies: `{json.dumps(agg['robustness_rule_based_verdict'])}`",
        "- Candidate-budget sweeps: "
        f"{agg['candidate_budget']['num_budgets']} settings; mean ΔnDCG@10="
        f"{_fmt(agg['candidate_budget']['mean_delta_ndcg_at_10'])}, mean latency ratio="
        f"{_fmt(agg['candidate_budget']['mean_latency_ratio_kalman_over_mean'])}.",
        "- Embedding-regime comparisons: proxied through negative-result diagnostics and reversal criteria.",
        "",
        "## Core questions",
        "",
        "### 1) What exactly failed?",
        ans["what_failed"],
        "",
        "### 2) Why likely failed?",
    ]
    lines.extend([f"- {item}" for item in ans["why_likely_failed"]])
    lines += ["", "### 3) Which assumptions were unsupported?"]
    lines.extend([f"- {item}" for item in ans["unsupported_assumptions"]])
    lines += ["", "### 4) Which components still showed promise?"]
    lines.extend([f"- {item}" for item in ans["promising_components"]])
    lines += [
        "",
        "### 5) What evidence would be required to overturn the current conclusion?",
    ]
    lines.extend([f"- {item}" for item in ans["overturn_evidence_required"]])
    lines += ["", "## Source files", ""]
    lines.extend([f"- `{k}`: `{v}`" for k, v in summary["sources"].items()])
    OUTPUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _escape_pdf_text(text: str) -> str:
    return text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def _write_minimal_pdf(lines: list[str], out_path: Path) -> None:
    content = ["BT", "/F1 10 Tf", "50 790 Td", "12 TL"]
    for i, line in enumerate(lines[:55]):
        if i > 0:
            content.append("T*")
        content.append(f"({_escape_pdf_text(line[:120])}) Tj")
    content.append("ET")
    stream = "\n".join(content).encode("latin-1", errors="replace")

    objs: list[bytes] = []
    objs.append(b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n")
    objs.append(b"2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj\n")
    objs.append(
        b"3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Resources << /Font << /F1 4 0 R >> >> /Contents 5 0 R >> endobj\n"
    )
    objs.append(
        b"4 0 obj << /Type /Font /Subtype /Type1 /BaseFont /Helvetica >> endobj\n"
    )
    objs.append(
        f"5 0 obj << /Length {len(stream)} >> stream\n".encode("ascii")
        + stream
        + b"\nendstream endobj\n"
    )

    pdf = bytearray(b"%PDF-1.4\n")
    offsets = [0]
    for obj in objs:
        offsets.append(len(pdf))
        pdf.extend(obj)
    xref_pos = len(pdf)
    pdf.extend(f"xref\n0 {len(offsets)}\n".encode("ascii"))
    pdf.extend(b"0000000000 65535 f \n")
    for off in offsets[1:]:
        pdf.extend(f"{off:010d} 00000 n \n".encode("ascii"))
    pdf.extend(
        f"trailer << /Size {len(offsets)} /Root 1 0 R >>\nstartxref\n{xref_pos}\n%%EOF\n".encode(
            "ascii"
        )
    )
    out_path.write_bytes(pdf)


def main() -> None:
    summary = _build_summary()
    OUTPUT_JSON.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    _write_markdown(summary)
    md_lines = OUTPUT_MD.read_text(encoding="utf-8").splitlines()
    _write_minimal_pdf(md_lines, OUTPUT_PDF)
    print(f"Wrote {OUTPUT_MD}")
    print(f"Wrote {OUTPUT_PDF}")
    print(f"Wrote {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
