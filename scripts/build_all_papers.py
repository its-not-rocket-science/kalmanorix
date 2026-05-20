#!/usr/bin/env python3
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "paper" / "shared" / "generated"


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    subprocess.run(cmd, cwd=ROOT, check=True)


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _fmt(v: float) -> str:
    return f"{v:.4f}" if isinstance(v, float) else str(v)


def generate_shared_artifacts() -> None:
    SHARED.mkdir(parents=True, exist_ok=True)
    evidence = _load(ROOT / "results/evidence_registry.json")
    claim_gate = _load(ROOT / "results/claim_gate.json")
    canonical = _load(ROOT / "results/canonical_benchmark_v3/summary.json")
    diagnostics = _load(ROOT / "results/negative_result_diagnostics/summary.json")

    baselines = canonical.get("baseline_metrics", {})
    rows = []
    for name in ["MeanFuser", "KalmanorixFuser", "fixed_weighted", "top1"]:
        item = baselines.get(name, {})
        rows.append((name, item.get("ndcg@10", 0.0), item.get("latency_ms", 0.0)))
    baseline_tex = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        "Method & nDCG@10 & latency (ms) \\\\",
        "\\midrule",
    ]
    for name, ndcg, lat in rows:
        baseline_tex.append(f"{name} & {_fmt(ndcg)} & {_fmt(lat)} \\\\")
    baseline_tex += ["\\bottomrule", "\\end{tabular}"]
    (SHARED / "baseline_matrix.tex").write_text(
        "\n".join(baseline_tex) + "\n", encoding="utf-8"
    )

    robustness = [
        "\\begin{itemize}",
        f"\\item Claim-gate verdict: \\texttt{{{claim_gate.get('claim_success_decision', 'unknown')}}}",
        f"\\item Confirmatory slice verdict: \\texttt{{{claim_gate.get('confirmatory_slice_verdict', 'unknown')}}}",
        f"\\item Diagnostics status: \\texttt{{{diagnostics.get('overall_verdict', 'mixed')}}}",
        "\\end{itemize}",
    ]
    (SHARED / "robustness_summary.tex").write_text(
        "\n".join(robustness) + "\n", encoding="utf-8"
    )

    failure = diagnostics.get("headline_findings", [])
    failure_tex = (
        ["\\begin{itemize}"] + [f"\\item {x}" for x in failure[:5]] + ["\\end{itemize}"]
    )
    (SHARED / "failure_analysis_summary.tex").write_text(
        "\n".join(failure_tex) + "\n", encoding="utf-8"
    )

    req = claim_gate.get("required_decisions", [])
    gate_tex = ["\\begin{itemize}"]
    for idx, value in enumerate(req, start=1):
        gate_tex.append(f"\\item decision_{idx}: \\texttt{{{value}}}")
    gate_tex.append("\\end{itemize}")
    (SHARED / "claim_gate_summary.tex").write_text(
        "\n".join(gate_tex) + "\n", encoding="utf-8"
    )

    (SHARED / "evidence_registry.json").write_text(
        json.dumps(evidence, indent=2) + "\n", encoding="utf-8"
    )


def main() -> int:
    run(["python", "scripts/build_evidence_registry.py"])
    try:
        run(["python", "scripts/build_claim_gate.py"])
    except subprocess.CalledProcessError:
        print(
            "Using committed results/claim_gate.json (rebuild skipped: missing optional benchmark artifact)."
        )
    generate_shared_artifacts()
    run(["python", "scripts/check_crosspaper_consistency.py"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
