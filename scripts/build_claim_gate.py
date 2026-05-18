#!/usr/bin/env python3
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = REPO / "results/canonical_benchmark_v3_fast_1193_c100_balanced"


def build_claim_gate(summary: dict) -> dict:
    decision = summary["decision"]["kalman_vs_mean"]
    bm_status = summary["benchmark_status"]["status"]
    cs_verdict = summary["confirmatory_slice_results"]["decision"]["verdict"]
    csd = summary["claim_success_decision"]

    return {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifact_source": str(ARTIFACT_DIR.relative_to(REPO) / "summary.json"),
        "benchmark_status": bm_status,
        "confirmatory_slice_verdict": cs_verdict,
        "claim_success_decision": csd["status"],
        "required_decisions": csd["required_decisions"],
        "primary_comparison": {
            "verdict": decision["verdict"],
            "rules": decision["rules"],
            "observed": decision["observed"],
            "checks": decision["checks"],
        },
    }


def main() -> int:
    summary_path = ARTIFACT_DIR / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing artifact: {summary_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    gate = build_claim_gate(summary)

    out = REPO / "results/claim_gate.json"
    out.write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
