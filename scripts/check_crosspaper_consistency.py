#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SHARED = ROOT / "paper/shared/generated"
FILES = {
    "tmlr": ROOT / "paper/tmlr/main.tex",
    "arxiv": ROOT / "paper/arxiv/kalmanorix_negative_result.tex",
    "joss": ROOT / "paper/joss/paper.md",
}


def main() -> int:
    errors: list[str] = []
    shared_needed = [
        "baseline_matrix.tex",
        "robustness_summary.tex",
        "failure_analysis_summary.tex",
        "claim_gate_summary.tex",
        "evidence_registry.json",
    ]
    for name in shared_needed:
        if not (SHARED / name).exists():
            errors.append(f"Missing shared artefact: paper/shared/generated/{name}")

    claim_gate = json.loads((ROOT / "results/claim_gate.json").read_text())
    observed = claim_gate["primary_comparison"]["observed"]
    delta = str(observed["primary_metric_delta"])
    pval = str(observed["primary_metric_adjusted_p_value"])

    texts = {k: p.read_text(encoding="utf-8") for k, p in FILES.items()}

    for venue in ("tmlr", "arxiv"):
        text = texts[venue]
        has_delta = (
            delta in text
            or "-9.258801070226193\\times10^{-6}" in text
            or "-9.259\\times10^{-6}" in text
            or "\\DeltaNDCG" in text
        )
        if not has_delta:
            errors.append(f"Headline number mismatch: delta {delta} missing in {venue}")
        has_pval = pval in text or "\\HolmP" in text
        if not has_pval:
            errors.append(
                f"Headline number mismatch: p-value {pval} missing in {venue}"
            )

    unsupported = [
        "outperforms mean fusion",
        "superior to monolith",
        "state-of-the-art",
    ]
    for venue, text in texts.items():
        tl = text.lower()
        for term in unsupported:
            if (
                term in tl
                and "not"
                not in tl[max(0, tl.index(term) - 40) : tl.index(term) + len(term) + 40]
            ):
                errors.append(f"Unsupported claim wording in {venue}: '{term}'")

    tmlr_lower = texts["tmlr"].lower()
    arxiv_lower = texts["arxiv"].lower()
    if ("no reliable" in arxiv_lower) != ("no demonstrated reliable" in tmlr_lower):
        errors.append("Potential contradiction between TMLR/arXiv claim polarity")

    if (
        "software" not in texts["joss"].lower()
        or "negative result" in texts["joss"].lower()
    ):
        errors.append(
            "JOSS must stay software-capability scoped and avoid empirical positioning"
        )

    for venue, text in texts.items():
        if (
            "../shared/generated/baseline_matrix" not in text
            and "paper/shared/generated/baseline_matrix.tex" not in text
        ):
            errors.append(
                f"{venue} does not reference shared baseline_matrix.tex source"
            )

    if errors:
        for e in errors:
            print(f"ERROR: {e}")
        return 1
    print("Cross-paper consistency checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
