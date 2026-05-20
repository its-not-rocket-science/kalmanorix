# Cross-paper consistency report

This report describes the synchronization pipeline that keeps TMLR, arXiv, and JOSS aligned to shared evidence artefacts.

## Shared generated artefacts

All venue manuscripts now consume generated files under `paper/shared/generated/`:

- `baseline_matrix.tex`
- `robustness_summary.tex`
- `failure_analysis_summary.tex`
- `claim_gate_summary.tex`
- `evidence_registry.json`

Generation entrypoint: `python scripts/build_all_papers.py`.
Validation entrypoint: `python scripts/check_crosspaper_consistency.py`.

## Venue positioning guardrails

- **TMLR**: full methodological interpretation and claim-gate framing.
- **arXiv**: concise empirical negative-result interpretation.
- **JOSS**: software-capability and infrastructure scope only.

## Automated checks

The consistency script enforces:

1. Presence of all shared generated artefacts.
2. Headline metric synchronization (Δ nDCG@10 and adjusted p-value) between TMLR/arXiv.
3. Unsupported claim language detection.
4. Coarse contradiction detection between TMLR and arXiv polarity.
5. JOSS scope protection (software-only positioning).
6. Baseline table source unification through shared generated baseline matrix.
