# JOSS paper materials for Kalmanorix

> Evidence registry source of truth: `results/evidence_registry.json` (generated via `PYTHONPATH=src python scripts/build_evidence_registry.py`).

This directory contains the active JOSS submission materials for Kalmanorix research software:

- `paper.md` — JOSS-format manuscript with YAML metadata and section structure.
- `paper.bib` — bibliography used by the manuscript.

## Scope note (submission-facing)

This draft is intentionally software-centric. It documents Kalmanorix as public MIT-licensed research software for specialist embedding fusion, uncertainty-aware retrieval experiments, routing evaluation, benchmark governance, and reproducible evidence tracking. Empirical findings are documented separately in repository experiment artefacts and should not be the focus of the JOSS submission.

## JOSS-readiness caveats checklist

Before submission, verify all of the following:

1. Check documentation completeness (installation, usage, architecture, contribution, and governance pages).
2. Check test coverage and CI status for core workflows and CLI pathways.
3. Create a tagged release that corresponds to the submitted software version.
4. Archive that release on Zenodo (or similar) and add the DOI to the paper metadata before submission.
5. Ensure the public issue tracker is enabled and actively usable by reviewers.
6. Add an explicit AI usage disclosure in `paper.md` with accurate details.
7. Confirm the manuscript remains software-focused and does not centre on new research results.

## Archival/non-submission materials

Any manuscripts outside `paper/joss/` (for example `paper/paper.md`) are archival drafts and are **not** part of the JOSS submission package.
