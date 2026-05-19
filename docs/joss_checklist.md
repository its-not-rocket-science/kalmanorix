# JOSS checklist status (Kalmanorix)

This checklist maps common JOSS expectations to concrete repository evidence and next actions.

| JOSS expectation | Status | Repository path | Action needed |
|---|---|---|---|
| Open-source licence is present and clear | ready | `LICENSE` | None. |
| Software citation metadata exists (`CITATION.cff`) | ready | `CITATION.cff` | Keep version/date aligned with next tagged release. |
| JOSS paper exists and focuses on software contribution | ready | `paper/joss/paper.md` | None. |
| Bibliography and references are included for paper claims | ready | `paper/joss/paper.bib` | None. |
| Installation instructions include editable install with required extras | ready | `paper/joss/paper.md`; `docs/getting-started/installation.md` | Keep command exactly `pip install -e ".[dev,benchmark]"` in docs and paper. |
| Minimal smoke-test command is documented | ready | `paper/joss/paper.md`; `tests/e2e/test_toy_pipeline.py` | Keep command `python -m pytest tests/e2e/test_toy_pipeline.py` documented and passing. |
| Full test command is documented | ready | `paper/joss/paper.md` | Keep command `python -m pytest` documented. |
| Documentation pointer is explicit in paper | ready | `paper/joss/paper.md`; `docs/index.md` | None. |
| Example/quickstart pointer is explicit in paper | ready | `paper/joss/paper.md`; `docs/getting-started/quickstart.md`; `docs/examples/minimal-fusion.md` | None. |
| Licence statement appears in paper | ready | `paper/joss/paper.md`; `LICENSE` | None. |
| CI status statement included | partial | `paper/joss/paper.md` | Confirm/update with explicit CI badge or workflow URL before submission. |
| Archive/release DOI plan documented (Zenodo or equivalent) | partial | `paper/joss/paper.md`; `paper/joss/README.md` | Mint/archive DOI at release time and add DOI to paper metadata. |
| Version number is explicit and consistent | partial | `pyproject.toml`; `CITATION.cff` | Reconcile version mismatch (`0.2.0` vs `0.1.0`) before submission. |
| Authors and affiliations are complete in manuscript metadata | ready | `paper/joss/paper.md` | Add co-authors/affiliations if authorship changes before submission. |
| Public issue tracker is available for reviewer interaction | partial | `README.md` (or repository settings) | Verify issue tracker is enabled and link is visible in repository front matter/docs. |

## Pre-submission TODO summary

1. Align release version metadata across `pyproject.toml`, `CITATION.cff`, and release tag.
2. Create tagged release and archive it (e.g., Zenodo), then include the resulting DOI in JOSS submission metadata.
3. Add a definitive CI status reference (badge and/or workflow URL) once workflow target branch is finalized.

## Reviewer-facing manuscript risk note

Remaining risks for `paper/joss/paper.md` before submission:

- **Evidence-registry alignment risk:** statements about supported/unsupported/inconclusive effects should be cross-checked against `results/evidence_registry.json` at release cut time to avoid stale claim wording.
- **External-stack claim risk:** references to SentenceTransformers/BEIR/FAISS integration should remain descriptive unless benchmark logs in the evidence registry explicitly demonstrate those paths in the tagged release.
- **CI wording risk:** the manuscript names local CI gates; add a concrete workflow URL or badge before final submission so reviewers can verify automation status quickly.
- **Version/DOI consistency risk:** ensure manuscript metadata, `pyproject.toml`, `CITATION.cff`, and release DOI metadata agree on version and release date.
