# JOSS checklist status (Kalmanorix)

This checklist tracks common JOSS software-paper requirements and current repository status as of 2026-05-11.

| Requirement | Status | Evidence |
|---|---|---|
| OSS license present | ✅ Complete | `LICENSE` (MIT license text). |
| `CITATION.cff` present | ✅ Complete | `CITATION.cff` added at repository root. |
| Software-focused paper (`paper/joss/paper.md`) | ✅ Complete | Includes Summary, Statement of need, Core functionality, Example usage, Reproducibility and testing, Relationship to research papers, References. |
| Bibliography present (`paper/joss/paper.bib`) | ✅ Complete | JOSS bibliography file exists with core IR/evaluation citations. |
| Installation instructions | ✅ Complete | `docs/getting-started/installation.md`. |
| Minimal example | ✅ Complete | `docs/examples/minimal-fusion.md` and `paper/joss/paper.md` example usage snippet. |
| API documentation link | ✅ Complete | `docs/index.md` links to API Reference (`docs/api-reference/*`). |
| Automated tests documented | ✅ Complete | `docs/contributing/testing.md`; CI-critical command: `pytest -m "not integration and not stress"`. |
| Archived release instructions | ⚠️ In progress | `paper/joss/README.md` documents release + Zenodo archiving requirement; final DOI and archived release record must be added at submission time. |
| Public issue tracker guidance | ⚠️ In progress | Repository-level operational check required prior to submission. |

## Submission-time actions still required

1. Create a tagged release matching the submitted version.
2. Archive the release (e.g., Zenodo) and update manuscript metadata with DOI.
3. Confirm issue tracker availability for reviewer interaction.
