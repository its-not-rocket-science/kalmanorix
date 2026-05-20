# arXiv Submission Readiness

Date: 2026-05-19 (UTC)

## 1. Build status

- **LaTeX build status:** **blocked in this environment** (`pdflatex` not installed).
- **Bibliography build status:** **blocked in this environment** (`bibtex` not installed).
- Source manuscript and bibliography files are present:
  - `paper/arxiv/kalmanorix_negative_result.tex`
  - `paper/arxiv/references.bib`

Commands run:

```bash
cd paper/arxiv
pdflatex -interaction=nonstopmode kalmanorix_negative_result.tex
bibtex kalmanorix_negative_result
```

Observed output:

- `/bin/bash: pdflatex: command not found`
- `/bin/bash: bibtex: command not found`

## 2. Missing references

Automated cite-key scan of `paper/arxiv/kalmanorix_negative_result.tex` against `paper/arxiv/references.bib` found:

- `total_cites = 9`
- `unique_cites = 9`
- `missing_keys = []`

**Assessment:** no missing BibTeX keys detected by static scan.

## 3. Placeholder checks

Static placeholder/token scan (`TODO`, `TBD`, `XXX`, `??`, `placeholder`, empty `\\cite{}`) over `paper/arxiv/` returned no matches.

**Assessment:** no obvious placeholder text remains in the arXiv manuscript sources.

## 4. UK English consistency

Targeted spelling scan for common US/UK divergences (e.g., `color`, `behavior`, `center`, `modeling`) returned no hits in `paper/arxiv/kalmanorix_negative_result.tex`.

**Assessment:** no obvious US-spelling inconsistencies were found in the scanned terms; full language-edit pass still recommended before upload.

## 5. Evidence-registry consistency

Publication-wide claim/evidence consistency check passes.

Command run:

```bash
python scripts/check_publication_consistency.py
```

Result:

- `Publication consistency checks passed.`

Additional claim-gate validation also passes for arXiv, TMLR, and JOSS manuscripts.

Command run:

```bash
python scripts/validate_publication_claims.py --claim-gate results/claim_gate.json paper/arxiv/kalmanorix_negative_result.tex paper/tmlr/main.tex paper/joss/paper.md
```

Result:

- JSON status: `"pass"`

## 6. Unsupported-claim scan

No unsupported-claim violations were reported by `scripts/check_publication_consistency.py`.

**Assessment:** unsupported claim phrases appear to be controlled relative to current `results/evidence_registry.json` / `results/claim_gate.json` statuses.

## 7. Remaining LLM-writing risks

Residual risks before upload:

1. **Build-unverified rendering risk**: without local TeX toolchain, final PDF layout, references, and line breaks remain unverified.
2. **Numeric drift risk**: hard-coded metrics in manuscript can drift if `results/` artefacts are regenerated post-edit.
3. **Scope-claim wording risk**: although automated checks pass, nuanced phrasing could still be interpreted as broader generalisation than intended.
4. **Version pinning risk**: manuscript date and artefact paths should be rechecked against the exact uploaded commit/tag.

## 8. Distinction from TMLR/JOSS

The arXiv manuscript explicitly states role separation:

- **TMLR**: methodology paper (protocol + claim-gated framework)
- **arXiv**: empirical negative-result report for one benchmark setup
- **JOSS**: software/infrastructure paper

Automated consistency check includes explicit cross-venue distinction checks and currently passes.

## 9. Recommendation

- **Current recommendation:** **revise before upload**.
- Rationale: scientific/claim consistency checks pass, but TeX/Bib build verification is blocked in this environment due to missing `pdflatex` and `bibtex`. arXiv readiness should only be marked “ready for arXiv” after a successful local (or CI) PDF+Bib build and visual inspection of the produced PDF.

---

## Commands used (audit trail)

### LaTeX build

```bash
cd paper/arxiv
pdflatex -interaction=nonstopmode kalmanorix_negative_result.tex
```

### Bibliography build

```bash
cd paper/arxiv
bibtex kalmanorix_negative_result
```

### Claim consistency checks

```bash
python scripts/check_publication_consistency.py
python scripts/validate_publication_claims.py --claim-gate results/claim_gate.json paper/arxiv/kalmanorix_negative_result.tex paper/tmlr/main.tex paper/joss/paper.md
```
