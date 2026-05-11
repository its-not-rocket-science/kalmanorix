# TMLR Draft (Anonymous by Default)

> Evidence registry source of truth: `results/evidence_registry.json` (generated via `PYTHONPATH=src python scripts/build_evidence_registry.py`).

This directory contains a **TMLR-style LaTeX submission draft** for the Kalmanorix negative empirical study.

## Files

- `main.tex`: Main manuscript source using the TMLR style.
- `references.bib`: BibTeX references (real, published sources only).
- `tables/main_results.tex`: Core result table used in the main text.
- `tables/decision_rules.tex`: Claim-gating and decision-rule table.

## Template requirements

TMLR submissions should use the official template assets:

- `tmlr.sty`
- `tmlr.bst`

Place these files in this directory (or another LaTeX-visible path) before building.

TMLR uses OpenReview and double-blind review. This draft is set up to stay anonymous by default.

## Build (anonymous submission mode)

From `paper/tmlr/`:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Switching to public preprint mode

If you want to post a non-anonymous preprint publicly, change:

```tex
\usepackage{tmlr}
```

to:

```tex
\usepackage[preprint]{tmlr}
```

Then add explicit author metadata in `main.tex` as permitted by the public version workflow.

## Caution on evidence readiness

This manuscript explicitly labels itself a **TMLR draft**. The present evidence is likely not submission-ready until replicated with at least one:

1. non-fast-local neural embedding run, or
2. external benchmark configuration.
