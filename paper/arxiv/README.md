# Kalmanorix negative-result preprint (arXiv package)

> Evidence registry source of truth: `results/evidence_registry.json` (generated via `PYTHONPATH=src python scripts/build_evidence_registry.py`).

This directory contains an arXiv-ready UK-English preprint source for the canonical negative result:
Kalman fusion did not beat mean fusion on the claim-ready mixed-domain retrieval benchmark, while hard routing was the strongest observed baseline.

## Files
- `kalmanorix_negative_result.tex` — main LaTeX source (article class, arXiv-compatible).
- `references.bib` — BibTeX database used by the paper.

## Build locally
Example:

```bash
cd paper/arxiv
pdflatex kalmanorix_negative_result.tex
bibtex kalmanorix_negative_result
pdflatex kalmanorix_negative_result.tex
pdflatex kalmanorix_negative_result.tex
```

## arXiv submission note
arXiv uploads need:
- the `.tex` source,
- the `.bib` file or generated `.bbl`,
- all figure files,
- and any style/class files if custom styles are used.

This preprint currently uses standard packages expected in typical arXiv LaTeX environments.
