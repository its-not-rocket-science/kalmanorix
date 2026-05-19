# Publication bundle

Mode: full

## Generated files

- `results/evidence_registry.json`
- `paper/tmlr/tables/main_results.tex`
- `paper/tmlr/tables/statistical_tests.tex`
- `paper/tmlr/tables/baseline_matrix.tex`
- `paper/tmlr/includes/generated_metrics.tex`
- `paper/arxiv/tables/main_results.tex`
- `paper/arxiv/tables/statistical_tests.tex`
- `paper/joss/results_summary.md`
- `paper/tmlr/main.pdf`
- `paper/arxiv/kalmanorix_negative_result.pdf`

## Source artifacts

- `results/canonical_benchmark_v3_fast_1193_c100_balanced/summary.json`
- `results/canonical_benchmark_v3_fast_1193_c100_balanced/report.md`

## Known limitations

- TMLR builds require `paper/tmlr/tmlr.sty` and `paper/tmlr/tmlr.bst` to be available.
- PDF builds require `pdflatex` and `bibtex` in PATH.
- Fast mode uses committed artifacts and skips expensive benchmark regeneration.
- Confirmatory slice is underpowered (n_pairs=1 < 20 minimum); slice verdict is inconclusive, not negative.

## Submission readiness checklist

### Automated (verified by CI / pre-push hooks)

- [x] Evidence registry rebuilt from current committed artifacts.
- [x] Publication claim consistency check passes.
- [x] LaTeX tables regenerated from `canonical_benchmark_v3_fast_1193_c100_balanced`.
- [x] TMLR PDF builds successfully.
- [x] arXiv PDF builds successfully.
- [x] JOSS example invocation (`python examples/minimal_fusion_demo.py`) runs without error.
- [x] All paper files reference `results/evidence_registry.json`.
- [x] Data consistency: all three venues use the same benchmark run.

### arXiv

- [ ] Replace placeholder author `Kalmanorix Project Team` in `paper/arxiv/kalmanorix_negative_result.tex` with real author name(s).
- [ ] Add affiliation(s) and optional ORCID(s) to arXiv submission metadata (entered in the arXiv web form, not the .tex file).
- [ ] Confirm no anonymisation required (arXiv preprints are non-anonymous).
- [ ] Upload source bundle: `.tex`, `.bib`, and `tables/` directory.
- [ ] Verify compiled PDF renders correctly after arXiv processing.

### TMLR

- [ ] Confirm `\author{Paul Schleifer}` and affiliation in `paper/tmlr/main.tex` are final.
- [ ] Add ORCID to TMLR submission portal author fields (not in the .tex file).
- [ ] Write cover letter: state the contribution is a methodology paper (claim-gated evaluation protocol), not a positive empirical result; reference the negative finding explicitly.
- [ ] Confirm submission is non-anonymous (TMLR uses open review).
- [ ] Verify all `\input{}` paths resolve in the TMLR submission bundle.
- [ ] Check references compile cleanly (`bibtex main` produces no missing-citation warnings).

### JOSS

- [ ] Add ORCID to `paper/joss/paper.md` YAML front matter under the author entry (`orcid: 0000-0000-0000-0000`).
- [ ] Confirm `date:` field in front matter is the intended submission date.
- [ ] Create a tagged release (e.g. `v1.0.0`) on GitHub corresponding to the submitted version.
- [ ] Archive that release on Zenodo and add the resulting DOI to `paper/joss/paper.md` front matter.
- [ ] Ensure public issue tracker is open and usable by reviewers.
- [ ] Verify AI usage disclosure section in `paper/joss/paper.md` is accurate and complete.
- [ ] Confirm manuscript remains software-focused (no new empirical claims beyond what is in the evidence registry).

### All venues

- [ ] Run `python scripts/build_publication_bundle.py` and confirm exit 0.
- [ ] Manually read all three compiled PDFs end-to-end before submitting.
