# Zenodo + JOSS release checklist

## Pre-release

- [ ] Bump version in `pyproject.toml` and changelog/release notes.
- [ ] Run format/lint/tests locally, including smoke e2e path.
- [ ] Verify package install path from clean environment:
  - `pip install -e ".[dev,benchmark]"`
- [ ] Confirm documented imports execute without import errors.

## JOSS paper package

- [ ] Update `paper/joss/paper.md` metadata and summary if needed.
- [ ] Ensure bibliography compiles and references resolve.
- [ ] Re-check installation and minimal usage instructions for reviewers.

## Zenodo archival

- [ ] Create GitHub release tag.
- [ ] Ensure `.zenodo.json` metadata (if present) is current.
- [ ] Wait for Zenodo archive generation and note DOI.
- [ ] Add DOI badge/links in README and JOSS materials.

## Final verification

- [ ] Re-run CI-critical checks.
- [ ] Record exact commit SHA used for release and paper.
- [ ] Save artifact hashes/paths for reproducibility notes.
