---
title: "Kalmanorix: A Python toolkit for specialist-embedding routing, uncertainty-aware fusion, and claim-gated retrieval evaluation"
tags:
  - Python
  - information retrieval
  - embeddings
  - uncertainty quantification
  - evaluation
  - reproducibility
authors:
  - name: "Paul Schleifer"
    orcid: "0009-0004-7972-3566"
    affiliation: "1"
affiliations:
  - name: "Independent Research Software Engineer, United Kingdom"
    index: 1
date: "2026-05-19"
bibliography: paper.bib
---

# Summary

Kalmanorix is an MIT-licensed Python package for retrieval benchmarks with specialist embedding models. It defines specialist embedders (`SEF`), groups them in a `Village`, routes queries with `ScoutRouter`, and compares fusion baselines through `Panoramix`. It also writes claim-gated outputs that label findings as supported, unsupported, or inconclusive in machine-readable files.

# Statement of need

Researchers evaluating specialist embedding systems need to compare routing, fusion, uncertainty estimation, and baseline methods under one protocol. Existing tools cover parts of that workflow: SentenceTransformers for embedding models, BEIR for retrieval benchmarks, FAISS for efficient vector search, and statistical testing guidance in separate evaluation literature [@reimers2019sentencebert; @thakur2021beir; @johnson2017faiss; @dror2017replicability; @smucker2007significance; @demsar2006statistical].

Kalmanorix fills this integration gap with reusable software for packaging specialists, routing queries to specialists, comparing fusion rules, generating claim-gated evidence reports, and preserving negative and null outcomes.

This paper positions Kalmanorix as research software infrastructure in the JOSS sense: software-first scholarship with auditable interfaces, artifacts, and governance that support companion empirical studies [@katz2018joss].

# Core functionality

Kalmanorix provides five core features:

1. Packages specialist embedders with explicit configuration for multi-domain retrieval runs.
2. Evaluates query-to-specialist routing, including slice-level reporting.
3. Runs baseline comparisons for single-model, weighted, and uncertainty-aware fusion.
4. Applies rule-based claim gating to metric outputs.
5. Exports deterministic JSON/Markdown artifacts for audit and paper appendices.

# Installation, quickstart, and verification

Install the package with development and benchmark extras:

```bash
pip install -e ".[dev,benchmark]"
```

Minimal smoke test (fast path used in JOSS readiness checks):

```bash
python -m pytest tests/e2e/test_toy_pipeline.py
```

Full test suite:

```bash
python -m pytest
```

For setup details, see `docs/getting-started/installation.md`. For a runnable example, see `docs/examples/minimal-fusion.md` and `examples/minimal_fusion_demo.py`.

# Example usage

A minimal workflow in this repository is:

```bash
# 1) install package and dev+benchmark tools
pip install -e ".[dev,benchmark]"

# 2) run the minimal fusion example
python examples/minimal_fusion_demo.py

# 3) run the JOSS smoke test
python -m pytest tests/e2e/test_toy_pipeline.py
```

Advanced usage combines benchmark manifests, router configuration, fusion baseline sweeps, and report generation into one reproducible run directory with JSON and Markdown outputs. The main orchestration entrypoint is `kalmanorix.panoramix.Panoramix`; the older `kalmanorix.kalman_engine.fuser.Panoramix` path remains as a deprecated shim.

# Software architecture and functionality

The public API maps directly to benchmark steps:

- **SEF** (`kalmanorix.village.SEF`): defines one specialist embedding backend and metadata.
- **Village** (`kalmanorix.village.Village`): stores the specialist set used for routing.
- **ScoutRouter** (`kalmanorix.scout.ScoutRouter`): picks a specialist per query or query batch.
- **Panoramix** (`kalmanorix.panoramix.Panoramix`): orchestrates routing, fusion, and artifact output.
- **MeanFuser** (`kalmanorix.MeanFuser`): arithmetic-mean baseline.
- **KalmanorixFuser** (`kalmanorix.KalmanorixFuser`): uncertainty-aware Kalman-filter baseline.
- **claim-gate scripts** (`scripts/build_claim_gate.py`, `results/evidence_registry.json`): convert metrics into claim status records with provenance.

Kalmanorix complements existing retrieval libraries instead of replacing them. In common setups, encoding comes from SentenceTransformers, dataset/task structure from BEIR-style benchmarks, and vector search from FAISS. Kalmanorix adds routing/fusion evaluation and claim reporting around that stack [@reimers2019sentencebert; @thakur2021beir; @johnson2017faiss].

```python
import numpy as np
from kalmanorix import SEF, Village, ScoutRouter, Panoramix, MeanFuser, KalmanorixFuser

# 1) define SEF specialists (embed callable + constant sigma2 uncertainty)
tech = SEF(name="tech", embed=lambda q: np.random.randn(128), sigma2=0.5)
cook = SEF(name="cook", embed=lambda q: np.random.randn(128), sigma2=1.0)

# 2) group into a Village
village = Village([tech, cook])

# 3) route and fuse with Panoramix
scout = ScoutRouter(mode="all")
panoramix = Panoramix(fuser=KalmanorixFuser())
potion = panoramix.brew("example query", village, scout)
print(potion.vector.shape, potion.weights)

# For mean-fusion comparison:
mean_panoramix = Panoramix(fuser=MeanFuser())
mean_potion = mean_panoramix.brew("example query", village, scout)
```

# Reproducibility, documentation, and release readiness

Reproducibility in Kalmanorix depends on versioned manifests, explicit run metadata, deterministic output paths, and scripted artifact export. The repository also includes automated tests and formatting checks.

Documentation entrypoint: `docs/index.md`.

Quickstart/example entrypoints: `docs/getting-started/quickstart.md` and `docs/examples/minimal-fusion.md`.

Licence statement: Kalmanorix is distributed under the MIT License (`LICENSE`).

CI gates referenced in this repository: `ruff format --check .` and `pytest -m "not integration and not stress"`.

Release metadata is tracked in `CITATION.cff`, `pyproject.toml`, and this manuscript front matter. The software is archived on Zenodo at [10.5281/zenodo.20318011](https://doi.org/10.5281/zenodo.20318011).

The reporting pipeline keeps unsupported and null findings instead of filtering them out. This aligns with recommendations for multi-dataset significance testing and robust cross-system comparison in NLP/IR evaluation [@dror2017replicability; @smucker2007significance; @demsar2006statistical].

The uncertainty-aware baseline (`KalmanorixFuser`) is motivated by calibration-aware modeling practice and Kalman-style filtering ideas [@guo2017calibration; @kalman1960filtering].

# Relationship to empirical papers

The JOSS paper presents the software package only. Companion manuscripts (TMLR methodology and arXiv empirical reporting) consume shared generated artifacts for reproducibility (`paper/shared/generated/evidence_registry.json` and `paper/shared/generated/baseline_matrix.tex`), while this JOSS paper remains restricted to software capabilities, interfaces, and release governance. The software supports Kalman-style fusion experiments but does not require, assume, or claim retrieval-quality improvement.

# AI usage disclosure

Generative AI tools were used as drafting assistants for parts of documentation and manuscript wording. The human author reviewed claims, citations, and scope statements before submission.

# Acknowledgements

The author thanks maintainers of open-source Python IR and scientific computing ecosystems, and contributors who shared reproducibility feedback during development.

# References
