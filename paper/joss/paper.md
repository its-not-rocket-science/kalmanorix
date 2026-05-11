---
title: "Kalmanorix: a Python toolkit for specialist-embedding fusion, routing evaluation, and claim-gated retrieval benchmarking"
tags:
  - information retrieval
  - embeddings
  - evaluation
  - benchmarking
  - reproducibility
authors:
  - name: "Paul Schleifer"
    affiliation: "1"
affiliations:
  - name: "Independent Research Software Engineer, United States"
    index: 1
date: "2026-05-11"
bibliography: paper.bib
---

# Summary

Kalmanorix is an MIT-licensed Python toolkit for designing and evaluating uncertainty-aware retrieval systems. It packages workflows for specialist embedding composition, semantic routing evaluation, and fusion baseline comparison under one reproducible command-line and artifact pipeline. The package is designed to produce auditable evidence for what works, what does not work, and what remains inconclusive, rather than to present a single winning method [@thakur2021beir; @jarvelin2002cumulated].

# Statement of need

IR and embedding research teams often maintain fragmented evaluation scripts across notebooks, ad hoc benchmark slices, and inconsistent statistical settings. This fragmentation makes it difficult to reproduce claims, compare routing or fusion decisions fairly, and preserve negative results.

Kalmanorix addresses this need by standardizing the full loop from benchmark specification to publishable evidence artifacts. The toolkit is intended for:

- information retrieval researchers,
- embedding researchers,
- evaluation-methodology researchers.

# Core functionality

Kalmanorix provides the following software capabilities:

1. **Specialist embedding packaging**: reproducible assembly and configuration of specialist embedders for multi-domain retrieval workflows.
2. **Routing evaluation**: evaluation of query-to-specialist routing behavior with domain and slice-aware reporting.
3. **Fusion baselines**: baseline matrix support for single-model, weighted, and uncertainty-aware fusion variants.
4. **Claim-gated benchmark reporting**: rule-based claim readiness outputs that distinguish supported improvements from negligible or inconclusive effects.
5. **Reproducible artifact generation**: deterministic manifests and machine-readable reports suitable for audit, reanalysis, and publication handoff.

These components help researchers evaluate uncertainty-aware fusion and routing in both positive and negative-result settings.

# Example usage

A minimal workflow in this repository is:

```bash
# 1) install package and dev tools
pip install -e .

# 2) run the minimal fusion example
python -m kalmanorix.examples.minimal_fusion

# 3) run non-integration tests
pytest -m "not integration and not stress"
```

Typical advanced usage combines benchmark manifests, routing configuration, fusion baseline sweeps, and report generation into one reproducible run directory with JSON + Markdown outputs.
The canonical Python orchestration API is `kalmanorix.panoramix.Panoramix`; the older `kalmanorix.kalman_engine.fuser.Panoramix` import path is retained as a deprecated compatibility shim.

# Reproducibility and testing

Kalmanorix emphasizes reproducibility through versioned benchmark manifests, explicit run metadata, deterministic output layout, and scripted artifact export. The repository includes automated tests and style checks, and project documentation includes installation, examples, API reference pages, and release/archival guidance.

The reporting pipeline intentionally preserves unsupported and null findings so that negative-result reporting remains first-class in empirical records.

# Relationship to research papers

Kalmanorix is research software infrastructure, not a research-results manuscript. The package supports experiments reported elsewhere, including studies of uncertainty-aware fusion and routing behavior. Its role is to make those studies reproducible and claim-disciplined, including when results are negative or inconclusive.

# AI usage disclosure

Generative AI tools were used as drafting assistants for portions of repository documentation and manuscript wording. All claims, citations, and scope statements were reviewed by the human author before submission.

# Acknowledgements

The author thanks maintainers of open-source Python IR and scientific computing ecosystems, and contributors who provided reproducibility feedback during development.

# References
