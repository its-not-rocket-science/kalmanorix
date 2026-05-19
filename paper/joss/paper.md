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
    orcid: "0009-0004-7972-3566"
    affiliation: "1"
affiliations:
  - name: "Independent Research Software Engineer, United Kingdom"
    index: 1
date: "2026-05-11"
bibliography: paper.bib
---

# Summary

Kalmanorix is an MIT-licensed Python package for **retrieval-system evaluation** with specialist embedding models, semantic routing policies, and uncertainty-aware fusion baselines. The software provides a reproducible workflow from benchmark manifest definition to machine-readable evidence artifacts, including claim-gated summaries of supported, unsupported, and inconclusive outcomes. Its focus is software infrastructure for disciplined benchmark reporting rather than promotion of a single retrieval strategy [@thakur2021beir; @jarvelin2002cumulated].

# Statement of need

Modern retrieval stacks provide strong building blocks---embedding models, vector databases, ANN search libraries, and ranking frameworks---but these tools typically stop at indexing and retrieval execution. They do not, by themselves, provide a unified protocol for (i) evaluating **specialist-embedding routing** decisions, (ii) comparing **uncertainty-aware fusion** baselines under consistent reporting rules, and (iii) translating metric outputs into **claim-gated** evidence suitable for publication and audit.

Research teams therefore often assemble bespoke scripts across notebooks and one-off benchmark slices, which makes it difficult to reproduce claim decisions, preserve negative results, and maintain stable evidence records across reruns. Kalmanorix addresses this gap by standardizing the loop from benchmark specification to claim-ready artifacts. The toolkit is intended for:

In this work, we position Kalmanorix as research software infrastructure for transparent, claim-gated retrieval benchmarking rather than as evidence that any single fusion strategy is universally best.

- information retrieval researchers,
- embedding researchers,
- evaluation-methodology researchers.

# Core functionality

Kalmanorix provides the following software capabilities:

1. **Specialist embedding packaging**: reproducible assembly and configuration of specialist embedders for multi-domain retrieval workflows.
2. **Routing evaluation**: evaluation of query-to-specialist routing behavior with domain and slice-aware reporting.
3. **Fusion baselines**: baseline-matrix support for single-model, weighted, and uncertainty-aware fusion variants, with explicit comparative reporting rather than quality guarantees.
4. **Claim-gated benchmark reporting**: rule-based claim readiness outputs that distinguish supported improvements from negligible or inconclusive effects.
5. **Reproducible artifact generation**: deterministic manifests and machine-readable reports suitable for audit, reanalysis, and publication handoff.

These components help researchers evaluate specialist-embedding routing and uncertainty-aware fusion behavior in both positive and negative-result settings, with claims constrained to reported evidence artifacts.

# Example usage

A minimal workflow in this repository is:

```bash
# 1) install package and dev tools
pip install -e .

# 2) run the minimal fusion example
python examples/minimal_fusion_demo.py

# 3) run non-integration tests
pytest -m "not integration and not stress"
```

Typical advanced usage combines benchmark manifests, specialist-router configuration, fusion baseline sweeps, and claim report generation into one reproducible run directory with JSON and Markdown outputs. In practice this is expressed through package components such as the `kalmanorix.panoramix.Panoramix` orchestration API, the `kalmanorix.run_claim_gate` command-line entrypoint, and repository checks such as `pytest -m "not integration and not stress"` and `ruff format --check .`.
The older `kalmanorix.kalman_engine.fuser.Panoramix` import path is retained as a deprecated compatibility shim.

# Reproducibility and testing

Kalmanorix emphasizes reproducibility through versioned benchmark manifests, explicit run metadata, deterministic output layout, and scripted artifact export. The repository includes automated tests and style checks, and project documentation includes installation, examples, API reference pages, and release/archival guidance.

The reporting pipeline intentionally preserves unsupported and null findings so that negative-result reporting remains first-class in empirical records.

# Relationship to companion papers

This JOSS manuscript documents the **software contribution**: package architecture, reproducible workflows, and evidence-generation interfaces. Companion TMLR/arXiv manuscripts report **empirical and methodological results** obtained using this tooling. Across these outputs, claim language is governed by the repository evidence registry (`results/evidence_registry.json`), so statements about supported or unsupported effects track recorded benchmark evidence rather than narrative preference.

# AI usage disclosure

Generative AI tools were used as drafting assistants for portions of repository documentation and manuscript wording. All claims, citations, and scope statements were reviewed by the human author before submission.

# Acknowledgements

The author thanks maintainers of open-source Python IR and scientific computing ecosystems, and contributors who provided reproducibility feedback during development.

# References
