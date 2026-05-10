> **Deprecated (May 10, 2026):** This manuscript is superseded for JOSS submission by `paper/joss/paper.md`. Keep this file only as archival draft material to avoid reviewer confusion.

---
title: "TODO: Kalmanorix JOSS Title"
tags:
  - TODO: tag1
  - routing
  - uncertainty quantification
authors:
  - name: TODO: Author Name 1
    orcid: TODO: 0000-0000-0000-0000
    affiliation: 1
  - name: TODO: Author Name 2
    orcid: TODO: 0000-0000-0000-0000
    affiliation: 1
affiliations:
  - name: TODO: Affiliation 1
    index: 1
date: TODO: YYYY-MM-DD
bibliography: paper.bib
---

# Summary

Kalmanorix is research software for evaluating routing strategies under uncertainty and for studying uncertainty-aware sensor fusion in navigation-like systems. The project provides a reproducible software environment for constructing controlled experiments, generating or loading benchmark-style datasets, running comparative evaluations, and inspecting route-level and aggregate outcomes with uncertainty-sensitive metrics. Its design emphasizes transparent experimentation so that researchers can isolate assumptions, vary uncertainty models, and compare methods using a consistent workflow.

This JOSS paper describes Kalmanorix as a software contribution. Empirical findings, including experimental outcomes and interpretation, are documented separately in the project’s empirical paper and associated artifacts.

# Statement of Need

Routing and decision systems often rely on noisy inputs (e.g., map features, sensor estimates, model predictions), but many evaluation stacks treat uncertainty as an afterthought. Researchers need practical tooling that supports both (1) route evaluation and (2) uncertainty-aware fusion analysis within one coherent framework. Existing ad hoc scripts can be difficult to reproduce, extend, or compare across studies.

Kalmanorix addresses this need by offering a software platform that connects data preparation, routing evaluation, and uncertainty-aware fusion experiments in a single, testable codebase. It is intended for research teams that require repeatable computational experiments and clear interfaces for swapping routing heuristics, uncertainty models, and fusion components. By reducing setup overhead and standardizing experiment structure, Kalmanorix helps researchers focus on methodological questions rather than custom pipeline glue code.

# State of the Field

Research on routing, probabilistic estimation, and sensor/data fusion spans multiple communities, with established methods for shortest-path search, state estimation, and uncertainty propagation [@TODO-routing-citation; @TODO-fusion-citation; @TODO-uncertainty-citation]. However, the software landscape is fragmented: route optimisation tooling, probabilistic filtering libraries, and experiment harnesses are often developed independently and integrated only for specific projects.

Kalmanorix contributes at this integration layer. It is positioned as research infrastructure that bridges routing evaluation and uncertainty-aware fusion analysis with reproducible experiment management. Rather than proposing a single closed-form algorithm as its primary contribution, the software supports systematic method comparison and ablation-oriented investigation in uncertainty-aware routing contexts.

# Functionality

Kalmanorix provides:

1. **Experiment scaffolding for routing evaluation**: structured workflows for defining scenarios, running route computations, and collecting comparable outputs across methods.
2. **Uncertainty-aware fusion hooks**: modular components for incorporating and evaluating uncertainty-aware fusion stages in the routing pipeline.
3. **Metric and analysis utilities**: tools for summarizing route quality, robustness, and uncertainty-related behaviour across runs.
4. **Reproducibility-oriented organisation**: configuration-driven execution, dataset handling conventions, and artifact-friendly outputs suitable for research reporting.

The project is designed to support iterative experimentation: researchers can vary routing assumptions, fusion strategies, and uncertainty treatments while preserving a consistent evaluation protocol. This makes Kalmanorix suitable for baseline construction, method benchmarking, and exploratory analysis in uncertainty-aware routing studies.

# Acknowledgements

TODO: Add funding sources, institutional support, and contributor acknowledgements.

# References
