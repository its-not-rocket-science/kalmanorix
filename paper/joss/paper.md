---
title: "Kalmanorix: specialist embedding fusion and uncertainty-aware routing research software"
tags:
  - information retrieval
  - embeddings
  - uncertainty quantification
  - semantic routing
  - benchmarking
authors:
  - name: "TODO: Author name"
    affiliation: "TODO: Affiliation"
    orcid: "TODO: ORCID"
  - name: "TODO: Additional author(s)"
    affiliation: "TODO: Affiliation"
    orcid: "TODO: ORCID"
affiliations:
  - name: "TODO: Affiliation"
    index: 1
date: "2026-05-06"
bibliography: paper.bib
---

# Summary

Kalmanorix is public research software for specialist embedding fusion, uncertainty-aware retrieval experimentation, routing evaluation, and benchmark governance. The project is distributed under the MIT Licence and exposes package metadata and build configuration in `pyproject.toml`, including command-line entry points that support full experiment workflows from benchmark construction to report generation. Rather than presenting one-off experiment scripts, Kalmanorix provides a reproducible software substrate for comparing fusion and routing choices under explicit decision rules.

The framework addresses a common issue in retrieval research: experimental pipelines are often difficult to reproduce, and decision criteria are frequently under-specified. Kalmanorix focuses on transparent benchmark artefacts, repeatable command-line workflows, and status-oriented reporting that captures both positive and negative outcomes. Empirical findings are documented in repository artefacts and reports, but those findings are intentionally treated as separate from this software paper; the contribution here is the research software itself and its governance scaffolding.

Kalmanorix includes CLI pathways for mixed benchmark building, canonical benchmark execution, report generation, routing evaluation, and benchmark utilities. These interfaces support disciplined comparisons across specialist fusion variants and routing settings while preserving provenance for later audit or re-analysis. The software therefore functions both as an experimentation platform and as an evidence-management layer for retrieval research [@thakur2021beir; @jarvelin2002cumulated].

# Statement of need

Modern embedding retrieval systems increasingly combine multiple specialist models, yet robust methods for evaluating such systems remain fragmented. Researchers need software that can: (i) build and run heterogeneous benchmarks; (ii) compare fusion strategies under uncertainty; (iii) evaluate routing policies with quality-versus-efficiency trade-offs; and (iv) preserve decision context, not only raw metric outputs.

Kalmanorix is designed to meet this need by providing a coherent, scriptable interface for uncertainty-aware retrieval research. The package operationalises specialist fusion workflows inspired by state-estimation thinking [@kalman1960] while remaining practical for information retrieval evaluation. It explicitly supports reproducible benchmark artefacts and explicit decision rules, enabling teams to track whether a claim is supported, unsupported, or inconclusive under declared thresholds. This is particularly important for negative-result stewardship: null or mixed outcomes remain first-class artefacts rather than disappearing from project history.

The software is also useful for research governance. By standardising experiment execution and report generation, Kalmanorix reduces ambiguity between “code that ran once” and “evidence that can be revisited”. This helps research groups maintain continuity across contributors, model updates, and benchmark revisions.

# State of the field

The retrieval community has established strong benchmark traditions, including shared evaluation suites such as BEIR [@thakur2021beir] and ranking metrics such as nDCG that are sensitive to graded relevance [@jarvelin2002cumulated]. At the same time, practical research software often lags behind methodological advances: benchmark recipes, routing heuristics, and experimental decision criteria are frequently distributed across ad hoc notebooks or bespoke scripts.

Kalmanorix sits in this gap. It is not a benchmark in itself; rather, it is software infrastructure for running and governing benchmarked retrieval experiments with specialist and fused embeddings. The project builds on mature scientific Python components (notably NumPy and SciPy) for numerical operations and optimisation foundations [@harris2020array; @virtanen2020scipy]. Optional training and evaluation pathways can also leverage widely used machine-learning tooling (for example, sentence-transformers and scikit-learn) where relevant to specialist model development and analysis [@reimers2019sentencebert; @pedregosa2011scikit].

By combining these established foundations with explicit benchmark lifecycle tooling, Kalmanorix contributes a missing layer: software process for reproducible retrieval experimentation, especially where uncertainty-aware fusion and routing decisions must be evaluated together.

# Software design

Kalmanorix is organised as a Python package with metadata declared in `pyproject.toml`, allowing repeatable installation and environment specification. The design emphasises executable research workflows through CLI entry points that map onto distinct lifecycle stages:

- mixed benchmark building;
- canonical benchmark running;
- report generation;
- routing evaluation;
- benchmark utility commands for fusion and calibration workflows.

This separation helps keep experimental responsibilities explicit: data/benchmark construction, method execution, and evidence reporting are addressable independently but composable in pipelines.

At the methodological layer, Kalmanorix supports specialist embedding fusion with uncertainty-aware components and routing mechanisms that trade coverage, quality, and computational cost. Importantly, evaluation outputs are structured as reproducible artefacts rather than transient logs. Decision-rule reporting is integrated so that conclusions can be interpreted relative to declared thresholds and evidence-readiness states. The result is a software workflow where outcome labels (including null and regression cases) are governed, versionable objects.

From a reproducibility perspective, this design provides several advantages:

1. **Determinable execution surfaces** via documented CLI entry points.
2. **Traceable artefacts** suitable for re-inspection and longitudinal comparisons.
3. **Decision transparency** through explicit rule-based status outputs.
4. **Balanced evidence tracking** that retains negative and positive results alike.

These properties make Kalmanorix appropriate for iterative research programmes in which conclusions evolve with additional data, stronger controls, or revised benchmark definitions.

# Research impact statement

Kalmanorix contributes impact primarily through research practice rather than a single algorithmic claim. It enables research teams to run specialist fusion and routing experiments under consistent interfaces, preserve benchmark provenance, and document decision outcomes with auditable criteria. This supports methodological rigour in settings where small metric deltas, compute constraints, and domain heterogeneity can otherwise make interpretation fragile.

A key impact area is benchmark governance. The software encourages explicit distinction between evidence generation and claim acceptance, reducing risk of over-interpreting underpowered or context-specific runs. It also improves institutional memory by preserving negative results and inconclusive outcomes as reproducible artefacts, which can prevent duplicate effort and support better hypothesis refinement.

Empirical findings produced with Kalmanorix are intentionally documented separately in repository reports and artefacts, and should not be treated as the focus of this JOSS submission. The software paper instead documents Kalmanorix as reusable infrastructure for uncertainty-aware retrieval experimentation and routing evaluation.

# AI usage disclosure

TODO: Replace this placeholder with an honest, specific disclosure of AI-system usage in software development and manuscript preparation (including model/tool names, tasks performed, and human verification steps).

# Acknowledgements

TODO: Add acknowledgements for contributors, maintainers, institutional support, and any non-financial assistance. Do not add funding claims unless they are verified and can be documented.

# References
