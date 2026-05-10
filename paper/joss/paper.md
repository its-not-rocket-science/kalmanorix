---
title: "Kalmanorix: reproducible benchmarking and uncertainty-aware evaluation infrastructure for retrieval research"
tags:
  - information retrieval
  - embeddings
  - uncertainty quantification
  - semantic routing
  - benchmarking
authors:
  - name: "Paul Schleifer"
    affiliation: "1"
affiliations:
  - name: "Independent Research Software Engineer, United States"
    index: 1
date: "2026-05-10"
bibliography: paper.bib
---

# Summary

Kalmanorix is MIT-licensed research software for reproducible retrieval benchmarking and evaluation governance. It provides a command-line framework that links benchmark construction, paired retrieval evaluation, uncertainty-aware analysis, statistical testing, and publication-ready artifact generation in one reproducible workflow. The contribution is infrastructure for producing auditable evidence, not a claim of state-of-the-art retrieval performance [@thakur2021beir; @jarvelin2002cumulated].

The software preserves null and negative outcomes as first-class artifacts and propagates them through reports, helping teams avoid overclaiming when observed differences are negligible or uncertain.

# Statement of need

Retrieval projects often lack standardized, reproducible evaluation pipelines. Benchmark manifests, slice definitions, statistical settings, and reporting logic are frequently distributed across ad hoc scripts and notebooks, which makes replication and governance difficult.

Kalmanorix addresses this gap by providing reusable infrastructure for benchmark lifecycle management and claim discipline. It codifies benchmark manifests, paired test workflows, multiplicity-aware inference, and uncertainty-aware reporting so teams can distinguish detectable improvement from negligible or inconclusive change under explicit decision criteria.

# Research application

Kalmanorix is intended for retrieval research programs that compare model variants, routing strategies, or efficiency/quality trade-offs across evolving benchmark slices. In these settings, the software supports:

- reproducible execution of paired retrieval evaluations,
- consistent statistical reporting with multiple-testing control,
- generation of machine-readable and narrative artifacts for audit and publication,
- uncertainty-aware interpretation that retains unsupported findings.

This positions Kalmanorix as research infrastructure for evidence production rather than a method paper centered on new empirical results.

Kalmanorix also standardizes how final empirical readouts are represented in project artifacts. In the current canonical domain-balanced C100 confirmatory run ($n_{\text{pairs}}=1193$), the recorded outcome is a practically null result (nDCG@10 delta $-9.258801070226193\times10^{-6}$, adjusted $p=1.0$, recall@100 delta $0.0$, latency ratio $1.0722551296204925$, FLOPs ratio $1.0$, verdict `inconclusive_sufficiently_powered`). In JOSS scope, this is cited only as an example of software-supported reporting, not as the manuscript's core scientific claim.

# Software architecture and workflows

Kalmanorix exposes a modular CLI and artifact pipeline:

- **Benchmark orchestration**: commands generate and version domain-balanced benchmark artifacts from declarative manifests.
- **Evaluation execution**: commands run paired retrieval comparisons (baseline, fusion, routing, and slices) under consistent protocols.
- **Statistical workflow**: paired tests, multiple-comparison correction, confidence intervals, and effect-size thresholds are applied and persisted as machine-readable outputs.
- **Reporting generation**: structured summaries and narrative reports are generated from artifacts, including benchmark status gates and claim-readiness indicators.
- **Reproducibility surfaces**: deterministic artifact layout, pinned benchmark identifiers, explicit run metadata, and re-runnable CLI entry points support audit and rerun.

These components are designed to compose into reproducibility-focused experimentation pipelines rather than one-off leaderboard runs.

# Conclusions

Kalmanorix contributes reusable infrastructure for reproducible retrieval benchmarking, uncertainty-aware evaluation, and claim-governed reporting. Its core value is enabling transparent, auditable, and repeatable evidence production across benchmark versions and contributors.

By preserving null/negative outcomes and coupling empirical summaries to explicit statistical decision rules, the software supports more trustworthy retrieval experimentation and reduces pressure toward overstated claims.

# AI usage disclosure

Generative AI tools were used as drafting assistants for portions of repository documentation and manuscript wording (including iterative edits prepared with OpenAI GPT-family tooling). All manuscript claims, citations, and software-scoping statements were reviewed and validated by the human author before submission. No AI system was treated as an authoritative source for empirical conclusions.

# Acknowledgements

The author thanks open-source maintainers whose scientific Python and retrieval tooling ecosystems make this project possible, as well as external users and reviewers who provided issue reports and reproducibility feedback during development.

# References
