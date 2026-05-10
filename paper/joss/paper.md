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

Current canonical evidence is a **powered negative result** in a **domain-balanced** confirmatory slice: paired evaluation at $n\_pairs=1193$ with `max_candidates=100` reports negligible $\Delta$ nDCG@10, Holm-adjusted $p=1.0$, recall@100 delta $0.0$, and documented latency/FLOPs ratios (from a local Windows CPU-only reference run; environment-dependent). This supports **claim-ready** governance with a blocked improvement claim under explicit **claim-gating** and **uncertainty-aware evaluation**.

Kalmanorix preserves these null/negative outcomes as first-class outputs and propagates them through reports to discourage overclaiming.

# Statement of need

Retrieval projects often lack standardized, reproducible evaluation pipelines. Benchmark manifests, slice definitions, statistical settings, and reporting logic are frequently distributed across ad hoc scripts and notebooks, which makes replication and governance difficult.

Kalmanorix addresses this gap by providing reusable infrastructure for benchmark lifecycle management and claim discipline. It codifies benchmark manifests, paired test workflows, multiplicity-aware inference, and uncertainty-aware reporting so teams can distinguish detectable improvement from negligible or inconclusive change under explicit decision criteria.

# Software architecture and workflows

Kalmanorix exposes a modular CLI and artifact pipeline:

- **Benchmark orchestration**: commands generate and version domain-balanced benchmark artifacts from declarative manifests.
- **Evaluation execution**: commands run paired retrieval comparisons (baseline, fusion, routing, and slices) under consistent protocols.
- **Statistical workflow**: paired tests, multiple-comparison correction, confidence intervals, and effect-size thresholds are applied and persisted as machine-readable outputs.
- **Reporting generation**: structured summaries and narrative reports are generated from artifacts, including benchmark status gates and claim-readiness indicators.
- **Reproducibility surfaces**: deterministic artifact layout, pinned benchmark identifiers, explicit run metadata, and re-runnable CLI entry points support audit and rerun.

These components are designed to compose into reproducibility-focused experimentation pipelines rather than one-off leaderboard runs.

# Repository usability

The repository is organized for direct operational use by retrieval practitioners and research engineers:

- documented CLI workflows for end-to-end benchmark creation, execution, and reporting,
- benchmark manifests and schema-backed artifacts for governed experiment configuration,
- structured outputs for downstream analysis and archival,
- reproducible execution patterns that enable rerun/diff/audit across revisions,
- reporting artifacts that preserve both positive and null outcomes with explicit statistical context.

This design supports team workflows in benchmarking governance, replication checks, and uncertainty-aware interpretation.

# Research-software positioning

Kalmanorix targets users who need trustworthy retrieval evaluation infrastructure: retrieval researchers, applied ML engineers, and maintainers of benchmark/reporting pipelines. The software emphasizes:

1. reproducibility-first experimentation,
2. governance of benchmark definitions and statistical decisions,
3. uncertainty-aware and efficiency-aware interpretation,
4. preservation of unsupported or negative findings as durable artifacts.

Accordingly, Kalmanorix should be read as research infrastructure that improves how retrieval evidence is generated and communicated, not as a universal performance-improvement method.

# Conclusions

Kalmanorix contributes reusable infrastructure for reproducible retrieval benchmarking, uncertainty-aware evaluation, and claim-governed reporting. Its core value is enabling transparent, auditable, and repeatable evidence production across benchmark versions and contributors.

By preserving null/negative outcomes and coupling empirical summaries to explicit statistical decision rules, the software supports more trustworthy retrieval experimentation and reduces pressure toward overstated claims.

# AI usage disclosure

Generative AI tools were used as drafting assistants for portions of repository documentation and manuscript wording (including iterative edits prepared with OpenAI GPT-family tooling). All manuscript claims, citations, and software-scoping statements were reviewed and validated by the human author before submission. No AI system was treated as an authoritative source for empirical conclusions.

# Acknowledgements

The author thanks open-source maintainers whose scientific Python and retrieval tooling ecosystems make this project possible, as well as external users and reviewers who provided issue reports and reproducibility feedback during development.

# References
