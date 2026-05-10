---
title: "Kalmanorix: reproducible benchmarking and uncertainty-aware evaluation infrastructure for retrieval research"
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

Kalmanorix is public MIT-licensed research software for reproducible retrieval evaluation and benchmarking. The package provides command-line workflows that connect benchmark construction, paired retrieval evaluation, uncertainty calibration, statistical testing, and publication-oriented artifact generation in a single reproducible pipeline. Its primary contribution is infrastructure: it standardizes how retrieval experiments are run, audited, and reported across benchmark versions and contributors.

Recent powered benchmark outcomes in this repository indicate negligible effect sizes, statistically insignificant retrieval deltas after multiple-comparison correction, latency-neutral operation, and acceptable compute overhead. In that setting, the software contribution is not a claim of superior retrieval quality, but a practical framework for producing interpretable negative results with the same rigor as positive findings. Kalmanorix is therefore positioned as tooling for trustworthy evaluation rather than as a new state-of-the-art retrieval method [@thakur2021beir; @jarvelin2002cumulated].

# Statement of need

Retrieval teams frequently face a reproducibility gap between exploratory notebooks and claim-grade evidence. In many projects, benchmark recipes, slice definitions, statistical decisions, and efficiency measurements are scattered across ad hoc scripts, making replication and governance difficult. This is especially problematic when observed metric changes are small and sensitive to benchmark composition, multiple testing, or compute budget.

Kalmanorix addresses this need with a reproducibility-first evaluation framework that integrates domain-balanced benchmark generation, powered paired testing, uncertainty-aware analysis, and replication-oriented reporting. The framework is designed to prevent overclaiming small retrieval gains. It codifies decision rules around Holm-corrected paired tests, confidence intervals, and effect-size thresholds so teams can distinguish “detectable improvement” from “practically negligible change” under declared criteria. It also preserves unsupported and inconclusive outcomes as first-class artifacts, improving institutional memory and reducing selective reporting.

# Feature overview

Kalmanorix exposes scriptable CLI pathways for the full evaluation lifecycle:

- **Benchmark construction and governance**: mixed-domain benchmark builders and schema-backed artifacts that preserve provenance and versioning.
- **Canonical and slice evaluation**: repeatable paired evaluations for baseline, fusion, and routing strategies, including confirmatory slice evaluation.
- **Statistical guardrails**: built-in support for Holm-corrected paired testing, confidence-interval reporting, and effect-size thresholding for practical significance decisions.
- **Uncertainty and calibration tooling**: utilities for uncertainty-aware retrieval analysis and calibration summaries to contextualize ranking behavior.
- **Compute-aware experimentation**: latency and overhead tracking integrated with quality reporting so retrieval deltas are interpreted alongside efficiency constraints.
- **Automatic artifact generation**: machine-readable summaries plus narrative reports that capture methods, assumptions, outcomes, and claim-readiness status.

Together, these features reposition Kalmanorix as evaluation infrastructure that emphasizes reproducibility, interpretability, and governance under realistic benchmarking constraints.

# Benchmarking

Kalmanorix structures benchmarking as a governed process rather than a single leaderboard run. Benchmarks are constructed as versioned, domain-balanced artifacts and evaluated with paired protocols designed for small-delta retrieval settings. The reporting layer explicitly couples hypothesis tests with uncertainty intervals and practical-effect criteria, reducing ambiguity in interpretation.

In the final powered benchmark setting, observed retrieval differences were negligible in effect size and statistically non-significant after correction, while runtime behavior remained latency-neutral with acceptable compute overhead. Kalmanorix treats this powered negative-result outcome as informative evidence, not failure: the framework makes it straightforward to document when improvements are not substantiated and to propagate that decision through downstream reporting.

This benchmark orientation supports replication-aware conclusions by requiring confirmatory slice evaluation, separating exploratory from confirmatory analyses, and preserving benchmark-specific scope limitations. As a result, the package helps teams avoid method overstatement when observed deltas are too small or too uncertain to justify superiority claims.

# Reproducibility

Reproducibility is a core design goal of Kalmanorix. The package provides stable execution surfaces through documented CLI entry points, deterministic artifact layouts, and machine-readable outputs that can be re-run, diffed, and audited across revisions.

Key reproducibility mechanisms include:

1. **Benchmark provenance controls**: explicit benchmark versions, split definitions, and artifact lineage.
2. **Decision-rule traceability**: persisted statistical settings (including multiplicity correction), confidence-interval outputs, and effect-size thresholds used for interpretation.
3. **Replication-aware reporting**: automatic inclusion of confirmatory slice outcomes and benchmark-status gates in generated summaries.
4. **Calibration and uncertainty records**: persisted calibration artifacts that support uncertainty-aware interpretation and re-analysis.
5. **Compute context capture**: integrated latency/overhead summaries so quality findings are reproducible within resource constraints.

These mechanisms align the software with JOSS expectations for reusable, inspectable research infrastructure and support transparent benchmark governance over time.

# Conclusions

Kalmanorix contributes practical research infrastructure for retrieval benchmarking, not a universal retrieval-performance improvement claim. Its value is in enabling powered statistical evaluation, uncertainty-aware analysis, benchmark governance, and automatic reproducibility artifacts in one coherent workflow.

By making negative and null outcomes straightforward to preserve and communicate, the framework strengthens scientific practice in retrieval engineering. In particular, it supports cautious interpretation when effect sizes are negligible, retrieval deltas are statistically inconclusive, and efficiency trade-offs must be reported alongside quality metrics. This infrastructure-focused framing is consistent with JOSS: the software contribution is a reproducible evaluation and benchmarking framework that improves how retrieval evidence is produced, audited, and replicated.

# AI usage disclosure

TODO: Replace this placeholder with an honest, specific disclosure of AI-system usage in software development and manuscript preparation (including model/tool names, tasks performed, and human verification steps).

# Acknowledgements

TODO: Add acknowledgements for contributors, maintainers, institutional support, and any non-financial assistance. Do not add funding claims unless they are verified and can be documented.

# References
