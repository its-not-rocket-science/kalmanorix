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

For user-facing setup details and environment notes, see the installation guide in `docs/getting-started/installation.md`. For a minimal runnable example, see `docs/examples/minimal-fusion.md` and `examples/minimal_fusion_demo.py`.

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

Typical advanced usage combines benchmark manifests, specialist-router configuration, fusion baseline sweeps, and claim report generation into one reproducible run directory with JSON and Markdown outputs. In practice this is expressed through package components such as the `kalmanorix.panoramix.Panoramix` orchestration API, the `kalmanorix.run_claim_gate` command-line entrypoint, and repository checks such as `pytest -m "not integration and not stress"` and `ruff format --check .`.
The older `kalmanorix.kalman_engine.fuser.Panoramix` import path is retained as a deprecated compatibility shim.


# Software architecture and functionality

Kalmanorix exposes a small set of public components that map directly to the specialist-routing and fusion workflow:

- **SEF** (`kalmanorix.sef.SEF`): defines a specialist embedding function, including a concrete model backend and metadata for routing/evaluation.
- **Village** (`kalmanorix.village.Village`): container for multiple SEF specialists, used as the candidate pool for query-time dispatch and benchmarking.
- **ScoutRouter** (`kalmanorix.router.ScoutRouter`): routing policy that selects a specialist from a Village for each query (or query batch) using router features/scores.
- **Panoramix** (`kalmanorix.panoramix.Panoramix`): orchestration layer that runs retrieval/fusion experiments and produces metric and evidence artifacts.
- **MeanFuser** (`kalmanorix.fusers.MeanFuser`): baseline fuser that combines specialist signals by arithmetic mean.
- **KalmanorixFuser** (`kalmanorix.fusers.KalmanorixFuser`): uncertainty-aware fusion baseline implemented in this package for controlled comparisons against simpler alternatives.
- **routing evaluator CLI** (`kalmanorix.run_routing_eval`): command-line entrypoint for routing-quality evaluation and slice-aware reporting.
- **evidence registry / claim-gated reporting tools** (`kalmanorix.run_claim_gate` and `results/evidence_registry.json`): transforms metric outputs into claim-status artifacts (supported/unsupported/inconclusive) with machine-readable provenance.

Kalmanorix is designed to complement, not replace, mature retrieval libraries. In typical usage, specialist embedders rely on SentenceTransformers for encoding, BEIR-style datasets/tasks for benchmark structure, and FAISS (or equivalent ANN backends) for vector indexing/search; Kalmanorix adds routing/fusion evaluation and claim-gated reporting around those components [@thakur2021beir].

```python
from kalmanorix.sef import SEF
from kalmanorix.village import Village
from kalmanorix.router import ScoutRouter
from kalmanorix.fusers import MeanFuser, KalmanorixFuser
from kalmanorix.panoramix import Panoramix

# 1) define SEF specialists
biomed = SEF(name="biomed", encoder="sentence-transformers/all-mpnet-base-v2")
legal = SEF(name="legal", encoder="sentence-transformers/multi-qa-mpnet-base-dot-v1")

# 2) place them in a Village
village = Village([biomed, legal])

# 3) select with ScoutRouter
router = ScoutRouter(village=village)
selected = router.route("what is first-line treatment for atrial fibrillation?")

# 4) fuse with Panoramix
panoramix = Panoramix(village=village, fusers=[MeanFuser(), KalmanorixFuser()])
run_dir = panoramix.run(queries=["sample query"], selected_specialists=[selected])

# 5) run routing/benchmark evaluation from CLI
# python -m kalmanorix.run_routing_eval --manifest manifests/routing_minimal.yaml
# python -m kalmanorix.run_claim_gate --run-dir <run_dir>
```


# Reproducibility, documentation, and release readiness

Kalmanorix emphasizes reproducibility through versioned benchmark manifests, explicit run metadata, deterministic output layout, and scripted artifact export. The repository includes automated tests and style checks, and project documentation includes installation, examples, API reference pages, and release/archival guidance.

Documentation entrypoint: `docs/index.md`.

Quickstart/example entrypoints: `docs/getting-started/quickstart.md` and `docs/examples/minimal-fusion.md`.

Licence statement: Kalmanorix is distributed under the MIT License (`LICENSE`).

CI status statement: the repository includes CI-oriented local gates (`ruff format --check .` and `pytest -m "not integration and not stress"`), with automation status tracked in repository CI configuration and badges when published.

Archive/release readiness: repository metadata includes `CITATION.cff` (with versioned software citation metadata), project versioning in `pyproject.toml`, and author/affiliation metadata in this manuscript front matter; release archiving is intended via a tagged release and Zenodo DOI minting workflow.

The reporting pipeline intentionally preserves unsupported and null findings so that negative-result reporting remains first-class in empirical records.

# Relationship to companion papers

This JOSS manuscript documents the **software contribution**: package architecture, reproducible workflows, and evidence-generation interfaces. Companion TMLR/arXiv manuscripts report **empirical and methodological results** obtained using this tooling. Across these outputs, claim language is governed by the repository evidence registry (`results/evidence_registry.json`), so statements about supported or unsupported effects track recorded benchmark evidence rather than narrative preference.

# AI usage disclosure

Generative AI tools were used as drafting assistants for portions of repository documentation and manuscript wording. All claims, citations, and scope statements were reviewed by the human author before submission.

# Acknowledgements

The author thanks maintainers of open-source Python IR and scientific computing ecosystems, and contributors who provided reproducibility feedback during development.

# References
