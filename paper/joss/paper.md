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

Kalmanorix is an MIT-licensed Python package for retrieval benchmarks with specialist embedding models. It defines specialist embedders (`SEF`), groups them in a `Village`, routes queries with `ScoutRouter`, and compares fusion baselines through `Panoramix`. It also writes claim-gated outputs that label findings as supported, unsupported, or inconclusive in machine-readable files.

# Statement of need

Teams can already encode text, build ANN indexes, and run search with libraries such as SentenceTransformers, BEIR-style tasks, and FAISS [@thakur2021beir; @jarvelin2002cumulated]. What is often missing is a stable evaluation loop for routing and fusion decisions. Many projects still depend on notebook-only scripts, so result interpretation changes across reruns and negative findings are lost.

Kalmanorix addresses this by standardizing benchmark manifests, run metadata, and claim rules in one package. It is built for IR researchers, embedding researchers, and evaluation researchers who need repeatable benchmark outputs they can audit.

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

Advanced usage combines manifests, router setup, fusion sweeps, and claim reporting in one run directory. Main entrypoints are `kalmanorix.panoramix.Panoramix`, `kalmanorix.run_routing_eval`, and `kalmanorix.run_claim_gate`. The older `kalmanorix.kalman_engine.fuser.Panoramix` path remains as a deprecated shim.

# Software architecture and functionality

The public API maps directly to benchmark steps:

- **SEF** (`kalmanorix.sef.SEF`): defines one specialist embedding backend and metadata.
- **Village** (`kalmanorix.village.Village`): stores the specialist set used for routing.
- **ScoutRouter** (`kalmanorix.router.ScoutRouter`): picks a specialist per query or query batch.
- **Panoramix** (`kalmanorix.panoramix.Panoramix`): runs retrieval/fusion experiments and writes outputs.
- **MeanFuser** (`kalmanorix.fusers.MeanFuser`): arithmetic-mean baseline.
- **KalmanorixFuser** (`kalmanorix.fusers.KalmanorixFuser`): uncertainty-aware baseline for comparison.
- **routing evaluator CLI** (`kalmanorix.run_routing_eval`): evaluates routing quality by domain/slice.
- **claim-gate tools** (`kalmanorix.run_claim_gate` and `results/evidence_registry.json`): convert metrics into claim status records with provenance.

Kalmanorix complements existing retrieval libraries instead of replacing them. In common setups, encoding comes from SentenceTransformers, dataset/task structure from BEIR-style benchmarks, and vector search from FAISS. Kalmanorix adds routing/fusion evaluation and claim reporting around that stack [@thakur2021beir].

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

Reproducibility in Kalmanorix depends on versioned manifests, explicit run metadata, deterministic output paths, and scripted artifact export. The repository also includes automated tests and formatting checks.

Documentation entrypoint: `docs/index.md`.

Quickstart/example entrypoints: `docs/getting-started/quickstart.md` and `docs/examples/minimal-fusion.md`.

Licence statement: Kalmanorix is distributed under the MIT License (`LICENSE`).

CI gates referenced in this repository: `ruff format --check .` and `pytest -m "not integration and not stress"`.

Release metadata is tracked in `CITATION.cff`, `pyproject.toml`, and this manuscript front matter. Archive publication is planned through a tagged release with Zenodo DOI minting.

The reporting pipeline keeps unsupported and null findings instead of filtering them out.

# Relationship to companion papers

This JOSS manuscript describes the software: package structure, interfaces, and reproducible workflows. Companion TMLR/arXiv manuscripts report empirical outcomes produced with this software. Claim wording across outputs is intended to follow `results/evidence_registry.json`.

# AI usage disclosure

Generative AI tools were used as drafting assistants for parts of documentation and manuscript wording. The human author reviewed claims, citations, and scope statements before submission.

# Acknowledgements

The author thanks maintainers of open-source Python IR and scientific computing ecosystems, and contributors who shared reproducibility feedback during development.

# References
