# Real Specialist Embedding Models: Implementation Plan

This document proposes a concrete, config-driven integration path for adding real public specialist embedding models to Kalmanorix while preserving the existing `SEF` + `Village` API.

## Current codebase constraints (scan summary)

- `SEF` currently requires `name`, `embed`, and `sigma2`, where `sigma2` may be a constant or query-dependent callable. This already supports uncertainty hooks required for specialists.
- `Village` is a thin container over a list of `SEF` modules and is already suitable for dynamic specialist composition.
- `create_huggingface_sef(...)` exists and can instantiate local/public Hugging Face models, but current registration is mostly code-driven rather than YAML-driven.
- `ModelRegistry` discovers persisted SEF artefacts from disk, but does not yet provide first-class loading from a specialist config that references public model IDs.

## 1) Exact model choices and justification

Use **four genuinely different specialists** with different training data and inductive biases:

### A. General semantic specialist
- **Model**: `sentence-transformers/all-mpnet-base-v2`
- **Domain**: general semantic retrieval / web + mixed corpora
- **Why**: strong general STS/retrieval baseline, robust for mixed-domain queries
- **Inductive bias**: broad sentence-level semantic similarity

### B. Biomedical specialist
- **Model**: `cambridgeltl/SapBERT-from-PubMedBERT-fulltext`
- **Domain**: biomedical entities (clinical terms, UMLS-like synonyms, scientific biomedical language)
- **Why**: domain-pretrained on biomedical text and medical concept synonymy
- **Inductive bias**: biomedical terminology + concept-level alignment

### C. Legal specialist
- **Model**: `nlpaueb/legal-bert-base-uncased`
- **Domain**: legal/contracts/statutory text
- **Why**: legal-domain BERT pretrained on legal corpora; better legal phrase semantics than general models
- **Inductive bias**: legal vocabulary and long-form formal syntax

### D. Code specialist
- **Model**: `microsoft/codebert-base`
- **Domain**: source code + code-documentation pairs
- **Why**: trained for NL-code alignment and code understanding; suitable for code search/routing
- **Inductive bias**: programming-language token structure + NL↔code association

> Optional future fifth specialist: multilingual (`intfloat/multilingual-e5-base`) when cross-lingual routing is needed.

## Uncertainty (`sigma²`) method per specialist

All specialists must expose `sigma2_for(query)` through `SEF.sigma2` callable. Start with a shared heuristic:

- Compute specialist centroid from domain calibration texts.
- Let `sim = cosine(query_emb, centroid)`.
- Define `sigma2(query) = base_sigma2 + scale * (1 - max(sim, -1)) / 2`.

This maps high in-domain similarity to lower variance and OOD inputs to higher variance.

Initial defaults:
- General: `base_sigma2=0.06`, `scale=0.20`
- Biomedical: `base_sigma2=0.05`, `scale=0.30`
- Legal: `base_sigma2=0.05`, `scale=0.28`
- Code: `base_sigma2=0.04`, `scale=0.32`

## 2) Code changes needed to integrate cleanly

### 2.1 Add typed config schema for specialists

**New file**: `src/kalmanorix/specialist_config.py`

Add dataclasses:
- `UncertaintyConfig(method, base_sigma2, scale, min_sigma2, max_sigma2)`
- `SpecialistConfig(name, domain, provider, model_name_or_path, pooling, normalize, device, calibration_texts_path, uncertainty, metadata)`
- `SpecialistRegistryConfig(specialists: list[SpecialistConfig])`

Include:
- `from_yaml(path)` / `to_yaml(path)`
- validation for unique names, supported providers, and required domain labels.

### 2.2 Add config-driven specialist factory

**New file**: `src/kalmanorix/specialist_factory.py`

Core functions:
- `build_sigma2_callable(embedder, uncertainty_cfg, calibration_texts)`
- `build_sef_from_config(spec_cfg) -> SEF`
- `build_village_from_config(path_or_config) -> Village`

Implementation notes:
- For `provider: huggingface`, call existing `create_huggingface_sef(...)`.
- Replace constant `sigma2` with a callable generated from calibration centroid.
- Attach `meta` fields: `{domain, model_id, provider, uncertainty_method}`.
- Optionally call `.with_domain_centroid(calibration_texts)` for semantic routing.

### 2.3 Register config loader in public API

**Modify**: `src/kalmanorix/__init__.py`

Export:
- `SpecialistConfig`
- `SpecialistRegistryConfig`
- `build_village_from_config`

### 2.4 Add tests for config-driven specialists

**New tests**:
- `tests/test_specialist_config.py`
  - YAML parsing/validation
  - duplicate specialist name rejection
- `tests/e2e/test_real_specialist_registry.py`
  - build 2-4 mocked specialists from config
  - assert every specialist has callable `sigma2` and non-empty domain metadata

## 3) Config-driven registration (no hardcoding)

### Proposed YAML

**New file**: `examples/configs/real_specialists.yaml`

```yaml
specialists:
  - name: general_mpnet
    domain: general
    provider: huggingface
    model_name_or_path: sentence-transformers/all-mpnet-base-v2
    pooling: mean
    normalize: true
    device: cpu
    calibration_texts_path: examples/data/calibration/general.txt
    uncertainty:
      method: centroid_distance
      base_sigma2: 0.06
      scale: 0.20
      min_sigma2: 0.03
      max_sigma2: 0.60

  - name: biomedical_sapbert
    domain: biomedical
    provider: huggingface
    model_name_or_path: cambridgeltl/SapBERT-from-PubMedBERT-fulltext
    pooling: mean
    normalize: true
    device: cpu
    calibration_texts_path: examples/data/calibration/biomedical.txt
    uncertainty:
      method: centroid_distance
      base_sigma2: 0.05
      scale: 0.30
      min_sigma2: 0.03
      max_sigma2: 0.70

  - name: legal_bert
    domain: legal
    provider: huggingface
    model_name_or_path: nlpaueb/legal-bert-base-uncased
    pooling: mean
    normalize: true
    device: cpu
    calibration_texts_path: examples/data/calibration/legal.txt
    uncertainty:
      method: centroid_distance
      base_sigma2: 0.05
      scale: 0.28
      min_sigma2: 0.03
      max_sigma2: 0.65

  - name: code_codebert
    domain: code
    provider: huggingface
    model_name_or_path: microsoft/codebert-base
    pooling: mean
    normalize: true
    device: cpu
    calibration_texts_path: examples/data/calibration/code.txt
    uncertainty:
      method: centroid_distance
      base_sigma2: 0.04
      scale: 0.32
      min_sigma2: 0.02
      max_sigma2: 0.75
```

## 4) Minimal working example wiring them into a Village

**New file**: `examples/real_specialists_village_demo.py`

```python
from kalmanorix import build_village_from_config

village = build_village_from_config("examples/configs/real_specialists.yaml")

print(village.list())
for m in village.modules:
    q = "Summarize indemnification clause for software license."
    print(m.name, m.meta.get("domain"), m.sigma2_for(q))
```

This produces a `Village` with four real public specialists and per-query uncertainty, with zero hardcoded model IDs in Python.

## Rollout plan

1. Land config schema + loader + tests with one HF specialist fixture.
2. Add 4-specialist example config and docs.
3. Add optional cache/offline notes for model weights.
4. Add benchmark script comparing specialist routing vs general-only baseline.
