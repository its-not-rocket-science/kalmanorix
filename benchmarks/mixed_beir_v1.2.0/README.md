# mixed_beir_v1.2.0 benchmark (hard-query expansion)

This benchmark version preserves prior artifacts (`v1.0.0`, `v1.1.0`) and introduces a harder held-out split with explicit query metadata.

## Intended upgrades

- Larger held-out test pool via higher per-domain caps.
- Additional difficult query categories:
  - ambiguous_cross_domain
  - misleading_lexical_overlap
  - long_tail_domain_terms
  - mixed_intent
  - adversarial_near_miss
- Query metadata fields in each row:
  - `dominant_domain`
  - `secondary_domain`
  - `ambiguity_category`
  - `ambiguity_score`
  - `fusion_usefulness_bucket`
  - `query_category`
  - `is_synthetic`
  - `provenance_note`

## Provenance rules

- Rows with `is_synthetic=true` are synthetic expansions derived from held-out BEIR test queries.
- Rows with `is_synthetic=false` are original BEIR rows.
- Synthetic rows are **not** represented as non-synthetic real data.

## Rebuild command

```bash
PYTHONPATH=src python scripts/build_mixed_benchmark.py \
  --output-dir benchmarks/mixed_beir_v1.2.0 \
  --seed 1337 \
  --max-candidates 80 \
  --cross-domain-negative-ratio 0.60 \
  --max-queries-per-domain 1800 \
  --max-test-queries-per-domain 360 \
  --hard-queries-per-category-per-domain 20
```

Expected artifacts are emitted by the builder (`corpus`, `queries`, `qrels`, `splits`, benchmark file, and manifest).
