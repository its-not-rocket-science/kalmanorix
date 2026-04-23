# Mixed-Domain Retrieval Benchmark (Real-Data, Reproducible)

This benchmark is designed for **Kalmanorix retrieval/fusion evaluation** with real corpora and query→relevance labels.

## 1) Exact datasets to use (links + licenses)

Use three BEIR tasks with retrieval triplets `(corpus, queries, qrels)` and distinct domains.
Implementation note: benchmark configs must treat these components as independently sourced
(dataset/config/split), because some BEIR Hugging Face datasets do not expose `qrels` as a
builder config on the same dataset entry as `corpus`/`queries`.

| Domain | Dataset | Why it fits | Link | License |
|---|---|---|---|---|
| General QA | **BEIR / NQ** (Natural Questions retrieval split) | Open-domain QA over Wikipedia-scale corpus. | https://huggingface.co/datasets/BeIR/nq | `cc-by-sa-4.0` (as published on the dataset card) |
| Biomedical / Scientific | **BEIR / SciFact** | Scientific claim verification retrieval (biomed-heavy literature). | https://huggingface.co/datasets/BeIR/scifact | `cc-by-4.0` (as published on the dataset card) |
| Finance | **BEIR / FiQA** | Financial question answering / retrieval over financial text. | https://huggingface.co/datasets/BeIR/fiqa | `cc-by-nc-4.0` (as published on the dataset card) |

### Version pinning (required for reproducibility)

- Pin Python dependencies in `requirements-benchmark.txt` (or equivalent):
  - `datasets==2.19.0`
  - `ir-datasets==0.5.9`
  - `pyarrow==15.0.2`
- Create `benchmark_manifest.json` with:
  - dataset name (`BeIR/nq`, `BeIR/scifact`, `BeIR/fiqa`)
  - download timestamp (UTC)
  - source URL
  - file checksums (`sha256`) for every downloaded shard/file
  - row counts for corpus/queries/qrels
- Treat that manifest as the benchmark version record (e.g., `v1.0.0`).

This gives deterministic replay even if upstream hosting moves.

---

## 2) How to unify them into a single benchmark

### Canonical approach

1. Load each BEIR subset into a canonical schema:
   - `corpus(doc_id, title, text)`
   - `queries(query_id, text)`
   - `qrels(query_id, doc_id, relevance)`
2. Namespace all IDs to avoid collisions:
   - `nq:doc123`, `scifact:doc55`, `fiqa:doc901`
   - `nq:q12`, `scifact:q7`, `fiqa:q3`
3. Concatenate corpora into one **global corpus**.
4. Concatenate queries into one **global query table**.
5. Concatenate qrels unchanged except for namespaced IDs.
6. Keep one split file per domain (e.g., BEIR test split), then publish a global split map:
   - `split=eval_mixed_v1`
   - optionally domain-balanced micro-splits (e.g., 1k queries/domain).

### Evaluation modes

- **Mixed retrieval**: retrieve over the full combined corpus.
- **Domain-aware slice metrics**: compute nDCG@10/Recall@k by domain label.
- **Calibration slice**: compare in-domain vs cross-domain confusion for routed specialists.

---

## 3) How to label domain membership per query

Add explicit domain metadata at query level:

- `domain`: one of `{general_qa, biomedical, finance}`
- `source_dataset`: one of `{beir_nq, beir_scifact, beir_fiqa}`
- `domain_id`: integer mapping for model input convenience (e.g., `0,1,2`)

Rule is deterministic:
- all queries from `BeIR/nq` → `general_qa`
- all queries from `BeIR/scifact` → `biomedical`
- all queries from `BeIR/fiqa` → `finance`

No classifier needed; labels are inherited from source dataset identity.

---

## 4) Data preprocessing pipeline

Use a single reproducible pipeline script (e.g., `scripts/build_mixed_benchmark.py`).

### Pipeline steps

1. **Fetch**
   - Download each dataset split via Hugging Face `datasets`.
2. **Normalize text**
   - Unicode NFKC normalization
   - collapse repeated whitespace
   - strip null bytes/control chars
3. **Canonicalize fields**
   - missing titles → empty string
   - document body from `text`
4. **Namespace IDs**
   - prefix all query/doc IDs by dataset key
5. **Filter invalid rows**
   - drop empty queries
   - drop qrels pointing to missing docs/queries
6. **Deduplicate exact duplicates**
   - dedup docs by `(title, text)` hash *within each source*
   - maintain `duplicate_of` mapping for auditability
7. **Assemble global artifacts**
   - write `corpus.parquet`, `queries.parquet`, `qrels.parquet`
   - write `benchmark_manifest.json`
8. **Validate constraints**
   - every query has ≥1 relevant document
   - every qrel doc exists in corpus
   - no ID collisions
9. **Freeze version**
   - compute final artifact checksums
   - tag benchmark release (e.g., `mixed_beir_v1.0.0`)

---

## 5) Final dataset schema

Recommended folder structure:

```text
benchmarks/mixed_beir_v1/
  corpus.parquet
  queries.parquet
  qrels.parquet
  splits.parquet
  benchmark_manifest.json
  LICENSES/
    beir_nq_LICENSE.txt
    beir_scifact_LICENSE.txt
    beir_fiqa_LICENSE.txt
```

### `corpus.parquet`

- `doc_id: string` (namespaced, unique)
- `source_doc_id: string`
- `source_dataset: string`
- `domain: string`
- `title: string`
- `text: string`
- `text_hash: string` (sha256 of normalized title+text)

### `queries.parquet`

- `query_id: string` (namespaced, unique)
- `source_query_id: string`
- `source_dataset: string`
- `domain: string`
- `domain_id: int8`
- `query_text: string`
- `split: string` (e.g., `eval_mixed_v1`)

### `qrels.parquet`

- `query_id: string`
- `doc_id: string`
- `relevance: int8` (keep source relevance scale)
- `source_dataset: string`

### `splits.parquet`

- `query_id: string`
- `split: string`
- `domain: string`

### `benchmark_manifest.json`

- benchmark version
- build timestamp
- tool/package versions
- dataset URLs and declared licenses
- per-file checksums and row counts
- validation summary

---

## Why this benchmark is suitable for Kalmanorix

- It is **truly mixed-domain** (general + biomedical + finance).
- It is built from **real retrieval datasets** with official qrels.
- It supports **global retrieval** and **per-domain diagnostics**, which is exactly what specialist routing + fusion needs.
- It is **versioned and reproducible** via manifest + checksums + pinned dependencies.
