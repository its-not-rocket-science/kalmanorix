"""Build a deterministic mixed-domain retrieval benchmark from BEIR subsets."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

BENCHMARK_VERSION = "mixed_beir_v1.0.0"
DEFAULT_SEED = 1337
DEFAULT_OUTPUT_DIR = Path("benchmarks") / BENCHMARK_VERSION
DATASET_SPECS = (
    {
        "hf_name": "BeIR/nq",
        "key": "nq",
        "source_dataset": "beir_nq",
        "domain": "general_qa",
        "domain_id": 0,
        "url": "https://huggingface.co/datasets/BeIR/nq",
        "license": "cc-by-sa-4.0",
    },
    {
        "hf_name": "BeIR/scifact",
        "key": "scifact",
        "source_dataset": "beir_scifact",
        "domain": "biomedical",
        "domain_id": 1,
        "url": "https://huggingface.co/datasets/BeIR/scifact",
        "license": "cc-by-4.0",
    },
    {
        "hf_name": "BeIR/fiqa",
        "key": "fiqa",
        "source_dataset": "beir_fiqa",
        "domain": "finance",
        "domain_id": 2,
        "url": "https://huggingface.co/datasets/BeIR/fiqa",
        "license": "cc-by-nc-4.0",
    },
)


@dataclass(frozen=True)
class SplitRatios:
    """Train/validation/test split ratios."""

    train: float = 0.8
    validation: float = 0.1
    test: float = 0.1


def _normalize_text(text: str) -> str:
    cleaned = unicodedata.normalize("NFKC", text or "")
    cleaned = cleaned.replace("\x00", " ")
    cleaned = re.sub(r"[\x00-\x1F\x7F]", " ", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _load_split(dataset_name: str, config: str, split_name: str):
    from datasets import load_dataset

    return load_dataset(dataset_name, config, split=split_name)


def _load_beir_triplet(dataset_name: str) -> tuple[Any, Any, Any]:
    candidates = {
        "corpus": ["corpus"],
        "queries": ["queries"],
        "qrels": ["qrels", "default"],
    }

    corpus = None
    queries = None
    qrels = None
    last_error = None

    for qrels_config in candidates["qrels"]:
        try:
            corpus = _load_split(dataset_name, "corpus", "corpus")
            queries = _load_split(dataset_name, "queries", "queries")
            qrels = _load_split(dataset_name, qrels_config, "test")
            return corpus, queries, qrels
        except Exception as exc:  # pragma: no cover - network/data source variation
            last_error = exc

    raise RuntimeError(f"Unable to load BEIR triplet for {dataset_name}: {last_error}")


def _build_domain_tables(
    spec: dict[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, str], dict[str, int]]:
    corpus_ds, queries_ds, qrels_ds = _load_beir_triplet(spec["hf_name"])

    doc_map: dict[str, dict[str, Any]] = {}
    duplicate_of: dict[str, str] = {}
    content_index: dict[str, str] = {}

    for row in corpus_ds:
        src_doc_id = str(row.get("_id") or row.get("doc_id") or row.get("id"))
        if not src_doc_id:
            continue
        title = _normalize_text(str(row.get("title") or ""))
        text = _normalize_text(str(row.get("text") or row.get("body") or ""))
        if not text:
            continue

        key = _sha256(f"{title}\n{text}")
        namespaced_doc_id = f"{spec['key']}:{src_doc_id}"

        if key in content_index:
            duplicate_of[namespaced_doc_id] = content_index[key]
            continue

        content_index[key] = namespaced_doc_id
        doc_map[namespaced_doc_id] = {
            "doc_id": namespaced_doc_id,
            "source_doc_id": src_doc_id,
            "source_dataset": spec["source_dataset"],
            "domain": spec["domain"],
            "title": title,
            "text": text,
            "text_hash": key,
        }

    query_map: dict[str, dict[str, Any]] = {}
    for row in queries_ds:
        src_query_id = str(row.get("_id") or row.get("query_id") or row.get("id"))
        if not src_query_id:
            continue
        query_text = _normalize_text(str(row.get("text") or row.get("query") or ""))
        if not query_text:
            continue

        query_id = f"{spec['key']}:{src_query_id}"
        query_map[query_id] = {
            "query_id": query_id,
            "source_query_id": src_query_id,
            "source_dataset": spec["source_dataset"],
            "domain": spec["domain"],
            "domain_id": spec["domain_id"],
            "query_text": query_text,
        }

    qrels: list[dict[str, Any]] = []
    relevance_counts: dict[str, int] = {}

    for row in qrels_ds:
        src_query_id = str(row.get("query-id") or row.get("query_id") or row.get("qid"))
        src_doc_id = str(row.get("corpus-id") or row.get("doc_id") or row.get("docid"))
        rel = int(row.get("score") or row.get("relevance") or 0)

        query_id = f"{spec['key']}:{src_query_id}"
        doc_id = f"{spec['key']}:{src_doc_id}"

        canonical_doc_id = duplicate_of.get(doc_id, doc_id)

        if rel <= 0 or query_id not in query_map or canonical_doc_id not in doc_map:
            continue

        qrels.append(
            {
                "query_id": query_id,
                "doc_id": canonical_doc_id,
                "relevance": rel,
                "source_dataset": spec["source_dataset"],
            }
        )
        relevance_counts[query_id] = relevance_counts.get(query_id, 0) + 1

    # remove queries without positives
    query_map = {
        qid: qrow
        for qid, qrow in query_map.items()
        if relevance_counts.get(qid, 0) > 0
    }

    qrels = [row for row in qrels if row["query_id"] in query_map]

    return doc_map, query_map, qrels, duplicate_of, relevance_counts


def _deterministic_split(query_rows: list[dict[str, Any]], seed: int, ratios: SplitRatios) -> dict[str, str]:
    if not query_rows:
        return {}

    if abs((ratios.train + ratios.validation + ratios.test) - 1.0) > 1e-9:
        raise ValueError("Split ratios must sum to 1.0")

    grouped: dict[str, list[str]] = {}
    for row in query_rows:
        grouped.setdefault(row["domain"], []).append(row["query_id"])

    rng = random.Random(seed)
    split_map: dict[str, str] = {}

    for domain, query_ids in grouped.items():
        ordered = sorted(query_ids)
        rng.shuffle(ordered)
        n_total = len(ordered)
        n_train = int(n_total * ratios.train)
        n_validation = int(n_total * ratios.validation)
        n_test = n_total - n_train - n_validation

        if n_total >= 3:
            n_train = max(1, n_train)
            n_validation = max(1, n_validation)
            n_test = max(1, n_test)
            overflow = (n_train + n_validation + n_test) - n_total
            if overflow > 0:
                n_train = max(1, n_train - overflow)

        for idx, query_id in enumerate(ordered):
            if idx < n_train:
                split_map[query_id] = "train"
            elif idx < n_train + n_validation:
                split_map[query_id] = "validation"
            else:
                split_map[query_id] = "test"

    return split_map


def _build_query_records(
    query_rows: list[dict[str, Any]],
    doc_map: dict[str, dict[str, Any]],
    qrels_rows: list[dict[str, Any]],
    split_map: dict[str, str],
    max_candidates: int,
    seed: int,
) -> list[dict[str, Any]]:
    positives_by_query: dict[str, set[str]] = {}
    for row in qrels_rows:
        positives_by_query.setdefault(row["query_id"], set()).add(row["doc_id"])

    docs_by_domain: dict[str, list[str]] = {}
    for doc_id, doc in doc_map.items():
        docs_by_domain.setdefault(doc["domain"], []).append(doc_id)

    rng = random.Random(seed)
    records: list[dict[str, Any]] = []

    for row in sorted(query_rows, key=lambda r: r["query_id"]):
        query_id = row["query_id"]
        positive_ids = sorted(positives_by_query.get(query_id, set()))
        if not positive_ids:
            continue

        domain_docs = sorted(docs_by_domain.get(row["domain"], []))
        negatives = [doc_id for doc_id in domain_docs if doc_id not in positive_ids]
        needed_negatives = max(0, max_candidates - len(positive_ids))
        sampled_negatives = rng.sample(negatives, k=min(needed_negatives, len(negatives)))

        candidate_ids = positive_ids + sampled_negatives
        candidate_docs = [
            {
                "doc_id": doc_map[doc_id]["doc_id"],
                "title": doc_map[doc_id]["title"],
                "text": doc_map[doc_id]["text"],
                "domain": doc_map[doc_id]["domain"],
                "source_dataset": doc_map[doc_id]["source_dataset"],
            }
            for doc_id in candidate_ids
        ]

        records.append(
            {
                "query_id": query_id,
                "query_text": row["query_text"],
                "candidate_documents": candidate_docs,
                "ground_truth_relevant_ids": positive_ids,
                "domain_label": row["domain"],
                "source_dataset": row["source_dataset"],
                "split": split_map.get(query_id, "train"),
            }
        )

    return records


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_mixed_domain_benchmark(
    output_dir: Path | str = DEFAULT_OUTPUT_DIR,
    seed: int = DEFAULT_SEED,
    max_candidates: int = 50,
    split_ratios: SplitRatios = SplitRatios(),
) -> dict[str, Any]:
    """Download, preprocess, split, and persist a mixed-domain benchmark."""
    from datasets import __version__ as datasets_version
    import pyarrow as pa
    import pyarrow.parquet as pq

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_docs: dict[str, dict[str, Any]] = {}
    all_queries: dict[str, dict[str, Any]] = {}
    all_qrels: list[dict[str, Any]] = []
    duplicate_map: dict[str, str] = {}
    source_row_counts: dict[str, dict[str, int]] = {}

    for spec in DATASET_SPECS:
        docs, queries, qrels, duplicates, _ = _build_domain_tables(spec)

        if set(all_docs).intersection(docs):
            raise ValueError("Document ID collision detected")
        if set(all_queries).intersection(queries):
            raise ValueError("Query ID collision detected")

        all_docs.update(docs)
        all_queries.update(queries)
        all_qrels.extend(qrels)
        duplicate_map.update(duplicates)

        source_row_counts[spec["source_dataset"]] = {
            "corpus": len(docs),
            "queries": len(queries),
            "qrels": len([r for r in qrels if r["source_dataset"] == spec["source_dataset"]]),
        }

    query_rows = list(all_queries.values())
    split_map = _deterministic_split(query_rows, seed, split_ratios)

    for query in query_rows:
        query["split"] = split_map[query["query_id"]]

    split_rows = [
        {"query_id": q["query_id"], "split": q["split"], "domain": q["domain"]}
        for q in sorted(query_rows, key=lambda r: r["query_id"])
    ]

    benchmark_records = _build_query_records(
        query_rows=query_rows,
        doc_map=all_docs,
        qrels_rows=all_qrels,
        split_map=split_map,
        max_candidates=max_candidates,
        seed=seed,
    )

    # Persist canonical tables + single benchmark file
    corpus_path = output_dir / "corpus.parquet"
    queries_path = output_dir / "queries.parquet"
    qrels_path = output_dir / "qrels.parquet"
    splits_path = output_dir / "splits.parquet"
    benchmark_path = output_dir / "mixed_benchmark.parquet"

    pq.write_table(pa.Table.from_pylist(sorted(all_docs.values(), key=lambda r: r["doc_id"])), corpus_path)
    pq.write_table(pa.Table.from_pylist(sorted(query_rows, key=lambda r: r["query_id"])), queries_path)
    pq.write_table(pa.Table.from_pylist(sorted(all_qrels, key=lambda r: (r["query_id"], r["doc_id"]))), qrels_path)
    pq.write_table(pa.Table.from_pylist(split_rows), splits_path)
    pq.write_table(pa.Table.from_pylist(benchmark_records), benchmark_path)

    output_files = [corpus_path, queries_path, qrels_path, splits_path, benchmark_path]

    manifest = {
        "benchmark_version": BENCHMARK_VERSION,
        "build_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": seed,
        "split_ratios": {
            "train": split_ratios.train,
            "validation": split_ratios.validation,
            "test": split_ratios.test,
        },
        "max_candidates_per_query": max_candidates,
        "packages": {
            "datasets": datasets_version,
            "pyarrow": pa.__version__,
        },
        "datasets": [
            {
                "name": spec["source_dataset"],
                "hf_dataset": spec["hf_name"],
                "url": spec["url"],
                "license": spec["license"],
                "row_counts": source_row_counts[spec["source_dataset"]],
            }
            for spec in DATASET_SPECS
        ],
        "artifacts": {
            path.name: {
                "sha256": _file_sha256(path),
                "rows": len(benchmark_records) if path == benchmark_path else None,
            }
            for path in output_files
        },
        "validation": {
            "queries_with_labels": len(query_rows),
            "qrels": len(all_qrels),
            "documents": len(all_docs),
            "duplicates_removed": len(duplicate_map),
            "all_queries_have_relevant": True,
        },
    }

    for path, rows in (
        (corpus_path, len(all_docs)),
        (queries_path, len(query_rows)),
        (qrels_path, len(all_qrels)),
        (splits_path, len(split_rows)),
        (benchmark_path, len(benchmark_records)),
    ):
        manifest["artifacts"][path.name]["rows"] = rows

    manifest_path = output_dir / "benchmark_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["artifacts"][manifest_path.name] = {
        "sha256": _file_sha256(manifest_path),
        "rows": None,
    }

    return manifest


def main() -> None:
    """CLI entry point for benchmark regeneration."""
    parser = argparse.ArgumentParser(
        description="Download and build the mixed-domain BEIR benchmark"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=50,
        help="Maximum candidate documents stored per query",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio",
    )
    parser.add_argument(
        "--validation-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.1,
        help="Test split ratio",
    )
    args = parser.parse_args()

    split_ratios = SplitRatios(
        train=args.train_ratio,
        validation=args.validation_ratio,
        test=args.test_ratio,
    )

    manifest = build_mixed_domain_benchmark(
        output_dir=args.output_dir,
        seed=args.seed,
        max_candidates=args.max_candidates,
        split_ratios=split_ratios,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
