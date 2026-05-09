"""Dataset loaders for benchmark registry experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import random
from typing import Any

from kalmanorix.toy_corpus import build_toy_corpus


def load_dataset(
    kind: str,
    path: Path | None,
    split: str,
    max_queries: int | None,
    max_candidates: int | None = None,
    stream: bool = False,
    row_batch_size: int = 4096,
    domain_balanced: bool = False,
    seed: int = 0,
) -> Any:
    """Load dataset payload by kind."""
    if kind == "synthetic_toy":
        return build_toy_corpus(british_spelling=True)

    if kind == "mixed_parquet":
        if path is None:
            raise ValueError("dataset.path is required for mixed_parquet")
        pyarrow_available = importlib.util.find_spec("pyarrow") is not None

        def _sample_domain_balanced(
            source_rows: list[dict[str, Any]],
        ) -> list[dict[str, Any]]:
            if max_queries is None or max_queries <= 0:
                return source_rows
            buckets: dict[str, list[dict[str, Any]]] = {}
            for row in source_rows:
                domain = str(row.get("domain", "__unknown__"))
                buckets.setdefault(domain, []).append(row)
            domains = sorted(buckets)
            if not domains:
                return source_rows[:max_queries]
            rng = random.Random(seed)
            for rows_for_domain in buckets.values():
                rng.shuffle(rows_for_domain)
            target = max_queries // len(domains)
            remainder = max_queries % len(domains)
            sampled: list[dict[str, Any]] = []
            for idx, domain in enumerate(domains):
                want = target + (1 if idx < remainder else 0)
                sampled.extend(buckets[domain][:want])
                buckets[domain] = buckets[domain][want:]
            if len(sampled) < max_queries:
                leftovers: list[dict[str, Any]] = []
                for domain in domains:
                    leftovers.extend(buckets[domain])
                rng.shuffle(leftovers)
                sampled.extend(leftovers[: max_queries - len(sampled)])
            return sampled

        if pyarrow_available and path.suffix == ".parquet":
            import pyarrow.parquet as pq

            if stream:
                rows = []
                for batch in pq.ParquetFile(path).iter_batches(
                    batch_size=max(1, row_batch_size)
                ):
                    for row in batch.to_pylist():
                        if row.get("split") != split:
                            continue
                        if max_candidates is not None:
                            row = {
                                **row,
                                "candidate_documents": row.get(
                                    "candidate_documents", []
                                )[:max_candidates],
                            }
                        rows.append(row)
            else:
                table = pq.read_table(path)
                source_rows = table.to_pylist()
                rows = [row for row in source_rows if row.get("split") == split]
                rows = (
                    _sample_domain_balanced(rows)
                    if domain_balanced
                    else (rows[:max_queries] if max_queries is not None else rows)
                )
                if max_candidates is not None:
                    rows = [
                        {
                            **row,
                            "candidate_documents": row.get("candidate_documents", [])[
                                :max_candidates
                            ],
                        }
                        for row in rows
                    ]
        else:
            source_rows = json.loads(path.read_text(encoding="utf-8"))
            rows = [row for row in source_rows if row.get("split") == split]
            rows = (
                _sample_domain_balanced(rows)
                if domain_balanced
                else (rows[:max_queries] if max_queries is not None else rows)
            )
            if max_candidates is not None:
                rows = [
                    {
                        **row,
                        "candidate_documents": row.get("candidate_documents", [])[
                            :max_candidates
                        ],
                    }
                    for row in rows
                ]
        if stream and max_queries is not None:
            rows = (
                _sample_domain_balanced(rows) if domain_balanced else rows[:max_queries]
            )
        if not rows:
            raise ValueError(f"No rows found for split='{split}' in {path}")
        return rows

    if kind == "efficiency_query":
        return {"query": (path.read_text(encoding="utf-8") if path else None)}

    raise ValueError(f"Unsupported dataset kind: {kind}")
