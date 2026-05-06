"""Dataset loaders for benchmark registry experiments."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
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
) -> Any:
    """Load dataset payload by kind."""
    if kind == "synthetic_toy":
        return build_toy_corpus(british_spelling=True)

    if kind == "mixed_parquet":
        if path is None:
            raise ValueError("dataset.path is required for mixed_parquet")
        pyarrow_available = importlib.util.find_spec("pyarrow") is not None
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
                        if max_queries is not None and len(rows) >= max_queries:
                            break
                    if max_queries is not None and len(rows) >= max_queries:
                        break
            else:
                table = pq.read_table(path)
                source_rows = table.to_pylist()
                rows = [row for row in source_rows if row.get("split") == split]
                if max_queries is not None:
                    rows = rows[:max_queries]
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
            if max_queries is not None:
                rows = rows[:max_queries]
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
        if not rows:
            raise ValueError(f"No rows found for split='{split}' in {path}")
        return rows

    if kind == "efficiency_query":
        return {"query": (path.read_text(encoding="utf-8") if path else None)}

    raise ValueError(f"Unsupported dataset kind: {kind}")
