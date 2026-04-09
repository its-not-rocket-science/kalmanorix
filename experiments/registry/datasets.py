"""Dataset loaders for benchmark registry experiments."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from kalmanorix.toy_corpus import build_toy_corpus


def load_dataset(kind: str, path: Path | None, split: str, max_queries: int | None) -> Any:
    """Load dataset payload by kind."""
    if kind == "synthetic_toy":
        return build_toy_corpus(british_spelling=True)

    if kind == "mixed_parquet":
        if path is None:
            raise ValueError("dataset.path is required for mixed_parquet")
        import pyarrow.parquet as pq

        table = pq.read_table(path)
        rows = [row for row in table.to_pylist() if row.get("split") == split]
        if max_queries is not None:
            rows = rows[:max_queries]
        if not rows:
            raise ValueError(f"No rows found for split='{split}' in {path}")
        return rows

    if kind == "efficiency_query":
        return {"query": (path.read_text(encoding="utf-8") if path else None)}

    raise ValueError(f"Unsupported dataset kind: {kind}")
