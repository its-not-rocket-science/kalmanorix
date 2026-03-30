# Scout Router API

*TODO: Auto‑generated API documentation for the `ScoutRouter` and semantic‑routing utilities.*

The `ScoutRouter` selects which specialists to consult for a given query. It supports three modes:

1. **`mode="all"`** – Use all specialists (fusion mode).
2. **`mode="hard"`** – Select exactly one specialist (winner‑takes‑all).
3. **`mode="semantic"`** – Select specialists whose domain centroid is sufficiently similar to the query (dynamic thresholding).

::: kalmanorix.scout
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Threshold Functions

- `threshold_top_k(k=1)` – Select top‑k closest specialists.
- `threshold_relative_to_max(ratio=0.8)` – Select specialists whose similarity is at least `ratio × max_similarity`.
- `threshold_adaptive_spread()` – Use spread‑based adaptive threshold.
- `threshold_query_length_adaptive()` – Adjust threshold based on query length.

*TODO: Add routing‑efficiency benchmarks and guidance on choosing a threshold.*
