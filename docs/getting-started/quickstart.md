# Quickstart

Get started with Kalmanorix in under 2 minutes on CPU.

## Run the executable quickstart

Use the included script (deterministic, no network required):

```bash
python examples/quickstart_cpu.py
```

This prints router selections and fusion weights for three toy queries.

## Core concepts

- **SEF**: Specialist Embedding Format wrapper (`name`, `embed`, `sigma2`)
- **Village**: runtime container of SEFs
- **ScoutRouter**: specialist selector (`all`, `hard`, `semantic`)
- **Panoramix**: orchestration layer for route/fuse
- **Fusers**: strategies such as `MeanFuser` and `KalmanorixFuser`

## Minimal import surface

```python
from kalmanorix import (
    SEF,
    Village,
    ScoutRouter,
    Panoramix,
    MeanFuser,
    KalmanorixFuser,
)
from kalmanorix.experimental import DiagonalKalmanFuser, threshold_top_k
```

## Next steps

- See `tests/e2e/test_toy_pipeline.py` for an end-to-end smoke path.
- See API pages under `docs/api-reference/` for class-level docs.
