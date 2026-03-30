# Panoramix API

*TODO: Auto‑generated API documentation for the `Panoramix` orchestrator and `Fuser` abstraction.*

`Panoramix` is the high‑level fusion orchestrator. It combines a `Village` (specialists), a `ScoutRouter` (selection), and a `Fuser` (fusion strategy) to produce fused embeddings.

::: kalmanorix.panoramix
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Fuser Implementations

- `MeanFuser` – Uniform averaging
- `KalmanorixFuser` – True Kalman fusion with diagonal covariance
- `EnsembleKalmanFuser` – Parallel Kalman updates
- `StructuredKalmanFuser` – Low‑rank covariance fusion
- `DiagonalKalmanFuser` – Scalar Kalman update (shared variance)
- `LearnedGateFuser` – Learned two‑way gating

*TODO: Add comparison table, fusion‑strategy selection guide, and batch‑fusion examples.*
