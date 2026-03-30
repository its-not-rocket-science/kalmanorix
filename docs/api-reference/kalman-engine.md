# Kalman Engine API

*TODO: Auto‑generated API documentation for the low‑level Kalman fusion algorithms.*

The `kalman_engine` module contains the core mathematical implementations of Kalman fusion for embeddings. These functions are used internally by the high‑level `KalmanorixFuser` and its variants.

::: kalmanorix.kalman_engine
    options:
      show_root_heading: true
      show_source: true
      heading_level: 3

## Core Functions

- `kalman_fuse_diagonal()` – Sequential updates with diagonal covariance.
- `kalman_fuse_diagonal_ensemble()` – Parallel updates for diagonal covariance.
- `kalman_fuse_structured()` – Sequential updates with structured (diagonal + low‑rank) covariance.
- Batch versions (`_batch` suffix) for processing multiple queries efficiently.

## Covariance Estimators

- `EmpiricalCovariance` – Compute per‑dimension variance from validation errors.
- `DistanceBasedCovariance` – Variance based on distance to domain centroid.
- `ConstantCovariance` – Fixed diagonal covariance.
- `ScalarCovariance` – Fixed scalar variance (same for all dimensions).

*TODO: Add mathematical derivation, pseudocode, and performance‑characterisation data.*
