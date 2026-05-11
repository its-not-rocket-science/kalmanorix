# Public API overview

This page summarizes the primary JOSS-facing API surface.

## Core classes and functions

- [`SEF`](api-reference/village.md)
- [`Village`](api-reference/village.md)
- [`ScoutRouter`](api-reference/scout-router.md)
- [`Panoramix`](api-reference/panoramix.md)
- [`MeanFuser`](api-reference/panoramix.md)
- [`KalmanorixFuser`](api-reference/panoramix.md)
- Routing evaluator: [`eval_retrieval`](api-reference/scout-router.md)

## Stability tiers

- Stable API: top-level `kalmanorix` imports.
- Experimental API: `kalmanorix.experimental` (faster iteration pace).
- Internal API: `kalmanorix.internal` (no compatibility guarantees).

Legacy top-level imports for moved experimental symbols are supported with
`DeprecationWarning` shims to preserve backward compatibility.
