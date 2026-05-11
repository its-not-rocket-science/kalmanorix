"""Verify documented import paths remain executable."""

from __future__ import annotations


def test_quickstart_imports() -> None:
    from kalmanorix import (  # noqa: F401
        KalmanorixFuser,
        MeanFuser,
        Panoramix,
        ScoutRouter,
        SEF,
        Village,
    )
    from kalmanorix.experimental import (  # noqa: F401
        DiagonalKalmanFuser,
        threshold_top_k,
    )
