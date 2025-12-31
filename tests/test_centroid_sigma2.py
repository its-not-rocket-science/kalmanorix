"""Kalmanorix public API."""

import numpy as np
from kalmanorix.uncertainty import CentroidDistanceSigma2


def test_centroid_sigma2_is_lower_near_centroid():
    """CentroidDistanceSigma2 should assign lower sigma² to more similar queries."""

    def embed(text: str) -> np.ndarray:
        t = text.lower()
        v = np.array([0.0, 0.0], dtype=np.float64)
        if "tech" in t:
            v += np.array([1.0, 0.0])
        if "cook" in t:
            v += np.array([0.0, 1.0])
        v = v / (np.linalg.norm(v) + 1e-12)
        return v

    sigma2 = CentroidDistanceSigma2.from_calibration(
        embed=embed,
        calibration_texts=["tech tech", "tech"],
        base_sigma2=0.2,
        scale=2.0,
    )

    assert sigma2("tech query") < sigma2("cook query")
