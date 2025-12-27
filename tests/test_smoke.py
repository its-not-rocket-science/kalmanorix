import numpy as np
from kalmanorix_scaffold import SEF, Village, ScoutRouter, Panoramix, KalmanorixFuser

def test_brew_runs():
    def e(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0])
    v = Village([SEF("a", e, sigma2=1.0)])
    p = Panoramix(fuser=KalmanorixFuser()).brew("q", v, ScoutRouter())
    assert p.vector.shape == (2,)
