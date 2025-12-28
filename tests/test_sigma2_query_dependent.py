import numpy as np
from kalmanorix import SEF, Village, ScoutRouter, Panoramix, KalmanorixFuser
from kalmanorix.uncertainty import KeywordSigma2


def test_kalman_weights_change_with_query_dependent_sigma2():
    # trivial embedders
    def tech_embed(_q: str) -> np.ndarray:
        return np.array([1.0, 0.0])

    def cook_embed(_q: str) -> np.ndarray:
        return np.array([0.0, 1.0])

    tech = SEF(
        name="tech",
        embed=tech_embed,
        sigma2=KeywordSigma2({"battery"}, in_domain_sigma2=0.2, out_domain_sigma2=5.0),
    )
    cook = SEF(
        name="cook",
        embed=cook_embed,
        sigma2=KeywordSigma2({"braise"}, in_domain_sigma2=0.2, out_domain_sigma2=5.0),
    )

    village = Village([tech, cook])
    scout = ScoutRouter(mode="all")
    pan = Panoramix(fuser=KalmanorixFuser())

    # Query looks tech-y
    p1 = pan.brew("battery lasts forever", village=village, scout=scout)
    assert p1.weights["tech"] > p1.weights["cook"]

    # Query looks cooking-y
    p2 = pan.brew("braise for hours", village=village, scout=scout)
    assert p2.weights["cook"] > p2.weights["tech"]
