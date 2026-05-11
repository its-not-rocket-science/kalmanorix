"""CPU-only quickstart script designed to run in under two minutes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from kalmanorix import KalmanorixFuser, MeanFuser, Panoramix, ScoutRouter, SEF, Village
from kalmanorix.experimental import threshold_top_k


@dataclass(frozen=True)
class ToyEmbedder:
    dim: int
    keyword: str
    axis: int

    def __call__(self, text: str) -> np.ndarray:
        t = text.lower()
        vec = np.zeros(self.dim, dtype=np.float64)
        vec[self.axis] = 1.0
        if self.keyword in t:
            vec[self.axis] += 2.0
        vec += 0.01
        return vec / (np.linalg.norm(vec) + 1e-12)


def main() -> None:
    med = SEF("medical", embed=ToyEmbedder(12, "patient", 0), sigma2=0.1)
    legal = SEF("legal", embed=ToyEmbedder(12, "contract", 1), sigma2=0.2)
    village = Village([med, legal])

    queries = [
        "Patient treatment update",
        "Contract breach summary",
        "Patient contract dispute",
    ]

    semantic_router = ScoutRouter(mode="semantic", threshold_fn=threshold_top_k(k=1))
    all_router = ScoutRouter(mode="all")

    mean_engine = Panoramix(fuser=MeanFuser())
    kalman_engine = Panoramix(fuser=KalmanorixFuser())

    print("== Kalmanorix quickstart ==")
    for query in queries:
        selected = semantic_router.select(query, village)
        mean_potion = mean_engine.brew(query, village=village, scout=all_router)
        kalman_potion = kalman_engine.brew(query, village=village, scout=all_router)
        print(f"\nquery: {query}")
        print("selected:", [s.name for s in selected])
        print("mean weights:", mean_potion.weights)
        print("kalman weights:", kalman_potion.weights)


if __name__ == "__main__":
    main()
