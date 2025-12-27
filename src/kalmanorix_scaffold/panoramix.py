from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol, Tuple
import numpy as np
from .village import SEF, Village
from .scout import ScoutRouter

Vec = np.ndarray

@dataclass(frozen=True)
class Potion:
    vector: Vec
    weights: Dict[str, float]
    meta: Optional[Dict[str, object]] = None

class Fuser(Protocol):
    def fuse(self, query: str, modules: List[SEF]) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]: ...

class MeanFuser:
    def fuse(self, query: str, modules: List[SEF]) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        Z = np.stack([m.embed(query) for m in modules], axis=0)
        w = {m.name: 1.0 / len(modules) for m in modules}
        return Z.mean(axis=0), w, None

class KalmanorixFuser:
    def fuse(self, query: str, modules: List[SEF]) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        Z = [m.embed(query) for m in modules]
        weights = np.array([1.0 / (m.sigma2 + 1e-12) for m in modules], dtype=np.float64)
        weights = weights / weights.sum()
        x = np.zeros_like(Z[0])
        out_w: Dict[str, float] = {}
        for m, wi, zi in zip(modules, weights, Z):
            x += wi * zi
            out_w[m.name] = float(wi)
        return x, out_w, None

@dataclass
class Panoramix:
    fuser: Fuser

    def brew(self, query: str, village: Village, scout: ScoutRouter) -> Potion:
        chosen = scout.select(query, village)
        vec, weights, meta = self.fuser.fuse(query, chosen)
        return Potion(vector=vec, weights=weights, meta=meta)
