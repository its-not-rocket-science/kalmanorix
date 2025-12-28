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

class LearnedGateFuser:
    """
    A tiny learned baseline: predicts α(query) ∈ [0,1] and returns:
        α * z_a + (1-α) * z_b

    This is a critical baseline for Kalmanorix-style fusion: if you can't beat
    a tiny learned gate, the Kalman framing likely isn't providing unique value.

    Implementation notes:
      - No external dependencies: hashed bag-of-words features
      - Logistic regression trained with simple gradient descent
      - Supports exactly two named modules for now
    """

    def __init__(
        self,
        module_a: str,
        module_b: str,
        *,
        n_features: int = 256,
        lr: float = 0.3,
        l2: float = 1e-3,
        steps: int = 300,
    ) -> None:
        self.module_a = module_a
        self.module_b = module_b
        self.n_features = int(n_features)
        self.lr = float(lr)
        self.l2 = float(l2)
        self.steps = int(steps)

        # weights for logistic regression: w[0] is bias
        self.w = np.zeros(self.n_features + 1, dtype=np.float64)

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        text = text.lower()
        out: list[str] = []
        buff: list[str] = []
        for ch in text:
            if ch.isalnum():
                buff.append(ch)
            else:
                if buff:
                    out.append("".join(buff))
                    buff.clear()
        if buff:
            out.append("".join(buff))
        return out

    @staticmethod
    def _stable_hash(token: str) -> int:
        import hashlib

        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], byteorder="little", signed=False)

    def _featurize(self, text: str) -> np.ndarray:
        x = np.zeros(self.n_features + 1, dtype=np.float64)
        x[0] = 1.0  # bias
        for tok in self._tokenize(text):
            idx = 1 + (self._stable_hash(tok) % self.n_features)
            x[idx] += 1.0
        # L2 normalize features (excluding bias)
        norm = np.linalg.norm(x[1:]) + 1e-12
        x[1:] /= norm
        return x

    @staticmethod
    def _sigmoid(t: float) -> float:
        if t >= 0:
            z = np.exp(-t)
            return float(1.0 / (1.0 + z))
        z = np.exp(t)
        return float(z / (1.0 + z))

    def fit(self, texts: list[str], y: list[int]) -> None:
        if len(texts) != len(y) or not texts:
            raise ValueError("texts and y must be the same non-zero length")

        X = np.stack([self._featurize(t) for t in texts], axis=0)  # (N, d)
        Y = np.array(y, dtype=np.float64)  # (N,)
        if not np.all((Y == 0.0) | (Y == 1.0)):
            raise ValueError("y must be binary labels 0/1")

        for _ in range(self.steps):
            logits = X @ self.w
            P = np.array([self._sigmoid(float(z)) for z in logits], dtype=np.float64)
            grad = (X.T @ (P - Y)) / len(Y)
            grad[1:] += self.l2 * self.w[1:]
            self.w -= self.lr * grad

    def predict_alpha(self, text: str) -> float:
        x = self._featurize(text)
        return self._sigmoid(float(x @ self.w))

    def fuse(self, query: str, modules: List[SEF]) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        by_name = {m.name: m for m in modules}
        if self.module_a not in by_name or self.module_b not in by_name:
            raise ValueError(
                f"LearnedGateFuser expects modules '{self.module_a}' and '{self.module_b}' "
                f"but got {list(by_name.keys())}"
            )

        a = by_name[self.module_a]
        b = by_name[self.module_b]

        z_a = a.embed(query)
        z_b = b.embed(query)

        alpha = self.predict_alpha(query)  # probability of choosing A
        x = (alpha * z_a) + ((1.0 - alpha) * z_b)

        weights = {a.name: float(alpha), b.name: float(1.0 - alpha)}
        meta = {"alpha": float(alpha), "gate": "hashed_logreg"}
        return x, weights, meta
