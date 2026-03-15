"""
Fusion orchestration for Kalmanorix.

This module defines:
- the `Potion` data structure (fused embedding + metadata)
- the fusion interface (`Fuser`)
- baseline fusion strategies
- the `Panoramix` orchestrator that combines routing and fusion

The design keeps routing, fusion, and uncertainty estimation cleanly separated.
"""

# pylint: disable=too-few-public-methods,too-many-arguments,invalid-name,import-outside-toplevel
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from .scout import ScoutRouter
from .village import SEF, Village

Vec = np.ndarray


@dataclass(frozen=True)
class Potion:
    """
    Result of a fusion operation.

    Attributes
    ----------
    vector:
        The fused embedding vector.
    weights:
        Per-module fusion weights.
    meta:
        Optional diagnostic metadata (e.g. gate values).
    """

    vector: Vec
    weights: Dict[str, float]
    meta: Optional[Dict[str, object]] = None


class Fuser(ABC):
    """
    Abstract base class for fusion strategies.

    A Fuser takes embeddings from multiple modules and combines them into
    a single vector, optionally producing diagnostic metadata.
    """

    @abstractmethod
    def fuse(
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        """
        Fuse embeddings from the given modules.

        Parameters
        ----------
        query:
            Input text query.
        modules:
            List of selected specialist modules.

        Returns
        -------
        vector:
            Fused embedding.
        weights:
            Per-module contribution weights.
        meta:
            Optional metadata.
        """
        raise NotImplementedError


class MeanFuser(Fuser):
    """
    Uniform averaging baseline.

    All modules contribute equally, regardless of uncertainty or query content.
    """

    def fuse(
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        z = np.stack([m.embed(query) for m in modules], axis=0)
        w = {m.name: 1.0 / len(modules) for m in modules}
        return z.mean(axis=0), w, None


class KalmanorixFuser(Fuser):
    """
    Precision-weighted fusion baseline.

    Each module contributes proportionally to the inverse of its
    query-dependent uncertainty (sigma²). This is a strong baseline,
    but it is not a sequential Kalman update.
    """

    def fuse(
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        z_list = [m.embed(query) for m in modules]
        weights = np.array(
            [1.0 / (m.sigma2_for(query) + 1e-12) for m in modules],
            dtype=np.float64,
        )
        weights = weights / weights.sum()

        x = np.zeros_like(z_list[0])
        out_w: Dict[str, float] = {}

        for m, wi, zi in zip(modules, weights, z_list):
            x += wi * zi
            out_w[m.name] = float(wi)

        return x, out_w, None


class DiagonalKalmanFuser(Fuser):
    """
    Sequential diagonal Kalman-style fusion.

    We treat the fused embedding as the latent state x and each specialist
    embedding z_i as a measurement with scalar noise R_i = sigma²_i(query).
    We maintain a scalar prior variance P (shared across dimensions).

    Update per module i:
        K_i = P / (P + R_i)
        x <- x + K_i * (z_i - x)
        P <- (1 - K_i) * P

    Notes
    -----
    - This is still "diagonal" / scalar P for simplicity (Phase 2-friendly).
    - The order of updates can matter slightly; we use increasing R_i
      (most confident first) for stability.
    - We return:
        * weights: normalized "effective K mass" per module (diagnostic)
        * meta: Ks, prior/posterior P, and update order
    """

    def __init__(
        self, *, prior_sigma2: float = 1.0, sort_by_sigma2: bool = True
    ) -> None:
        self.prior_sigma2 = float(prior_sigma2)
        self.sort_by_sigma2 = bool(sort_by_sigma2)

    def _kalman_updates(
        self, order: List[str], z_by_name: Dict[str, Vec], r_by_name: Dict[str, float]
    ) -> Tuple[Vec, Dict[str, float]]:
        x = np.zeros_like(z_by_name[order[0]])
        p = float(self.prior_sigma2)
        k_by_name: Dict[str, float] = {}
        for name in order:
            r = max(r_by_name[name], 1e-12)
            k = p / (p + r)
            x = x + (k * (z_by_name[name] - x))
            p = (1.0 - k) * p
            k_by_name[name] = float(k)
        return x, k_by_name

    def fuse(
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        if not modules:
            raise ValueError("DiagonalKalmanFuser requires at least one module")

        z_by_name: Dict[str, Vec] = {m.name: m.embed(query) for m in modules}
        r_by_name: Dict[str, float] = {
            m.name: float(m.sigma2_for(query)) for m in modules
        }

        order = [m.name for m in modules]
        if self.sort_by_sigma2:
            order = sorted(order, key=lambda n: r_by_name[n])

        # x = np.zeros_like(next(iter(z_by_name.values())))
        p = float(self.prior_sigma2)

        x, k_by_name = self._kalman_updates(order, z_by_name, r_by_name)

        # Convert K's into a stable weight readout (purely diagnostic)
        k_sum = sum(k_by_name.values()) + 1e-12
        weights = {name: float(k / k_sum) for name, k in k_by_name.items()}

        meta: Dict[str, object] = {
            "prior_sigma2": float(self.prior_sigma2),
            "post_sigma2": float(p),
            "kalman_gains": dict(k_by_name),
            "order": list(order),
            "sort_by_sigma2": self.sort_by_sigma2,
        }
        return x, weights, meta


@dataclass
class Panoramix:
    """
    Orchestrator that combines routing and fusion.

    Panoramix:
      1. asks a ScoutRouter which modules to consult
      2. delegates fusion to a Fuser
      3. packages the result as a Potion
    """

    fuser: Fuser

    def brew(self, query: str, village: Village, scout: ScoutRouter) -> Potion:
        """
        Produce a fused embedding for a query.

        Parameters
        ----------
        query:
            Input text query.
        village:
            Collection of available specialist modules.
        scout:
            Routing strategy.

        Returns
        -------
        Potion
            The fused embedding and diagnostics.
        """
        chosen = scout.select(query, village)
        vec, weights, meta = self.fuser.fuse(query, chosen)
        return Potion(vector=vec, weights=weights, meta=meta)


class LearnedGateFuser(Fuser):
    """
    Learned two-way gating baseline.

    Predicts a scalar α(query) ∈ [0, 1] and returns:
        α * z_a + (1 - α) * z_b

    This serves as a critical learned baseline against which Kalman-style
    fusion should be compared.
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

        # Logistic regression weights (w[0] is bias)
        self.w = np.zeros(self.n_features + 1, dtype=np.float64)

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        """Very simple alphanumeric tokenizer."""
        text = text.lower()
        out: List[str] = []
        buff: List[str] = []
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
        """Stable hash (independent of Python's hash randomization)."""
        import hashlib

        h = hashlib.md5(token.encode("utf-8")).digest()
        return int.from_bytes(h[:4], byteorder="little", signed=False)

    def _featurize(self, text: str) -> np.ndarray:
        """Convert text to a normalized hashed bag-of-words feature vector."""
        x = np.zeros(self.n_features + 1, dtype=np.float64)
        x[0] = 1.0  # bias

        for tok in self._tokenize(text):
            idx = 1 + (self._stable_hash(tok) % self.n_features)
            x[idx] += 1.0

        norm = np.linalg.norm(x[1:]) + 1e-12
        x[1:] /= norm
        return x

    @staticmethod
    def _sigmoid(t: float) -> float:
        """Numerically stable sigmoid."""
        if t >= 0:
            z = np.exp(-t)
            return float(1.0 / (1.0 + z))
        z = np.exp(t)
        return float(z / (1.0 + z))

    def fit(self, texts: List[str], y: List[int]) -> None:
        """
        Train the gate using binary labels:
          1 => prefer module_a
          0 => prefer module_b
        """
        if len(texts) != len(y) or not texts:
            raise ValueError("texts and y must be the same non-zero length")

        X = np.stack([self._featurize(t) for t in texts], axis=0)
        Y = np.array(y, dtype=np.float64)

        if not np.all((Y == 0.0) | (Y == 1.0)):
            raise ValueError("y must be binary labels 0/1")

        for _ in range(self.steps):
            logits = X @ self.w
            P = np.array([self._sigmoid(float(z)) for z in logits])
            grad = (X.T @ (P - Y)) / len(Y)
            grad[1:] += self.l2 * self.w[1:]
            self.w -= self.lr * grad

    def predict_alpha(self, text: str) -> float:
        """Predict α ∈ [0, 1], the probability of choosing module_a."""
        x = self._featurize(text)
        return self._sigmoid(float(x @ self.w))

    def fuse(
        self,
        query: str,
        modules: List[SEF],
    ) -> Tuple[Vec, Dict[str, float], Optional[Dict[str, object]]]:
        by_name = {m.name: m for m in modules}
        if self.module_a not in by_name or self.module_b not in by_name:
            raise ValueError(
                f"LearnedGateFuser expects modules '{self.module_a}' and "
                f"'{self.module_b}', got {list(by_name.keys())}"
            )

        a = by_name[self.module_a]
        b = by_name[self.module_b]

        z_a = a.embed(query)
        z_b = b.embed(query)

        alpha = self.predict_alpha(query)
        x = (alpha * z_a) + ((1.0 - alpha) * z_b)

        weights = {a.name: float(alpha), b.name: float(1.0 - alpha)}
        meta = {"alpha": float(alpha), "gate": "hashed_logreg"}
        return x, weights, meta
