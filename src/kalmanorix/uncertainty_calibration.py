"""Validation-only calibration of scalar uncertainty estimates (sigma2)."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable, Literal

import numpy as np
from scipy.stats import spearmanr


CalibratorName = Literal[
    "identity",
    "affine",
    "temperature",
    "isotonic",
    "piecewise_monotonic",
]


@dataclass(frozen=True)
class ScalarCalibrator:
    """Deterministic scalar calibrator for sigma2 mappings."""

    name: CalibratorName
    params: dict[str, float | list[float] | str]

    def transform(self, sigma2: np.ndarray) -> np.ndarray:
        x = np.asarray(sigma2, dtype=np.float64)
        if self.name == "identity":
            return np.maximum(x, 1e-12)
        if self.name == "affine":
            a = float(self.params["a"])
            b = float(self.params["b"])
            return np.maximum(a * x + b, 1e-12)
        if self.name == "temperature":
            t = float(self.params["temperature"])
            return np.maximum(x / max(t, 1e-12), 1e-12)
        if self.name in {"isotonic", "piecewise_monotonic"}:
            knots_x = np.asarray(self.params["knots_x"], dtype=np.float64)
            knots_y = np.asarray(self.params["knots_y"], dtype=np.float64)
            return np.maximum(np.interp(x, knots_x, knots_y), 1e-12)
        raise ValueError(f"Unsupported calibrator: {self.name}")

    def to_json(self, path: Path) -> None:
        payload = {"name": self.name, "params": self.params}
        path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "ScalarCalibrator":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(name=payload["name"], params=payload["params"])


@dataclass(frozen=True)
class CalibrationFit:
    calibrator: ScalarCalibrator
    n_train: int
    used_fallback: bool
    objective_mse: float


@dataclass(frozen=True)
class ReliabilitySummary:
    ece: float
    mean_predicted: float
    mean_realized: float
    bin_predicted: list[float]
    bin_realized: list[float]
    bin_counts: list[int]


def _pava(y: np.ndarray, w: np.ndarray) -> np.ndarray:
    y_hat = y.astype(np.float64).copy()
    weight = w.astype(np.float64).copy()
    n = len(y_hat)
    i = 0
    while i < n - 1:
        if y_hat[i] <= y_hat[i + 1]:
            i += 1
            continue
        total_w = weight[i] + weight[i + 1]
        pooled = (weight[i] * y_hat[i] + weight[i + 1] * y_hat[i + 1]) / total_w
        y_hat[i] = pooled
        y_hat[i + 1] = pooled
        weight[i] = total_w
        weight[i + 1] = total_w
        j = i
        while j > 0 and y_hat[j - 1] > y_hat[j]:
            total_w = weight[j - 1] + weight[j]
            pooled = (weight[j - 1] * y_hat[j - 1] + weight[j] * y_hat[j]) / total_w
            y_hat[j - 1] = pooled
            y_hat[j] = pooled
            weight[j - 1] = total_w
            weight[j] = total_w
            j -= 1
        i += 1
    return y_hat


def fit_scalar_calibrator(
    sigma2: np.ndarray,
    realized_error: np.ndarray,
    method: CalibratorName,
    min_samples: int = 8,
) -> CalibrationFit:
    x = np.asarray(sigma2, dtype=np.float64)
    y = np.asarray(realized_error, dtype=np.float64)
    if x.shape != y.shape:
        raise ValueError("sigma2 and realized_error must have identical shapes")
    if x.size < min_samples:
        cal = ScalarCalibrator(
            name="identity",
            params={
                "reason": f"too_few_samples:{x.size}",
                "min_samples": float(min_samples),
            },
        )
        return CalibrationFit(
            calibrator=cal,
            n_train=int(x.size),
            used_fallback=True,
            objective_mse=float(np.mean((x - y) ** 2)) if x.size else 0.0,
        )

    order = np.argsort(x)
    xs = x[order]
    ys = y[order]

    if method == "affine":
        design = np.column_stack([xs, np.ones_like(xs)])
        sol, *_ = np.linalg.lstsq(design, ys, rcond=None)
        a = float(max(sol[0], 0.0))
        b = float(sol[1])
        cal = ScalarCalibrator(name="affine", params={"a": a, "b": b})
    elif method == "temperature":
        denom = float(np.sum(xs * ys))
        if abs(denom) < 1e-12:
            t = 1.0
        else:
            t = float(max(np.sum(xs * xs) / denom, 1e-6))
        cal = ScalarCalibrator(name="temperature", params={"temperature": t})
    elif method == "isotonic":
        y_iso = _pava(ys, np.ones_like(ys))
        cal = ScalarCalibrator(
            name="isotonic",
            params={"knots_x": xs.tolist(), "knots_y": y_iso.tolist()},
        )
    elif method == "piecewise_monotonic":
        n_bins = min(12, max(4, x.size // 4))
        edges = np.quantile(xs, np.linspace(0.0, 1.0, n_bins + 1))
        mids: list[float] = []
        vals: list[float] = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            mask = (xs >= lo) & (xs <= hi)
            if not np.any(mask):
                continue
            mids.append(float(np.mean(xs[mask])))
            vals.append(float(np.mean(ys[mask])))
        vals_np = np.maximum.accumulate(np.asarray(vals, dtype=np.float64))
        cal = ScalarCalibrator(
            name="piecewise_monotonic",
            params={"knots_x": mids, "knots_y": vals_np.tolist()},
        )
    elif method == "identity":
        cal = ScalarCalibrator(name="identity", params={})
    else:
        raise ValueError(f"Unsupported method: {method}")

    preds = cal.transform(x)
    return CalibrationFit(
        calibrator=cal,
        n_train=int(x.size),
        used_fallback=False,
        objective_mse=float(np.mean((preds - y) ** 2)),
    )


def reliability_summary(
    predicted: np.ndarray,
    realized: np.ndarray,
    n_bins: int = 10,
) -> ReliabilitySummary:
    p = np.asarray(predicted, dtype=np.float64)
    r = np.asarray(realized, dtype=np.float64)
    if p.shape != r.shape:
        raise ValueError("predicted and realized must have identical shapes")

    lo = float(np.min(p))
    hi = float(np.max(p))
    if hi - lo < 1e-12:
        hi = lo + 1e-12
    edges = np.linspace(lo, hi, n_bins + 1)
    idx = np.digitize(p, edges, right=True) - 1
    idx = np.clip(idx, 0, n_bins - 1)

    bin_pred, bin_real, counts = [], [], []
    ece = 0.0
    for b in range(n_bins):
        mask = idx == b
        cnt = int(np.sum(mask))
        counts.append(cnt)
        if cnt == 0:
            bin_pred.append(float("nan"))
            bin_real.append(float("nan"))
            continue
        mp = float(np.mean(p[mask]))
        mr = float(np.mean(r[mask]))
        bin_pred.append(mp)
        bin_real.append(mr)
        ece += cnt * abs(mp - mr)

    return ReliabilitySummary(
        ece=float(ece / max(np.sum(counts), 1)),
        mean_predicted=float(np.mean(p)),
        mean_realized=float(np.mean(r)),
        bin_predicted=bin_pred,
        bin_realized=bin_real,
        bin_counts=counts,
    )


def uncertainty_rank_correlation(predicted: np.ndarray, realized: np.ndarray) -> float:
    corr, _ = spearmanr(predicted, realized)
    return 0.0 if np.isnan(corr) else float(corr)


def apply_calibrator_to_sigma2_fn(
    sigma2_fn: Callable[[str], float],
    calibrator: ScalarCalibrator,
) -> Callable[[str], float]:
    def _wrapped(query: str) -> float:
        raw = float(sigma2_fn(query))
        return float(calibrator.transform(np.array([raw], dtype=np.float64))[0])

    return _wrapped
