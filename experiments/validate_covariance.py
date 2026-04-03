"""Validate covariance calibration quality on synthetic data.

Outputs:
  - results/covariance_calibration.png
  - results/covariance_validation.json
"""

import json
import sys
import zlib
import struct
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from kalmanorix.kalman_engine.covariance import (  # pylint: disable=wrong-import-position
    calibrate_uncertainty,
    estimate_covariance,
)


def _synthetic_embeddings(
    n_samples: int, dimension: int, noise_multiplier: float, seed: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create synthetic predicted/ground-truth embeddings with known noise."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(0.0, 1.0, size=(n_samples, dimension))
    base_std = np.exp(rng.normal(0.0, 0.4, size=dimension)) * noise_multiplier

    difficulty = 0.5 + np.abs(latent[:, : min(8, dimension)]).mean(axis=1)
    sample_std = np.outer(difficulty, base_std)

    noise = rng.normal(0.0, 1.0, size=(n_samples, dimension)) * sample_std
    predicted = latent + noise
    return predicted, latent, sample_std


def _reliability_diagram(
    pred_uncertainty: np.ndarray, actual_error: np.ndarray, output_path: Path
) -> None:
    """Create a lightweight reliability diagram PNG without external deps."""
    q = np.quantile(pred_uncertainty, np.linspace(0.0, 1.0, 11))
    x_vals = []
    y_vals = []
    for lo, hi in zip(q[:-1], q[1:]):
        if hi <= lo:
            continue
        mask = (pred_uncertainty >= lo) & (pred_uncertainty <= hi)
        if np.sum(mask) < 5:
            continue
        x_vals.append(float(np.mean(pred_uncertainty[mask])))
        y_vals.append(float(np.mean(actual_error[mask])))

    width, height = 800, 600
    canvas = np.full((height, width, 3), 255, dtype=np.uint8)
    margin = 60
    x0, y0 = margin, height - margin
    x1, y1 = width - margin, margin

    # Axes
    canvas[y1:y0 + 1, x0 - 1 : x0 + 1] = 0
    canvas[y0 - 1 : y0 + 1, x0:x1 + 1] = 0

    if x_vals:
        max_val = max(max(x_vals), max(y_vals), 1e-8)
        norm = lambda v: v / max_val

        # Perfect calibration line (gray diagonal)
        for t in np.linspace(0, 1, 200):
            px = int(x0 + t * (x1 - x0))
            py = int(y0 - t * (y0 - y1))
            canvas[max(py - 1, 0) : min(py + 2, height), max(px - 1, 0) : min(px + 2, width)] = [180, 180, 180]

        # Empirical curve points (blue)
        pts = []
        for xv, yv in zip(x_vals, y_vals):
            px = int(x0 + norm(xv) * (x1 - x0))
            py = int(y0 - norm(yv) * (y0 - y1))
            pts.append((px, py))
            canvas[max(py - 3, 0) : min(py + 4, height), max(px - 3, 0) : min(px + 4, width)] = [46, 117, 182]

        # Connect points
        for (ax, ay), (bx, by) in zip(pts[:-1], pts[1:]):
            steps = max(abs(bx - ax), abs(by - ay), 1)
            for t in np.linspace(0, 1, steps):
                px = int(ax + t * (bx - ax))
                py = int(ay + t * (by - ay))
                canvas[max(py - 1, 0) : min(py + 2, height), max(px - 1, 0) : min(px + 2, width)] = [46, 117, 182]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    _save_png(canvas, output_path)


def _save_png(rgb: np.ndarray, path: Path) -> None:
    """Save uint8 RGB image to PNG using only stdlib."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[row].tobytes() for row in range(h))
    compressed = zlib.compress(raw, level=9)

    def chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack("!I", len(data))
            + tag
            + data
            + struct.pack("!I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    png = b"\x89PNG\r\n\x1a\n"
    ihdr = struct.pack("!IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png += chunk(b"IHDR", ihdr)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    path.write_bytes(png)


def run_validation(
    n_train: int = 2000,
    n_val: int = 800,
    dimension: int = 128,
    seed: int = 7,
) -> None:
    """Run validation across required methods and write artifacts."""
    train_pred, train_gt, _ = _synthetic_embeddings(n_train, dimension, 0.8, seed)
    val_pred, val_gt, _ = _synthetic_embeddings(n_val, dimension, 1.1, seed + 1)
    actual_error = np.mean((val_pred - val_gt) ** 2, axis=1)
    raw_uncertainty = np.linalg.norm(val_pred - np.mean(val_pred, axis=0), axis=1)
    calibrated_uncertainty, slope, bias = calibrate_uncertainty(
        raw_uncertainty, actual_error
    )

    val_cov, val_corr = estimate_covariance(
        validation_embeddings=val_pred,
        ground_truth_embeddings=val_gt,
        method="validation_residual",
    )

    dist_cov, dist_corr = estimate_covariance(
        validation_embeddings=val_pred,
        ground_truth_embeddings=val_gt,
        train_embeddings=train_pred,
        method="distance",
    )

    # Simulated MC dropout samples around validation embeddings.
    rng = np.random.default_rng(seed + 2)
    dropout_passes = np.stack(
        [val_pred + rng.normal(0.0, 0.03, size=val_pred.shape) for _ in range(30)], axis=0
    )
    mc_cov = np.mean(np.var(dropout_passes, axis=0, ddof=1), axis=0)
    mc_uncertainty = np.mean(np.var(dropout_passes, axis=0, ddof=1), axis=1)
    mc_uncertainty, _, _ = calibrate_uncertainty(mc_uncertainty, actual_error)
    mc_corr = float(np.corrcoef(mc_uncertainty, actual_error)[0, 1])

    results = {
        "validation_residual": {
            "correlation": float(val_corr),
            "mean_covariance": float(np.mean(val_cov)),
        },
        "distance_fallback": {
            "correlation": float(dist_corr),
            "mean_covariance": float(np.mean(dist_cov)),
        },
        "mc_dropout": {
            "correlation": float(mc_corr),
            "mean_covariance": float(np.mean(mc_cov)),
            "passes": 30,
        },
        "calibration": {
            "slope": float(slope),
            "bias": float(bias),
            "mean_uncertainty": float(np.mean(calibrated_uncertainty)),
        },
    }

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)
    with open(results_dir / "covariance_validation.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    _reliability_diagram(
        pred_uncertainty=calibrated_uncertainty,
        actual_error=actual_error,
        output_path=results_dir / "covariance_calibration.png",
    )

    print(json.dumps(results, indent=2))
    if val_corr < 0.5:
        raise SystemExit(
            f"Validation residual correlation below target: {val_corr:.3f} < 0.5"
        )


if __name__ == "__main__":
    run_validation()
