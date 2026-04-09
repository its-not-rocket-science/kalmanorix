"""Calibration utilities for uncertainty quantification in Kalmanorix.

The core design goal is *cross-run comparability*:
- Avoid per-batch target normalization.
- Evaluate a consistent event with an absolute tolerance.
- Map predicted sigma² to confidence via a probabilistic error model.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np


@dataclass
class CalibrationResult:
    """Results of calibration evaluation."""

    ece: float  # Expected Calibration Error (lower is better)
    brier_score: float  # Brier score (lower is better)
    n_samples: int  # Number of samples used
    bin_edges: np.ndarray  # Bin edges for reliability diagram
    bin_centers: np.ndarray  # Bin centers
    bin_accuracies: np.ndarray  # Actual accuracy/recall in each bin (NaN for empty bins)
    bin_confidences: np.ndarray  # Average predicted confidence in each bin (NaN for empty bins)
    bin_counts: np.ndarray  # Number of samples in each bin
    mean_confidence: float  # Mean predicted confidence
    mean_accuracy: float  # Mean empirical accuracy/event rate


def _compute_errors(
    specialist_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    norm: Literal["l2", "cosine"],
) -> np.ndarray:
    """Compute per-sample embedding error distances."""
    if norm == "l2":
        return np.linalg.norm(specialist_embeddings - reference_embeddings, axis=1)

    if norm == "cosine":
        # Cosine distance in [0, 2] for normalized vectors.
        from scipy.spatial.distance import cdist

        return cdist(
            specialist_embeddings,
            reference_embeddings,
            metric="cosine",
        ).diagonal()

    raise ValueError(f"Unsupported norm: {norm}")


def _variance_to_confidence(
    predicted_variances: np.ndarray,
    *,
    embedding_dim: int,
    error_tolerance: float,
    norm: Literal["l2", "cosine"],
) -> np.ndarray:
    """Map sigma² to confidence P(error <= tolerance).

    Assumes isotropic Gaussian embedding noise:
      (||e||² / sigma²) ~ ChiSquare(df=embedding_dim)

    so confidence is CDF_chi2(tolerance² / sigma²).

    For cosine distance tolerance τ_cos, we approximate via chord distance
    in normalized space: ||u - v||² = 2 * d_cos.
    """
    if embedding_dim <= 0:
        raise ValueError("embedding_dim must be positive")
    if error_tolerance <= 0:
        raise ValueError("error_tolerance must be positive")

    from scipy.stats import chi2

    safe_sigma2 = np.maximum(predicted_variances, 1e-12)

    if norm == "l2":
        tol_sq = float(error_tolerance) ** 2
    elif norm == "cosine":
        tol_sq = 2.0 * float(error_tolerance)
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    scaled = tol_sq / safe_sigma2
    confidences = chi2.cdf(scaled, df=embedding_dim)
    return np.clip(confidences, 0.0, 1.0)


def compute_embedding_calibration(
    specialist_embeddings: np.ndarray,
    reference_embeddings: np.ndarray,
    predicted_variances: np.ndarray,
    n_bins: int = 10,
    norm: Literal["l2", "cosine"] = "l2",
    error_tolerance: Optional[float] = None,
) -> CalibrationResult:
    """Compute calibration for embedding errors vs predicted variance.

    Calibration event:
      accuracy_i := 1[error_i <= error_tolerance]

    Predicted confidence:
      confidence_i := P(error_i <= error_tolerance | sigma²_i)

    This avoids per-batch min/max normalization and produces comparable
    calibration metrics across runs and datasets when error_tolerance is fixed.
    """
    n_samples = specialist_embeddings.shape[0]
    if n_samples == 0:
        raise ValueError("No samples provided")

    if reference_embeddings.shape != specialist_embeddings.shape:
        raise ValueError(
            f"Shapes mismatch: specialist {specialist_embeddings.shape}, "
            f"reference {reference_embeddings.shape}"
        )
    if predicted_variances.shape != (n_samples,):
        raise ValueError(
            f"predicted_variances shape {predicted_variances.shape} "
            f"does not match n_samples {n_samples}"
        )

    errors = _compute_errors(
        specialist_embeddings=specialist_embeddings,
        reference_embeddings=reference_embeddings,
        norm=norm,
    )

    if error_tolerance is None:
        # Fixed defaults for run-to-run comparability.
        error_tolerance = 1.0 if norm == "l2" else 0.25

    accuracies = (errors <= error_tolerance).astype(float)

    embedding_dim = specialist_embeddings.shape[1]
    confidences = _variance_to_confidence(
        predicted_variances,
        embedding_dim=embedding_dim,
        error_tolerance=error_tolerance,
        norm=norm,
    )

    return _compute_calibration_metrics(
        accuracies=accuracies,
        confidences=confidences,
        n_bins=n_bins,
    )


def compute_retrieval_calibration(
    query_embeddings: np.ndarray,
    doc_embeddings: np.ndarray,
    query_variances: np.ndarray,
    true_indices: list[int],
    k: int = 10,
    n_bins: int = 10,
    distance_metric: Literal["cosine", "l2"] = "cosine",
) -> CalibrationResult:
    """Compute calibration for retrieval success vs predicted uncertainty."""
    n_queries = query_embeddings.shape[0]
    if n_queries == 0:
        raise ValueError("No queries provided")

    if len(true_indices) != n_queries:
        raise ValueError(
            f"true_indices length {len(true_indices)} "
            f"does not match n_queries {n_queries}"
        )
    if query_variances.shape != (n_queries,):
        raise ValueError(
            f"query_variances shape {query_variances.shape} "
            f"does not match n_queries {n_queries}"
        )

    successes_list: list[float] = []
    for i in range(n_queries):
        true_idx = true_indices[i]
        if true_idx < 0 or true_idx >= doc_embeddings.shape[0]:
            successes_list.append(0.0)
            continue

        if distance_metric == "cosine":
            from scipy.spatial.distance import cdist

            distances = cdist(
                query_embeddings[i : i + 1],
                doc_embeddings,
                metric="cosine",
            )[0]
        elif distance_metric == "l2":
            distances = np.linalg.norm(doc_embeddings - query_embeddings[i], axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        top_k_indices = np.argsort(distances)[:k]
        success = 1.0 if true_idx in top_k_indices else 0.0
        successes_list.append(success)

    successes = np.array(successes_list, dtype=float)

    # Retrieval confidence mapping is less model-specific; keep bounded monotone map.
    confidences = 1.0 / (1.0 + np.maximum(query_variances, 1e-12))

    return _compute_calibration_metrics(
        accuracies=successes,
        confidences=confidences,
        n_bins=n_bins,
    )


def _compute_calibration_metrics(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """Compute ECE/Brier and reliability-bin summaries."""
    n_samples = len(accuracies)
    if n_samples == 0:
        raise ValueError("No samples provided")

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = np.full(n_bins, np.nan, dtype=float)
    bin_confidences = np.full(n_bins, np.nan, dtype=float)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        count = int(np.sum(mask))
        bin_counts[b] = count
        if count > 0:
            bin_accuracies[b] = float(np.mean(accuracies[mask]))
            bin_confidences[b] = float(np.mean(confidences[mask]))

    non_empty = bin_counts > 0
    if not np.any(non_empty):
        warnings.warn("All bins are empty", RuntimeWarning)
        return CalibrationResult(
            ece=0.0,
            brier_score=0.0,
            n_samples=n_samples,
            bin_edges=bin_edges,
            bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2,
            bin_accuracies=bin_accuracies,
            bin_confidences=bin_confidences,
            bin_counts=bin_counts,
            mean_confidence=float(np.mean(confidences)),
            mean_accuracy=float(np.mean(accuracies)),
        )

    ece = float(
        np.sum(
            bin_counts[non_empty]
            * np.abs(bin_accuracies[non_empty] - bin_confidences[non_empty])
        )
        / np.sum(bin_counts[non_empty])
    )
    brier_score = float(np.mean((accuracies - confidences) ** 2))

    return CalibrationResult(
        ece=ece,
        brier_score=brier_score,
        n_samples=n_samples,
        bin_edges=bin_edges,
        bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
        mean_confidence=float(np.mean(confidences)),
        mean_accuracy=float(np.mean(accuracies)),
    )


def calibration_summary(result: CalibrationResult) -> dict[str, float]:
    """Compact calibration summary used in experiment reports."""
    return {
        "n_samples": float(result.n_samples),
        "ece": float(result.ece),
        "brier_score": float(result.brier_score),
        "mean_confidence": float(result.mean_confidence),
        "mean_accuracy": float(result.mean_accuracy),
        "overconfidence_gap": float(result.mean_confidence - result.mean_accuracy),
    }


def plot_reliability_diagram(
    result: CalibrationResult,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """Plot reliability diagram for calibration visualization."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib not available, skipping reliability diagram",
            ImportWarning,
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    non_empty = result.bin_counts > 0
    if np.any(non_empty):
        bin_accuracies = result.bin_accuracies[non_empty]
        bin_confidences = result.bin_confidences[non_empty]
        bin_counts = result.bin_counts[non_empty]

        total = np.sum(bin_counts)
        widths = bin_counts / max(total, 1) * 0.8

        for center, acc, conf, width in zip(
            result.bin_centers[non_empty], bin_accuracies, bin_confidences, widths
        ):
            ax.bar(
                center,
                height=acc - conf,
                bottom=conf,
                width=width,
                align="center",
                edgecolor="black",
                alpha=0.7,
                color="red" if acc < conf else "blue",
            )

        ax.scatter(
            bin_confidences,
            bin_accuracies,
            s=50,
            c="black",
            marker="o",
            label=f"Bins (ECE={result.ece:.3f})",
            zorder=5,
        )

    ax.set_xlabel("Predicted confidence")
    ax.set_ylabel("Actual accuracy/recall")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.text(
        0.05,
        0.95,
        (
            f"ECE = {result.ece:.3f}\n"
            f"Brier = {result.brier_score:.3f}\n"
            f"Mean conf = {result.mean_confidence:.3f}\n"
            f"Mean acc = {result.mean_accuracy:.3f}"
        ),
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
        plt.close()
    else:
        plt.show()
