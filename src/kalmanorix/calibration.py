"""
Calibration metrics for uncertainty quantification in Kalmanorix.

Measures how well predicted uncertainties match actual errors:
- Embedding calibration: Compare predicted variance vs L2 distance to reference.
- Retrieval calibration: Compare predicted uncertainty vs retrieval success.

Calibration is crucial for trustworthiness: a well-calibrated model's
uncertainty estimates can be used to decide when to defer to human judgment.
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
    bin_accuracies: np.ndarray  # Actual accuracy/recall in each bin
    bin_confidences: np.ndarray  # Average predicted confidence in each bin
    bin_counts: np.ndarray  # Number of samples in each bin


def compute_embedding_calibration(
    specialist_embeddings: np.ndarray,  # (n_samples, d)
    reference_embeddings: np.ndarray,  # (n_samples, d) "ground truth"
    predicted_variances: np.ndarray,  # (n_samples,) scalar variance per sample
    n_bins: int = 10,
    norm: Literal["l2", "cosine"] = "l2",
) -> CalibrationResult:
    """
    Compute calibration for embedding errors vs predicted variance.

    For each sample, we compute:
    - Actual error: distance between specialist embedding and reference
    - Predicted uncertainty: scalar variance (transformed to confidence)

    We then bin samples by predicted confidence and compare average confidence
    vs average accuracy (where accuracy = 1 - normalized error).

    Parameters
    ----------
    specialist_embeddings : np.ndarray, shape (n_samples, d)
        Embeddings from a specialist model (or fused model).
    reference_embeddings : np.ndarray, shape (n_samples, d)
        Reference "ground truth" embeddings (e.g., from monolith or ensemble).
    predicted_variances : np.ndarray, shape (n_samples,)
        Predicted scalar variance for each embedding.
    n_bins : int, optional
        Number of bins for calibration curve. Default: 10.
    norm : {"l2", "cosine"}, optional
        Distance metric between embeddings. Default: "l2".

    Returns
    -------
    CalibrationResult
        Calibration metrics and reliability diagram data.
    """
    n_samples = specialist_embeddings.shape[0]
    if n_samples == 0:
        raise ValueError("No samples provided")

    # Validate shapes
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

    # Compute actual errors
    if norm == "l2":
        errors = np.linalg.norm(specialist_embeddings - reference_embeddings, axis=1)
    elif norm == "cosine":
        # Cosine distance = 1 - cosine similarity
        from scipy.spatial.distance import cdist

        errors = cdist(
            specialist_embeddings, reference_embeddings, metric="cosine"
        ).diagonal()
    else:
        raise ValueError(f"Unsupported norm: {norm}")

    # Normalize errors to [0, 1] range for accuracy calculation
    # We use min-max normalization across the batch
    if errors.max() > errors.min():
        normalized_errors = (errors - errors.min()) / (errors.max() - errors.min())
    else:
        normalized_errors = np.zeros_like(errors)  # all errors equal

    accuracies = 1.0 - normalized_errors  # Higher accuracy = smaller error

    # Convert variances to confidences
    # For Gaussian assumption: confidence = 1 / (1 + variance)
    # This maps variance=0 -> confidence=1, variance→∞ -> confidence→0
    confidences = 1.0 / (1.0 + predicted_variances)

    return _compute_calibration_metrics(
        accuracies=accuracies,
        confidences=confidences,
        n_bins=n_bins,
    )


def compute_retrieval_calibration(
    query_embeddings: np.ndarray,  # (n_queries, d)
    doc_embeddings: np.ndarray,  # (n_docs, d)
    query_variances: np.ndarray,  # (n_queries,) per-query uncertainty
    true_indices: list[int],  # true document index for each query
    k: int = 10,
    n_bins: int = 10,
    distance_metric: Literal["cosine", "l2"] = "cosine",
) -> CalibrationResult:
    """
    Compute calibration for retrieval success vs predicted uncertainty.

    For each query, we compute:
    - Actual success: whether true document is in top-k retrieved results
    - Predicted confidence: derived from query variance

    We then bin queries by predicted confidence and compare average confidence
    vs average recall (success rate).

    Parameters
    ----------
    query_embeddings : np.ndarray, shape (n_queries, d)
        Query embeddings.
    doc_embeddings : np.ndarray, shape (n_docs, d)
        Document embeddings.
    query_variances : np.ndarray, shape (n_queries,)
        Predicted variance for each query.
    true_indices : list[int]
        Index of true document for each query (-1 indicates no true document).
    k : int, optional
        Recall@k threshold. Default: 10.
    n_bins : int, optional
        Number of bins for calibration curve. Default: 10.
    distance_metric : {"cosine", "l2"}, optional
        Distance metric for retrieval. Default: "cosine".

    Returns
    -------
    CalibrationResult
        Calibration metrics and reliability diagram data.
    """
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

    # Compute retrieval success for each query
    successes_list: list[float] = []
    for i in range(n_queries):
        true_idx = true_indices[i]
        if true_idx < 0 or true_idx >= doc_embeddings.shape[0]:
            # No true document or invalid index
            successes_list.append(0.0)
            continue

        # Compute distances between query i and all documents
        if distance_metric == "cosine":
            from scipy.spatial.distance import cdist

            distances = cdist(
                query_embeddings[i : i + 1], doc_embeddings, metric="cosine"
            )[0]
        elif distance_metric == "l2":
            distances = np.linalg.norm(doc_embeddings - query_embeddings[i], axis=1)
        else:
            raise ValueError(f"Unsupported distance metric: {distance_metric}")

        # Get indices of top-k nearest documents (lowest distance)
        top_k_indices = np.argsort(distances)[:k]
        success = 1.0 if true_idx in top_k_indices else 0.0
        successes_list.append(success)

    successes = np.array(successes_list, dtype=float)

    # Convert variances to confidences
    confidences = 1.0 / (1.0 + query_variances)

    return _compute_calibration_metrics(
        accuracies=successes,  # binary success/failure
        confidences=confidences,
        n_bins=n_bins,
    )


def _compute_calibration_metrics(
    accuracies: np.ndarray,
    confidences: np.ndarray,
    n_bins: int = 10,
) -> CalibrationResult:
    """
    Compute calibration metrics given accuracies and confidences.

    Parameters
    ----------
    accuracies : np.ndarray, shape (n_samples,)
        Actual accuracy/success for each sample (0-1).
    confidences : np.ndarray, shape (n_samples,)
        Predicted confidence for each sample (0-1).
    n_bins : int, optional
        Number of bins. Default: 10.

    Returns
    -------
    CalibrationResult
        Calibration metrics.
    """
    n_samples = len(accuracies)
    if n_samples == 0:
        raise ValueError("No samples provided")

    # Bin samples by confidence
    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(confidences, bin_edges, right=True) - 1
    # digitize returns 1..n_bins, shift to 0..n_bins-1
    # Handle edge case where confidence == 1.0
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    bin_accuracies = np.zeros(n_bins)
    bin_confidences = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for b in range(n_bins):
        mask = bin_indices == b
        if np.any(mask):
            bin_accuracies[b] = np.mean(accuracies[mask])
            bin_confidences[b] = np.mean(confidences[mask])
            bin_counts[b] = np.sum(mask)

    # Remove empty bins
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
        )

    bin_accuracies = bin_accuracies[non_empty]
    bin_confidences = bin_confidences[non_empty]
    bin_counts = bin_counts[non_empty]
    # bin_edges kept original for plotting

    # Expected Calibration Error (ECE)
    ece = np.sum(bin_counts * np.abs(bin_accuracies - bin_confidences)) / np.sum(
        bin_counts
    )

    # Brier score (mean squared error)
    brier_score = np.mean((accuracies - confidences) ** 2)

    return CalibrationResult(
        ece=ece,
        brier_score=brier_score,
        n_samples=n_samples,
        bin_edges=bin_edges,
        bin_centers=(bin_edges[:-1] + bin_edges[1:]) / 2,
        bin_accuracies=bin_accuracies,
        bin_confidences=bin_confidences,
        bin_counts=bin_counts,
    )


def plot_reliability_diagram(
    result: CalibrationResult,
    save_path: Optional[str] = None,
    title: str = "Reliability Diagram",
) -> None:
    """
    Plot reliability diagram for calibration visualization.

    Requires matplotlib (optional dependency).

    Parameters
    ----------
    result : CalibrationResult
        Calibration results to plot.
    save_path : Optional[str], optional
        If provided, save figure to this path.
    title : str, optional
        Plot title. Default: "Reliability Diagram".
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        warnings.warn(
            "matplotlib not available, skipping reliability diagram",
            ImportWarning,
        )
        return

    fig, ax = plt.subplots(figsize=(8, 6))

    # Plot perfect calibration line
    ax.plot([0, 1], [0, 1], "k--", label="Perfect calibration", alpha=0.5)

    # Plot reliability curve
    non_empty = result.bin_counts > 0
    if np.any(non_empty):
        bin_centers = result.bin_centers[non_empty]
        bin_accuracies = result.bin_accuracies[non_empty]
        bin_confidences = result.bin_confidences[non_empty]
        bin_counts = result.bin_counts[non_empty]

        # Bar width proportional to bin count
        total = np.sum(bin_counts)
        widths = bin_counts / total * 0.8  # normalized width

        # Plot bars
        for i, (center, acc, conf, width) in enumerate(
            zip(bin_centers, bin_accuracies, bin_confidences, widths)
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

        # Scatter points for bin averages
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

    # Add Brier score annotation
    ax.text(
        0.05,
        0.95,
        f"ECE = {result.ece:.3f}\nBrier = {result.brier_score:.3f}",
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
