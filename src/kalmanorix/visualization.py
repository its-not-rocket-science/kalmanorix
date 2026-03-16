"""
Visualisation tools for embedding uncertainty.

This module provides functions to visualise uncertainty estimates and
fusion results. Requires matplotlib (optional dependency).
"""

from typing import Optional, List, Tuple
import numpy as np

try:
    import matplotlib.pyplot as plt  # type: ignore[import-not-found]

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def plot_embedding_with_uncertainty(
    embedding: np.ndarray,
    uncertainty: np.ndarray,
    *,
    title: str = "Embedding with Uncertainty",
    dims_to_plot: Optional[List[int]] = None,
    ax=None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """
    Plot embedding dimensions with uncertainty bars.

    Args:
        embedding: Vector of shape (d,)
        uncertainty: Diagonal covariance vector of shape (d,)
        title: Plot title
        dims_to_plot: List of dimension indices to plot (default: first 20)
        ax: Matplotlib axes to plot on (if None, creates new figure)
        show: Whether to call plt.show()

    Returns:
        Matplotlib axes if show=False, else None

    Raises:
        ImportError: If matplotlib not installed
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )

    d = embedding.shape[0]
    if dims_to_plot is None:
        dims_to_plot = list(range(min(20, d)))

    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 6))
        own_fig = True
    else:
        own_fig = False

    indices = np.array(dims_to_plot)
    x_pos = np.arange(len(indices))

    # Values and errors
    values = embedding[indices]
    errors = np.sqrt(uncertainty[indices])  # Convert variance to std dev

    ax.bar(x_pos, values, yerr=errors, capsize=5, alpha=0.7, color="steelblue")
    ax.axhline(y=0, color="black", linewidth=0.5, linestyle="--", alpha=0.5)

    ax.set_xlabel("Dimension Index")
    ax.set_ylabel("Embedding Value ± 1σ")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in indices])

    # Add text with total uncertainty
    total_var = np.sum(uncertainty)
    ax.text(
        0.02,
        0.98,
        f"Total variance: {total_var:.4f}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    if show and own_fig:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_covariance_comparison(
    covariances: List[np.ndarray],
    labels: List[str],
    *,
    title: str = "Covariance Comparison",
    dims_to_plot: Optional[List[int]] = None,
    ax=None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """
    Compare multiple covariance vectors.

    Args:
        covariances: List of diagonal covariance vectors, each shape (d,)
        labels: Names for each covariance
        title: Plot title
        dims_to_plot: Dimensions to plot (default: first 10)
        ax: Matplotlib axes
        show: Whether to call plt.show()

    Returns:
        Matplotlib axes if show=False, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )

    if len(covariances) != len(labels):
        raise ValueError("covariances and labels must have same length")

    d = covariances[0].shape[0]
    if dims_to_plot is None:
        dims_to_plot = list(range(min(10, d)))

    if ax is None:
        _fig, ax = plt.subplots(figsize=(12, 6))
        own_fig = True
    else:
        own_fig = False

    indices = np.array(dims_to_plot)
    x_pos = np.arange(len(indices))
    width = 0.8 / len(covariances)

    for i, (cov, label) in enumerate(zip(covariances, labels)):
        offset = (i - len(covariances) / 2 + 0.5) * width
        values = cov[indices]
        ax.bar(x_pos + offset, values, width, label=label, alpha=0.7)

    ax.set_xlabel("Dimension Index")
    ax.set_ylabel("Variance")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels([str(i) for i in indices])
    ax.legend()
    ax.grid(True, alpha=0.3, linestyle="--")

    if show and own_fig:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_fusion_weights(
    weights: List[Tuple[str, float]],
    *,
    title: str = "Fusion Weights",
    ax=None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """
    Plot fusion weights as a bar chart.

    Args:
        weights: List of (module_name, weight) tuples
        title: Plot title
        ax: Matplotlib axes
        show: Whether to call plt.show()

    Returns:
        Matplotlib axes if show=False, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 6))
        own_fig = True
    else:
        own_fig = False

    names, values = zip(*weights)
    x_pos = np.arange(len(names))

    bars = ax.bar(x_pos, values, color="coral", alpha=0.7)
    ax.set_xlabel("Module")
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(names, rotation=45, ha="right")

    # Add value labels on bars
    for bar_item in bars:
        height = bar_item.get_height()
        ax.text(
            bar_item.get_x() + bar_item.get_width() / 2.0,
            height + 0.01,
            f"{height:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylim(0, max(values) * 1.2)

    if show and own_fig:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_alignment_matrix(
    alignment_matrix: np.ndarray,
    *,
    title: str = "Alignment Matrix",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """Plot alignment matrix as heatmap.

    Args:
        alignment_matrix: Orthogonal matrix of shape (d, d)
        title: Plot title
        ax: Matplotlib axes to plot on (if None, creates new figure)
        show: Whether to call plt.show()

    Returns:
        Matplotlib axes if show=False, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )

    if ax is None:
        _fig, ax = plt.subplots(figsize=(8, 6))
        own_fig = True
    else:
        own_fig = False

    # Heatmap
    im = ax.imshow(alignment_matrix, cmap="RdBu_r", aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("Target Dimension")
    ax.set_ylabel("Source Dimension")
    ax.set_title(title)

    # Colorbar
    plt.colorbar(im, ax=ax, shrink=0.8)

    # Add text with orthogonality error
    ortho_error = np.linalg.norm(
        alignment_matrix.T @ alignment_matrix - np.eye(alignment_matrix.shape[0]),
        ord="fro",
    )
    ax.text(
        0.02,
        0.98,
        f"Orthogonality error: {ortho_error:.2e}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    if show and own_fig:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_similarity_comparison(
    similarities_before: np.ndarray,
    similarities_after: np.ndarray,
    *,
    title: str = "Similarity Before/After Alignment",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """Plot distribution of similarities before and after alignment.

    Args:
        similarities_before: Array of cosine similarities before alignment
        similarities_after: Array of cosine similarities after alignment
        title: Plot title
        ax: Matplotlib axes
        show: Whether to call plt.show()

    Returns:
        Matplotlib axes if show=False, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )

    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 6))
        own_fig = True
    else:
        own_fig = False

    # Box plot or violin plot
    data = [similarities_before, similarities_after]
    positions = [1, 2]
    labels = ["Before", "After"]

    # Box plot
    bp = ax.boxplot(data, positions=positions, patch_artist=True, widths=0.6)
    # Color boxes
    colors = ["lightblue", "lightcoral"]
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)

    # Add individual points with jitter
    for i, (pos, vals) in enumerate(zip(positions, data)):
        jitter = np.random.uniform(-0.1, 0.1, size=len(vals))
        ax.scatter(pos + jitter, vals, alpha=0.4, s=20, color="black", zorder=3)

    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Cosine Similarity")
    ax.set_title(title)
    ax.grid(True, alpha=0.3, linestyle="--")

    # Add mean lines
    for i, vals in enumerate(data):
        ax.axhline(
            np.mean(vals),
            xmin=(i + 0.6) / len(positions),
            xmax=(i + 1.4) / len(positions),
            color="red",
            linestyle="--",
            alpha=0.8,
        )

    # Add text with improvement
    mean_before = np.mean(similarities_before)
    mean_after = np.mean(similarities_after)
    improvement = mean_after - mean_before
    ax.text(
        0.02,
        0.98,
        f"Δ mean = {improvement:.3f} ({improvement / mean_before * 100:.1f}%)",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    if show and own_fig:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def plot_alignment_error(
    true_matrix: np.ndarray,
    estimated_matrix: np.ndarray,
    *,
    title: str = "Alignment Error",
    ax: Optional[plt.Axes] = None,
    show: bool = True,
) -> Optional[plt.Axes]:
    """Plot error between true and estimated alignment matrices.

    Args:
        true_matrix: True orthogonal matrix (d, d)
        estimated_matrix: Estimated alignment matrix (d, d)
        title: Plot title
        ax: Matplotlib axes
        show: Whether to call plt.show()

    Returns:
        Matplotlib axes if show=False, else None
    """
    if not HAS_MATPLOTLIB:
        raise ImportError(
            "matplotlib is required for visualisation. "
            "Install with: pip install matplotlib"
        )

    error_matrix = estimated_matrix - true_matrix

    if ax is None:
        _fig, ax = plt.subplots(figsize=(10, 4))
        own_fig = True
    else:
        own_fig = False

    # Plot error heatmap
    im = ax.imshow(
        error_matrix,
        cmap="RdBu_r",
        aspect="auto",
        vmin=-np.max(np.abs(error_matrix)),
        vmax=np.max(np.abs(error_matrix)),
    )
    ax.set_xlabel("Dimension")
    ax.set_ylabel("Dimension")
    ax.set_title(title)
    plt.colorbar(im, ax=ax, shrink=0.8, label="Error")

    # Add text with Frobenius norm
    fro_error = np.linalg.norm(error_matrix, ord="fro")
    rel_error = fro_error / np.linalg.norm(true_matrix, ord="fro")
    ax.text(
        0.02,
        0.98,
        f"Frobenius error: {fro_error:.2e}\nRelative: {rel_error:.2e}",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox={"boxstyle": "round", "facecolor": "wheat", "alpha": 0.5},
    )

    if show and own_fig:
        plt.tight_layout()
        plt.show()
        return None

    return ax


def demo_visualisation():
    """Demo the visualisation functions with synthetic data."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed, skipping demo")
        return

    np.random.seed(42)
    d = 50

    # Create synthetic embedding and uncertainty
    embedding = np.random.randn(d)
    uncertainty = np.exp(np.random.randn(d))  # log-normal variances

    print("Plotting embedding with uncertainty...")
    plot_embedding_with_uncertainty(
        embedding, uncertainty, title="Synthetic Embedding with Uncertainty", show=True
    )

    # Compare multiple covariances
    covariances = [uncertainty, np.ones(d) * 0.5, np.exp(np.random.randn(d) * 0.5)]
    labels = ["Empirical", "Uniform", "Another"]

    print("Plotting covariance comparison...")
    plot_covariance_comparison(
        covariances, labels, title="Covariance Comparison", show=True
    )

    # Fusion weights
    weights = [("Tech", 0.6), ("Cooking", 0.3), ("Medical", 0.1)]
    print("Plotting fusion weights...")
    plot_fusion_weights(weights, title="Example Fusion Weights", show=True)

    # Alignment visualization
    print("\n--- Alignment visualisation ---")
    d_align = 20  # smaller for visibility
    # Generate random orthogonal matrix
    Q, _ = np.linalg.qr(np.random.randn(d_align, d_align))
    if np.linalg.det(Q) < 0:
        Q[:, -1] *= -1
    print("Plotting alignment matrix...")
    plot_alignment_matrix(Q, title="Random Orthogonal Matrix", show=True)

    # Simulate similarity improvement
    n_samples = 100
    similarities_before = np.random.uniform(0.1, 0.5, n_samples)
    similarities_after = similarities_before + np.random.uniform(0.2, 0.4, n_samples)
    similarities_after = np.clip(similarities_after, 0, 1)
    print("Plotting similarity comparison...")
    plot_similarity_comparison(
        similarities_before,
        similarities_after,
        title="Simulated Similarity Improvement",
        show=True,
    )

    # Alignment error
    noise = np.random.randn(d_align, d_align) * 0.1
    estimated = Q + noise
    # Re-orthogonalize (simple approximation)
    U, _, Vt = np.linalg.svd(estimated)
    estimated_ortho = U @ Vt
    print("Plotting alignment error...")
    plot_alignment_error(
        Q, estimated_ortho, title="Alignment Error (with noise)", show=True
    )


if __name__ == "__main__":
    demo_visualisation()
