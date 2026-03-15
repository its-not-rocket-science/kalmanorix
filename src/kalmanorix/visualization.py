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


if __name__ == "__main__":
    demo_visualisation()
