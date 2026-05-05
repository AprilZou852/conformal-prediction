"""Plotting utilities for conformal prediction evaluation."""

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d


def kernel_smooth_1d(
    x: np.ndarray,
    y: np.ndarray,
    sigma: float = 1.0,
) -> np.ndarray:
    """
    Smooth 1D data using Gaussian kernel.

    Applies Gaussian filter along the sorted x-axis.

    Args:
        x: X-coordinates, shape (n,).
        y: Y-values, shape (n,).
        sigma: Gaussian kernel width (standard deviation).

    Returns:
        Smoothed y-values, shape (n,).
    """
    sort_idx = np.argsort(x)
    x_sorted = x[sort_idx]
    y_sorted = y[sort_idx]
    y_smooth = gaussian_filter1d(y_sorted, sigma=sigma)
    # Unsort back to original order
    unsort_idx = np.argsort(sort_idx)
    return y_smooth[unsort_idx]


def plot_conditional_coverage_scatter(
    pc1_groups: np.ndarray,
    coverages: np.ndarray,
    target_coverage: float = 0.9,
    title: str = "Conditional Coverage by PC1 Group",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot coverage vs PC1 groups as scatter plot with target line.

    Creates a scatter plot showing coverage by PC1-based group with a
    horizontal line at target coverage.

    Args:
        pc1_groups: PC1 group assignments, shape (n_test,).
        coverages: Coverage indicator per sample, shape (n_test,).
        target_coverage: Target coverage level (e.g., 0.9).
        title: Plot title.
        figsize: Figure size (width, height).
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Compute group-wise coverage
    unique_groups = np.unique(pc1_groups)
    group_coverages = []
    for g in unique_groups:
        mask = pc1_groups == g
        cov = np.mean(coverages[mask])
        group_coverages.append(cov)

    ax.scatter(unique_groups, group_coverages, s=100, alpha=0.7, edgecolors="black")
    ax.axhline(y=target_coverage, color="r", linestyle="--", linewidth=2, label=f"Target: {target_coverage}")
    ax.set_xlabel("PC1 Group", fontsize=12)
    ax.set_ylabel("Coverage", fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim([-0.05, 1.05])
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_smoothed_curves(
    x: np.ndarray,
    coverage_data: dict,
    width_data: dict = None,
    title: str = "Smoothed Coverage and Width",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot smoothed coverage and width curves for multiple methods.

    Creates line plots with kernel-smoothed coverage/width curves.

    Args:
        x: X-axis values (e.g., PC1 values or sample index).
        coverage_data: Dict mapping method names to coverage arrays, shape (n,).
        width_data: Dict mapping method names to width arrays, shape (n,).
                   If None, only coverage is plotted.
        title: Plot title.
        figsize: Figure size.
        save_path: If provided, save figure to this path.
    """
    if width_data is None:
        n_plots = 1
    else:
        n_plots = 2

    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    if n_plots == 1:
        axes = [axes]

    # Plot coverage
    ax = axes[0]
    for method, cov in coverage_data.items():
        y_smooth = kernel_smooth_1d(x, cov, sigma=1.0)
        ax.plot(x, y_smooth, label=method, alpha=0.8, linewidth=2)

    ax.set_xlabel("PC1 or Index", fontsize=11)
    ax.set_ylabel("Coverage", fontsize=11)
    ax.set_title("Smoothed Coverage", fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim([-0.05, 1.05])

    # Plot width if provided
    if width_data is not None:
        ax = axes[1]
        for method, wid in width_data.items():
            y_smooth = kernel_smooth_1d(x, wid, sigma=1.0)
            ax.plot(x, y_smooth, label=method, alpha=0.8, linewidth=2)

        ax.set_xlabel("PC1 or Index", fontsize=11)
        ax.set_ylabel("Mean Width", fontsize=11)
        ax.set_title("Smoothed Interval Width", fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)


def plot_bar_coverage_width(
    methods: list,
    coverages: np.ndarray,
    widths: np.ndarray,
    target_coverage: float = 0.9,
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
) -> None:
    """
    Plot grouped bar chart for coverage and width across methods.

    Creates side-by-side bar plots showing coverage and width metrics.

    Args:
        methods: List of method names.
        coverages: Array of coverage values, shape (n_methods,).
        widths: Array of width values, shape (n_methods,).
        target_coverage: Target coverage line.
        figsize: Figure size.
        save_path: If provided, save figure to this path.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    x = np.arange(len(methods))
    width = 0.6

    # Coverage bars
    colors = ["green" if c >= target_coverage else "orange" for c in coverages]
    ax1.bar(x, coverages, width, color=colors, alpha=0.7, edgecolor="black")
    ax1.axhline(y=target_coverage, color="r", linestyle="--", linewidth=2, label=f"Target: {target_coverage}")
    ax1.set_ylabel("Coverage", fontsize=11)
    ax1.set_title("Empirical Coverage by Method", fontsize=12)
    ax1.set_xticks(x)
    ax1.set_xticklabels(methods, rotation=45, ha="right")
    ax1.set_ylim([0, 1.05])
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis="y")

    # Width bars
    ax2.bar(x, widths, width, color="steelblue", alpha=0.7, edgecolor="black")
    ax2.set_ylabel("Mean Width", fontsize=11)
    ax2.set_title("Mean Interval Width by Method", fontsize=12)
    ax2.set_xticks(x)
    ax2.set_xticklabels(methods, rotation=45, ha="right")
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()

    plt.close(fig)
