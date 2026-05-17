"""Evaluation metrics for conformal prediction."""

from typing import Tuple, Optional
import numpy as np
from sklearn.decomposition import PCA

try:
    from covmetrics import ERT
    _HAS_COVMETRICS = True
except ImportError:
    _HAS_COVMETRICS = False


def compute_coverage(y_test: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """
    Compute empirical coverage: fraction of test points in [lo, hi].

    Args:
        y_test: Test targets, shape (n,).
        lo: Lower bounds, shape (n,).
        hi: Upper bounds, shape (n,).

    Returns:
        Coverage rate in [0, 1].
    """
    return float(np.mean((y_test >= lo) & (y_test <= hi)))


def compute_width(lo: np.ndarray, hi: np.ndarray) -> float:
    """
    Compute mean interval width.

    Args:
        lo: Lower bounds, shape (n,).
        hi: Upper bounds, shape (n,).

    Returns:
        Mean width across all intervals.
    """
    return float(np.mean(hi - lo))


def compute_wmad(
    coverages: np.ndarray,
    weights: np.ndarray,
    target: float,
) -> float:
    """
    Compute Weighted Mean Absolute Deviation from target coverage.

    WMAD = sum(weights * |coverage - target|) / sum(weights)

    Args:
        coverages: Coverage rates by group, shape (n_groups,).
        weights: Group weights (e.g., counts), shape (n_groups,).
        target: Target coverage rate (e.g., 0.9).

    Returns:
        WMAD value.
    """
    deviations = np.abs(coverages - target)
    return float(np.average(deviations, weights=weights))


def compute_worst_bin_coverage(coverages: np.ndarray) -> float:
    """
    Compute worst (minimum) bin coverage.

    Args:
        coverages: Coverage rates by group/bin, shape (n_groups,).

    Returns:
        Minimum coverage rate among all groups.
    """
    return float(np.min(coverages))


def compute_l1_ert(
    X_test: np.ndarray,
    y_test: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    alpha: float = 0.10,
) -> float:
    """
    Compute L1-ERT metric for conditional coverage evaluation.

    Estimates E[ | P(Y in C(X) | X) - (1-alpha) | ] using the ERT
    (Estimated Risk of the Threshold) approach from Angelopoulos et al.
    (arXiv:2512.11779). Lower values indicate better conditional coverage.

    Requires the `covmetrics` package: pip install covmetrics.

    Args:
        X_test: Test features, shape (n, p).
        y_test: Test targets, shape (n,).
        lo: Lower bounds of prediction intervals, shape (n,).
        hi: Upper bounds, shape (n,).
        alpha: Nominal miscoverage level (default: 0.10).

    Returns:
        L1-ERT value (float). Returns NaN if covmetrics is not installed.
    """
    if not _HAS_COVMETRICS:
        return float("nan")
    cover = ((y_test >= lo) & (y_test <= hi)).astype(float)
    return float(ERT().evaluate(X_test, cover, alpha))


def compute_pc1_groups(
    X: np.ndarray,
    n_groups: int = 4,
) -> Tuple[np.ndarray, PCA]:
    """
    Compute PC1 (first principal component) and divide samples into groups.

    Useful for conditional coverage evaluation: partitions samples
    along the direction of maximum variance.

    Args:
        X: Features, shape (n, p).
        n_groups: Number of groups (e.g., 4 for quartiles).

    Returns:
        (pc1_values, pca_model) where:
        - pc1_values: PC1 coordinates, shape (n,).
        - pca_model: Fitted PCA model for later projection.
    """
    pca = PCA(n_components=1)
    pc1_values = pca.fit_transform(X).squeeze()
    return pc1_values, pca


def assign_pc1_group(
    pc1_values: np.ndarray,
    n_groups: int = 4,
) -> np.ndarray:
    """
    Assign samples to groups based on PC1 quantiles.

    Creates n_groups equally-sized groups by dividing PC1 into quantiles.

    Args:
        pc1_values: PC1 coordinates, shape (n,).
        n_groups: Number of groups.

    Returns:
        Group assignments, shape (n,), with values in {0, 1, ..., n_groups-1}.
    """
    quantiles = np.linspace(0, 1, n_groups + 1)
    bin_edges = np.quantile(pc1_values, quantiles)
    # Use searchsorted to assign to groups
    groups = np.digitize(pc1_values, bin_edges) - 1
    # Ensure groups are in [0, n_groups-1]
    groups = np.clip(groups, 0, n_groups - 1)
    return groups
