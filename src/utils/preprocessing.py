"""Data preprocessing and standardization utilities."""

from typing import Tuple
import numpy as np


def fit_standardizer(
    X_train: np.ndarray,
    y_train: np.ndarray,
    eps: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute standardization statistics for features and targets.

    Computes mean and standard deviation for X and y. Ensures std >= eps
    to avoid division by zero.

    Args:
        X_train: Training features, shape (n, p).
        y_train: Training targets, shape (n,).
        eps: Minimum standard deviation threshold.

    Returns:
        (x_mu, x_sd, y_mu, y_sd) where:
        - x_mu: Feature means, shape (p,), dtype float32.
        - x_sd: Feature std devs, shape (p,), dtype float32.
        - y_mu: Target mean, scalar, dtype float32.
        - y_sd: Target std dev, scalar, dtype float32.
    """
    x_mu = X_train.mean(axis=0).astype("float32")
    x_sd = X_train.std(axis=0).astype("float32")
    x_sd[x_sd < eps] = 1.0

    y_mu = np.float32(np.mean(y_train))
    y_sd = np.float32(np.std(y_train))
    if y_sd < eps:
        y_sd = np.float32(1.0)

    return x_mu, x_sd, y_mu, y_sd


def transform_X(
    X: np.ndarray,
    x_mu: np.ndarray,
    x_sd: np.ndarray,
) -> np.ndarray:
    """
    Standardize features using pre-computed statistics.

    Args:
        X: Features to standardize.
        x_mu: Feature means from fit_standardizer.
        x_sd: Feature std devs from fit_standardizer.

    Returns:
        Standardized features, dtype float32.
    """
    return ((X - x_mu) / x_sd).astype("float32")


def transform_y(
    y: np.ndarray,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
) -> np.ndarray:
    """
    Standardize targets using pre-computed statistics.

    Args:
        y: Targets to standardize.
        y_mu: Target mean from fit_standardizer.
        y_sd: Target std dev from fit_standardizer.

    Returns:
        Standardized targets, dtype float32.
    """
    return ((y - y_mu) / y_sd).astype("float32")


def inverse_y(
    y_std: np.ndarray,
    y_mu: np.ndarray,
    y_sd: np.ndarray,
) -> np.ndarray:
    """
    Inverse standardization: convert standardized targets back to original scale.

    Args:
        y_std: Standardized targets.
        y_mu: Target mean from fit_standardizer.
        y_sd: Target std dev from fit_standardizer.

    Returns:
        Targets in original scale, dtype float32.
    """
    return (y_mu + y_sd * y_std).astype("float32")
