"""
Standard conformal prediction baselines (Residual, Rescaled, CQR).

These methods serve as benchmarks for the main CPI/DCP approaches.
"""

from typing import Tuple

import numpy as np


def calibrate_residual(
    y_cal: np.ndarray, y_hat_cal: np.ndarray, alpha: float, n_cal: int = None
) -> float:
    """
    Calibrate residual-based conformal prediction.

    Computes the (1-alpha)(1+1/n) quantile of absolute residuals.

    Parameters
    ----------
    y_cal : np.ndarray
        True values on calibration set, shape (n_cal,).
    y_hat_cal : np.ndarray
        Predicted values on calibration set, shape (n_cal,).
    alpha : float
        Miscoverage level.
    n_cal : int, optional
        Calibration size. If None, inferred from y_cal.

    Returns
    -------
    float
        Quantile q_hat.
    """
    if n_cal is None:
        n_cal = len(y_cal)

    q_level = (1 - alpha) * (1 + 1 / n_cal)
    q_level = min(q_level, 1.0 - 1e-6)

    scores = np.abs(y_cal - y_hat_cal)
    q_hat = float(np.quantile(scores, q_level))
    return q_hat


def predict_residual(y_hat_test: np.ndarray, q_hat: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate residual-based prediction intervals.

    Parameters
    ----------
    y_hat_test : np.ndarray
        Predicted values on test set, shape (n_test,).
    q_hat : float
        Quantile from calibration.

    Returns
    -------
    lo : np.ndarray
        Lower bounds, shape (n_test,).
    hi : np.ndarray
        Upper bounds, shape (n_test,).
    """
    lo = (y_hat_test - q_hat).astype("float32")
    hi = (y_hat_test + q_hat).astype("float32")
    return lo, hi


def calibrate_rescaled(
    y_cal: np.ndarray,
    mean_cal: np.ndarray,
    sigma_cal: np.ndarray,
    alpha: float,
    n_cal: int = None,
    sigma_min: float = 1e-3,
    sigma_max: float = 10.0,
) -> float:
    """
    Calibrate rescaled conformal prediction.

    Computes the (1-alpha)(1+1/n) quantile of rescaled residuals |y - mu| / sigma.

    Parameters
    ----------
    y_cal : np.ndarray
        True values on calibration set, shape (n_cal,).
    mean_cal : np.ndarray
        Predicted means, shape (n_cal,).
    sigma_cal : np.ndarray
        Predicted standard deviations, shape (n_cal,).
    alpha : float
        Miscoverage level.
    n_cal : int, optional
        Calibration size.
    sigma_min : float, default=1e-3
        Minimum clipping for sigma.
    sigma_max : float, default=10.0
        Maximum clipping for sigma.

    Returns
    -------
    float
        Quantile q_hat.
    """
    if n_cal is None:
        n_cal = len(y_cal)

    q_level = (1 - alpha) * (1 + 1 / n_cal)
    q_level = min(q_level, 1.0 - 1e-6)

    sigma_cal = np.clip(sigma_cal, sigma_min, sigma_max)
    scores = np.abs(y_cal - mean_cal) / (sigma_cal + 1e-6)
    q_hat = float(np.quantile(scores, q_level))
    return q_hat


def predict_rescaled(
    mean_test: np.ndarray,
    sigma_test: np.ndarray,
    q_hat: float,
    sigma_min: float = 1e-3,
    sigma_max: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate rescaled prediction intervals.

    Parameters
    ----------
    mean_test : np.ndarray
        Predicted means on test set, shape (n_test,).
    sigma_test : np.ndarray
        Predicted standard deviations on test set, shape (n_test,).
    q_hat : float
        Quantile from calibration.
    sigma_min : float, default=1e-3
        Minimum clipping for sigma.
    sigma_max : float, default=10.0
        Maximum clipping for sigma.

    Returns
    -------
    lo : np.ndarray
        Lower bounds, shape (n_test,).
    hi : np.ndarray
        Upper bounds, shape (n_test,).
    """
    sigma_test = np.clip(sigma_test, sigma_min, sigma_max)
    lo = (mean_test - q_hat * sigma_test).astype("float32")
    hi = (mean_test + q_hat * sigma_test).astype("float32")
    return lo, hi


def calibrate_cqr(
    y_cal: np.ndarray,
    q_lo_cal: np.ndarray,
    q_hi_cal: np.ndarray,
    alpha: float,
    n_cal: int = None,
) -> float:
    """
    Calibrate conditional quantile regression (CQR).

    Computes the (1-alpha)(1+1/n) quantile of max(q_lo - y, y - q_hi).

    Parameters
    ----------
    y_cal : np.ndarray
        True values on calibration set, shape (n_cal,).
    q_lo_cal : np.ndarray
        Predicted lower quantiles, shape (n_cal,).
    q_hi_cal : np.ndarray
        Predicted upper quantiles, shape (n_cal,).
    alpha : float
        Miscoverage level.
    n_cal : int, optional
        Calibration size.

    Returns
    -------
    float
        Quantile q_hat.
    """
    if n_cal is None:
        n_cal = len(y_cal)

    q_level = (1 - alpha) * (1 + 1 / n_cal)
    q_level = min(q_level, 1.0 - 1e-6)

    scores = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
    q_hat = float(np.quantile(scores, q_level))
    return q_hat


def predict_cqr(
    q_lo_test: np.ndarray, q_hi_test: np.ndarray, q_hat: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate CQR prediction intervals.

    Parameters
    ----------
    q_lo_test : np.ndarray
        Predicted lower quantiles on test set, shape (n_test,).
    q_hi_test : np.ndarray
        Predicted upper quantiles on test set, shape (n_test,).
    q_hat : float
        Quantile from calibration.

    Returns
    -------
    lo : np.ndarray
        Lower bounds, shape (n_test,).
    hi : np.ndarray
        Upper bounds, shape (n_test,).
    """
    lo = (q_lo_test - q_hat).astype("float32")
    hi = (q_hi_test + q_hat).astype("float32")
    return lo, hi
