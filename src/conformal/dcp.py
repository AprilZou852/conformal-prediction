"""
DCP-NNCDE (Distribution-free Conformal Prediction) calibration.

Distribution-free variant leveraging optimal z* for centering.
"""

from typing import Tuple, Union, List

import numpy as np

from .hazard_inference import compute_quantile


def calibrate_dcp(
    pit_cal: np.ndarray,
    z_star_cal: np.ndarray,
    z_star_test: np.ndarray,
    alpha: float,
    n_cal: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate DCP method using optimal z-centering.

    For calibration:
        center_cal[i] = z_star_cal[i] + (1 - alpha) / 2
        scores[i] = |PIT[i] - center_cal[i]|
        q_hat = quantile(scores, level) where level = (1-alpha)(1+1/n_cal)

    For each test point i:
        center_test[i] = z_star_test[i] + (1 - alpha) / 2
        u_lo[i] = clip(center_test[i] - q_hat, 0, 1)
        u_hi[i] = clip(center_test[i] + q_hat, 0, 1)

    Parameters
    ----------
    pit_cal : np.ndarray
        PIT values on calibration set, shape (n_cal,).
    z_star_cal : np.ndarray
        Optimal z* values on calibration set, shape (n_cal,).
    z_star_test : np.ndarray
        Optimal z* values on test set, shape (n_test,).
    alpha : float
        Miscoverage level.
    n_cal : int
        Size of calibration set.

    Returns
    -------
    u_lo_dcp : np.ndarray
        Lower quantile levels for each test point, shape (n_test,).
    u_hi_dcp : np.ndarray
        Upper quantile levels for each test point, shape (n_test,).
    """
    pit_cal = pit_cal.astype("float64")
    z_star_cal = z_star_cal.astype("float64")
    z_star_test = z_star_test.astype("float64")

    # Compute quantile level with conservative correction
    q_level = (1 - alpha) * (1 + 1 / n_cal)
    q_level = min(q_level, 1.0 - 1e-6)

    # Calibration: compute q_hat as quantile of centered absolute deviations
    center_cal = z_star_cal + (1.0 - alpha) / 2.0
    scores_dcp = np.abs(pit_cal - center_cal)
    q_hat_dcp = float(np.quantile(scores_dcp, q_level))

    # Test: compute u_lo and u_hi for each test point
    center_test = z_star_test + (1.0 - alpha) / 2.0
    u_lo_dcp = np.clip(center_test - q_hat_dcp, 0.0, 1.0).astype("float32")
    u_hi_dcp = np.clip(center_test + q_hat_dcp, 0.0, 1.0).astype("float32")

    return u_lo_dcp, u_hi_dcp


def predict_dcp(
    hz_models_or_cache: Union[Tuple, List, Tuple[List, List, List]],
    t_edges: np.ndarray,
    X_test: np.ndarray,
    u_lo_dcp: np.ndarray,
    u_hi_dcp: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate DCP prediction intervals.

    Parameters
    ----------
    hz_models_or_cache : Union[Tuple, List, Tuple[List, List, List]]
        Either:
        - (net, mu, sg) tuple for single model
        - List of (net, mu, sg) for ensemble
        - (test_Fgrids, test_cumLs, test_lams) precomputed cache
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    X_test : np.ndarray
        Test covariates, shape (n_test, n_features). Ignored if cache provided.
    u_lo_dcp : np.ndarray
        Lower quantile levels for each test point, shape (n_test,).
    u_hi_dcp : np.ndarray
        Upper quantile levels for each test point, shape (n_test,).

    Returns
    -------
    lo : np.ndarray
        Lower prediction bounds, shape (n_test,).
    hi : np.ndarray
        Upper prediction bounds, shape (n_test,).
    """
    # Check if precomputed cache is provided
    if isinstance(hz_models_or_cache, tuple) and len(hz_models_or_cache) == 3:
        test_Fgrids, test_cumLs, test_lams = hz_models_or_cache
        is_cache = all(isinstance(x, list) for x in hz_models_or_cache)
        if is_cache:
            n_test = len(test_Fgrids)
            lo = np.empty(n_test, dtype="float32")
            hi = np.empty(n_test, dtype="float32")
            for i in range(n_test):
                lo[i] = compute_quantile(
                    test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_lo_dcp[i]
                )
                hi[i] = compute_quantile(
                    test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_hi_dcp[i]
                )
            return lo, hi

    # Otherwise compute from X_test
    from .hazard_inference import precompute_distributions

    test_cumLs, test_lams, test_Fgrids = precompute_distributions(
        hz_models_or_cache, t_edges, X_test
    )

    n_test = len(X_test)
    lo = np.empty(n_test, dtype="float32")
    hi = np.empty(n_test, dtype="float32")
    for i in range(n_test):
        lo[i] = compute_quantile(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_lo_dcp[i]
        )
        hi[i] = compute_quantile(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_hi_dcp[i]
        )

    return lo, hi
