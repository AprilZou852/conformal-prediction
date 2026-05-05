"""
CPI-NNCDE (Conditional Probability Integral) calibration.

The paper's main proposed method for conformal prediction with NNCDE.
"""

from typing import Tuple, Union, List

import numpy as np

from .hazard_inference import compute_quantile


def calibrate_cpi(
    pit_cal: np.ndarray, z_star_test: np.ndarray, alpha: float
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calibrate CPI method to find coverage-adjusted quantiles.

    For each test point i with optimal z_i*, computes:
        u_lo[i] = quantile(pit_cal, z_i*)
        u_hi[i] = quantile(pit_cal, z_i* + 1 - alpha)

    Parameters
    ----------
    pit_cal : np.ndarray
        PIT values on calibration set, shape (n_cal,).
    z_star_test : np.ndarray
        Optimal z* values for test points, shape (n_test,).
    alpha : float
        Miscoverage level (e.g., 0.1 for 90% coverage).

    Returns
    -------
    u_lo : np.ndarray
        Lower quantile levels for each test point, shape (n_test,).
    u_hi : np.ndarray
        Upper quantile levels for each test point, shape (n_test,).
    """
    eps_u = 1e-6
    pit_cal = pit_cal.astype("float64")
    z_star_test = z_star_test.astype("float64")

    qvec_lo = np.clip(z_star_test, eps_u, 1.0 - eps_u)
    qvec_hi = np.clip((z_star_test + 1.0 - alpha), eps_u, 1.0 - eps_u)

    u_lo = np.quantile(pit_cal, qvec_lo).astype("float32")
    u_hi = np.quantile(pit_cal, qvec_hi).astype("float32")

    u_lo = np.clip(u_lo, eps_u, 1.0 - eps_u)
    u_hi = np.clip(u_hi, eps_u, 1.0 - eps_u)

    return u_lo, u_hi


def predict_cpi(
    hz_models_or_cache: Union[Tuple, List, Tuple[List, List, List]],
    t_edges: np.ndarray,
    X_test: np.ndarray,
    u_lo: np.ndarray,
    u_hi: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate CPI prediction intervals.

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
    u_lo : np.ndarray
        Lower quantile levels for each test point, shape (n_test,).
    u_hi : np.ndarray
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
                    test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_lo[i]
                )
                hi[i] = compute_quantile(
                    test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_hi[i]
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
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_lo[i]
        )
        hi[i] = compute_quantile(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_hi[i]
        )

    return lo, hi
