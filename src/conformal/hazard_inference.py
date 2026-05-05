"""
Core hazard-based conformal prediction inference functions.

This module implements the probabilistic integral transform (PIT) and
quantile inversion routines for NNCDE-based conformal methods.
"""

import math
from typing import Tuple, Optional, Union, List

import numpy as np
import torch

# Constants
CLAMP_MAX_ETA = 10.0
EPS = 1e-8
chunk_rows = 32768

device = "cuda" if torch.cuda.is_available() else "cpu"


def make_coarse_time_grid(
    uniq_y: np.ndarray, max_grid: int, mode: str = "quantile"
) -> np.ndarray:
    """
    Create a coarse time grid from unique y values.

    Parameters
    ----------
    uniq_y : np.ndarray
        Unique sorted y values.
    max_grid : int
        Maximum number of grid points.
    mode : str, default="quantile"
        Grid reduction method: "quantile" or "linear".

    Returns
    -------
    np.ndarray
        Time grid edges with t[0] = uniq_y[0] - 1.0, shape (n_edges,).
    """
    uniq = np.sort(np.unique(uniq_y))
    if len(uniq) < 2:
        uniq = np.array([uniq.min(), uniq.min() + 1.0], dtype=uniq.dtype)

    if len(uniq) > max_grid:
        if mode == "quantile":
            qs = np.linspace(0, 1, num=max_grid, endpoint=True)
            uniq = np.quantile(uniq, qs, method="nearest")
        else:
            idx = np.linspace(0, len(uniq) - 1, num=max_grid, dtype=int)
            uniq = uniq[idx]

    t0 = uniq[0] - 1.0
    t_edges = np.r_[t0, uniq]
    return t_edges.astype("float32")


@torch.no_grad()
def hazard_grid_for_one_x(
    net: torch.nn.Module,
    mu: np.ndarray,
    sg: np.ndarray,
    t_edges: np.ndarray,
    x_cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative hazard and hazard rates for one test point.

    Parameters
    ----------
    net : torch.nn.Module
        Trained hazard neural network.
    mu : np.ndarray
        Feature means, shape (n_features,).
    sg : np.ndarray
        Feature standard deviations, shape (n_features,).
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    x_cov : np.ndarray
        Covariate vector, shape (n_features,).

    Returns
    -------
    cum_L : np.ndarray
        Cumulative hazard, shape (n_intervals,).
    t_edges : np.ndarray
        Time grid edges (returned as-is).
    lam_all : np.ndarray
        Hazard rates, shape (n_intervals,).
    """
    M = len(t_edges) - 1
    starts = t_edges[:-1]
    ends = t_edges[1:]
    dt = ends - starts
    time_feat = np.c_[starts.reshape(-1, 1), ends.reshape(-1, 1)]
    cov_rep = np.repeat(x_cov.reshape(1, -1), M, axis=0)
    rows = np.concatenate([time_feat, cov_rep], axis=1).astype("float32")
    rows_norm = (rows - mu) / sg

    lam_list = []
    for i in range(0, M, chunk_rows):
        xb = torch.tensor(rows_norm[i : i + chunk_rows], device=device)
        eta = torch.clamp(net(xb).squeeze(1), max=CLAMP_MAX_ETA)
        lam = torch.exp(eta)
        lam_list.append(lam.detach().cpu().numpy())

    lam_all = np.concatenate(lam_list, axis=0).astype("float32")
    contrib = lam_all * dt
    cum_L = np.cumsum(contrib).astype("float32")
    return cum_L, t_edges, lam_all


@torch.no_grad()
def hazard_grid_ensemble(
    models: List[Tuple[torch.nn.Module, np.ndarray, np.ndarray]],
    t_edges: np.ndarray,
    x_cov: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute cumulative hazard via ensemble averaging.

    Averages hazard rates across multiple trained models and computes
    cumulative hazard from the averaged rates.

    Parameters
    ----------
    models : List[Tuple[torch.nn.Module, np.ndarray, np.ndarray]]
        List of (net, mu, sg) tuples for each ensemble member.
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    x_cov : np.ndarray
        Covariate vector, shape (n_features,).

    Returns
    -------
    cum_L : np.ndarray
        Cumulative hazard, shape (n_intervals,).
    t_edges : np.ndarray
        Time grid edges.
    lam_avg : np.ndarray
        Averaged hazard rates, shape (n_intervals,).
    """
    lam_accum = None
    for net, mu, sg in models:
        _, _, lam_i = hazard_grid_for_one_x(net, mu, sg, t_edges, x_cov)
        if lam_accum is None:
            lam_accum = lam_i.astype("float64")
        else:
            lam_accum += lam_i.astype("float64")

    lam_avg = (lam_accum / max(1, len(models))).astype("float32")
    dt = (t_edges[1:] - t_edges[:-1]).astype("float32")
    cum_L = np.cumsum(lam_avg * dt).astype("float32")
    return cum_L, t_edges, lam_avg


def compute_pit(
    cum_L: np.ndarray, t_edges: np.ndarray, lam: np.ndarray, y_val: float
) -> float:
    """
    Compute probability integral transform (PIT) value.

    Evaluates the CDF F(y_val | x) using precomputed cumulative hazard.

    Parameters
    ----------
    cum_L : np.ndarray
        Cumulative hazard, shape (n_intervals,).
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    lam : np.ndarray
        Hazard rates, shape (n_intervals,).
    y_val : float
        Point at which to evaluate CDF.

    Returns
    -------
    float
        PIT value in (eps, 1-eps) where eps=1e-12.
    """
    if y_val >= t_edges[-1]:
        base_H = float(cum_L[-1]) if len(cum_L) > 0 else 0.0
        lam_last = float(lam[-1]) if len(lam) > 0 else 0.0
        extra = lam_last * max(0.0, (y_val - float(t_edges[-1])))
        H = base_H + extra
        Fval = -math.expm1(-H)
        return float(np.clip(Fval, 1e-12, 1.0 - 1e-12))

    k = int(np.searchsorted(t_edges[1:], y_val, side="left"))
    s_k = float(t_edges[k]) if k > 0 else float(t_edges[0])
    base_H = float(cum_L[k - 1]) if k > 0 else 0.0
    lam_k = float(lam[k]) if k < len(lam) else 0.0
    part = lam_k * max(0.0, (y_val - s_k))
    H = base_H + part
    Fval = -math.expm1(-H)
    return float(np.clip(Fval, 1e-12, 1.0 - 1e-12))


def compute_quantile(
    F_grid: np.ndarray,
    cum_L: np.ndarray,
    t_edges: np.ndarray,
    lam: np.ndarray,
    u: float,
) -> float:
    """
    Compute quantile via precomputed F_grid (inverse CDF).

    Parameters
    ----------
    F_grid : np.ndarray
        Precomputed CDF values on time grid, shape (n_intervals,).
    cum_L : np.ndarray
        Cumulative hazard, shape (n_intervals,).
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    lam : np.ndarray
        Hazard rates, shape (n_intervals,).
    u : float
        Quantile level in (0, 1).

    Returns
    -------
    float
        Quantile value Q(u | x).
    """
    u = float(np.clip(u, 1e-12, 1.0 - 1e-12))
    k = int(np.searchsorted(F_grid, u, side="left"))

    if k >= len(F_grid):
        F_end = float(F_grid[-1]) if len(F_grid) > 0 else 0.0
        lam_last = float(lam[-1]) if len(lam) > 0 else 0.0
        if lam_last <= 0.0 or u <= F_end:
            return float(t_edges[-1])
        frac = min(max((u - F_end) / (1.0 - F_end), 0.0), 1.0 - 1e-12)
        delta = -math.log1p(-frac) / lam_last
        return float(t_edges[-1] + max(0.0, delta))

    s_k = float(t_edges[k])
    base_H = float(cum_L[k - 1]) if k > 0 else 0.0
    lam_k = float(lam[k]) if k < len(lam) else 0.0
    F_left = -math.expm1(-base_H)
    num, den = (u - F_left), (1.0 - F_left)

    if den <= 1e-15 or lam_k <= 0.0 or num <= 0.0:
        return s_k

    frac = min(max(num / den, 0.0), 1.0 - 1e-12)
    delta = -math.log1p(-frac) / lam_k
    return float(min(s_k + max(0.0, delta), t_edges[k + 1]))


def precompute_distributions(
    hz_models: Union[Tuple, List],
    t_edges: np.ndarray,
    X_arr: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Batch precompute F_grid, cumulative hazard, and lam for array of X.

    Parameters
    ----------
    hz_models : Union[Tuple, List]
        Either (net, mu, sg) tuple or list of such tuples for ensemble.
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    X_arr : np.ndarray
        Array of covariates, shape (n_samples, n_features).

    Returns
    -------
    cumLs : List[np.ndarray]
        Cumulative hazards, each shape (n_intervals,).
    lams : List[np.ndarray]
        Hazard rates, each shape (n_intervals,).
    Fgrids : List[np.ndarray]
        CDF values on grid, each shape (n_intervals,).
    """
    cumLs, lams, Fgrids = [], [], []

    if isinstance(hz_models, list):
        # Ensemble case
        for x in X_arr:
            cumL, _, lam = hazard_grid_ensemble(hz_models, t_edges, x)
            F_grid = -np.expm1(-cumL.astype(np.float64))
            cumLs.append(cumL.astype("float32"))
            lams.append(lam.astype("float32"))
            Fgrids.append(F_grid.astype("float32"))
    else:
        # Single model case
        net, mu, sg = hz_models
        for x in X_arr:
            cumL, _, lam = hazard_grid_for_one_x(net, mu, sg, t_edges, x)
            F_grid = -np.expm1(-cumL.astype(np.float64))
            cumLs.append(cumL.astype("float32"))
            lams.append(lam.astype("float32"))
            Fgrids.append(F_grid.astype("float32"))

    return cumLs, lams, Fgrids


def compute_optimal_z_star(
    F_grid: np.ndarray,
    cumL: np.ndarray,
    t_edges: np.ndarray,
    lam: np.ndarray,
    alpha: float,
    grid_size: int = 41,
    eps_u: float = 1e-6,
) -> float:
    """
    Compute optimal z* via grid search on shortest interval width.

    Solves z*(x) = argmin_{z in (eps_u, alpha-eps_u)} {Q(z+1-alpha|x) - Q(z|x)}.

    Parameters
    ----------
    F_grid : np.ndarray
        Precomputed CDF values on grid, shape (n_intervals,).
    cumL : np.ndarray
        Cumulative hazard, shape (n_intervals,).
    t_edges : np.ndarray
        Time grid edges, shape (n_edges,).
    lam : np.ndarray
        Hazard rates, shape (n_intervals,).
    alpha : float
        Miscoverage level.
    grid_size : int, default=41
        Number of grid points for z.
    eps_u : float, default=1e-6
        Numerical clipping for quantile levels.

    Returns
    -------
    float
        Optimal z* value.
    """
    z_left = float(eps_u)
    z_right = float(alpha - eps_u)

    if z_right <= z_left:
        return float(alpha / 2.0)

    z_grid = np.linspace(z_left, z_right, int(grid_size)).astype("float64")

    best_z = float(z_grid[0])
    best_w = float("inf")

    for z in z_grid:
        u1 = float(np.clip(z, eps_u, 1.0 - eps_u))
        u2 = float(np.clip(z + 1.0 - alpha, eps_u, 1.0 - eps_u))
        q1 = compute_quantile(F_grid, cumL, t_edges, lam, u1)
        q2 = compute_quantile(F_grid, cumL, t_edges, lam, u2)
        w = q2 - q1
        if w < best_w:
            best_w = w
            best_z = float(z)

    return best_z
