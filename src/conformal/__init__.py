"""Conformal prediction methods (CPI, DCP, baselines)."""

from .hazard_inference import (
    make_coarse_time_grid,
    hazard_grid_for_one_x,
    hazard_grid_ensemble,
    compute_pit,
    compute_quantile,
    precompute_distributions,
    compute_optimal_z_star,
)
from .cpi import calibrate_cpi, predict_cpi
from .dcp import calibrate_dcp, predict_dcp
from .baselines import (
    calibrate_residual,
    predict_residual,
    calibrate_rescaled,
    predict_rescaled,
    calibrate_cqr,
    predict_cqr,
)

__all__ = [
    "make_coarse_time_grid",
    "hazard_grid_for_one_x",
    "hazard_grid_ensemble",
    "compute_pit",
    "compute_quantile",
    "precompute_distributions",
    "compute_optimal_z_star",
    "calibrate_cpi",
    "predict_cpi",
    "calibrate_dcp",
    "predict_dcp",
    "calibrate_residual",
    "predict_residual",
    "calibrate_rescaled",
    "predict_rescaled",
    "calibrate_cqr",
    "predict_cqr",
]
