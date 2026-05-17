#!/usr/bin/env python3
"""
Real data experiment runner for CPI (Conformalized Percentile Interval).

This script runs the full pipeline on real datasets: feature selection,
model training, calibration, and evaluation with PC1 grouping.
"""

import argparse
import os
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import load_real_data
from src.data.feature_selection import select_features
from src.models.hazard_net import train_hazard_ensemble, expand_long_table
from src.models.baseline_nets import (
    train_mean_variance_nn,
    train_quantile_nn,
)
from src.models.residual import train_residual_regressor
from src.conformal.hazard_inference import (
    predict_cdf,
    find_quantile,
    compute_z_star_gridsearch,
    _precompute_hazard_distributions,
    _stable_pit,
    _compute_shortest_z_star_from_cache,
    _hazard_quantile_from_cache,
    _make_coarse_time_grid,
)
from src.utils.helpers import set_all_seeds
from src.utils.metrics import compute_l1_ert
from src.utils.preprocessing import (
    fit_standardizer_xy,
    transform_X,
    transform_y,
    inverse_y,
)


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_device(device_arg):
    """Determine device (cuda/cpu)."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def run_one_split(X_all, y_all, feature_names, split_seed, config, output_dir):
    """
    Run one train/cal/test split for real data.

    Args:
        X_all, y_all: Full dataset
        feature_names: Feature names
        split_seed: Random seed for this split
        config: Configuration dict
        output_dir: Output directory

    Returns:
        List of result dicts (one per group and method)
    """
    set_all_seeds(split_seed)

    alpha = config["alpha"]
    test_frac = config["data"]["test_frac"]
    cal_frac = config["data"]["cal_frac"]
    hazard_cfg = config["hazard"]
    baseline_cfg = config["baselines"]
    z_grid_cfg = config["z_grid"]
    pc1_n_groups = config["pc1_groups"]
    device = setup_device(config["device"])

    # Train/Cal/Test split
    idx_all = np.arange(len(y_all), dtype=int)
    idx_rest, idx_test = train_test_split(
        idx_all, test_size=test_frac, random_state=split_seed
    )

    n = len(idx_all)
    test_frac_actual = len(idx_test) / max(1, n)
    rest_frac = 1.0 - test_frac_actual
    if rest_frac <= 0:
        raise ValueError(f"Empty rest set (test size={len(idx_test)} equals n={n}).")

    cal_frac_within_rest = cal_frac / rest_frac
    cal_frac_within_rest = float(np.clip(cal_frac_within_rest, 0.0, 0.99))

    idx_train, idx_cal = train_test_split(
        idx_rest,
        test_size=cal_frac_within_rest,
        random_state=split_seed,
    )

    X_train = X_all[idx_train]
    y_train = y_all[idx_train]
    X_cal = X_all[idx_cal]
    y_cal = y_all[idx_cal]
    X_test = X_all[idx_test]
    y_test = y_all[idx_test]

    print(f"  Train: {len(X_train)}, Cal: {len(X_cal)}, Test: {len(X_test)}")

    # Feature selection
    fs_cfg = config["feature_selection"]
    sel_idx, sel_names = select_features(
        X_train, y_train, feature_names,
        random_state=split_seed,
        method=fs_cfg.get("method", "stability_lgbm"),
        bootstraps=fs_cfg.get("bootstraps", 50),
        subsample=fs_cfg.get("subsample", 0.7),
        freq_thr=fs_cfg.get("freq_thr", 0.6),
        max_features=fs_cfg.get("max_features"),
    )

    X_train_sel = X_train[:, sel_idx]
    X_cal_sel = X_cal[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]

    print(f"  Features selected: {len(sel_idx)}/{len(feature_names)}")

    # Residual model (for meta-features)
    mdl_residual = None
    residual_backend = None
    try:
        mdl_residual, residual_backend = train_residual_regressor(
            X_train_sel, y_train, random_state=split_seed, weak_level="medium"
        )
        print(f"  [Residual] backend={residual_backend}")
    except Exception as e:
        print(f"  [Residual] failed: {e}")

    # Hazard model training with ensemble
    print(f"  [HazardNN] Training ensemble K={hazard_cfg['ensemble_k']} ...")
    df_long, uniq_y_train = expand_long_table(X_train_sel, y_train)

    hz_models = train_hazard_ensemble(
        df_long, uniq_y_train,
        epochs=hazard_cfg["epochs"],
        hidden=hazard_cfg["hidden"],
        patience=hazard_cfg["patience"],
        batch=hazard_cfg["batch"],
        lr=hazard_cfg["lr"],
        ensemble_k=hazard_cfg["ensemble_k"],
        ensemble_topk=hazard_cfg["ensemble_topk"],
        seed=split_seed,
    )

    # Precompute time grid
    t_edges = _make_coarse_time_grid(
        uniq_y_train,
        hazard_cfg.get("max_grid", 1024),
        mode=hazard_cfg.get("timegrid_mode", "coarse"),
    )

    # Precompute hazard distributions
    cal_cumLs, cal_lams, cal_Fgrids = _precompute_hazard_distributions(
        hz_models, t_edges, X_cal_sel
    )
    test_cumLs, test_lams, test_Fgrids = _precompute_hazard_distributions(
        hz_models, t_edges, X_test_sel
    )

    # Compute PIT values
    pit_cal = np.array([
        _stable_pit(cal_cumLs[i], t_edges, cal_lams[i], float(y_cal[i]))
        for i in range(len(X_cal_sel))
    ], dtype="float32")

    # Compute z*(x) for test and cal
    z_star_test = np.array([
        _compute_shortest_z_star_from_cache(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i],
            alpha=alpha, grid_size=z_grid_cfg["size"], eps_u=z_grid_cfg["eps_u"]
        )
        for i in range(len(X_test_sel))
    ], dtype="float32")

    z_star_cal = np.array([
        _compute_shortest_z_star_from_cache(
            cal_Fgrids[i], cal_cumLs[i], t_edges, cal_lams[i],
            alpha=alpha, grid_size=z_grid_cfg["size"], eps_u=z_grid_cfg["eps_u"]
        )
        for i in range(len(X_cal_sel))
    ], dtype="float32")

    # CPI
    qvec_lo = np.clip(z_star_test.astype("float64"), z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])
    qvec_hi = np.clip((z_star_test + 1.0 - alpha).astype("float64"), z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])

    u_lo_x = np.quantile(pit_cal.astype("float64"), qvec_lo).astype("float32")
    u_hi_x = np.quantile(pit_cal.astype("float64"), qvec_hi).astype("float32")

    u_lo_x = np.clip(u_lo_x, z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])
    u_hi_x = np.clip(u_hi_x, z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])

    lo_cpi = np.array([
        _hazard_quantile_from_cache(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_lo_x[i]
        )
        for i in range(len(X_test_sel))
    ], dtype="float32")

    hi_cpi = np.array([
        _hazard_quantile_from_cache(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_hi_x[i]
        )
        for i in range(len(X_test_sel))
    ], dtype="float32")

    # DCP
    center_cal = z_star_cal.astype(np.float64) + (1.0 - alpha) / 2.0
    scores_dcp = np.abs(pit_cal.astype(np.float64) - center_cal)
    q_level = (1 - alpha) * (1 + 1 / len(y_cal))
    q_hat_dcp = float(np.quantile(scores_dcp, q_level))

    center_test = z_star_test.astype(np.float64) + (1.0 - alpha) / 2.0
    u_lo_dcp = np.clip(center_test - q_hat_dcp, 0.0, 1.0)
    u_hi_dcp = np.clip(center_test + q_hat_dcp, 0.0, 1.0)

    lo_dcp = np.array([
        _hazard_quantile_from_cache(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], float(u_lo_dcp[i])
        )
        for i in range(len(X_test_sel))
    ], dtype="float32")

    hi_dcp = np.array([
        _hazard_quantile_from_cache(
            test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], float(u_hi_dcp[i])
        )
        for i in range(len(X_test_sel))
    ], dtype="float32")

    # Rescaled and CQR use standardized data
    x_mu, x_sd, y_mu, y_sd = fit_standardizer_xy(X_train_sel, y_train)

    X_train_std = transform_X(X_train_sel, x_mu, x_sd)
    X_cal_std = transform_X(X_cal_sel, x_mu, x_sd)
    X_test_std = transform_X(X_test_sel, x_mu, x_sd)

    y_train_std = transform_y(y_train, y_mu, y_sd)
    y_cal_std = transform_y(y_cal, y_mu, y_sd)

    # Rescaled
    mdl_rescaled = train_mean_variance_nn(
        X_train_std, y_train_std,
        epochs=baseline_cfg["epochs"],
        hidden=baseline_cfg["hidden"],
        patience=baseline_cfg["patience"],
        batch=baseline_cfg["batch"],
        seed_split=split_seed,
    )

    with torch.no_grad():
        mean_cal_res, sigma_cal_res = mdl_rescaled(
            torch.tensor(X_cal_std, dtype=torch.float32, device=device)
        )
        mean_cal_res = mean_cal_res.cpu().squeeze().numpy().astype("float32")
        sigma_cal_res = sigma_cal_res.cpu().squeeze().numpy().astype("float32")

    sigma_cal_res = np.clip(
        sigma_cal_res,
        baseline_cfg.get("rescaled_sigma_min", 1e-8),
        baseline_cfg.get("rescaled_sigma_max", 10.0),
    )
    scores_rescaled = np.abs(y_cal_std - mean_cal_res) / (sigma_cal_res + baseline_cfg.get("std_eps", 1e-8))
    q_rescaled = np.quantile(scores_rescaled, q_level)

    with torch.no_grad():
        mean_test_res, sigma_test_res = mdl_rescaled(
            torch.tensor(X_test_std, dtype=torch.float32, device=device)
        )
        mean_test_res = mean_test_res.cpu().squeeze().numpy().astype("float32")
        sigma_test_res = sigma_test_res.cpu().squeeze().numpy().astype("float32")

    sigma_test_res = np.clip(
        sigma_test_res,
        baseline_cfg.get("rescaled_sigma_min", 1e-8),
        baseline_cfg.get("rescaled_sigma_max", 10.0),
    )

    lo_rescaled_std = mean_test_res - q_rescaled * sigma_test_res
    hi_rescaled_std = mean_test_res + q_rescaled * sigma_test_res

    lo_rescaled = inverse_y(lo_rescaled_std, y_mu, y_sd)
    hi_rescaled = inverse_y(hi_rescaled_std, y_mu, y_sd)

    # CQR
    mdl_cqr = train_quantile_nn(
        X_train_std, y_train_std,
        quantiles=[alpha / 2, 1 - alpha / 2],
        epochs=baseline_cfg["epochs"],
        hidden=baseline_cfg["hidden"],
        patience=baseline_cfg["patience"],
        batch=baseline_cfg["batch"],
        seed_split=split_seed,
    )

    with torch.no_grad():
        q_cal_pred_std = mdl_cqr(
            torch.tensor(X_cal_std, dtype=torch.float32, device=device)
        ).cpu().numpy()

    q_lo_cal_std = q_cal_pred_std[:, 0].astype("float32")
    q_hi_cal_std = q_cal_pred_std[:, 1].astype("float32")
    q_lo_cal_std = np.minimum(q_lo_cal_std, q_hi_cal_std)
    q_hi_cal_std = np.maximum(q_lo_cal_std, q_hi_cal_std)

    scores_cqr = np.maximum(q_lo_cal_std - y_cal_std, y_cal_std - q_hi_cal_std)
    q_cqr = np.quantile(scores_cqr, q_level)

    with torch.no_grad():
        q_test_pred_std = mdl_cqr(
            torch.tensor(X_test_std, dtype=torch.float32, device=device)
        ).cpu().numpy()

    q_lo_test_std = q_test_pred_std[:, 0].astype("float32")
    q_hi_test_std = q_test_pred_std[:, 1].astype("float32")
    q_lo_test_std = np.minimum(q_lo_test_std, q_hi_test_std)
    q_hi_test_std = np.maximum(q_lo_test_std, q_hi_test_std)

    lo_cqr_std = q_lo_test_std - q_cqr
    hi_cqr_std = q_hi_test_std + q_cqr

    lo_cqr = inverse_y(lo_cqr_std, y_mu, y_sd)
    hi_cqr = inverse_y(hi_cqr_std, y_mu, y_sd)

    # Residual
    lo_residual = None
    hi_residual = None
    if mdl_residual is not None:
        y_hat_cal_res = mdl_residual.predict(X_cal_sel)
        scores_resid = np.abs(y_cal - y_hat_cal_res)
        q_resid = np.quantile(scores_resid, q_level)
        y_hat_test_res = mdl_residual.predict(X_test_sel)
        lo_residual = (y_hat_test_res - q_resid).astype("float32")
        hi_residual = (y_hat_test_res + q_resid).astype("float32")

    # PC1 grouping
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_sel)
    pca = PCA(n_components=1, random_state=split_seed).fit(X_train_scaled)

    X_test_scaled = scaler.transform(X_test_sel)
    pc1_test = pca.transform(X_test_scaled).flatten()

    def assign_pc1_groups(pc1_vals, n_groups):
        """Assign PC1 values to quantile-based groups."""
        edges = np.quantile(pc1_vals, np.linspace(0, 1, n_groups + 1))
        # Make edges unique
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + 1e-6
        return np.digitize(pc1_vals, edges[1:-1])

    X_cal_scaled = scaler.transform(X_cal_sel)
    pc1_cal = pca.transform(X_cal_scaled).flatten()
    edges = np.quantile(pc1_cal, np.linspace(0, 1, pc1_n_groups + 1))

    # Ensure unique edges
    for i in range(1, len(edges)):
        if edges[i] <= edges[i - 1]:
            edges[i] = edges[i - 1] + 1e-6

    group_test = np.digitize(pc1_test, edges[1:-1])

    # Evaluate per group
    all_group_results = []
    test_range = float(np.max(y_test) - np.min(y_test))
    if test_range <= 0:
        test_range = 1.0

    methods = {
        "CPI": (lo_cpi, hi_cpi),
        "DCP": (lo_dcp, hi_dcp),
        "Rescaled": (lo_rescaled, hi_rescaled),
        "CQR": (lo_cqr, hi_cqr),
    }

    if lo_residual is not None:
        methods["Residual"] = (lo_residual, hi_residual)

    group_labels = [f"PC1_Group_{i+1}" for i in range(pc1_n_groups)]

    for gid in range(pc1_n_groups):
        mask = group_test == gid
        if np.sum(mask) == 0:
            continue

        y_test_g = y_test[mask]
        n_g = len(y_test_g)

        X_test_g = X_test_sel[mask]

        for method_name, (lo, hi) in methods.items():
            lo_g = lo[mask]
            hi_g = hi[mask]

            valid = np.isfinite(lo_g) & np.isfinite(hi_g)
            if valid.sum() == 0:
                continue

            yy = y_test_g[valid]
            lo_v = lo_g[valid]
            hi_v = hi_g[valid]

            cov = float(np.mean((yy >= lo_v) & (yy <= hi_v)))
            width_mean = float(np.mean(hi_v - lo_v))
            width_norm_mean = float(np.mean((hi_v - lo_v) / test_range))

            # L1-ERT conditional coverage metric
            try:
                l1_ert = compute_l1_ert(X_test_g[valid], yy, lo_v, hi_v, alpha)
            except Exception:
                l1_ert = float("nan")

            all_group_results.append({
                "seed": int(split_seed),
                "group": group_labels[gid],
                "method": method_name,
                "coverage_mean": cov,
                "width_mean": width_mean,
                "width_norm_mean": width_norm_mean,
                "l1_ert": l1_ert,
                "n_group": n_g,
                "n_valid": int(valid.sum()),
                "test_size": len(y_test),
            })

    # Also add marginal results
    for method_name, (lo, hi) in methods.items():
        valid = np.isfinite(lo) & np.isfinite(hi)
        if valid.sum() == 0:
            continue

        yy = y_test[valid]
        lo_v = lo[valid]
        hi_v = hi[valid]

        cov = float(np.mean((yy >= lo_v) & (yy <= hi_v)))
        width_mean = float(np.mean(hi_v - lo_v))
        width_norm_mean = float(np.mean((hi_v - lo_v) / test_range))

        # L1-ERT conditional coverage metric
        try:
            l1_ert = compute_l1_ert(X_test_sel[valid], yy, lo_v, hi_v, alpha)
        except Exception:
            l1_ert = float("nan")

        all_group_results.append({
            "seed": int(split_seed),
            "group": "Marginal",
            "method": method_name,
            "coverage_mean": cov,
            "width_mean": width_mean,
            "width_norm_mean": width_norm_mean,
            "l1_ert": l1_ert,
            "n_group": len(y_test),
            "n_valid": int(valid.sum()),
            "test_size": len(y_test),
        })

    return all_group_results


def main():
    parser = argparse.ArgumentParser(
        description="Run real data experiments for CPI paper"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/realdata.yaml",
        help="Path to real data config YAML",
    )
    parser.add_argument(
        "--x-path",
        type=str,
        required=True,
        help="Path to X.csv",
    )
    parser.add_argument(
        "--y-path",
        type=str,
        required=True,
        help="Path to y.csv",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=None,
        help="Override seeds",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (cuda/cpu/auto)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/realdata",
        help="Output directory",
    )
    parser.add_argument(
        "--fs-method",
        type=str,
        default=None,
        help="Feature selection method",
    )
    parser.add_argument(
        "--hz-ensemble-k",
        type=int,
        default=None,
        help="Hazard ensemble size K",
    )
    parser.add_argument(
        "--no-pcs",
        action="store_true",
        help="Disable PCS",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.seeds:
        config["seeds"] = args.seeds
    if args.alpha:
        config["alpha"] = args.alpha
    if args.device:
        config["device"] = args.device
    if args.fs_method:
        config["feature_selection"]["method"] = args.fs_method
    if args.hz_ensemble_k:
        config["hazard"]["ensemble_k"] = args.hz_ensemble_k

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load data
    print(f"Loading data from {args.x_path}, {args.y_path}")
    X_all, y_all, feature_names = load_real_data(args.x_path, args.y_path)
    print(f"Data shape: n={len(X_all)}, p={X_all.shape[1]}")

    # Run splits
    print(f"Running {len(config['seeds'])} seeds: {config['seeds']}")
    all_results = []

    for i, seed in enumerate(config["seeds"], 1):
        print(f"\n=== Split {i}/{len(config['seeds'])} (seed={seed}) ===")
        try:
            results = run_one_split(
                X_all, y_all, feature_names, seed, config, args.output_dir
            )
            all_results.extend(results)

            # Print summary for this seed
            df = pd.DataFrame(results)
            marginal = df[df["group"] == "Marginal"]
            if len(marginal) > 0:
                print("\nMarginal results for this seed:")
                cols = ["method", "coverage_mean", "width_norm_mean", "l1_ert"]
                cols = [c for c in cols if c in marginal.columns]
                print(marginal[cols].to_string(index=False, float_format="%.4f"))
        except Exception as e:
            print(f"Error in seed {seed}: {e}")
            import traceback
            traceback.print_exc()
            continue

    # Aggregate results
    if len(all_results) > 0:
        summary_df = pd.DataFrame(all_results)

        agg_dict = {
            "coverage_mean": ["mean", "std"],
            "width_mean": ["mean", "std"],
            "width_norm_mean": ["mean", "std"],
            "n_group": "mean",
            "n_valid": "mean",
            "test_size": "mean",
            "seed": "nunique",
        }
        col_names = [
            "group", "method",
            "coverage_mean", "coverage_std",
            "width_mean", "width_std",
            "width_norm_mean", "width_norm_std",
            "n_group_mean", "n_valid_mean", "test_size_mean", "n_seeds",
        ]
        if "l1_ert" in summary_df.columns:
            agg_dict["l1_ert"] = ["mean", "std"]
            col_names = col_names[:8] + ["l1_ert_mean", "l1_ert_std"] + col_names[8:]

        final_summary = (
            summary_df.groupby(["group", "method"])
            .agg(agg_dict)
            .reset_index()
        )
        final_summary.columns = col_names

        # Fill NaN std with 0
        for col in ["coverage_std", "width_std", "width_norm_std"]:
            final_summary[col] = final_summary[col].fillna(0.0)
        if "l1_ert_std" in final_summary.columns:
            final_summary["l1_ert_std"] = final_summary["l1_ert_std"].fillna(0.0)

        out_csv = os.path.join(args.output_dir, "summary_realdata.csv")
        final_summary.to_csv(out_csv, index=False)
        print(f"\nSaved summary to {out_csv}")

        print("\nFinal Summary:")
        print(final_summary.to_string(index=False, float_format="%.4f"))
    else:
        print("No results to save.")


if __name__ == "__main__":
    main()
