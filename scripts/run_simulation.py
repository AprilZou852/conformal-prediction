#!/usr/bin/env python3
"""
Simulation experiment runner for CPI (Conformalized Percentile Interval).

This script runs the main simulation loop over Monte Carlo replicates,
trains baseline and hazard models, and generates results for each setup.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import yaml
import matplotlib.pyplot as plt

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.dgp import generate_data, generate_y_for_x
from src.models.hazard_net import train_hazard_ensemble, expand_long_table
from src.models.baseline_nets import (
    train_simple_nn,
    train_mean_variance_nn,
    train_quantile_nn,
)
from src.conformal.hazard_inference import (
    predict_cdf,
    find_quantile,
    compute_z_star_gridsearch,
)
from src.utils.helpers import set_all_seeds


def load_config(config_path):
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def setup_device(device_arg):
    """Determine device (cuda/cpu)."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device_arg


def run_simulation(
    setup,
    config,
    output_dir=".",
):
    """
    Run simulation for a given setup.

    Args:
        setup: Setup ID (e.g., "0a", "0c1")
        config: Configuration dict
        output_dir: Output directory for results
    """
    alpha = config["alpha"]
    n_train = config["data"]["n_train"]
    n_cal = config["data"]["n_cal"]
    n_test = config["data"]["n_test"]
    n_rep = config["data"]["n_rep"]
    base_seed = config["seed"]
    device = setup_device(config["device"])

    baseline_cfg = config["baseline_config"]
    hazard_cfg = config["hazard"]
    z_grid_cfg = config["z_grid"]

    print("=" * 70)
    print(f"STARTING SIMULATION FOR SETUP: {setup}")
    print(f"Device: {device}")
    print(f"Baseline config: hidden={baseline_cfg['hidden']}, epochs={baseline_cfg['epochs']}, "
          f"patience={baseline_cfg['patience']}, batch={baseline_cfg['batch']}, lr={baseline_cfg['lr']}")
    print("=" * 70)

    set_all_seeds(base_seed)

    # Step 1: Train models on training data
    print(f"\n[{setup}] --- Step 1: Training Models ---")
    X_train, y_train = generate_data(n_train, setup=setup, seed=base_seed, is_train=True)

    # Train SimpleNN for Residual
    mdl_resid = train_simple_nn(
        X_train, y_train,
        epochs=baseline_cfg["epochs"],
        batch=baseline_cfg["batch"],
        val_frac=baseline_cfg["val_frac"],
        patience=baseline_cfg["patience"],
        seed_split=base_seed,
        hidden=baseline_cfg["hidden"],
        lr=baseline_cfg["lr"],
        weight_decay=baseline_cfg.get("weight_decay", 0.0),
    )

    # Train MeanVarianceNN for Rescaled
    mdl_rescaled = train_mean_variance_nn(
        X_train, y_train,
        epochs=baseline_cfg["epochs"],
        batch=baseline_cfg["batch"],
        val_frac=baseline_cfg["val_frac"],
        patience=baseline_cfg["patience"],
        seed_split=base_seed,
        hidden=baseline_cfg["hidden"],
        lr=baseline_cfg["lr"],
        weight_decay=baseline_cfg.get("weight_decay", 0.0),
        clamp_logvar=baseline_cfg.get("clamp_logvar", False),
    )

    # Train QuantileNN for CQR
    mdl_cqr = train_quantile_nn(
        X_train, y_train,
        quantiles=[alpha / 2, 1 - alpha / 2],
        epochs=baseline_cfg["epochs"],
        batch=baseline_cfg["batch"],
        val_frac=baseline_cfg["val_frac"],
        patience=baseline_cfg["patience"],
        seed_split=base_seed,
        hidden=baseline_cfg["cqr_hidden"],
        lr=baseline_cfg["lr"],
        weight_decay=baseline_cfg.get("weight_decay", 0.0),
    )

    # Train Hazard model
    print(f"[{setup}] --- Training Hazard Model ---")
    df_long, uniq_y_train = expand_long_table(X_train, y_train)
    hz_models = train_hazard_ensemble(
        df_long, uniq_y_train,
        epochs=hazard_cfg["epochs"],
        hidden=hazard_cfg["hidden"],
        patience=hazard_cfg["patience"],
        batch=hazard_cfg["batch"],
        lr=hazard_cfg["lr"],
        seed=base_seed,
    )

    # Fixed test set
    X_test_fixed, _ = generate_data(n_test, setup=setup, seed=base_seed + 1)
    y_min, y_max = np.quantile(y_train, [0.0, 1.0])
    y_range_extension = (y_max - y_min) * 10
    y_min -= y_range_extension
    y_max += y_range_extension

    # Precompute z* on test set
    print(f"[{setup}] --- Computing z* (shortest width grid search) ---")
    z_test_optimal = compute_z_star_gridsearch(
        hz_models, uniq_y_train, X_test_fixed,
        y_min, y_max, alpha=alpha,
        grid_size=z_grid_cfg["size"],
        eps_u=z_grid_cfg["eps_u"],
    )

    # Step 2: Monte Carlo loop
    print(f"\n[{setup}] --- Step 2: Monte Carlo ({n_rep} Reps) ---")

    method_names = [
        "CPI",
        "DCP",
        "Residual",
        "Rescaled",
        "CQR",
    ]

    results = {
        name: {
            "C": np.zeros((n_test, n_rep), dtype=np.float32),
            "W": np.zeros((n_test, n_rep), dtype=np.float32),
        }
        for name in method_names
    }

    X_test_fixed_t = torch.tensor(X_test_fixed, dtype=torch.float32, device=device)

    for k in tqdm(range(n_rep), desc=f"Sim {setup}"):
        rep_seed = base_seed * 1000 + k
        X_cal, y_cal = generate_data(n_cal, setup=setup, seed=rep_seed)
        rng_test = np.random.default_rng(rep_seed + 1)
        y_test = generate_y_for_x(X_test_fixed, setup=setup, rng=rng_test)

        q_level = (1 - alpha) * (1 + 1 / n_cal)
        q_level = min(float(q_level), 0.999999)

        # Compute PIT values for calibration
        pit_cal = np.array([
            predict_cdf(hz_models, uniq_y_train, X_cal[i], t_end=y_cal[i])
            for i in range(n_cal)
        ], dtype=np.float32)

        # CPI: quantile-based bounds on PIT
        qvec_lo = np.clip(z_test_optimal.astype("float64"), z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])
        qvec_hi = np.clip((z_test_optimal + 1.0 - alpha).astype("float64"), z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])

        u_lo_x = np.quantile(pit_cal.astype("float64"), qvec_lo).astype("float32")
        u_hi_x = np.quantile(pit_cal.astype("float64"), qvec_hi).astype("float32")

        u_lo_x = np.clip(u_lo_x, z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])
        u_hi_x = np.clip(u_hi_x, z_grid_cfg["eps_u"], 1.0 - z_grid_cfg["eps_u"])

        q_lo_cpi = np.array([
            find_quantile(hz_models, uniq_y_train, X_test_fixed[i], float(u_lo_x[i]), y_min, y_max)
            for i in range(n_test)
        ], dtype=np.float32)
        q_hi_cpi = np.array([
            find_quantile(hz_models, uniq_y_train, X_test_fixed[i], float(u_hi_x[i]), y_min, y_max)
            for i in range(n_test)
        ], dtype=np.float32)

        results["CPI"]["W"][:, k] = q_hi_cpi - q_lo_cpi
        results["CPI"]["C"][:, k] = ((y_test >= q_lo_cpi) & (y_test <= q_hi_cpi)).astype(np.float32)

        # DCP: center-based bounds
        center_cal = z_test_optimal + (1.0 - alpha) / 2.0
        scores_dcp = np.abs(pit_cal.astype(np.float64) - center_cal.astype(np.float64))
        q_hat_dcp = float(np.quantile(scores_dcp, q_level))

        center_test = z_test_optimal.astype(np.float64) + (1.0 - alpha) / 2.0
        u_lo_dcp = np.clip(center_test - q_hat_dcp, 0.0, 1.0)
        u_hi_dcp = np.clip(center_test + q_hat_dcp, 0.0, 1.0)

        q_lo_dcp = np.array([
            find_quantile(hz_models, uniq_y_train, X_test_fixed[i], float(u_lo_dcp[i]), y_min, y_max)
            for i in range(n_test)
        ], dtype=np.float32)
        q_hi_dcp = np.array([
            find_quantile(hz_models, uniq_y_train, X_test_fixed[i], float(u_hi_dcp[i]), y_min, y_max)
            for i in range(n_test)
        ], dtype=np.float32)

        results["DCP"]["W"][:, k] = q_hi_dcp - q_lo_dcp
        results["DCP"]["C"][:, k] = ((y_test >= q_lo_dcp) & (y_test <= q_hi_dcp)).astype(np.float32)

        # Residual
        with torch.no_grad():
            y_hat_cal = mdl_resid(torch.tensor(X_cal, dtype=torch.float32, device=device)).squeeze().cpu().numpy()
            y_hat_test = mdl_resid(X_test_fixed_t).squeeze().cpu().numpy()

        scores_resid = np.abs(y_cal - y_hat_cal)
        q_resid = np.quantile(scores_resid, q_level)
        lo_r, hi_r = y_hat_test - q_resid, y_hat_test + q_resid

        results["Residual"]["W"][:, k] = (hi_r - lo_r).astype(np.float32)
        results["Residual"]["C"][:, k] = ((y_test >= lo_r) & (y_test <= hi_r)).astype(np.float32)

        # Rescaled
        with torch.no_grad():
            mean_cal, log_var_cal = mdl_rescaled(torch.tensor(X_cal, dtype=torch.float32, device=device))
            if baseline_cfg.get("clamp_logvar"):
                log_var_cal = torch.clamp(log_var_cal, min=-1.0, max=1.0)
            sigma_cal = torch.exp(0.5 * log_var_cal).squeeze().cpu().numpy()
            mean_cal_np = mean_cal.squeeze().cpu().numpy()

        sigma_cal = np.maximum(sigma_cal, 1e-8)
        scores_rescaled = np.abs(y_cal - mean_cal_np) / sigma_cal
        q_rescaled = np.quantile(scores_rescaled, q_level)

        with torch.no_grad():
            mean_test, log_var_test = mdl_rescaled(X_test_fixed_t)
            if baseline_cfg.get("clamp_logvar"):
                log_var_test = torch.clamp(log_var_test, min=-1.0, max=1.0)
            sigma_test = torch.exp(0.5 * log_var_test).squeeze().cpu().numpy()
            mean_test_np = mean_test.squeeze().cpu().numpy()

        sigma_test = np.maximum(sigma_test, 1e-8)
        lo_s = mean_test_np - q_rescaled * sigma_test
        hi_s = mean_test_np + q_rescaled * sigma_test

        results["Rescaled"]["W"][:, k] = (hi_s - lo_s).astype(np.float32)
        results["Rescaled"]["C"][:, k] = ((y_test >= lo_s) & (y_test <= hi_s)).astype(np.float32)

        # CQR
        with torch.no_grad():
            q_preds_cal = mdl_cqr(torch.tensor(X_cal, dtype=torch.float32, device=device)).cpu().numpy()

        q_lo_cal, q_hi_cal = q_preds_cal[:, 0], q_preds_cal[:, 1]
        scores_cqr = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
        q_cqr = np.quantile(scores_cqr, q_level)

        with torch.no_grad():
            q_preds_test = mdl_cqr(X_test_fixed_t).cpu().numpy()

        q_lo_test, q_hi_test = q_preds_test[:, 0], q_preds_test[:, 1]
        lo_c, hi_c = q_lo_test - q_cqr, q_hi_test + q_cqr

        results["CQR"]["W"][:, k] = (hi_c - lo_c).astype(np.float32)
        results["CQR"]["C"][:, k] = ((y_test >= lo_c) & (y_test <= hi_c)).astype(np.float32)

    # Step 3: Save results
    print(f"\n[{setup}] --- Step 3: Saving Results ---")
    results_df = pd.DataFrame({"x1": X_test_fixed[:, 0]})

    print(f"\nFinal Aggregated Metrics (Mean over all test points):")
    for name, data in results.items():
        mean_cov = data["C"].mean()
        mean_wid = data["W"].mean()
        print(f"  {name:20s} | Coverage: {mean_cov:.4f} | Width: {mean_wid:.4f}")
        results_df[f"coverage_{name}"] = data["C"].mean(axis=1)
        results_df[f"width_{name}"] = data["W"].mean(axis=1)

    out_csv = os.path.join(output_dir, f"simulation_results_{setup}.csv")
    results_df.to_csv(out_csv, index=False)
    print(f"[{setup}] Results saved to {out_csv}")

    # Plotting
    plt.style.use("seaborn-v0_8-whitegrid")
    sorted_indices = np.argsort(results_df["x1"].values)
    x1_sorted = results_df["x1"].iloc[sorted_indices].values

    # Width plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    markers = ["o", "s", "D", "^", "x", "v"]
    for i, name in enumerate(method_names):
        col = f"width_{name}"
        if col in results_df.columns:
            width_sorted = results_df[col].iloc[sorted_indices].values
            ax.plot(x1_sorted, width_sorted, marker=markers[i % len(markers)],
                   linestyle="-", label=name, markersize=4, alpha=0.9)

    ax.set_title(f"Setup {setup}: Mean Interval Width vs. x1")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_width_{setup}.png"), dpi=300)
    plt.close(fig)

    # Coverage plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for i, name in enumerate(method_names):
        col = f"coverage_{name}"
        if col in results_df.columns:
            coverage_sorted = results_df[col].iloc[sorted_indices].values
            ax.plot(x1_sorted, coverage_sorted, marker=markers[i % len(markers)],
                   linestyle="-", label=name, markersize=4, alpha=0.9)

    ax.axhline(y=1 - alpha, color="r", linestyle="--", label=f"Target={1-alpha:.2f}")
    ax.set_title(f"Setup {setup}: Coverage vs. x1")
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"plot_coverage_{setup}.png"), dpi=300)
    plt.close(fig)

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Run simulation experiments for CPI paper"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/simulation.yaml",
        help="Path to simulation config YAML",
    )
    parser.add_argument(
        "--setups",
        nargs="+",
        type=str,
        default=None,
        help="Setups to run (e.g., 0a 0c1)",
    )
    parser.add_argument(
        "--n-train",
        type=int,
        default=None,
        help="Override n_train in config",
    )
    parser.add_argument(
        "--n-cal",
        type=int,
        default=None,
        help="Override n_cal in config",
    )
    parser.add_argument(
        "--n-test",
        type=int,
        default=None,
        help="Override n_test in config",
    )
    parser.add_argument(
        "--n-rep",
        type=int,
        default=None,
        help="Override n_rep in config",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=None,
        help="Override alpha in config",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Override seed in config",
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
        default="results/simulation",
        help="Output directory",
    )
    parser.add_argument(
        "--baseline-strength",
        type=str,
        default=None,
        help="Baseline strength preset (light/medium/heavy)",
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Override with CLI args
    if args.setups:
        config["setups"] = args.setups
    if args.n_train:
        config["data"]["n_train"] = args.n_train
    if args.n_cal:
        config["data"]["n_cal"] = args.n_cal
    if args.n_test:
        config["data"]["n_test"] = args.n_test
    if args.n_rep:
        config["data"]["n_rep"] = args.n_rep
    if args.alpha:
        config["alpha"] = args.alpha
    if args.seed is not None:
        config["seed"] = args.seed
    if args.device:
        config["device"] = args.device
    if args.baseline_strength:
        config["baseline_strength"] = args.baseline_strength

    # Create output dir
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Run simulations
    print(f"Device: {setup_device(config['device'])}")
    print(f"Running experiments for Setups: {config['setups']}")

    for setup in config["setups"]:
        try:
            run_simulation(setup, config, output_dir=args.output_dir)
        except Exception as e:
            print(f"Error running setup {setup}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\nAll requested simulations completed.")


if __name__ == "__main__":
    main()
