#!/usr/bin/env python3
"""
Plotting script for CPI paper results.

Generates coverage/width plots for simulation and conditional evaluation plots for real data.
"""

import argparse
import os
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


def plot_simulation_results(input_dir, output_dir):
    """
    Plot simulation results: width and coverage vs x1 for each setup.

    Args:
        input_dir: Directory containing simulation_results_*.csv files
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    csv_files = list(Path(input_dir).glob("simulation_results_*.csv"))
    if not csv_files:
        print(f"No simulation results found in {input_dir}")
        return

    print(f"Found {len(csv_files)} simulation result files")

    for csv_file in csv_files:
        print(f"\nProcessing {csv_file.name}")
        df = pd.read_csv(csv_file)

        # Extract setup from filename
        setup = csv_file.stem.replace("simulation_results_", "")

        # Sort by x1
        df = df.sort_values("x1")
        x1 = df["x1"].values

        # Width plot
        fig, ax = plt.subplots(figsize=(12, 7))
        markers = ["o", "s", "D", "^", "x", "v", "P", "*"]
        width_cols = [col for col in df.columns if col.startswith("width_")]

        for i, col in enumerate(width_cols):
            method = col.replace("width_", "")
            ax.plot(x1, df[col], marker=markers[i % len(markers)],
                   linestyle="-", label=method, markersize=4, alpha=0.8)

        ax.set_xlabel("x1", fontsize=12)
        ax.set_ylabel("Mean Prediction Interval Width", fontsize=12)
        ax.set_title(f"Setup {setup}: Interval Width vs. x1", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = Path(output_dir) / f"plot_width_setup_{setup}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)

        # Coverage plot
        fig, ax = plt.subplots(figsize=(12, 7))
        cov_cols = [col for col in df.columns if col.startswith("coverage_")]

        for i, col in enumerate(cov_cols):
            method = col.replace("coverage_", "")
            ax.plot(x1, df[col], marker=markers[i % len(markers)],
                   linestyle="-", label=method, markersize=4, alpha=0.8)

        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=2, label="Target (0.90)")
        ax.set_xlabel("x1", fontsize=12)
        ax.set_ylabel("Coverage", fontsize=12)
        ax.set_title(f"Setup {setup}: Coverage vs. x1", fontsize=14)
        ax.set_ylim([-0.1, 1.1])
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = Path(output_dir) / f"plot_coverage_setup_{setup}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close(fig)


def plot_conditional_results(input_dir, output_dir):
    """
    Plot conditional results: coverage vs width scatter for PC1 groups.

    Args:
        input_dir: Directory containing summary_realdata.csv files
        output_dir: Output directory for plots
    """
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # Find all summary CSV files
    summary_files = list(Path(input_dir).rglob("summary_realdata.csv"))
    if not summary_files:
        print(f"No summary files found in {input_dir}")
        return

    print(f"Found {len(summary_files)} summary files")

    # Combine all summaries
    all_dfs = []
    for csv_file in summary_files:
        df = pd.read_csv(csv_file)
        dataset = csv_file.parent.name
        df["dataset"] = dataset
        all_dfs.append(df)

    if not all_dfs:
        return

    combined = pd.concat(all_dfs, ignore_index=True)

    # Group evaluation plots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Coverage by group
    ax = axes[0]
    groups = combined["group"].unique()
    methods = combined["method"].unique()

    x = np.arange(len(groups))
    width = 0.8 / max(1, len(methods))

    for i, method in enumerate(sorted(methods)):
        sub = combined[combined["method"] == method]
        data_dict = {g: None for g in groups}
        for _, row in sub.iterrows():
            data_dict[row["group"]] = row["coverage_mean"]

        y = [data_dict.get(g, np.nan) for g in sorted(groups)]
        pos = x + (i - len(methods) / 2) * width
        ax.bar(pos, y, width=width, label=method, alpha=0.8)

    ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1.5, label="Target")
    ax.set_xlabel("PC1 Group", fontsize=11)
    ax.set_ylabel("Mean Coverage", fontsize=11)
    ax.set_title("Coverage by PC1 Group", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(groups), rotation=45, ha="right")
    ax.legend(loc="best", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim([0.7, 1.0])

    # Width by group
    ax = axes[1]
    for i, method in enumerate(sorted(methods)):
        sub = combined[combined["method"] == method]
        data_dict = {g: None for g in groups}
        for _, row in sub.iterrows():
            data_dict[row["group"]] = row["width_norm_mean"]

        y = [data_dict.get(g, np.nan) for g in sorted(groups)]
        pos = x + (i - len(methods) / 2) * width
        ax.bar(pos, y, width=width, label=method, alpha=0.8)

    ax.set_xlabel("PC1 Group", fontsize=11)
    ax.set_ylabel("Mean Normalized Width", fontsize=11)
    ax.set_title("Normalized Prediction Interval Width by PC1 Group", fontsize=13)
    ax.set_xticks(x)
    ax.set_xticklabels(sorted(groups), rotation=45, ha="right")
    ax.legend(loc="best", ncol=2)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out_path = Path(output_dir) / "plot_conditional_results.png"
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path}")
    plt.close(fig)

    # Scatter: coverage vs width for each method
    marginal = combined[combined["group"] == "Marginal"].copy()

    if len(marginal) > 0:
        fig, ax = plt.subplots(figsize=(10, 7))

        marker_map = {
            "Residual": "^",
            "Rescaled": "v",
            "CQR": "o",
            "CPI": "s",
            "DCP": "P",
        }

        for method in sorted(marginal["method"].unique()):
            sub = marginal[marginal["method"] == method]
            marker = marker_map.get(method, "o")

            for _, row in sub.iterrows():
                ax.scatter(row["width_norm_mean"], row["coverage_mean"],
                          marker=marker, s=150, alpha=0.7, label=method if _ == 0 else "")

        ax.axhline(y=0.9, color="red", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.set_xlabel("Mean Normalized Width", fontsize=12)
        ax.set_ylabel("Coverage", fontsize=12)
        ax.set_title("Coverage vs. Width (Marginal Results)", fontsize=14)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        out_path = Path(output_dir) / "plot_cov_vs_width.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved: {out_path}")
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Plot CPI paper results")
    parser.add_argument(
        "--type",
        type=str,
        choices=["simulation", "realdata", "conditional"],
        required=True,
        help="Plot type",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Input directory",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="figures",
        help="Output directory",
    )

    args = parser.parse_args()

    if args.type == "simulation":
        plot_simulation_results(args.input_dir, args.output_dir)
    elif args.type == "conditional":
        plot_conditional_results(args.input_dir, args.output_dir)
    else:
        print(f"Unknown plot type: {args.type}")


if __name__ == "__main__":
    main()
