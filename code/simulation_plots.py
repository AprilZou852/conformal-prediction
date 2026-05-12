
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import display

# =========================
# User parameters
# =========================
CSV_PATHS = [
    r"/Users/lenovo/Desktop/coverage/simulation_results_setup_0c1_baseline_plus_nozC2C3_plus_shortestC.csv",
    # r"/path/to/another.csv",
]

FEATURE_COL = "x1"     # feature used for conditional plots
ALPHA       = 0.1      # target coverage = 1 - ALPHA
NUM_BINS    = 8        # equal-width bins on [0,1]
WARNING_COV = 0.8      # warning line threshold

# Plot saving
SAVE_FIGS = True
OUTDIR    = "figures"
DPI       = 300  # Increased DPI for publication quality

# Optional label for titles (Titles are now removed from plots, but kept for file naming/logging)
SETUP_NAME = "Setup"

# Coverage distribution thresholds
UNDER_COVERAGE_THRESHOLDS = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
OVER_COVERAGE_THRESHOLDS  = []  # e.g. [0.98, 0.99]

# If True, fill empty bins by forward/backward filling
FILL_EMPTY_BINS = True

# make grouped bar plots across feature bins (mean +/- std)
MAKE_BIN_BAR_PLOTS = True
# make box plots per bin (disabled per request)
MAKE_BIN_BOX_PLOTS = False


# =========================
# Display name mapping (FINAL)
# =========================
METHOD_DISPLAY_NAME = {
    "HazardNN": "CPI-NNCDE",
    "hazardnn_Shortest": "CPI-opt-NNCDE",
    "HazardNN_Shortest": "CPI-opt-NNCDE",  # just in case capitalization differs

    "HazardNN_UCentered_Fixed": "DCP-NNCDE",

    # --- DCP-opt-NNCDE aliases ---
    "HazardNN_UCentered_Shortest": "DCP-opt-NNCDE",
    "hazardnn_UCentered_Shortest": "DCP-opt-NNCDE",
    "HazardNN_UCentered_shortest": "DCP-opt-NNCDE",
    "hazardnn_UCentered_shortest": "DCP-opt-NNCDE",
    "HazardNN_UCentered_Opt": "DCP-opt-NNCDE",
    "hazardnn_UCentered_Opt": "DCP-opt-NNCDE",
    "HazardNN_UCentered_opt": "DCP-opt-NNCDE",
    "hazardnn_UCentered_opt": "DCP-opt-NNCDE",

    # --- likely names for the "tr" variant (add more aliases if your CSV uses another suffix) ---
    "hazardnn_Shortest_tr": "CPI-opt-tr-NNCDE",
    "HazardNN_Shortest_tr": "CPI-opt-tr-NNCDE",
    "hazardnn_Shortest_TR": "CPI-opt-tr-NNCDE",
    "HazardNN_Shortest_TR": "CPI-opt-tr-NNCDE",
    "hazardnn_Shortest_train": "CPI-opt-tr-NNCDE",
    "HazardNN_Shortest_train": "CPI-opt-tr-NNCDE",

    # "HazardNN_YCQR_Fixed": (dropped)
}

def display_name(method: str) -> str:
    """Map internal method key -> display name (legend/tables)."""
    return METHOD_DISPLAY_NAME.get(method, method)


# =========================
# Desired showing sequence (by DISPLAY name)
# Only keep the five requested methods
# =========================
DESIRED_DISPLAY_ORDER = [
    "CPI-opt-NNCDE",
    "DCP-opt-NNCDE",
    "Residual",
    "Rescaled",
    "CQR",
]


# =========================
# Helper utilities
# =========================
def find_methods(df: pd.DataFrame):
    """Find methods that have BOTH coverage_* and width_* columns."""
    cov_cols = [c for c in df.columns if c.startswith("coverage_")]
    wid_cols = [c for c in df.columns if c.startswith("width_")]
    methods_cov = [c.replace("coverage_", "", 1) for c in cov_cols]
    methods_wid = [c.replace("width_", "", 1) for c in wid_cols]
    methods = sorted(set(methods_cov).intersection(set(methods_wid)))
    methods = [m.strip() for m in methods]
    if not methods:
        raise ValueError("No methods found with both coverage_* and width_* columns.")
    return methods

def is_residual_cond_cal(method_name: str) -> bool:
    """
    Remove 'Residual (Cond. Cal.)' from presentation.
    """
    s = method_name.lower()
    return ("residual" in s) and ("cond" in s) and ("cal" in s)

def preferred_method_order(methods):
    """
    Keep only:
      - CPI-opt-NNCDE
      - DCP-opt-NNCDE
      - Residual
      - Rescaled
      - CQR

    Also:
      - Drop CPI-NNCDE
      - Drop DCP-NNCDE
      - Drop CPI-opt-tr-NNCDE
      - Drop HazardNN_YCQR_Fixed
      - Drop Residual (Cond. Cal.)
    """
    # 0) drop YCQR fixed variant
    methods = [m for m in methods if ("ycqr" not in m.lower()) and ("cqr_fixed" not in m.lower())]

    # 1) filter out "Residual (Cond. Cal.)"
    methods = [m for m in methods if not is_residual_cond_cal(m)]

    # 2) keep only desired display names
    keep_display_names = {
        "CPI-opt-NNCDE",
        "DCP-opt-NNCDE",
        "Residual",
        "Rescaled",
        "CQR",
    }
    methods = [m for m in methods if display_name(m) in keep_display_names]

    # 3) sort by desired DISPLAY order
    order_index = {name: i for i, name in enumerate(DESIRED_DISPLAY_ORDER)}

    def sort_key(m):
        dn = display_name(m)
        return (order_index.get(dn, 10**9), dn.lower())

    return sorted(methods, key=sort_key)

def parse_title_suffix(csv_path: str):
    """
    Try to parse 'setup' and 'c' from filename, if present.
    """
    fname = os.path.basename(csv_path)
    setup = None
    cval  = None
    m = re.search(r"setup_([^_]+)", fname)
    if m:
        setup = m.group(1)
    m = re.search(r"_c_([0-9.]+)", fname)
    if m:
        cval = m.group(1)

    parts = []
    if setup is not None:
        parts.append(f"Setup {setup}")
    if cval is not None:
        parts.append(f"c={cval}")
    return ("  |  " + " , ".join(parts)) if parts else ""

def summary_table(df: pd.DataFrame, prefix: str, methods):
    """Compute Mean / SD / 25% / 75% for each method's column."""
    out = pd.DataFrame()
    for m in methods:
        colname = f"{prefix}{m}"
        if colname not in df.columns:
            continue
        col = df[colname]
        s = pd.Series({
            "Mean": col.mean(),
            "Std Dev": col.std(ddof=1),
            "25% Quantile": col.quantile(0.25),
            "75% Quantile": col.quantile(0.75),
        }, name=display_name(m))
        out = pd.concat([out, s.to_frame()], axis=1)
    return out.T

def build_style_map(methods):
    """
    Build unique style per method:
    - color cycles first
    - NNCDE methods get solid lines, others get dashed lines
    - then marker changes
    """
    # Custom palette matching the user's requested premium publication look
    custom_colors = ["#EE9B00", "#AE615C", "#6D84AB", "#709E5B", "#8A8A8A", "#D4AF37", "#6C5B7B"]

    colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    if len(custom_colors) >= 1:
        colors = custom_colors

    markers = ["o", "s", "^", "D", "v", "P", "X", "*", "+", "x"]

    style_map = {}
    for i, m in enumerate(methods):
        c = colors[i % len(colors)]

        dn = display_name(m).lower()
        internal_name = m.lower()
        if "nncde" in dn or "hazardnn" in internal_name:
            ls = "-"   # solid
        else:
            ls = "--"  # dashed

        mk = markers[i % len(markers)]
        style_map[m] = {"color": c, "linestyle": ls, "marker": mk}

    return style_map


# =========================
# Binning helpers
# =========================
def make_uniform_bins_01(num_bins: int):
    """Uniform edges on [0,1]. Adds tiny epsilon to include 1.0."""
    edges = np.linspace(0.0, 1.0, num_bins + 1)
    edges[-1] += 1e-12
    labels = [f"[{edges[i]:.2f}, {edges[i+1]:.2f})" for i in range(num_bins)]
    return edges, labels

def add_bin_column(df: pd.DataFrame, feature_col: str, edges, labels):
    df = df.copy()
    df["__bin__"] = pd.cut(
        df[feature_col],
        bins=edges,
        labels=labels,
        right=False,
        include_lowest=True,
    )
    return df

def compute_binned_mean_std(df: pd.DataFrame, feature_col: str, methods, num_bins: int, fill_empty_bins=True):
    edges, labels = make_uniform_bins_01(num_bins)
    dfb = add_bin_column(df, feature_col, edges, labels)

    count_per_bin = (
        dfb.groupby("__bin__")[feature_col]
        .size()
        .reindex(labels)
        .fillna(0)
        .astype(int)
    )

    width_mean, width_std = {}, {}
    cov_mean, cov_std = {}, {}

    for m in methods:
        wcol = f"width_{m}"
        ccol = f"coverage_{m}"
        if wcol not in dfb.columns or ccol not in dfb.columns:
            continue

        w_mean = (
            dfb[["__bin__", wcol]]
            .dropna()
            .groupby("__bin__")[wcol]
            .mean()
            .reindex(labels)
            .astype(float)
        )
        w_std_ = (
            dfb[["__bin__", wcol]]
            .dropna()
            .groupby("__bin__")[wcol]
            .std(ddof=1)
            .reindex(labels)
            .astype(float)
        )

        c_mean = (
            dfb[["__bin__", ccol]]
            .dropna()
            .groupby("__bin__")[ccol]
            .mean()
            .reindex(labels)
            .astype(float)
        )
        c_std_ = (
            dfb[["__bin__", ccol]]
            .dropna()
            .groupby("__bin__")[ccol]
            .std(ddof=1)
            .reindex(labels)
            .astype(float)
        )

        if fill_empty_bins:
            w_mean = w_mean.ffill().bfill()
            c_mean = c_mean.ffill().bfill()
            w_std_ = w_std_.fillna(0.0)
            c_std_ = c_std_.fillna(0.0)

        width_mean[m] = w_mean.values
        width_std[m]  = w_std_.values
        cov_mean[m]   = c_mean.values
        cov_std[m]    = c_std_.values

    stats = {
        "width_mean": width_mean,
        "width_std":  width_std,
        "cov_mean":   cov_mean,
        "cov_std":    cov_std,
        "edges":      edges,
        "labels":     labels,
        "count_per_bin": count_per_bin,
    }
    return stats


# =========================
# Grouped bar plots across bins (mean +/- std)
# =========================
def bar_plots_across_feature_bins(
    df: pd.DataFrame,
    csv_path: str,
    feature_col: str,
    methods,
    style_map,
    alpha: float = 0.1,
    num_bins: int = 5,
    fill_empty_bins=True,
    save=True,
    outdir="figures",
    dpi=150,
    warning_cov: float = 0.80,
):
    if feature_col not in df.columns:
        raise ValueError(f"CSV must contain column '{feature_col}'.")

    stats = compute_binned_mean_std(df, feature_col, methods, num_bins, fill_empty_bins=fill_empty_bins)
    labels = stats["labels"]
    count_per_bin = stats["count_per_bin"]
    target_cov = 1.0 - alpha

    if save:
        os.makedirs(outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]

    with plt.style.context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 16,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.grid': True,
        'grid.alpha': 0.3,
        'axes.grid.axis': 'y',
        'lines.linewidth': 1.5,
    }):
        groups = labels
        x = np.arange(len(groups))
        bar_w = 0.85 / max(1, len(methods))

        # ---- Coverage bar plot (mean +/- std)
        plt.figure(figsize=(12, 6), dpi=300)

        for i, m in enumerate(methods):
            if m not in stats["cov_mean"]:
                continue
            y = stats["cov_mean"][m]
            err = stats["cov_std"][m]
            xpos = x + i * bar_w - (len(methods) - 1) * bar_w / 2

            plt.bar(
                xpos, y, width=bar_w, label=display_name(m),
                yerr=err, capsize=3,
                color=style_map[m]["color"], alpha=0.95,
                edgecolor='white', linewidth=0.5,
                error_kw=dict(lw=1.0, capsize=3, capthick=1.0)
            )

        plt.axhline(target_cov, linestyle="--", linewidth=1.5, color="#555555")
        plt.axhline(warning_cov, linestyle=":", linewidth=1.5, color="black")

        plt.ylabel("Coverage")
        plt.xlabel("")
        plt.xticks(x, [])
        plt.ylim(0.6, 1.05)

        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.05),
            ncol=min(len(methods), 4),
            frameon=False
        )

        plt.tight_layout()

        if save:
            out_cov = os.path.join(outdir, f"{base}__binned_coverage_bar__{feature_col}__bins{num_bins}.png")
            plt.savefig(out_cov, dpi=dpi, bbox_inches='tight')
            print(f"[Saved] {out_cov}")
        plt.show()

        # ---- Width bar plot (mean +/- std)
        plt.figure(figsize=(12, 6), dpi=300)
        for i, m in enumerate(methods):
            if m not in stats["width_mean"]:
                continue
            y = stats["width_mean"][m]
            err = stats["width_std"][m]
            xpos = x + i * bar_w - (len(methods) - 1) * bar_w / 2

            plt.bar(
                xpos, y, width=bar_w, label=display_name(m),
                yerr=err, capsize=3,
                color=style_map[m]["color"], alpha=0.95,
                edgecolor='white', linewidth=0.5,
                error_kw=dict(lw=1.0, capsize=3, capthick=1.0)
            )

        plt.ylabel("Mean Width")
        plt.xlabel(f"{feature_col} bins")
        clean_labels = [l.replace("[", "").replace(")", "").replace(", ", "-") for l in groups]
        plt.xticks(x, clean_labels, rotation=35, ha="right")

        plt.legend(
            loc='upper center',
            bbox_to_anchor=(0.5, -0.2),
            ncol=min(len(methods), 4),
            frameon=False
        )
        plt.tight_layout()

        if save:
            out_wid = os.path.join(outdir, f"{base}__binned_width_bar__{feature_col}__bins{num_bins}.png")
            plt.savefig(out_wid, dpi=dpi, bbox_inches='tight')
            print(f"[Saved] {out_wid}")
        plt.show()

    print("\n[Bin counts]")
    display(pd.DataFrame({"bin": labels, "count": count_per_bin.values}))


# =========================
# Box plots per bin
# (kept for completeness; disabled by default; FutureWarning fixed)
# =========================
def box_plots_per_bin(
    df: pd.DataFrame,
    csv_path: str,
    feature_col: str,
    methods,
    style_map,
    num_bins: int = 8,
    save=True,
    outdir="figures",
    dpi=150
):
    """
    Generate grouped boxplots for 'Coverage' in each bin.
    FutureWarning fix: use hue=x with legend disabled.
    """
    if save:
        os.makedirs(outdir, exist_ok=True)

    base = os.path.splitext(os.path.basename(csv_path))[0]

    edges, labels = make_uniform_bins_01(num_bins)
    dfb = add_bin_column(df, feature_col, edges, labels)

    keep_cols = ["__bin__"] + [f"coverage_{m}" for m in methods if f"coverage_{m}" in dfb.columns]
    subset = dfb[keep_cols].copy()

    unique_bins = subset["__bin__"].cat.categories

    plt.rcParams['axes.grid'] = True
    plt.rcParams['grid.alpha'] = 0.3

    for i, bin_label in enumerate(unique_bins):
        bin_data = subset[subset["__bin__"] == bin_label]
        if bin_data.empty:
            continue

        melting_cols = [f"coverage_{m}" for m in methods if f"coverage_{m}" in subset.columns]
        if not melting_cols:
            continue

        melted = bin_data.melt(
            id_vars=["__bin__"],
            value_vars=melting_cols,
            var_name="Method",
            value_name="Coverage"
        )
        melted["Method"] = melted["Method"].str.replace("coverage_", "", regex=False)
        melted["DisplayMethod"] = melted["Method"].apply(display_name)

        display_order = [display_name(m) for m in methods]
        melted["DisplayMethod"] = pd.Categorical(melted["DisplayMethod"], categories=display_order, ordered=True)

        plt.figure(figsize=(10, 6), dpi=dpi)

        palette = {display_name(m): style_map[m]["color"] for m in methods}

        plt.axhline(1.0 - ALPHA, linestyle="--", linewidth=1.5, color="red")

        sns.boxplot(
            data=melted,
            x="DisplayMethod",
            y="Coverage",
            hue="DisplayMethod",
            palette=palette,
            showfliers=False,
            linewidth=1.2,
            width=0.6,
            dodge=False,
            legend=False
        )

        plt.ylabel("Coverage")
        plt.xlabel("Method")
        plt.ylim(-0.05, 1.05)
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()

        if save:
            sanitized_label = str(bin_label).replace("[", "").replace("]", "").replace(", ", "_")
            out_name = os.path.join(outdir, f"{base}__boxplot_coverage_bin_{i}_{sanitized_label}.png")
            plt.savefig(out_name, dpi=dpi)
            print(f"[Saved] {out_name}")

        plt.show()


# =========================
# Coverage distribution diagnostics
# =========================
def coverage_distribution_table(df: pd.DataFrame, methods, under_thresholds=None, over_thresholds=None):
    if under_thresholds is None:
        under_thresholds = [0.75, 0.70, 0.65, 0.60, 0.55, 0.50]
    if over_thresholds is None:
        over_thresholds = []

    rows = []
    for m in methods:
        colname = f"coverage_{m}"
        if colname not in df.columns:
            continue
        col = df[colname]
        row = {"Method": display_name(m)}

        for t in under_thresholds:
            row[f"Count < {int(t*100)}%"] = int((col < t).sum())
        for t in over_thresholds:
            row[f"Count > {int(t*100)}%"] = int((col > t).sum())

        rows.append(row)

    out = pd.DataFrame(rows)
    col_order = ["Method"] + \
        [f"Count < {int(t*100)}%" for t in sorted(under_thresholds, reverse=True)] + \
        [f"Count > {int(t*100)}%" for t in sorted(over_thresholds)]
    return out[col_order]


# =========================
# Smoothed / Raw plots (4 per CSV)
# =========================
def _kernel_bw_silverman(x):
    x = np.asarray(x)
    n = len(x)
    if n <= 1:
        return 1.0
    std = np.std(x)
    bw = 1.06 * std * (n ** (-1/5))
    if not np.isfinite(bw) or bw <= 1e-12:
        rng_ = np.max(x) - np.min(x)
        bw = max(rng_ / 20.0, 1e-3)
    return bw

def kernel_smooth_1d(x, y, x_grid=None, bw=None):
    x = np.asarray(x).ravel()
    y = np.asarray(y).ravel()
    if x_grid is None:
        x_grid = np.linspace(x.min(), x.max(), 300)
    if bw is None:
        bw = _kernel_bw_silverman(x)
    diff = (x_grid[:, None] - x[None, :]) / bw
    W = np.exp(-0.5 * diff**2)
    Wy = W @ y
    Ws = W.sum(axis=1) + 1e-12
    y_smooth = Wy / Ws
    return x_grid, y_smooth

def plot_four_panels(df, csv_path, feature_col, methods, style_map, alpha=0.1,
                     save=True, outdir="figures", dpi=150, warning_cov: float = 0.80):
    if feature_col not in df.columns:
        raise ValueError(f"CSV must contain column '{feature_col}'.")

    idx = np.argsort(df[feature_col].values)
    x_sorted = df[feature_col].values[idx]

    target_cov = 1 - alpha

    if save:
        os.makedirs(outdir, exist_ok=True)
    base = os.path.splitext(os.path.basename(csv_path))[0]

    # --- 1) Width - Smoothed ---
    with plt.style.context({
        'font.family': 'serif',
        'font.serif': ['Times New Roman', 'DejaVu Serif'],
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 14,
        'axes.grid': False,
        'axes.spines.right': False,
        'axes.spines.top': False,
    }):
        fig1, ax1 = plt.subplots(figsize=(8, 5))

        for m in methods:
            col = f"width_{m}"
            if col not in df.columns:
                continue
            y = df[col].values[idx]
            xg, ys = kernel_smooth_1d(x_sorted, y)
            st = style_map[m]
            ax1.plot(xg, ys, label=display_name(m),
                     color=st["color"], linestyle=st["linestyle"], linewidth=3.0, alpha=0.9)

        ax1.set_xlabel(f"Feature {feature_col}")
        ax1.set_ylabel("Interval Width")
        ax1.legend(frameon=False, loc='best')
        plt.tight_layout()

    # 2) Width - Raw
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    for m in methods:
        col = f"width_{m}"
        if col not in df.columns:
            continue
        y = df[col].values[idx]
        st = style_map[m]
        ax2.plot(x_sorted, y, label=display_name(m), linewidth=1.2, color=st["color"],
                 linestyle=st["linestyle"], alpha=0.9)
    ax2.set_xlabel(f"Feature {feature_col}", fontsize=12)
    ax2.set_ylabel("Mean Prediction Interval Width", fontsize=12)
    ax2.legend()
    ax2.grid(True, linestyle="--", linewidth=0.5)
    fig2.tight_layout()

    # 3) Coverage - Smoothed
    fig3, ax3 = plt.subplots(figsize=(12, 7))
    for m in methods:
        col = f"coverage_{m}"
        if col not in df.columns:
            continue
        y = df[col].values[idx]
        xg, ys = kernel_smooth_1d(x_sorted, y)
        st = style_map[m]
        ax3.plot(xg, ys, label=display_name(m), linewidth=2, color=st["color"], linestyle=st["linestyle"])
    ax3.axhline(y=target_cov, linestyle="--", color="gray", linewidth=1.5, label=f"Target {target_cov:.2f}")
    ax3.axhline(y=warning_cov, linestyle=":", color="black", linewidth=1.5, label=f"Warning {warning_cov:.2f}")
    ax3.set_xlabel(f"Feature {feature_col}", fontsize=12)
    ax3.set_ylabel("Conditional Coverage Probability", fontsize=12)
    ax3.set_ylim([0.5, 1.0])
    ax3.legend()
    ax3.grid(True, linestyle="--", linewidth=0.5)
    fig3.tight_layout()

    # 4) Coverage - Raw
    fig4, ax4 = plt.subplots(figsize=(12, 7))
    for m in methods:
        col = f"coverage_{m}"
        if col not in df.columns:
            continue
        y = df[col].values[idx]
        st = style_map[m]
        ax4.plot(x_sorted, y, label=display_name(m), linewidth=1.2, color=st["color"],
                 linestyle=st["linestyle"], alpha=0.9)
    ax4.axhline(y=target_cov, linestyle="--", color="gray", linewidth=1.5, label=f"Target {target_cov:.2f}")
    ax4.axhline(y=warning_cov, linestyle=":", color="black", linewidth=1.5, label=f"Warning {warning_cov:.2f}")
    ax4.set_xlabel(f"Feature {feature_col}", fontsize=12)
    ax4.set_ylabel("Conditional Coverage Probability", fontsize=12)
    ax4.set_ylim([0.5, 1.0])
    ax4.legend()
    ax4.grid(True, linestyle="--", linewidth=0.5)
    fig4.tight_layout()

    if save:
        out1 = os.path.join(outdir, f"{base}__width_vs_{feature_col}_smooth.png")
        out2 = os.path.join(outdir, f"{base}__width_vs_{feature_col}_raw.png")
        out3 = os.path.join(outdir, f"{base}__coverage_vs_{feature_col}_smooth.png")
        out4 = os.path.join(outdir, f"{base}__coverage_vs_{feature_col}_raw.png")
        fig1.savefig(out1, dpi=dpi)
        fig2.savefig(out2, dpi=dpi)
        fig3.savefig(out3, dpi=dpi)
        fig4.savefig(out4, dpi=dpi)
        print(f"[Saved] {out1}")
        print(f"[Saved] {out2}")
        print(f"[Saved] {out3}")
        print(f"[Saved] {out4}")

    plt.show()


# =========================
# Main runner
# =========================
def analyze_one_csv(csv_path: str):
    if not os.path.exists(csv_path):
        print(f"File not found: {csv_path}")
        return

    df = pd.read_csv(csv_path)

    raw_methods = find_methods(df)
    methods = preferred_method_order(raw_methods)
    style_map = build_style_map(methods)

    print("\n" + "="*80)
    print(f"File: {csv_path}")
    print("="*80)
    print(f"Detected methods (after filtering + reorder) ({len(methods)}):")
    for m in methods:
        dn = display_name(m)
        if dn != m:
            print(f"  - {m}  ->  {dn}")
        else:
            print(f"  - {m}")

    print("\n--- Detailed Conditional Coverage Summary (overall) ---")
    cov_sum = summary_table(df, "coverage_", methods)
    display(cov_sum)

    print("\n--- Detailed Conditional Interval Width Summary (overall) ---")
    wid_sum = summary_table(df, "width_", methods)
    display(wid_sum)

    if MAKE_BIN_BAR_PLOTS:
        print(f"\n--- Bar plots across '{FEATURE_COL}' bins ({NUM_BINS} bins) ---")
        bar_plots_across_feature_bins(
            df=df,
            csv_path=csv_path,
            feature_col=FEATURE_COL,
            methods=methods,
            style_map=style_map,
            alpha=ALPHA,
            num_bins=NUM_BINS,
            fill_empty_bins=FILL_EMPTY_BINS,
            save=SAVE_FIGS,
            outdir=OUTDIR,
            dpi=DPI,
            warning_cov=WARNING_COV,
        )

    print("\n--- Under/Over Coverage Count Table ---")
    dist = coverage_distribution_table(
        df, methods,
        under_thresholds=UNDER_COVERAGE_THRESHOLDS,
        over_thresholds=OVER_COVERAGE_THRESHOLDS
    )
    display(dist)

    print("\n--- Smoothed/Raw plots ---")
    plot_four_panels(
        df=df,
        csv_path=csv_path,
        feature_col=FEATURE_COL,
        methods=methods,
        style_map=style_map,
        alpha=ALPHA,
        save=SAVE_FIGS,
        outdir=OUTDIR,
        dpi=DPI,
        warning_cov=WARNING_COV,
    )

if __name__ == "__main__":
    for p in CSV_PATHS:
        analyze_one_csv(p)