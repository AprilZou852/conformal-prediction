"""
For each dataset, split test points into 4 groups by PC1 quartile.
For each group, plot Coverage vs Mean Width scatter (one point per method).
Each dataset produces a 2x2 figure.
"""
import json, os, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATASETS = ["Airfoil", "Computer", "Abalone", "concrete", "AutoMPG", "Crime"]
DISPLAY_NAMES = {"concrete": "Concrete"}
METHODS = ["Residual", "Rescaled", "CQR", "CPI-NNCDE", "DCP-NNCDE"]
METHOD_LABELS = ["Residual", "Rescaled", "CQR", "CPI", "DCP"]
SEEDS = list(range(777, 787))
TEST_FRAC = 0.20
N_GROUPS = 4

style = {
    "Residual":  {"marker": "^", "color": "#1f77b4"},
    "Rescaled":  {"marker": "v", "color": "#ff7f0e"},
    "CQR":       {"marker": "o", "color": "#2ca02c"},
    "CPI-NNCDE": {"marker": "s", "color": "#9467bd"},
    "DCP-NNCDE": {"marker": "P", "color": "#8c564b"},
}

BASE = "/sessions/affectionate-eloquent-heisenberg/mnt/outputs"

def load_xy(xp, yp):
    Xdf = pd.read_csv(xp)
    for c in Xdf.columns: Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    Xdf = Xdf.dropna(axis=1, how="all").apply(lambda c: c.fillna(c.mean()), axis=0)
    try:
        ydf = pd.read_csv(yp, header=None); y = pd.to_numeric(ydf.iloc[:,0], errors="coerce")
    except:
        ydf = pd.read_csv(yp); y = pd.to_numeric(ydf.iloc[:,0], errors="coerce")
    n = min(len(Xdf), len(y)); Xdf=Xdf.iloc[:n]; y=y.iloc[:n]
    mask = ~y.isna(); Xdf=Xdf.loc[mask].reset_index(drop=True); y=y.loc[mask].reset_index(drop=True)
    return Xdf.values.astype("float32"), y.values.astype("float32")

def train_test_split_np(n, test_frac, seed):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    nt = int(n * test_frac)
    return idx[nt:], idx[:nt]

def find_dataset(ds_name):
    bases = [
        "/sessions/affectionate-eloquent-heisenberg/mnt/outputs/bench_data",
        "/sessions/affectionate-eloquent-heisenberg/mnt/coverage/new_datasets",
        "/sessions/affectionate-eloquent-heisenberg/mnt/coverage",
    ]
    for base in bases:
        xp = os.path.join(base, ds_name, "X.csv")
        yp = os.path.join(base, ds_name, "y.csv")
        if os.path.exists(xp):
            if ds_name == "concrete" and not os.path.exists(yp):
                yp = os.path.join(base, ds_name, "y 2.csv")
            return xp, yp
    return None, None

for ds in DATASETS:
    dn = DISPLAY_NAMES.get(ds, ds)
    print(f"\n{'='*50}\n  Dataset: {dn}\n{'='*50}")
    
    xp, yp = find_dataset(ds)
    if xp is None:
        print(f"  Dataset {ds} not found, skipping")
        continue
    X_all, y_all = load_xy(xp, yp)
    n = len(y_all)
    print(f"  n={n}, p={X_all.shape[1]}")
    
    # Collect per-group, per-method, per-seed: coverage and width
    # group_data[group][method] = {"cov": [...], "width": [...]}
    group_data = {g: {m: {"cov": [], "width": []} for m in METHODS} for g in range(N_GROUPS)}
    
    for seed in SEEDS:
        pw_path = os.path.join(BASE, f"pointwise_{ds}_seed_{seed}.json")
        if not os.path.exists(pw_path):
            continue
        with open(pw_path) as f:
            pw = json.load(f)
        
        # Reproduce the exact same split
        idx_rest, idx_test = train_test_split_np(n, TEST_FRAC, seed)
        Xte = X_all[idx_test]
        
        # PCA: fit on training data, project test data
        Xtr = X_all[idx_rest]
        # Standardize using training stats
        mu = Xtr.mean(axis=0)
        sd = Xtr.std(axis=0)
        sd[sd < 1e-12] = 1.0
        Xtr_std = (Xtr - mu) / sd
        Xte_std = (Xte - mu) / sd
        
        # Compute PC1 via SVD on training data
        # Center (already done by standardization)
        U, S, Vt = np.linalg.svd(Xtr_std, full_matrices=False)
        pc1_direction = Vt[0]  # first principal component
        
        # Project test data onto PC1
        pc1_test = Xte_std @ pc1_direction
        
        # Split into 4 groups by quartile
        quartiles = np.percentile(pc1_test, [25, 50, 75])
        group_idx = np.digitize(pc1_test, quartiles)  # 0,1,2,3
        
        y_test = np.array(pw["y_test"])
        
        for method in METHODS:
            lo_key = method + "_lo"
            hi_key = method + "_hi"
            if lo_key not in pw:
                continue
            lo = np.array(pw[lo_key])
            hi = np.array(pw[hi_key])
            
            for g in range(N_GROUPS):
                mask = (group_idx == g)
                if mask.sum() == 0:
                    continue
                cov_g = float(np.mean((y_test[mask] >= lo[mask]) & (y_test[mask] <= hi[mask])))
                width_g = float(np.median(hi[mask] - lo[mask]))  # median to handle outliers
                group_data[g][method]["cov"].append(cov_g)
                group_data[g][method]["width"].append(width_g)
    
    # Compute all coverage values first to determine adaptive ylim
    all_covs = []
    plot_points = {g: {} for g in range(N_GROUPS)}
    for g in range(N_GROUPS):
        for method in METHODS:
            covs = np.array(group_data[g][method]["cov"])
            widths = np.array(group_data[g][method]["width"])
            if len(covs) == 0:
                continue
            med_cov = np.median(covs)
            med_width = np.median(widths)
            plot_points[g][method] = (med_width, med_cov)
            all_covs.append(med_cov)

    # Adaptive ylim
    if all_covs:
        cov_min = min(all_covs)
        cov_max = max(all_covs)
        margin = max((cov_max - cov_min) * 0.15, 0.01)
        ylim_lo = np.floor((cov_min - margin) * 100) / 100
        ylim_hi = np.ceil((cov_max + margin) * 100) / 100
    else:
        ylim_lo, ylim_hi = 0.85, 0.95

    # Plot 2x2
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    axes_flat = axes.flatten()

    for g in range(N_GROUPS):
        ax = axes_flat[g]
        for method in METHODS:
            if method not in plot_points[g]:
                continue
            med_width, med_cov = plot_points[g][method]
            s = style[method]
            ax.scatter(med_width, med_cov, marker=s["marker"], color=s["color"], s=140,
                       edgecolors='black', linewidths=0.6, zorder=5,
                       label=METHOD_LABELS[METHODS.index(method)])

        ax.axhline(y=0.90, color='gray', linestyle='--', linewidth=1, alpha=0.7)
        ax.set_xlabel('Mean Width', fontsize=10)
        ax.set_ylabel('Coverage', fontsize=10)
        ax.set_title(f'Group {g+1}', fontsize=11)
        ax.set_ylim(ylim_lo, ylim_hi)
        ax.tick_params(axis='both', labelsize=9)
        ax.grid(alpha=0.2)

    # Legend below
    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', ncol=5, fontsize=9.5,
               frameon=True, fancybox=True, markerscale=1.2,
               bbox_to_anchor=(0.5, -0.02))

    plt.tight_layout(rect=[0, 0.04, 1, 1.0])
    outname = f"pc1_coverage_vs_width_{ds}.png"
    plt.savefig(f"/sessions/affectionate-eloquent-heisenberg/mnt/coverage/{outname}",
                dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  Saved: {outname}")

print("\nAll done!")
