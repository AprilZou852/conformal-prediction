# -*- coding: utf-8 -*-
"""
HazardNN + baselines + PCS-UQ integration
(Align test split to PCS-UQ provided test_index)

Final display names in output table / CSV / plots:
  - Residual
  - Rescaled
  - CQR
  - PCS
  - CPI-NNCDE   (was CPI-NNCDE-opt: optimal z found in calibration & test)
  - DCP-NNCDE   (was DCP-NNCDE-opt: optimal z found in calibration & test)
"""

# ======================
# Safe-mode switches to avoid kernel crash
# ======================
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 默认先关 GPU（更稳）。如果你想启用 GPU，把它改成 "0" 或注释掉这一行。
os.environ["CUDA_VISIBLE_DEVICES"] = ""

import re
import math
import copy
import warnings
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader, random_split

# ========= L1-ERT conditional coverage metric =========
try:
    from covmetrics import ERT
    _HAS_COVMETRICS = True
except ImportError:
    _HAS_COVMETRICS = False

# ========= 变量选择依赖 =========
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import PCA

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

# ========= XGBoost（可选） =========
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

warnings.filterwarnings("ignore")
device = "cuda" if torch.cuda.is_available() else "cpu"

# ======================
# Global configs
# ======================
SEEDS = [777, 778, 779, 780, 781, 782, 783, 784, 785, 786]
X_PATH = "X.csv"
Y_PATH = "y.csv"

ALPHA = 0.10
TEST_FRAC = 0.20
CAL_FRAC = 0.35

# ===== PC1 grouping (evaluation strata) =====
PC1_N_GROUPS = 4

ENABLE_HAZARDNN = True

# 只想用现有CSV直接画图就把它改成 False，并设置 EXISTING_CSV_PATH
RUN_TRAINING = True
EXISTING_CSV_PATH = "./summary_seeds_777_786_with_hazard_grid_fs_stability_lgbm_PCA_GROUPS.csv"

# ========== PCS-UQ integration ==========
ENABLE_PCS_UQ = False
USE_PCS_TEST_SPLIT = False

PCS_UQ_PATHS = {
    777: "pcs_uq_seed_777_train_size_0.8_preds.csv",
    778: "pcs_uq_seed_778_train_size_0.8_preds.csv",
    779: "pcs_uq_seed_779_train_size_0.8_preds.csv",
    780: "pcs_uq_seed_780_train_size_0.8_preds.csv",
    781: "pcs_uq_seed_781_train_size_0.8_preds.csv",
    782: "pcs_uq_seed_782_train_size_0.8_preds.csv",
    783: "pcs_uq_seed_783_train_size_0.8_preds.csv",
    784: "pcs_uq_seed_784_train_size_0.8_preds.csv",
    785: "pcs_uq_seed_785_train_size_0.8_preds.csv",
    786: "pcs_uq_seed_786_train_size_0.8_preds.csv",
}

REQUIRE_PCS_ALL_SEEDS = False
PCS_APPEND_TO_EXISTING_CSV = True

# —— HazardNN 关键参数 —— #
HAZARD_MAX_GRID = 1024
PRED_CHUNK_ROWS = 32768
LONGTABLE_MAX_ROWS = 2_000_000

# —— HazardNN：时间设置 —— #
HZ_USE_TIME_REPARAM = False
HZ_TIMEGRID_MODE = "unique"
HZ_CLAMP_MAX_ETA = 10.0
HZ_EPS = 1e-8

# —— HazardNN 集成大小 K —— #
HZ_ENSEMBLE_K = 15
HZ_ENSEMBLE_SELECT_TOPK = 5
HZ_ENSEMBLE_VERBOSE_SELECTION = True

# —— 特征选择 —— #
FS_ENABLE = True
FS_METHOD = "stability_lgbm"
FS_BOOTSTRAPS = 50
FS_SUBSAMPLE = 0.7
FS_FREQ_THR = 0.6
FS_MAX_FEATURES = None
FS_COEF_THR = 1e-8
FS_ENET_L1RATIO = [0.1, 0.5, 0.9]
FS_ENET_ALPHAS = np.logspace(-4, 1, 50)

# —— HazardNN：XGB 预测作 meta 特征（可选） —— #
HZ_USE_XGB_META = False

# ======================
# CPI-opt / DCP-opt shortest configs
# ======================
RUN_SHORTEST_MODES = ["c"]
Z_EPS_U = 1e-6
Z_GRID_SIZE = 41

# ======================
# Rescaled / CQR stabilization configs
# ======================
STD_EPS = 1e-6

RESCALED_SIGMA_MIN = 1e-3
RESCALED_SIGMA_MAX = 10.0
RESCALED_DEBUG = False

CQR_DEBUG = False

# ======================
# Residual backend configs
# ======================
# 若必须强制 Residual 只能用 xgboost，把它改成 True
REQUIRE_XGBOOST_FOR_RESIDUAL = False

# xgboost 不可用时，Residual 是否允许 fallback
RESIDUAL_ALLOW_FALLBACK = True

# residual model weakening config (for xgboost)
XGB_WEAK_LEVEL = "medium"

# 复现 & 单线程
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.set_num_threads(1)

# ======================
# Final method names
# ======================
M_PCS = "PCS"
M_CQR = "CQR"
M_RESCALED = "Rescaled"
M_RESIDUAL = "Residual"
M_CPI = "CPI-NNCDE"   # was CPI-NNCDE-opt
M_DCP = "DCP-NNCDE"   # was DCP-NNCDE-opt

METHOD_ORDER_FINAL = [
    M_RESIDUAL,
    M_RESCALED,
    M_CQR,
    M_CPI,
    M_DCP,
]

# Old -> final method name mapping
METHOD_RENAME_MAP = {
    "residual-xgboost": M_RESIDUAL,
    "Residual": M_RESIDUAL,

    "Rescaled": M_RESCALED,
    "CQR": M_CQR,
    "PCS": M_PCS,
    "PCS_UQ": M_PCS,

    # All old names for the optimal-z CPI variant collapse into CPI-NNCDE
    "CPI-NNCDE": M_CPI,
    "CPI-NNCDE-opt": M_CPI,
    "HazardNN_Shortest": M_CPI,
    "HazardNN_Shortest_C_empiricalU": M_CPI,

    # All old names for the optimal-z DCP variant collapse into DCP-NNCDE
    "DCP-NNCDE": M_DCP,
    "DCP-NNCDE-opt": M_DCP,
    "DCP_OPT": M_DCP,
}


# ======================
# Utils
# ======================
def set_all_seeds(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def rename_method_name(x: str) -> str:
    return METHOD_RENAME_MAP.get(str(x), str(x))


def rename_methods_in_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "method" in df.columns:
        df["method"] = df["method"].astype(str).map(rename_method_name)
    return df


# ======================
# Data loading
# ======================
def load_real_xy_all_features(x_path, y_path):
    Xdf = pd.read_csv(x_path)
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    Xdf = Xdf.dropna(axis=1, how="all").copy()
    Xdf = Xdf.apply(lambda col: col.fillna(col.mean()), axis=0)

    try:
        ydf = pd.read_csv(y_path, header=None)
        y = pd.to_numeric(ydf.iloc[:, 0], errors="coerce")
    except Exception:
        ydf = pd.read_csv(y_path)
        y = pd.to_numeric(ydf.iloc[:, 0], errors="coerce")

    n = min(len(Xdf), len(y))
    Xdf = Xdf.iloc[:n, :].copy()
    y = y.iloc[:n].copy()

    mask = ~y.isna()
    Xdf = Xdf.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    p = Xdf.shape[1]
    feature_names = [f"x{i+1}" for i in range(p)]
    Xdf.columns = feature_names
    return Xdf.values.astype("float32"), y.values.astype("float32"), feature_names


# ======================
# PCS-UQ loading + alignment
# ======================
def _normalize_colname(s: str) -> str:
    s = str(s).strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    s = re.sub(r"__+", "_", s)
    return s


def load_pcs_uq_predictions(path: str) -> pd.DataFrame:
    if path is None:
        raise FileNotFoundError("PCS-UQ path is None")
    if not os.path.exists(path):
        raise FileNotFoundError(f"PCS-UQ file not found: {path}")

    ext = Path(path).suffix.lower()
    if ext in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        try:
            df = pd.read_csv(path)
        except Exception:
            df = pd.read_csv(path, sep="\t")

    col_map = {_normalize_colname(c): c for c in df.columns}

    def pick(*cands):
        for k in cands:
            if k in col_map:
                return col_map[k]
        return None

    c_test_index = pick("test_index", "test_idx", "idx_test", "index_test", "testindex")
    c_y_test = pick("y_test", "ytrue", "y_true", "y", "y_obs", "yobs")
    c_lb = pick("y_pred_lb", "y_lb", "pred_lb", "lower", "y_pred_lower", "y_lower", "lb")
    c_ub = pick("y_pred_ub", "y_ub", "pred_ub", "upper", "y_pred_upper", "y_upper", "ub")

    missing = [("test_index", c_test_index), ("y_test", c_y_test), ("y_pred_lb", c_lb), ("y_pred_ub", c_ub)]
    missing = [name for name, col in missing if col is None]
    if missing:
        raise ValueError(f"PCS-UQ file missing required columns {missing}. Columns seen: {list(df.columns)}")

    out = df[[c_test_index, c_y_test, c_lb, c_ub]].copy()
    out.columns = ["test_index", "y_test", "y_pred_lb", "y_pred_ub"]

    for c in ["test_index", "y_test", "y_pred_lb", "y_pred_ub"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["test_index", "y_pred_lb", "y_pred_ub"]).copy()
    out["test_index"] = out["test_index"].astype(int)
    out = out.drop_duplicates(subset=["test_index"], keep="first").reset_index(drop=True)
    return out


def infer_test_index_base(pcs_df: pd.DataFrame, y_all: np.ndarray, tol: float = 1e-6) -> int:
    n = len(y_all)
    idx = pcs_df["test_index"].values.astype(int)

    def score(base):
        j = idx - base
        ok = (j >= 0) & (j < n)
        if ok.sum() == 0:
            return -1.0
        y_ref = y_all[j[ok]]
        y_pcs = pcs_df.loc[ok, "y_test"].values.astype(float)
        if np.all(np.isnan(y_pcs)):
            return -1.0
        m = np.isfinite(y_pcs) & np.isfinite(y_ref)
        if m.sum() == 0:
            return -1.0
        return float(np.mean(np.abs(y_pcs[m] - y_ref[m]) <= tol))

    s0 = score(0)
    s1 = score(1)
    if s1 > s0 + 0.05:
        return 1
    return 0


def get_forced_test_indices_from_pcs(
    seed: int,
    y_all: np.ndarray,
    *,
    pcs_paths: dict,
    require_all: bool = False,
):
    if seed not in pcs_paths:
        msg = f"[PCS_TEST_SPLIT] seed={seed}: no PCS file in PCS_UQ_PATHS."
        if require_all:
            raise FileNotFoundError(msg)
        print(msg + " -> fall back to sklearn train_test_split.")
        return None

    pcs_file = pcs_paths[seed]
    if not os.path.exists(pcs_file):
        msg = f"[PCS_TEST_SPLIT] seed={seed}: PCS file not found on disk: {pcs_file}"
        if require_all:
            raise FileNotFoundError(msg)
        print(msg + " -> fall back to sklearn train_test_split.")
        return None
    pcs_df = load_pcs_uq_predictions(pcs_file)
    base = infer_test_index_base(pcs_df, y_all)
    idx0 = (pcs_df["test_index"].values.astype(int) - base).astype(int)

    n = len(y_all)
    ok = (idx0 >= 0) & (idx0 < n)
    if ok.sum() != len(idx0):
        bad = idx0[~ok]
        msg = f"[PCS_TEST_SPLIT] seed={seed}: {len(bad)} indices out of range after base={base} (n={n})."
        if require_all:
            raise ValueError(msg)
        print("[WARN] " + msg + " -> dropping out-of-range indices.")
        idx0 = idx0[ok]

    seen = set()
    idx_list = []
    for i in idx0.tolist():
        if int(i) not in seen:
            seen.add(int(i))
            idx_list.append(int(i))

    idx_test = np.array(idx_list, dtype=int)
    if len(idx_test) == 0:
        msg = f"[PCS_TEST_SPLIT] seed={seed}: empty idx_test after processing."
        if require_all:
            raise ValueError(msg)
        print("[WARN] " + msg + " -> fall back to sklearn train_test_split.")
        return None

    print(f"[PCS_TEST_SPLIT] seed={seed}: forced test_size={len(idx_test)} (base={base}).")
    return idx_test


def get_pcs_intervals_for_seed(
    seed: int,
    idx_test: np.ndarray,
    y_all: np.ndarray,
    *,
    pcs_paths: dict,
    require_all: bool = False,
):
    if seed not in pcs_paths:
        msg = f"[PCS_UQ] seed={seed}: no PCS file in PCS_UQ_PATHS."
        if require_all:
            raise FileNotFoundError(msg)
        print(msg + " -> skip PCS_UQ for this seed.")
        return None, None

    pcs_path = pcs_paths[seed]
    if not os.path.exists(pcs_path):
        msg = f"[PCS_UQ] seed={seed}: PCS file not found on disk: {pcs_path}"
        if require_all:
            raise FileNotFoundError(msg)
        print(msg + " -> skip PCS_UQ for this seed.")
        return None, None
    pcs_df = load_pcs_uq_predictions(pcs_path)
    base = infer_test_index_base(pcs_df, y_all)

    pcs_df = pcs_df.copy()
    pcs_df["idx0"] = pcs_df["test_index"].values.astype(int) - base

    set_pcs = set(pcs_df["idx0"].tolist())
    set_test = set(int(i) for i in idx_test.tolist())
    if set_pcs != set_test:
        inter = sorted(set_pcs.intersection(set_test))
        msg = (
            f"[PCS_UQ] seed={seed}: test index set mismatch. "
            f"pcs_size={len(set_pcs)} split_test_size={len(set_test)} inter={len(inter)} "
            f"(base={base})"
        )
        if require_all:
            raise ValueError(msg)
        print("[WARN] " + msg + " -> will evaluate on intersection only (missing -> NaN).")

    pcs_map_lb = pcs_df.set_index("idx0")["y_pred_lb"].to_dict()
    pcs_map_ub = pcs_df.set_index("idx0")["y_pred_ub"].to_dict()

    lo = np.full(len(idx_test), np.nan, dtype="float32")
    hi = np.full(len(idx_test), np.nan, dtype="float32")
    for k, idx in enumerate(idx_test.tolist()):
        if idx in pcs_map_lb and idx in pcs_map_ub:
            lo[k] = float(pcs_map_lb[idx])
            hi[k] = float(pcs_map_ub[idx])

    return lo, hi


# ======================
# Feature selection
# ======================
def _fallback_topk_by_corr(X_train, y_train, feature_names, k=1):
    corrs = []
    for i in range(X_train.shape[1]):
        xi = X_train[:, i]
        if np.std(xi) == 0:
            corrs.append(0.0)
        else:
            c = np.corrcoef(xi, y_train)[0, 1]
            if np.isnan(c):
                c = 0.0
            corrs.append(abs(c))
    order = np.argsort(corrs)[::-1][:max(1, k)]
    return [int(i) for i in order], [feature_names[i] for i in order]


def _select_stability_elasticnet(
    X_train,
    y_train,
    feature_names,
    random_state=0,
    alphas=None,
    l1_ratios=None,
    n_boot=FS_BOOTSTRAPS,
    subsample=FS_SUBSAMPLE,
    freq_thr=FS_FREQ_THR,
    max_features=FS_MAX_FEATURES,
    coef_thr=FS_COEF_THR,
    verbose=True,
):
    if alphas is None:
        alphas = FS_ENET_ALPHAS
    if l1_ratios is None:
        l1_ratios = FS_ENET_L1RATIO

    rng = np.random.default_rng(random_state)
    n, p = X_train.shape
    counts = np.zeros(p, dtype=int)

    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs_full = scaler.fit_transform(X_train.astype(np.float64))

    for b in range(n_boot):
        idx = rng.choice(n, size=int(n * subsample), replace=False)
        Xs = Xs_full[idx]
        ys = y_train[idx].astype(np.float64)

        enet = ElasticNetCV(
            alphas=alphas,
            l1_ratio=l1_ratios,
            cv=5,
            random_state=random_state + b,
            max_iter=5000,
        )
        enet.fit(Xs, ys)
        counts += (np.abs(enet.coef_) > coef_thr).astype(int)

    freq = counts / n_boot
    sel_idx = np.where(freq >= freq_thr)[0].tolist()

    if len(sel_idx) == 0:
        order = np.argsort(freq)[::-1]
        top = order[:max(1, min(5, p))].tolist()
        sel_idx = top
        if np.all(freq[top] == 0):
            sel_idx, _ = _fallback_topk_by_corr(X_train, y_train, feature_names, k=1)

    if max_features is not None and len(sel_idx) > max_features:
        sel_idx = sorted(sel_idx, key=lambda j: (-freq[j], j))[:max_features]

    sel_names = [feature_names[i] for i in sel_idx]
    if verbose:
        shown = ", ".join(f"{feature_names[i]}({freq[i]:.2f})" for i in sel_idx[:20])
        print(f"[FS/Stability-ENet] kept={len(sel_idx)} thr={freq_thr} -> {shown}")

    return sel_idx, sel_names


def _select_stability_lgbm(
    X_train,
    y_train,
    feature_names,
    random_state=0,
    n_boot=FS_BOOTSTRAPS,
    subsample=FS_SUBSAMPLE,
    max_features=FS_MAX_FEATURES,
    verbose=True,
):
    rng = np.random.default_rng(random_state)
    n, p = X_train.shape
    gains = np.zeros(p, dtype=float)

    for b in range(n_boot):
        idx = rng.choice(n, size=int(n * subsample), replace=False)
        Xs = X_train[idx]
        ys = y_train[idx]

        if _HAS_LGB:
            model = lgb.LGBMRegressor(
                n_estimators=600,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                max_depth=-1,
                random_state=random_state + b,
            )
            model.fit(Xs, ys)
            imp = model.booster_.feature_importance(importance_type="gain")
        else:
            rf = RandomForestRegressor(
                n_estimators=400,
                max_features="sqrt",
                random_state=random_state + b,
                n_jobs=1,
            )
            rf.fit(Xs, ys)
            imp = rf.feature_importances_

        gains += imp.astype(float)

    gains /= n_boot
    order = np.argsort(gains)[::-1]
    sel_idx = [i for i in order if gains[i] > 0]

    if len(sel_idx) == 0:
        sel_idx, _ = _fallback_topk_by_corr(X_train, y_train, feature_names, k=1)

    if max_features is not None and len(sel_idx) > max_features:
        sel_idx = sel_idx[:max_features]

    sel_names = [feature_names[i] for i in sel_idx]
    if verbose:
        tag = "LGBM" if _HAS_LGB else "RF"
        shown = ", ".join(f"{feature_names[i]}({gains[i]:.1f})" for i in sel_idx[:20])
        print(f"[FS/Stability-{tag}] kept={len(sel_idx)} -> {shown}")

    return sel_idx, sel_names


def _select_lasso(
    X_train,
    y_train,
    feature_names,
    random_state=0,
    alphas=np.logspace(-4, 1, 50),
    coef_thr=FS_COEF_THR,
    max_features=FS_MAX_FEATURES,
    verbose=True,
):
    scaler = StandardScaler(with_mean=True, with_std=True)
    Xs = scaler.fit_transform(X_train.astype(np.float64))

    lcv = LassoCV(alphas=alphas, cv=5, random_state=random_state, max_iter=5000)
    lcv.fit(Xs, y_train.astype(np.float64))

    coefs = np.asarray(lcv.coef_, dtype=float)
    sel_idx = np.where(np.abs(coefs) > coef_thr)[0].tolist()

    if len(sel_idx) == 0:
        sel_idx, _ = _fallback_topk_by_corr(X_train, y_train, feature_names, k=1)

    if max_features is not None and len(sel_idx) > max_features:
        order = np.argsort(-np.abs(coefs[sel_idx]))
        sel_idx = [sel_idx[i] for i in order[:max_features]]

    sel_names = [feature_names[i] for i in sel_idx]
    if verbose:
        print(f"[FS/LassoCV] kept={len(sel_idx)} -> {sel_names}")

    return sel_idx, sel_names


def select_features(X_train, y_train, feature_names, random_state=0, verbose=True):
    if not FS_ENABLE:
        return list(range(len(feature_names))), feature_names[:]

    if FS_METHOD == "stability_enet":
        return _select_stability_elasticnet(
            X_train, y_train, feature_names, random_state=random_state, verbose=verbose
        )
    if FS_METHOD == "stability_lgbm":
        return _select_stability_lgbm(
            X_train, y_train, feature_names, random_state=random_state, verbose=verbose
        )
    return _select_lasso(
        X_train, y_train, feature_names, random_state=random_state, verbose=verbose
    )


# ======================
# Long-table expand for HazardNN
# ======================
def expand_long(X, y, feature_names):
    uniq = np.sort(np.unique(y))
    if len(uniq) < 2:
        uniq = np.array([y.min(), y.min() + 1.0], dtype=y.dtype)
    t0 = float(uniq[0] - 1.0)
    rows = []
    for idx, (xi, yi) in enumerate(zip(X, y)):
        s = t0
        for t in uniq[uniq <= yi]:
            rows.append([idx, s, float(t), int(t == yi), *xi])
            s = float(t)
    cols = ["id", "start", "end", "delta"] + feature_names
    return pd.DataFrame(rows, columns=cols), uniq


# ======================
# Models: HazardNN + baselines
# ======================
class F1Net(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def train_F1_longtable(
    df,
    feats_list,
    *,
    epochs=500,
    batch=128,
    val_frac=0.2,
    patience=30,
    seed_split=0,
):
    set_all_seeds(seed_split)

    if len(df) > LONGTABLE_MAX_ROWS:
        idx = np.random.default_rng(seed_split).choice(
            len(df), size=LONGTABLE_MAX_ROWS, replace=False
        )
        df = df.iloc[idx].reset_index(drop=True)

    X = df[feats_list].values.astype("float32")
    Y = np.c_[
        df["delta"].values.astype("float32"),
        (df["end"] - df["start"]).values.astype("float32"),
    ]

    mu, sg = X.mean(0), X.std(0)
    sg[sg == 0] = 1
    Xt = (X - mu) / sg

    ds = TensorDataset(
        torch.tensor(Xt, device=device),
        torch.tensor(Y, device=device),
    )
    n_val = int(len(ds) * val_frac)
    gen = torch.Generator(device="cpu").manual_seed(seed_split)
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch, shuffle=False)

    net = F1Net(in_dim=X.shape[1]).to(device)
    opt = optim.Adam(net.parameters(), lr=1e-3)

    def loss_fn(y_true, y_pred):
        y_pred = torch.clamp(y_pred, max=HZ_CLAMP_MAX_ETA)
        delta = y_true[:, :1]
        dt = y_true[:, 1:2]
        return -(delta * y_pred).mean() + (torch.exp(y_pred) * dt).mean()

    best_state = copy.deepcopy(net.state_dict())
    best_val = math.inf
    best_epoch = -1
    no_gain = 0

    for ep in range(epochs):
        net.train()
        for xb, yb in dl_tr:
            opt.zero_grad()
            loss_fn(yb, net(xb)).backward()
            opt.step()

        net.eval()
        with torch.no_grad():
            v = np.mean([loss_fn(yb, net(xb)).item() for xb, yb in dl_va])

        if v < best_val - 1e-4:
            best_val = v
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = ep
            no_gain = 0
        else:
            no_gain += 1

        if no_gain >= patience:
            break

    net.load_state_dict(best_state)
    return net, mu.astype("float32"), sg.astype("float32"), float(best_val), int(best_epoch)


# ======================
# Shared standardization helpers
# ======================
def fit_standardizer_xy(X_train, y_train, eps=STD_EPS):
    x_mu = X_train.mean(axis=0).astype("float32")
    x_sd = X_train.std(axis=0).astype("float32")
    x_sd[x_sd < eps] = 1.0

    y_mu = np.float32(np.mean(y_train))
    y_sd = np.float32(np.std(y_train))
    if y_sd < eps:
        y_sd = np.float32(1.0)

    return x_mu, x_sd, y_mu, y_sd


def transform_X(X, x_mu, x_sd):
    return ((X - x_mu) / x_sd).astype("float32")


def transform_y(y, y_mu, y_sd):
    return ((y - y_mu) / y_sd).astype("float32")


def inverse_y(y_std, y_mu, y_sd):
    return (y_mu + y_sd * y_std).astype("float32")


# ======================
# Rescaled baseline
# ======================
class MeanScaleNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.scale_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.base_net(x)
        mean = self.mean_head(h)
        sigma = RESCALED_SIGMA_MIN + F.softplus(self.scale_head(h))
        sigma = torch.clamp(sigma, min=RESCALED_SIGMA_MIN, max=RESCALED_SIGMA_MAX)
        return mean, sigma


def gaussian_nll_loss(y_true, mean, sigma):
    sigma = torch.clamp(sigma, min=RESCALED_SIGMA_MIN, max=RESCALED_SIGMA_MAX)
    z = (y_true - mean) / sigma
    return (torch.log(sigma) + 0.5 * z.pow(2)).mean()


def train_mean_variance_NN(
    X, y, *, epochs=500, batch=128, val_frac=0.2, patience=30, seed_split=0
):
    set_all_seeds(seed_split)

    X = X.astype("float32")
    y = y.astype("float32")

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    n_val = int(len(ds) * val_frac)
    gen = torch.Generator(device="cpu").manual_seed(seed_split)
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch, shuffle=False)

    model = MeanScaleNN(in_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), math.inf, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            mean, sigma = model(xb)
            opt.zero_grad()
            gaussian_nll_loss(yb, mean, sigma).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses = [
                gaussian_nll_loss(yb.to(device), *model(xb.to(device))).item()
                for xb, yb in dl_va
            ]
            v = np.mean(losses)

        if v < best_val - 1e-4:
            best_val, best_state, no_gain = v, copy.deepcopy(model.state_dict()), 0
        else:
            no_gain += 1

        if no_gain >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ======================
# CQR baseline
# ======================
class QuantileNN(nn.Module):
    def __init__(self, in_dim, hidden=64):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.center_head = nn.Linear(hidden, 1)
        self.width_head = nn.Linear(hidden, 1)

    def forward(self, x):
        h = self.base(x)
        center = self.center_head(h)
        half_width = F.softplus(self.width_head(h))
        q_lo = center - half_width
        q_hi = center + half_width
        return torch.cat([q_lo, q_hi], dim=1)


def pinball_loss(y_true, y_pred, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        error = y_true - y_pred[:, i : i + 1]
        losses.append(torch.mean(torch.max((q - 1) * error, q * error)))
    return sum(losses)


def train_quantile_NN(
    X, y, *, quantiles, epochs=500, batch=128, val_frac=0.2, patience=30, seed_split=0
):
    set_all_seeds(seed_split)

    X = X.astype("float32")
    y = y.astype("float32")

    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
    n_val = int(len(ds) * val_frac)
    gen = torch.Generator(device="cpu").manual_seed(seed_split)
    tr, va = random_split(ds, [len(ds) - n_val, n_val], generator=gen)
    dl_tr = DataLoader(tr, batch_size=batch, shuffle=True)
    dl_va = DataLoader(va, batch_size=batch, shuffle=False)

    model = QuantileNN(in_dim=X.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)

    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), math.inf, 0

    for _ in range(epochs):
        model.train()
        for xb, yb in dl_tr:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            pinball_loss(yb, pred, quantiles).backward()
            opt.step()

        model.eval()
        with torch.no_grad():
            losses = [
                pinball_loss(yb.to(device), model(xb.to(device)), quantiles).item()
                for xb, yb in dl_va
            ]
            v = np.mean(losses)

        if v < best_val - 1e-4:
            best_val, best_state, no_gain = v, copy.deepcopy(model.state_dict()), 0
        else:
            no_gain += 1

        if no_gain >= patience:
            break

    model.load_state_dict(best_state)
    model.eval()
    return model


# ======================
# Residual regressor
# ======================
def train_xgb_regressor(X_train, y_train, random_state=0, weak_level=None):
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed")

    if weak_level is None:
        weak_level = XGB_WEAK_LEVEL

    use_gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    use_gpu = torch.cuda.is_available() and use_gpu_env not in ("", "none", "null")
    tree_method = "gpu_hist" if use_gpu else "hist"

    presets = {
        "orig": dict(
            n_estimators=300,
            learning_rate=0.1,
            max_depth=4,
            subsample=1.0,
            colsample_bytree=1.0,
            reg_lambda=1.0,
        ),
        "mild": dict(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=3,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=2.0,
            gamma=0.0,
            min_child_weight=1,
            max_bin=256,
        ),
        "medium": dict(
            n_estimators=120,
            learning_rate=0.08,
            max_depth=2,
            subsample=0.75,
            colsample_bytree=0.75,
            min_child_weight=5,
            gamma=0.1,
            reg_alpha=0.2,
            reg_lambda=5.0,
            max_bin=128,
        ),
        "heavy": dict(
            n_estimators=80,
            learning_rate=0.1,
            max_depth=2,
            subsample=0.65,
            colsample_bytree=0.65,
            min_child_weight=10,
            gamma=0.3,
            reg_alpha=0.5,
            reg_lambda=10.0,
            max_bin=64,
        ),
    }

    params = presets.get(str(weak_level).lower(), presets["medium"])

    model = xgb.XGBRegressor(
        **params,
        random_state=random_state,
        n_jobs=1,
        tree_method=tree_method,
        verbosity=0,
        objective="reg:squarederror",
    )
    model.fit(X_train, y_train)
    return model


def train_residual_regressor(X_train, y_train, random_state=0, weak_level=None):
    """
    Prefer xgboost.
    If unavailable, optionally fall back so Residual still has results.
    Returns: (model, backend_name)
    """
    if _HAS_XGB:
        model = train_xgb_regressor(X_train, y_train, random_state=random_state, weak_level=weak_level)
        return model, "xgboost"

    if REQUIRE_XGBOOST_FOR_RESIDUAL:
        raise ImportError(
            "xgboost is not installed, and REQUIRE_XGBOOST_FOR_RESIDUAL=True. "
            "Install xgboost or set REQUIRE_XGBOOST_FOR_RESIDUAL=False."
        )

    if not RESIDUAL_ALLOW_FALLBACK:
        raise ImportError(
            "xgboost is not installed, and RESIDUAL_ALLOW_FALLBACK=False. "
            "Cannot compute Residual."
        )

    if _HAS_LGB:
        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            max_depth=-1,
            random_state=random_state,
        )
        model.fit(X_train, y_train)
        return model, "lightgbm-fallback"

    model = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model, "rf-fallback"


def add_xgb_meta(X_arr, mdl, residual_backend=None):
    # only add meta feature if backend is truly xgboost
    if HZ_USE_XGB_META and (mdl is not None) and (residual_backend == "xgboost"):
        pred = mdl.predict(X_arr).reshape(-1, 1).astype("float32")
        return np.concatenate([X_arr, pred], axis=1), ["xgb_pred"]
    return X_arr, []


# ======================
# HazardNN inference: time grid + stable PIT/quantile
# ======================
def _make_coarse_time_grid(uniq_y_train: np.ndarray, max_grid: int, mode: str = "quantile"):
    uniq = np.sort(np.unique(uniq_y_train))
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


def _time_feats(starts, ends):
    if not HZ_USE_TIME_REPARAM:
        return np.c_[starts.reshape(-1, 1), ends.reshape(-1, 1)]
    t_mid = 0.5 * (starts + ends)
    dt = np.maximum(ends - starts, HZ_EPS)
    return np.c_[t_mid.reshape(-1, 1), np.log(dt).reshape(-1, 1)]


@torch.no_grad()
def _hazard_grid_for_one_x(net, mu, sg, t_edges, x_cov, *, chunk_rows=PRED_CHUNK_ROWS):
    M = len(t_edges) - 1
    starts = t_edges[:-1]
    ends = t_edges[1:]
    dt = ends - starts
    time_feat = _time_feats(starts, ends)
    cov_rep = np.repeat(x_cov.reshape(1, -1), M, axis=0)
    rows = np.concatenate([time_feat, cov_rep], axis=1).astype("float32")
    rows_norm = (rows - mu) / sg

    lam_list = []
    for i in range(0, M, chunk_rows):
        xb = torch.tensor(rows_norm[i : i + chunk_rows], device=device)
        eta = torch.clamp(net(xb).squeeze(1), max=HZ_CLAMP_MAX_ETA)
        lam = torch.exp(eta)
        lam_list.append(lam.detach().cpu().numpy())

    lam_all = np.concatenate(lam_list, axis=0).astype("float32")
    contrib = lam_all * dt
    cum_L = np.cumsum(contrib).astype("float32")
    return cum_L, t_edges, lam_all


@torch.no_grad()
def _hazard_grid_for_one_x_ensemble(models, t_edges, x_cov, *, chunk_rows=PRED_CHUNK_ROWS):
    lam_accum = None
    for net, mu, sg in models:
        _, _, lam_i = _hazard_grid_for_one_x(
            net, mu, sg, t_edges, x_cov, chunk_rows=chunk_rows
        )
        if lam_accum is None:
            lam_accum = lam_i.astype("float64")
        else:
            lam_accum += lam_i.astype("float64")

    lam_avg = (lam_accum / max(1, len(models))).astype("float32")
    dt = (t_edges[1:] - t_edges[:-1]).astype("float32")
    cum_L = np.cumsum(lam_avg * dt).astype("float32")
    return cum_L, t_edges, lam_avg


@torch.no_grad()
def _cumL_lam_for_one_x(hz_models, t_edges, x_cov):
    if isinstance(hz_models, list):
        return _hazard_grid_for_one_x_ensemble(hz_models, t_edges, x_cov)
    net, mu, sg = hz_models
    return _hazard_grid_for_one_x(net, mu, sg, t_edges, x_cov)


@torch.no_grad()
def _stable_pit(cum_L, t_edges, lam, y_val):
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


def _stable_quantile_precomp(F_grid, cum_L, t_edges, lam, u):
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


def _precompute_hazard_distributions(hz_models, t_edges, X_arr):
    cumLs, lams, Fgrids = [], [], []
    for x in X_arr:
        cumL, _, lam = _cumL_lam_for_one_x(hz_models, t_edges, x)
        F_grid = -np.expm1(-cumL.astype(np.float64))
        cumLs.append(cumL.astype("float32"))
        lams.append(lam.astype("float32"))
        Fgrids.append(F_grid.astype("float32"))
    return cumLs, lams, Fgrids


def _hazard_quantile_from_cache(F_grid, cumL, t_edges, lam, u):
    return _stable_quantile_precomp(F_grid, cumL, t_edges, lam, float(u))


# ======================
# Optimal-z (shortest width) helper
# Solves z*(x) in argmin_{z in (0, alpha)} { Q(z+1-alpha|x) - Q(z|x) } by grid search.
# ======================
def _compute_shortest_z_star_from_cache(F_grid, cumL, t_edges, lam, alpha, grid_size, eps_u):
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
        q1 = _stable_quantile_precomp(F_grid, cumL, t_edges, lam, u1)
        q2 = _stable_quantile_precomp(F_grid, cumL, t_edges, lam, u2)
        w = q2 - q1
        if w < best_w:
            best_w = w
            best_z = float(z)

    return best_z


# ======================
# PC1 grouping helpers
# ======================
def _pc1_group_labels_from_edges(edges: np.ndarray, include_marginal: bool = True):
    n_groups = max(1, int(len(edges) - 1))
    labels = [f"PC1_Group_{i}" for i in range(1, n_groups + 1)]
    if include_marginal:
        return ["Marginal"] + labels
    return labels


def _infer_pc1_group_order_from_df(df: pd.DataFrame):
    groups = sorted(df["group"].astype(str).unique().tolist())
    pc1 = []
    for g in groups:
        m = re.match(r"PC1_Group_(\d+)", str(g))
        if m:
            pc1.append((int(m.group(1)), g))
    pc1_sorted = [g for _, g in sorted(pc1, key=lambda x: x[0])]
    out = ["Marginal"] if "Marginal" in groups else []
    out += pc1_sorted
    out += [g for g in groups if g not in set(out)]
    return out


# ======================
# Main pipeline: one split
# ======================
def run_one_split(X_all, y_all, feature_names, split_seed):
    set_all_seeds(split_seed)
    idx_all = np.arange(len(y_all), dtype=int)

    # ---------- TEST split (forced by PCS if enabled) ----------
    idx_test_forced = None
    if USE_PCS_TEST_SPLIT and ENABLE_PCS_UQ:
        idx_test_forced = get_forced_test_indices_from_pcs(
            split_seed,
            y_all,
            pcs_paths=PCS_UQ_PATHS,
            require_all=REQUIRE_PCS_ALL_SEEDS,
        )

    if idx_test_forced is not None:
        idx_test = idx_test_forced
        set_test = set(int(i) for i in idx_test.tolist())
        idx_rest = np.array([i for i in idx_all.tolist() if int(i) not in set_test], dtype=int)
    else:
        idx_rest, idx_test = train_test_split(
            idx_all,
            test_size=TEST_FRAC,
            random_state=split_seed,
        )

    X_test, y_test = X_all[idx_test], y_all[idx_test]

    # ---------- TRAIN/CAL split on rest ----------
    n = len(idx_all)
    test_frac_actual = len(idx_test) / max(1, n)
    rest_frac = 1.0 - test_frac_actual
    if rest_frac <= 0:
        raise ValueError(f"Empty rest set (test size={len(idx_test)} equals n={n}).")

    cal_frac_within_rest = CAL_FRAC / rest_frac
    cal_frac_within_rest = float(np.clip(cal_frac_within_rest, 0.0, 0.99))

    idx_train, idx_cal = train_test_split(
        idx_rest,
        test_size=cal_frac_within_rest,
        random_state=split_seed,
    )
    X_train, y_train = X_all[idx_train], y_all[idx_train]
    X_cal, y_cal = X_all[idx_cal], y_all[idx_cal]

    # conformal quantile level (shared)
    q_level = (1 - ALPHA) * (1 + 1 / len(y_cal))
    q_level = min(float(q_level), 0.999999)

    # ===== Feature selection =====
    sel_idx, sel_names = select_features(
        X_train, y_train, feature_names, random_state=split_seed, verbose=True
    )
    X_train_sel = X_train[:, sel_idx]
    X_cal_sel = X_cal[:, sel_idx]
    X_test_sel = X_test[:, sel_idx]
    print(
        f"[FS] Using {len(sel_idx)}/{len(feature_names)} features: "
        f"{sel_names[:20]}{'...' if len(sel_names) > 20 else ''}"
    )

    # ===== Residual model =====
    mdl_residual = None
    residual_backend = None
    try:
        mdl_residual, residual_backend = train_residual_regressor(
            X_train_sel,
            y_train,
            random_state=split_seed,
            weak_level=XGB_WEAK_LEVEL,
        )
        print(f"[Residual] backend={residual_backend}")
    except Exception as e:
        print(f"[Residual] failed: {e}")
        mdl_residual = None
        residual_backend = None

    # ===== XGB meta (optional) for HazardNN =====
    X_train_hz, meta_names = add_xgb_meta(X_train_sel, mdl_residual, residual_backend=residual_backend)
    X_cal_hz, _ = add_xgb_meta(X_cal_sel, mdl_residual, residual_backend=residual_backend)
    X_test_hz, _ = add_xgb_meta(X_test_sel, mdl_residual, residual_backend=residual_backend)
    hz_feature_names = sel_names + meta_names

    # ===== PC1 grouping (for evaluation only) =====
    pca_scaler = StandardScaler()
    X_train_sel_s = pca_scaler.fit_transform(X_train_sel)
    pca = PCA(n_components=1, random_state=split_seed).fit(X_train_sel_s)

    pc1_cal = pca.transform(pca_scaler.transform(X_cal_sel)).flatten()
    pc1_test = pca.transform(pca_scaler.transform(X_test_sel)).flatten()

    def compute_edges(v, n_groups=PC1_N_GROUPS):
        qs = np.linspace(0, 1, n_groups + 1)
        edges = np.quantile(v, qs).astype(float)

        rng = float(np.max(edges) - np.min(edges))
        eps = max(1e-12, 1e-6 * (rng if rng > 0 else 1.0))
        for i in range(1, len(edges)):
            if edges[i] <= edges[i - 1]:
                edges[i] = edges[i - 1] + eps

        edges[0] -= eps
        edges[-1] += eps
        return edges

    mond_edges = compute_edges(pc1_cal)

    def assign_group(v, edges):
        idx = np.digitize(v, edges[1:-1], right=True)
        labels = _pc1_group_labels_from_edges(edges, include_marginal=False)
        return np.array([labels[i] for i in idx])

    group_test = assign_group(pc1_test, mond_edges)
    group_labels = _pc1_group_labels_from_edges(mond_edges, include_marginal=True)

    # ======================
    # Collect intervals
    # ======================
    all_method_intervals = {}

    # ===== PCS intervals (external) =====
    if ENABLE_PCS_UQ:
        lo_pcs, hi_pcs = get_pcs_intervals_for_seed(
            split_seed,
            idx_test,
            y_all,
            pcs_paths=PCS_UQ_PATHS,
            require_all=REQUIRE_PCS_ALL_SEEDS,
        )
        if (lo_pcs is not None) and (hi_pcs is not None):
            all_method_intervals[M_PCS] = {"lo": lo_pcs, "hi": hi_pcs}

    # ===== HazardNN training =====
    hz_models = None
    t_edges = None

    if ENABLE_HAZARDNN:
        df_long, uniq_y_train = expand_long(X_train_hz, y_train, hz_feature_names)

        if HZ_USE_TIME_REPARAM:
            df_long["t_mid"] = 0.5 * (df_long["start"] + df_long["end"])
            df_long["log_dt"] = np.log(
                np.maximum(df_long["end"] - df_long["start"], HZ_EPS)
            )
            feats_list = ["t_mid", "log_dt"] + hz_feature_names
        else:
            feats_list = ["start", "end"] + hz_feature_names

        if HZ_ENSEMBLE_K and HZ_ENSEMBLE_K > 1:
            topk = int(min(max(1, HZ_ENSEMBLE_SELECT_TOPK), HZ_ENSEMBLE_K))
            hz_models_info = []

            for k in range(HZ_ENSEMBLE_K):
                seed_k = split_seed * 100 + k
                net_k, mu_k, sg_k, best_val_k, best_ep_k = train_F1_longtable(
                    df_long,
                    feats_list,
                    epochs=500,
                    batch=128,
                    seed_split=seed_k,
                )
                hz_models_info.append(
                    (float(best_val_k), int(k), net_k, mu_k, sg_k, int(best_ep_k))
                )

            hz_models_info.sort(key=lambda x: (x[0], x[1]))
            selected = hz_models_info[:topk]
            hz_models = [(m[2], m[3], m[4]) for m in selected]

            if HZ_ENSEMBLE_VERBOSE_SELECTION:
                print(
                    f"[HazardNN-EnsSelect] seed={split_seed}: "
                    f"trained K={HZ_ENSEMBLE_K}, select TOPK={topk} by best val loss"
                )
                for rank, (v, kk, _, __, ___, ep) in enumerate(selected, 1):
                    print(f"  - rank{rank}: member k={kk}  best_val={v:.6f}  best_epoch={ep}")
        else:
            net_hz, mu_hz, sg_hz, best_val_hz, best_ep_hz = train_F1_longtable(
                df_long,
                feats_list,
                epochs=500,
                batch=128,
                seed_split=split_seed,
            )
            hz_models = (net_hz, mu_hz, sg_hz)
            if HZ_ENSEMBLE_VERBOSE_SELECTION:
                print(
                    f"[HazardNN-Single] seed={split_seed}: "
                    f"best_val={best_val_hz:.6f} best_epoch={best_ep_hz}"
                )

        t_edges = _make_coarse_time_grid(
            uniq_y_train,
            HAZARD_MAX_GRID,
            mode=HZ_TIMEGRID_MODE,
        )

        print("[HazardNN] Precomputing hazard distributions on CAL / TEST ...")
        cal_cumLs, cal_lams, cal_Fgrids = _precompute_hazard_distributions(
            hz_models, t_edges, X_cal_hz
        )
        test_cumLs, test_lams, test_Fgrids = _precompute_hazard_distributions(
            hz_models, t_edges, X_test_hz
        )

        # Calibration PIT values
        pit_cal = np.empty(len(X_cal_hz), dtype="float32")
        for i, yv in enumerate(y_cal):
            pit_cal[i] = _stable_pit(cal_cumLs[i], t_edges, cal_lams[i], float(yv))

        # ----- Optimal z*(x) on TEST set (shortest-width grid search) -----
        z_star_test = np.empty(len(X_test_hz), dtype="float32")
        for i in range(len(X_test_hz)):
            z_star_test[i] = _compute_shortest_z_star_from_cache(
                test_Fgrids[i],
                test_cumLs[i],
                t_edges,
                test_lams[i],
                alpha=ALPHA,
                grid_size=Z_GRID_SIZE,
                eps_u=Z_EPS_U,
            )

        # ----- Optimal z*(x) on CALIBRATION set (used by DCP-NNCDE) -----
        z_star_cal = np.empty(len(X_cal_hz), dtype="float32")
        for i in range(len(X_cal_hz)):
            z_star_cal[i] = _compute_shortest_z_star_from_cache(
                cal_Fgrids[i],
                cal_cumLs[i],
                t_edges,
                cal_lams[i],
                alpha=ALPHA,
                grid_size=Z_GRID_SIZE,
                eps_u=Z_EPS_U,
            )

        # --------------------------------------------------
        # CPI-NNCDE  (was CPI-NNCDE-opt)
        #   1) For each test x, choose z*(x) by shortest width grid search:
        #        z*(x) in argmin_z { Q(z+1-alpha|x) - Q(z|x) }, z in (0, alpha)
        #   2) Calibrate via empirical quantile function G^{-1} of calibration PITs:
        #        [ Q(G^{-1}(z*(x)) | x), Q(G^{-1}(z*(x)+1-alpha) | x) ]
        # --------------------------------------------------
        qvec_lo = np.clip(z_star_test.astype("float64"), Z_EPS_U, 1.0 - Z_EPS_U)
        qvec_hi = np.clip((z_star_test + 1.0 - ALPHA).astype("float64"), Z_EPS_U, 1.0 - Z_EPS_U)

        u_lo_x = np.quantile(pit_cal.astype("float64"), qvec_lo).astype("float32")
        u_hi_x = np.quantile(pit_cal.astype("float64"), qvec_hi).astype("float32")

        u_lo_x = np.clip(u_lo_x, Z_EPS_U, 1.0 - Z_EPS_U)
        u_hi_x = np.clip(u_hi_x, Z_EPS_U, 1.0 - Z_EPS_U)

        lo_cpi = np.empty(len(X_test_hz), dtype="float32")
        hi_cpi = np.empty(len(X_test_hz), dtype="float32")
        for i in range(len(X_test_hz)):
            lo_cpi[i] = _hazard_quantile_from_cache(
                test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_lo_x[i]
            )
            hi_cpi[i] = _hazard_quantile_from_cache(
                test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], u_hi_x[i]
            )
        all_method_intervals[M_CPI] = {"lo": lo_cpi, "hi": hi_cpi}

        # --------------------------------------------------
        # DCP-NNCDE  (was DCP-NNCDE-opt)
        # Calibration:
        #   z_cal_optimal = argmin width on CAL set (per cal point)
        #   center_cal_i  = z_cal_optimal_i + (1 - alpha) / 2
        #   scores_dcp_i  = | PIT_i - center_cal_i |
        #   q_hat_dcp     = empirical quantile of scores_dcp at level
        #                   q_level = (1-alpha)(1+1/n_cal)
        # Test (per test point i):
        #   center_test_i = z_star_test_i + (1 - alpha) / 2
        #   u_lo_dcp_i    = clip(center_test_i - q_hat_dcp, 0, 1)
        #   u_hi_dcp_i    = clip(center_test_i + q_hat_dcp, 0, 1)
        #   invert via NNCDE quantile.
        # --------------------------------------------------
        center_cal = z_star_cal.astype(np.float64) + (1.0 - ALPHA) / 2.0
        scores_dcp = np.abs(pit_cal.astype(np.float64) - center_cal)
        q_hat_dcp = float(np.quantile(scores_dcp, q_level))

        center_test = z_star_test.astype(np.float64) + (1.0 - ALPHA) / 2.0
        u_lo_dcp = np.clip(center_test - q_hat_dcp, 0.0, 1.0)
        u_hi_dcp = np.clip(center_test + q_hat_dcp, 0.0, 1.0)

        lo_dcp = np.empty(len(X_test_hz), dtype="float32")
        hi_dcp = np.empty(len(X_test_hz), dtype="float32")
        for i in range(len(X_test_hz)):
            lo_dcp[i] = _hazard_quantile_from_cache(
                test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], float(u_lo_dcp[i])
            )
            hi_dcp[i] = _hazard_quantile_from_cache(
                test_Fgrids[i], test_cumLs[i], t_edges, test_lams[i], float(u_hi_dcp[i])
            )
        all_method_intervals[M_DCP] = {"lo": lo_dcp, "hi": hi_dcp}

    # ======================
    # Shared standardization for Rescaled + CQR
    # ======================
    x_mu_std, x_sd_std, y_mu_std, y_sd_std = fit_standardizer_xy(X_train_sel, y_train)

    X_train_std = transform_X(X_train_sel, x_mu_std, x_sd_std)
    X_cal_std = transform_X(X_cal_sel, x_mu_std, x_sd_std)
    X_test_std = transform_X(X_test_sel, x_mu_std, x_sd_std)

    y_train_std = transform_y(y_train, y_mu_std, y_sd_std)
    y_cal_std = transform_y(y_cal, y_mu_std, y_sd_std)

    # ======================
    # Rescaled
    # ======================
    mdl_rescaled = train_mean_variance_NN(
        X_train_std, y_train_std, epochs=500, seed_split=split_seed
    )

    with torch.no_grad():
        mean_cal_res, sigma_cal_res = mdl_rescaled(
            torch.tensor(X_cal_std, dtype=torch.float32).to(device)
        )
        mean_cal_res = mean_cal_res.cpu().squeeze().numpy().astype("float32")
        sigma_cal_res = sigma_cal_res.cpu().squeeze().numpy().astype("float32")

    sigma_cal_res = np.clip(sigma_cal_res, RESCALED_SIGMA_MIN, RESCALED_SIGMA_MAX)
    scores_rescaled = np.abs(y_cal_std - mean_cal_res) / (sigma_cal_res + STD_EPS)
    q_rescaled = np.quantile(scores_rescaled, q_level)

    with torch.no_grad():
        mean_test_res, sigma_test_res = mdl_rescaled(
            torch.tensor(X_test_std, dtype=torch.float32).to(device)
        )
        mean_test_res = mean_test_res.cpu().squeeze().numpy().astype("float32")
        sigma_test_res = sigma_test_res.cpu().squeeze().numpy().astype("float32")

    sigma_test_res = np.clip(sigma_test_res, RESCALED_SIGMA_MIN, RESCALED_SIGMA_MAX)

    lo_test_res_std = mean_test_res - q_rescaled * sigma_test_res
    hi_test_res_std = mean_test_res + q_rescaled * sigma_test_res

    lo_test_res = inverse_y(lo_test_res_std, y_mu_std, y_sd_std)
    hi_test_res = inverse_y(hi_test_res_std, y_mu_std, y_sd_std)

    all_method_intervals[M_RESCALED] = {
        "lo": lo_test_res,
        "hi": hi_test_res,
    }

    # ======================
    # CQR
    # ======================
    mdl_cqr = train_quantile_NN(
        X_train_std,
        y_train_std,
        quantiles=[ALPHA / 2, 1 - ALPHA / 2],
        epochs=500,
        seed_split=split_seed,
    )

    with torch.no_grad():
        q_cal_pred_std = mdl_cqr(
            torch.tensor(X_cal_std, dtype=torch.float32).to(device)
        ).cpu().numpy()

    q_lo_cal_std = q_cal_pred_std[:, 0].astype("float32")
    q_hi_cal_std = q_cal_pred_std[:, 1].astype("float32")
    q_lo_cal_std, q_hi_cal_std = (
        np.minimum(q_lo_cal_std, q_hi_cal_std),
        np.maximum(q_lo_cal_std, q_hi_cal_std),
    )

    scores_cqr = np.maximum(q_lo_cal_std - y_cal_std, y_cal_std - q_hi_cal_std)
    q_cqr = np.quantile(scores_cqr, q_level)

    with torch.no_grad():
        q_test_pred_std = mdl_cqr(
            torch.tensor(X_test_std, dtype=torch.float32).to(device)
        ).cpu().numpy()

    q_lo_test_std = q_test_pred_std[:, 0].astype("float32")
    q_hi_test_std = q_test_pred_std[:, 1].astype("float32")
    q_lo_test_std, q_hi_test_std = (
        np.minimum(q_lo_test_std, q_hi_test_std),
        np.maximum(q_lo_test_std, q_hi_test_std),
    )

    lo_test_cqr_std = q_lo_test_std - q_cqr
    hi_test_cqr_std = q_hi_test_std + q_cqr

    lo_test_cqr = inverse_y(lo_test_cqr_std, y_mu_std, y_sd_std)
    hi_test_cqr = inverse_y(hi_test_cqr_std, y_mu_std, y_sd_std)

    all_method_intervals[M_CQR] = {
        "lo": lo_test_cqr,
        "hi": hi_test_cqr,
    }

    # ======================
    # Residual
    # ======================
    if mdl_residual is not None:
        y_hat_cal_resid = mdl_residual.predict(X_cal_sel)
        scores_resid = np.abs(y_cal - y_hat_cal_resid)
        q_resid = np.quantile(scores_resid, q_level)
        y_hat_test_resid = mdl_residual.predict(X_test_sel)
        all_method_intervals[M_RESIDUAL] = {
            "lo": y_hat_test_resid - q_resid,
            "hi": y_hat_test_resid + q_resid,
        }
        print(f"[Residual] Added intervals using backend={residual_backend}")
    else:
        print("[Residual] No intervals added.")

    # ======================
    # Group evaluation
    # ======================
    all_group_results = []
    test_range = float(np.max(y_test) - np.min(y_test))
    if test_range <= 0:
        test_range = 1.0

    groups_to_process = [("Marginal", np.ones(len(y_test), dtype=bool))]
    for g in group_labels[1:]:
        groups_to_process.append((g, group_test == g))

    for gname, mask in groups_to_process:
        if np.sum(mask) == 0:
            continue
        y_test_g = y_test[mask]
        X_test_g = X_test_sel[mask]
        n_g = int(len(y_test_g))

        for method_name, itv in all_method_intervals.items():
            lo_g = itv["lo"][mask]
            hi_g = itv["hi"][mask]

            valid = np.isfinite(lo_g) & np.isfinite(hi_g)
            if valid.sum() == 0:
                continue

            yy = y_test_g[valid]
            lo = lo_g[valid]
            hi = hi_g[valid]

            cov = float(np.mean((yy >= lo) & (yy <= hi)))
            width = (hi - lo).astype(np.float64)

            width_mean = float(np.mean(width))
            width_norm_mean = float(np.mean(width / test_range))

            # L1-ERT: conditional coverage metric
            l1_ert = np.nan
            if _HAS_COVMETRICS:
                try:
                    cover_bool = ((yy >= lo) & (yy <= hi)).astype(float)
                    X_g_valid = X_test_g[valid]
                    l1_ert = float(ERT().evaluate(X_g_valid, cover_bool, ALPHA))
                except Exception as e:
                    print(f"[L1-ERT] {method_name}/{gname}: {e}")

            all_group_results.append(
                {
                    "seed": int(split_seed),
                    "group": gname,
                    "method": method_name,
                    "coverage_mean": cov,
                    "width_mean": width_mean,
                    "width_norm_mean": width_norm_mean,
                    "l1_ert": l1_ert,
                    "n_group": n_g,
                    "n_valid": int(valid.sum()),
                    "test_size": int(len(y_test)),
                }
            )

    return all_group_results


# ======================
# Visualization
# ======================
def _scatter_cov_vs_width_by_pc1_groups(
    df,
    out_dir,
    target_cov,
    hide_methods=None,
    group_names=None,
    fname_prefix="cov_vs_width",
):
    if hide_methods is None:
        hide_methods = []

    if group_names is None:
        group_names = [g for g in _infer_pc1_group_order_from_df(df) if str(g).startswith("PC1_Group_")]

    marker_map = {
        M_RESIDUAL: "^",
        M_RESCALED: "v",
        M_CQR: "o",
        M_PCS: "X",
        M_CPI: "s",
        M_DCP: "P",
    }

    x_col = "width_mean" if ("width_mean" in df.columns) else "width_norm_mean"
    if x_col != "width_mean":
        print("[WARN] CSV 没有 width_mean，scatter 将使用 width_norm_mean 作为横轴（归一化宽度）。")

    df2 = df[~df["method"].isin(hide_methods)].copy()

    methods_present = [m for m in METHOD_ORDER_FINAL if m in df2["method"].unique()] + [
        m for m in df2["method"].unique() if m not in METHOD_ORDER_FINAL
    ]

    for g in group_names:
        sub = df2[df2["group"] == g].copy()
        if len(sub) == 0:
            continue

        plt.figure(figsize=(8.5, 5.5), dpi=180)
        plt.axhline(target_cov, linestyle="--", linewidth=1.5, color="black", alpha=0.8)

        for m in methods_present:
            row = sub[sub["method"] == m]
            if len(row) == 0:
                continue

            x = float(row.iloc[0][x_col])
            y = float(row.iloc[0]["coverage_mean"])
            mk = marker_map.get(m, "o")
            plt.scatter(x, y, s=220, marker=mk, alpha=0.9, label=m)

        plt.title(f"{g}: Coverage vs Mean Width", fontsize=14)
        plt.xlabel("Mean Width" if x_col == "width_mean" else "Mean Normalized Width")
        plt.ylabel("Coverage")

        xs = sub[x_col].values.astype(float)
        xmin, xmax = float(np.min(xs)), float(np.max(xs))
        pad = 0.08 * (xmax - xmin + 1e-12)
        plt.xlim(xmin - pad, xmax + pad)

        plt.ylim(0.8, 1.0)
        plt.grid(alpha=0.25)

        plt.legend(
            loc="upper center",
            bbox_to_anchor=(0.5, -0.18),
            ncol=2,
            frameon=False,
        )

        plt.tight_layout()
        out_path = Path(out_dir) / f"{fname_prefix}_{g}.png"
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        print(f"Saved Scatter plot: {out_path}")
        plt.show()


def visualize_results(
    csv_path,
    target_alpha=0.10,
    save_dir=".",
    hide_methods_in_plots=None,
    show_bar_labels=False,
    label_mode="marginal",
    make_scatter_cov_width=True,
):
    try:
        results_df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: Results file not found at {csv_path}")
        return

    results_df = rename_methods_in_df(results_df)
    results_df = results_df.drop_duplicates(subset=["group", "method"])

    out_dir = Path(save_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== Averaged Results Table (All Methods) ===")
    print(results_df.to_string(index=False, float_format="%.4f"))

    for c in ["coverage_std", "width_std", "width_norm_std"]:
        if c in results_df.columns:
            results_df[c] = results_df[c].fillna(0.0)

    if hide_methods_in_plots is None:
        hide_methods_in_plots = []

    plot_df = results_df[~results_df["method"].isin(hide_methods_in_plots)].copy()

    group_order = _infer_pc1_group_order_from_df(plot_df)
    plot_df["group"] = pd.Categorical(plot_df["group"], categories=group_order, ordered=True)
    plot_df["method"] = pd.Categorical(plot_df["method"], categories=METHOD_ORDER_FINAL, ordered=True)
    plot_df = plot_df.sort_values(["group", "method"])

    target_coverage = 1.0 - target_alpha
    methods = [m for m in METHOD_ORDER_FINAL if m in plot_df["method"].astype(str).unique()]
    groups = [g for g in group_order if g in plot_df["group"].astype(str).unique()]
    x = np.arange(len(groups))
    width = 0.85 / max(1, len(methods))

    def _maybe_label(bars, vals, gname, fmt):
        if not show_bar_labels:
            return
        if (label_mode == "marginal") and (gname != "Marginal"):
            return
        labels = [("" if np.isnan(v) else format(v, fmt)) for v in vals]
        plt.bar_label(
            bars,
            labels=labels,
            padding=2,
            fontsize=8,
            rotation=90,
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none", pad=0.2),
        )

    # ---- Coverage bar plot
    plt.figure(figsize=(14, 6), dpi=160)
    for i, m in enumerate(methods):
        sub = plot_df[plot_df["method"] == m].set_index("group").reindex(groups)
        y = sub["coverage_mean"].values
        err = sub["coverage_std"].values if "coverage_std" in sub.columns else None

        xpos = x + i * width - (len(methods) - 1) * width / 2
        bars = plt.bar(xpos, y, width=width, label=m, yerr=err, capsize=3)

        for gi, gname in enumerate(groups):
            _maybe_label([bars[gi]], [y[gi]], gname, ".2f")

    plt.axhline(
        target_coverage,
        linestyle="--",
        linewidth=1,
        color="red",
        label=f"Target={target_coverage:.2f}",
    )
    plt.title(f"Mean Coverage across PC1 Groups (Target={target_coverage:.2f})")
    plt.ylabel("Mean Coverage")
    plt.xlabel("Group (by PC1 quantiles)")
    plt.xticks(x, groups)
    plt.ylim(0, 1.03)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    cov_path = out_dir / (Path(csv_path).stem + "_coverage.png")
    plt.savefig(cov_path, dpi=300)
    print(f"Saved Coverage plot: {cov_path}")
    plt.show()

    # ---- Width bar plot
    plt.figure(figsize=(14, 6), dpi=160)
    for i, m in enumerate(methods):
        sub = plot_df[plot_df["method"] == m].set_index("group").reindex(groups)
        y = sub["width_norm_mean"].values
        err = sub["width_norm_std"].values if "width_norm_std" in sub.columns else None

        xpos = x + i * width - (len(methods) - 1) * width / 2
        bars = plt.bar(xpos, y, width=width, label=m, yerr=err, capsize=3)

        for gi, gname in enumerate(groups):
            _maybe_label([bars[gi]], [y[gi]], gname, ".3f")

    plt.title("Mean Normalized Prediction Interval Width across PC1 Groups")
    plt.ylabel("Mean Normalized Width")
    plt.xlabel("Group (by PC1 quantiles)")
    plt.xticks(x, groups)
    plt.grid(axis="y", alpha=0.25)
    plt.legend(bbox_to_anchor=(1.02, 1.0), loc="upper left", borderaxespad=0.0)
    plt.tight_layout(rect=[0, 0, 0.82, 1])

    wid_path = out_dir / (Path(csv_path).stem + "_width.png")
    plt.savefig(wid_path, dpi=300)
    print(f"Saved Normalized Width plot: {wid_path}")
    plt.show()

    if make_scatter_cov_width:
        _scatter_cov_vs_width_by_pc1_groups(
            results_df,
            out_dir,
            target_cov=target_coverage,
            hide_methods=hide_methods_in_plots,
            group_names=None,
            fname_prefix=Path(csv_path).stem + "_cov_vs_width",
        )


# ======================
# PCS-only (append to existing CSV without retraining heavy methods)
# ======================
def compute_pcs_summary_only(X_all, y_all, feature_names, seeds):
    all_rows = []

    for seed in seeds:
        set_all_seeds(seed)
        idx_all = np.arange(len(y_all), dtype=int)

        idx_test_forced = None
        if USE_PCS_TEST_SPLIT and ENABLE_PCS_UQ:
            idx_test_forced = get_forced_test_indices_from_pcs(
                seed,
                y_all,
                pcs_paths=PCS_UQ_PATHS,
                require_all=REQUIRE_PCS_ALL_SEEDS,
            )

        if idx_test_forced is not None:
            idx_test = idx_test_forced
            set_test = set(int(i) for i in idx_test.tolist())
            idx_rest = np.array([i for i in idx_all.tolist() if int(i) not in set_test], dtype=int)
        else:
            idx_rest, idx_test = train_test_split(idx_all, test_size=TEST_FRAC, random_state=seed)

        n = len(idx_all)
        test_frac_actual = len(idx_test) / max(1, n)
        rest_frac = 1.0 - test_frac_actual
        if rest_frac <= 0:
            continue

        cal_frac_within_rest = float(np.clip(CAL_FRAC / rest_frac, 0.0, 0.99))
        idx_train, idx_cal = train_test_split(
            idx_rest,
            test_size=cal_frac_within_rest,
            random_state=seed,
        )

        X_train, y_train = X_all[idx_train], y_all[idx_train]
        X_cal, y_cal = X_all[idx_cal], y_all[idx_cal]
        X_test, y_test = X_all[idx_test], y_all[idx_test]

        sel_idx, _ = select_features(X_train, y_train, feature_names, random_state=seed, verbose=False)
        X_train_sel = X_train[:, sel_idx]
        X_cal_sel = X_cal[:, sel_idx]
        X_test_sel = X_test[:, sel_idx]

        pca_scaler = StandardScaler()
        X_train_sel_s = pca_scaler.fit_transform(X_train_sel)
        pca = PCA(n_components=1, random_state=seed).fit(X_train_sel_s)
        pc1_cal = pca.transform(pca_scaler.transform(X_cal_sel)).flatten()
        pc1_test = pca.transform(pca_scaler.transform(X_test_sel)).flatten()

        def compute_edges(v, n_groups=PC1_N_GROUPS):
            qs = np.linspace(0, 1, n_groups + 1)
            edges = np.quantile(v, qs).astype(float)

            rng = float(np.max(edges) - np.min(edges))
            eps = max(1e-12, 1e-6 * (rng if rng > 0 else 1.0))
            for i in range(1, len(edges)):
                if edges[i] <= edges[i - 1]:
                    edges[i] = edges[i - 1] + eps

            edges[0] -= eps
            edges[-1] += eps
            return edges

        mond_edges = compute_edges(pc1_cal)

        def assign_group(v, edges):
            idx = np.digitize(v, edges[1:-1], right=True)
            labels = _pc1_group_labels_from_edges(edges, include_marginal=False)
            return np.array([labels[i] for i in idx])

        group_test = assign_group(pc1_test, mond_edges)
        group_labels = _pc1_group_labels_from_edges(mond_edges, include_marginal=True)

        lo, hi = get_pcs_intervals_for_seed(
            seed,
            idx_test,
            y_all,
            pcs_paths=PCS_UQ_PATHS,
            require_all=REQUIRE_PCS_ALL_SEEDS,
        )
        if lo is None or hi is None:
            continue

        test_range = float(np.max(y_test) - np.min(y_test))
        if test_range <= 0:
            test_range = 1.0

        groups_to_process = [("Marginal", np.ones(len(y_test), dtype=bool))]
        for g in group_labels[1:]:
            groups_to_process.append((g, group_test == g))

        for gname, mask in groups_to_process:
            if np.sum(mask) == 0:
                continue

            yy = y_test[mask]
            lo_g = lo[mask]
            hi_g = hi[mask]
            valid = np.isfinite(lo_g) & np.isfinite(hi_g)
            if valid.sum() == 0:
                continue

            yy2 = yy[valid]
            lo2 = lo_g[valid]
            hi2 = hi_g[valid]
            cov = float(np.mean((yy2 >= lo2) & (yy2 <= hi2)))
            w = (hi2 - lo2).astype(np.float64)

            all_rows.append(
                dict(
                    seed=int(seed),
                    group=gname,
                    method=M_PCS,
                    coverage_mean=cov,
                    width_mean=float(np.mean(w)),
                    width_norm_mean=float(np.mean(w / test_range)),
                    n_group=int(np.sum(mask)),
                    n_valid=int(valid.sum()),
                    n_test=int(len(y_test)),
                )
            )

    if len(all_rows) == 0:
        print("[WARN] No PCS rows computed (no PCS files found). Skipping PCS summary.")
        return pd.DataFrame()

    df = pd.DataFrame(all_rows)
    summary = (
        df.groupby(["group", "method"])
        .agg(
            coverage_mean=("coverage_mean", "mean"),
            coverage_std=("coverage_mean", "std"),
            width_mean=("width_mean", "mean"),
            width_std=("width_mean", "std"),
            width_norm_mean=("width_norm_mean", "mean"),
            width_norm_std=("width_norm_mean", "std"),
            n_group_mean=("n_group", "mean"),
            n_valid_mean=("n_valid", "mean"),
            n_test_mean=("n_test", "mean"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    for c in ["coverage_std", "width_std", "width_norm_std"]:
        summary[c] = summary[c].fillna(0.0)
    return summary


# ======================
# Main
# ======================
def main():
    print("Device:", device)
    print(f"_HAS_XGB={_HAS_XGB}, _HAS_LGB={_HAS_LGB}")

    X_all, y_all, feature_names = load_real_xy_all_features(X_PATH, Y_PATH)
    n, p = X_all.shape
    print(f"[Data] n={n}, p={p} (TEST_FRAC={TEST_FRAC}, CAL_FRAC={CAL_FRAC}, seeds={len(SEEDS)})")

    if FS_ENABLE:
        print(
            f"[FS] method={FS_METHOD} boot={FS_BOOTSTRAPS} "
            f"subsample={FS_SUBSAMPLE} thr={FS_FREQ_THR} "
            f"max_features={FS_MAX_FEATURES}"
        )

    print(f"[PC1] Quantile groups (excluding Marginal): PC1_N_GROUPS={PC1_N_GROUPS}")

    if ENABLE_PCS_UQ and USE_PCS_TEST_SPLIT:
        print("[PCS_TEST_SPLIT] ENABLED: will force test indices to match PCS test_index when available.")

    if ENABLE_HAZARDNN and (HZ_ENSEMBLE_K and HZ_ENSEMBLE_K > 1):
        topk = int(min(max(1, HZ_ENSEMBLE_SELECT_TOPK), HZ_ENSEMBLE_K))
        print(f"[HazardNN] Ensemble enabled: train K={HZ_ENSEMBLE_K} then select TOPK={topk} by best val loss.")

    print("[Methods] Keep only:", ", ".join(METHOD_ORDER_FINAL))

    if RUN_TRAINING:
        all_results_list = []

        for i, seed in enumerate(SEEDS, 1):
            print(f"\n=== Split {i}/{len(SEEDS)} (seed={seed}) ===")
            group_results_for_seed = run_one_split(X_all, y_all, feature_names, split_seed=seed)
            all_results_list.extend(group_results_for_seed)

            temp_df = pd.DataFrame(group_results_for_seed)
            temp_df = rename_methods_in_df(temp_df)
            marginal_df = temp_df[temp_df["group"] == "Marginal"]

            cols = ["method", "coverage_mean", "width_norm_mean", "width_mean", "l1_ert", "n_valid", "test_size"]
            cols = [c for c in cols if c in marginal_df.columns]
            if len(marginal_df):
                print(marginal_df.to_string(index=False, columns=cols, float_format="%.4f"))

        print("\n=== Averaged over seeds ===")
        summary_df = pd.DataFrame(all_results_list)
        summary_df = rename_methods_in_df(summary_df)

        agg_dict = dict(
            coverage_mean=("coverage_mean", "mean"),
            coverage_std=("coverage_mean", "std"),
            width_mean=("width_mean", "mean"),
            width_std=("width_mean", "std"),
            width_norm_mean=("width_norm_mean", "mean"),
            width_norm_std=("width_norm_mean", "std"),
            n_group_mean=("n_group", "mean"),
            n_valid_mean=("n_valid", "mean"),
            test_size_mean=("test_size", "mean"),
            n_seeds=("seed", "nunique"),
        )
        if "l1_ert" in summary_df.columns:
            agg_dict["l1_ert_mean"] = ("l1_ert", "mean")
            agg_dict["l1_ert_std"] = ("l1_ert", "std")

        final_summary = (
            summary_df.groupby(["group", "method"])
            .agg(**agg_dict)
            .reset_index()
        )

        for c in ["coverage_std", "width_std", "width_norm_std"]:
            final_summary[c] = final_summary[c].fillna(0.0)
        if "l1_ert_std" in final_summary.columns:
            final_summary["l1_ert_std"] = final_summary["l1_ert_std"].fillna(0.0)

        group_order = _infer_pc1_group_order_from_df(final_summary)
        final_summary["group"] = pd.Categorical(final_summary["group"], categories=group_order, ordered=True)
        final_summary["method"] = pd.Categorical(final_summary["method"], categories=METHOD_ORDER_FINAL, ordered=True)
        final_summary = final_summary.sort_values(["group", "method"]).reset_index(drop=True)

        print(final_summary.to_string(index=False, float_format="%.4f"))

        out_dir = os.path.dirname(X_PATH) if os.path.dirname(X_PATH) else "."
        ens_top = (
            min(HZ_ENSEMBLE_K, max(1, HZ_ENSEMBLE_SELECT_TOPK))
            if (HZ_ENSEMBLE_K and HZ_ENSEMBLE_K > 1)
            else 1
        )
        tag = (
            f"realdata_fs_{FS_METHOD if FS_ENABLE else 'none'}_"
            f"hazEnsK_{HZ_ENSEMBLE_K}_"
            f"hazEnsTop_{ens_top}_"
            f"hzmeta_{int(HZ_USE_XGB_META)}_"
            f"shortestCalGrid_{Z_GRID_SIZE}_"
            f"pcs_{int(ENABLE_PCS_UQ)}_"
            f"pcsTest_{int(USE_PCS_TEST_SPLIT)}_"
            f"pc1g_{PC1_N_GROUPS}"
        )
        results_csv_path = os.path.join(out_dir, f"summary_seeds_777_786_{tag}.csv")
        final_summary.to_csv(results_csv_path, index=False)
        print(f"\nSaved: {results_csv_path}")
    else:
        results_csv_path = EXISTING_CSV_PATH
        print(f"[Skip training] Use existing CSV: {results_csv_path}")

        if ENABLE_PCS_UQ and PCS_APPEND_TO_EXISTING_CSV:
            print("[PCS] Computing PCS summary and appending to existing CSV ...")
            pcs_summary = compute_pcs_summary_only(X_all, y_all, feature_names, SEEDS)

            base_df = pd.read_csv(results_csv_path)
            base_df = rename_methods_in_df(base_df)
            base_df = base_df[base_df["method"] != M_PCS].copy()

            combined = pd.concat([base_df, pcs_summary], axis=0, ignore_index=True)

            out_dir = os.path.dirname(results_csv_path) if os.path.dirname(results_csv_path) else "."
            combined_path = os.path.join(out_dir, Path(results_csv_path).stem + "_with_pcs.csv")
            combined.to_csv(combined_path, index=False)
            results_csv_path = combined_path
            print(f"[PCS] Saved combined CSV: {results_csv_path}")

    viz_dir = Path(os.path.dirname(results_csv_path) if results_csv_path else ".") / "pc1_group_outputs"
    visualize_results(
        results_csv_path,
        target_alpha=ALPHA,
        save_dir=str(viz_dir),
        hide_methods_in_plots=[],
        show_bar_labels=False,
        label_mode="marginal",
        make_scatter_cov_width=True,
    )


if __name__ == "__main__":
    main()