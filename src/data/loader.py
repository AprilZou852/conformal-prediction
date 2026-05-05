"""
Real data loading and PCS-UQ integration utilities.
"""

import re
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd


def load_real_data(x_path: str, y_path: str) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Load real data from CSV files with missing value handling.

    Parameters
    ----------
    x_path : str
        Path to X (features) CSV file.
    y_path : str
        Path to y (response) CSV file.

    Returns
    -------
    X : np.ndarray
        Feature matrix, shape (n, p), dtype float32.
    y : np.ndarray
        Response vector, shape (n,), dtype float32.
    feature_names : list
        List of feature names.
    """
    # Load X
    Xdf = pd.read_csv(x_path)
    for c in Xdf.columns:
        Xdf[c] = pd.to_numeric(Xdf[c], errors="coerce")
    Xdf = Xdf.dropna(axis=1, how="all").copy()
    Xdf = Xdf.apply(lambda col: col.fillna(col.mean()), axis=0)

    # Load y
    try:
        ydf = pd.read_csv(y_path, header=None)
        y = pd.to_numeric(ydf.iloc[:, 0], errors="coerce")
    except Exception:
        ydf = pd.read_csv(y_path)
        y = pd.to_numeric(ydf.iloc[:, 0], errors="coerce")

    # Align sizes
    n = min(len(Xdf), len(y))
    Xdf = Xdf.iloc[:n, :].copy()
    y = y.iloc[:n].copy()

    # Remove missing y values
    mask = ~y.isna()
    Xdf = Xdf.loc[mask].reset_index(drop=True)
    y = y.loc[mask].reset_index(drop=True)

    # Create feature names
    p = Xdf.shape[1]
    feature_names = [f"x{i+1}" for i in range(p)]
    Xdf.columns = feature_names

    return Xdf.values.astype("float32"), y.values.astype("float32"), feature_names


def _normalize_colname(s: str) -> str:
    """Normalize column name for flexible matching."""
    s = str(s).strip().lower()
    s = s.replace(" ", "_")
    s = s.replace("-", "_")
    s = re.sub(r"__+", "_", s)
    return s


def load_pcs_predictions(path: str) -> pd.DataFrame:
    """
    Load PCS-UQ predictions from file.

    Flexible loading supporting various column naming conventions.

    Parameters
    ----------
    path : str
        Path to PCS predictions file (.csv, .xlsx, .xls, .tsv).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: test_index, y_test, y_pred_lb, y_pred_ub.
    """
    if path is None:
        raise FileNotFoundError("PCS path is None")
    if not Path(path).exists():
        raise FileNotFoundError(f"PCS file not found: {path}")

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
        raise ValueError(f"PCS file missing columns {missing}. Columns seen: {list(df.columns)}")

    out = df[[c_test_index, c_y_test, c_lb, c_ub]].copy()
    out.columns = ["test_index", "y_test", "y_pred_lb", "y_pred_ub"]

    for c in ["test_index", "y_test", "y_pred_lb", "y_pred_ub"]:
        out[c] = pd.to_numeric(out[c], errors="coerce")

    out = out.dropna(subset=["test_index", "y_pred_lb", "y_pred_ub"]).copy()
    out["test_index"] = out["test_index"].astype(int)
    out = out.drop_duplicates(subset=["test_index"], keep="first").reset_index(drop=True)
    return out


def _infer_test_index_base(pcs_df: pd.DataFrame, y_all: np.ndarray, tol: float = 1e-6) -> int:
    """Infer whether test indices are 0-based or 1-based."""
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


def get_pcs_test_indices(
    seed: int, y_all: np.ndarray, pcs_paths: dict, require_all: bool = False
) -> Optional[np.ndarray]:
    """
    Extract forced test indices from PCS predictions.

    Parameters
    ----------
    seed : int
        Seed identifier.
    y_all : np.ndarray
        Full response vector.
    pcs_paths : dict
        Mapping from seed to PCS file path.
    require_all : bool, default=False
        If True, raise error if seed not in pcs_paths.

    Returns
    -------
    np.ndarray or None
        Forced test indices, or None if unavailable.
    """
    if seed not in pcs_paths:
        msg = f"[PCS_TEST_SPLIT] seed={seed}: no PCS file in pcs_paths."
        if require_all:
            raise FileNotFoundError(msg)
        print(msg + " -> falling back to sklearn train_test_split.")
        return None

    pcs_df = load_pcs_predictions(pcs_paths[seed])
    base = _infer_test_index_base(pcs_df, y_all)
    idx0 = (pcs_df["test_index"].values.astype(int) - base).astype(int)

    n = len(y_all)
    ok = (idx0 >= 0) & (idx0 < n)
    if ok.sum() != len(idx0):
        bad = idx0[~ok]
        msg = f"[PCS_TEST_SPLIT] seed={seed}: {len(bad)} indices out of range (n={n})."
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
        print("[WARN] " + msg + " -> falling back to sklearn train_test_split.")
        return None

    print(f"[PCS_TEST_SPLIT] seed={seed}: forced test_size={len(idx_test)} (base={base}).")
    return idx_test


def get_pcs_intervals(
    seed: int,
    idx_test: np.ndarray,
    y_all: np.ndarray,
    pcs_paths: dict,
    require_all: bool = False,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Get PCS prediction intervals for test indices.

    Parameters
    ----------
    seed : int
        Seed identifier.
    idx_test : np.ndarray
        Test indices.
    y_all : np.ndarray
        Full response vector.
    pcs_paths : dict
        Mapping from seed to PCS file path.
    require_all : bool, default=False
        If True, raise error if seed not in pcs_paths.

    Returns
    -------
    lo : np.ndarray or None
        Lower bounds, shape (len(idx_test),), or None.
    hi : np.ndarray or None
        Upper bounds, shape (len(idx_test),), or None.
    """
    if seed not in pcs_paths:
        msg = f"[PCS_UQ] seed={seed}: no PCS file in pcs_paths."
        if require_all:
            raise FileNotFoundError(msg)
        print(msg + " -> skipping PCS_UQ for this seed.")
        return None, None

    pcs_path = pcs_paths[seed]
    pcs_df = load_pcs_predictions(pcs_path)
    base = _infer_test_index_base(pcs_df, y_all)

    pcs_df = pcs_df.copy()
    pcs_df["idx0"] = pcs_df["test_index"].values.astype(int) - base

    set_pcs = set(pcs_df["idx0"].tolist())
    set_test = set(int(i) for i in idx_test.tolist())
    if set_pcs != set_test:
        inter = sorted(set_pcs.intersection(set_test))
        msg = (
            f"[PCS_UQ] seed={seed}: test index set mismatch. "
            f"pcs_size={len(set_pcs)} split_test_size={len(set_test)} inter={len(inter)}"
        )
        if require_all:
            raise ValueError(msg)
        print("[WARN] " + msg + " -> evaluating on intersection only.")

    pcs_map_lb = pcs_df.set_index("idx0")["y_pred_lb"].to_dict()
    pcs_map_ub = pcs_df.set_index("idx0")["y_pred_ub"].to_dict()

    lo = np.full(len(idx_test), np.nan, dtype="float32")
    hi = np.full(len(idx_test), np.nan, dtype="float32")
    for k, idx in enumerate(idx_test.tolist()):
        if idx in pcs_map_lb and idx in pcs_map_ub:
            lo[k] = float(pcs_map_lb[idx])
            hi[k] = float(pcs_map_ub[idx])

    return lo, hi
