"""
Feature selection methods for conformal prediction.

Implements stability-based methods (ElasticNet, LightGBM) and Lasso.
"""

from typing import Tuple, List, Optional

import numpy as np
from sklearn.linear_model import ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False


# Default configuration
FS_BOOTSTRAPS = 50
FS_SUBSAMPLE = 0.7
FS_FREQ_THR = 0.6
FS_MAX_FEATURES = None
FS_COEF_THR = 1e-8
FS_ENET_L1RATIO = [0.1, 0.5, 0.9]
FS_ENET_ALPHAS = np.logspace(-4, 1, 50)


def _fallback_topk_by_corr(
    X_train: np.ndarray, y_train: np.ndarray, feature_names: List[str], k: int = 1
) -> Tuple[List[int], List[str]]:
    """
    Fallback feature selection: top-k by absolute correlation with y.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n, p).
    y_train : np.ndarray
        Training response, shape (n,).
    feature_names : List[str]
        Feature names.
    k : int, default=1
        Number of features to select.

    Returns
    -------
    sel_idx : List[int]
        Selected feature indices.
    sel_names : List[str]
        Selected feature names.
    """
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
    order = np.argsort(corrs)[::-1][: max(1, k)]
    return [int(i) for i in order], [feature_names[i] for i in order]


def _select_stability_elasticnet(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    random_state: int = 0,
    alphas: Optional[np.ndarray] = None,
    l1_ratios: Optional[list] = None,
    n_boot: int = FS_BOOTSTRAPS,
    subsample: float = FS_SUBSAMPLE,
    freq_thr: float = FS_FREQ_THR,
    max_features: Optional[int] = FS_MAX_FEATURES,
    coef_thr: float = FS_COEF_THR,
    verbose: bool = True,
) -> Tuple[List[int], List[str]]:
    """
    Select features via ElasticNet stability selection.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n, p).
    y_train : np.ndarray
        Training response, shape (n,).
    feature_names : List[str]
        Feature names.
    random_state : int, default=0
        Random seed.
    alphas : np.ndarray, optional
        Regularization strengths. Defaults to logspace(-4, 1, 50).
    l1_ratios : list, optional
        L1 ratio values. Defaults to [0.1, 0.5, 0.9].
    n_boot : int, default=50
        Number of bootstrap samples.
    subsample : float, default=0.7
        Subsample fraction.
    freq_thr : float, default=0.6
        Frequency threshold for inclusion.
    max_features : int, optional
        Maximum features to keep.
    coef_thr : float, default=1e-8
        Coefficient threshold for nonzero.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    sel_idx : List[int]
        Selected feature indices.
    sel_names : List[str]
        Selected feature names.
    """
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
        top = order[: max(1, min(5, p))].tolist()
        sel_idx = top
        if np.all(freq[top] == 0):
            sel_idx, _ = _fallback_topk_by_corr(X_train, y_train, feature_names, k=1)

    if max_features is not None and len(sel_idx) > max_features:
        sel_idx = sorted(sel_idx, key=lambda j: (-freq[j], j))[: max_features]

    sel_names = [feature_names[i] for i in sel_idx]
    if verbose:
        shown = ", ".join(f"{feature_names[i]}({freq[i]:.2f})" for i in sel_idx[:20])
        print(f"[FS/Stability-ENet] kept={len(sel_idx)} thr={freq_thr} -> {shown}")

    return sel_idx, sel_names


def _select_stability_lgbm(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    random_state: int = 0,
    n_boot: int = FS_BOOTSTRAPS,
    subsample: float = FS_SUBSAMPLE,
    max_features: Optional[int] = FS_MAX_FEATURES,
    verbose: bool = True,
) -> Tuple[List[int], List[str]]:
    """
    Select features via LightGBM/RandomForest stability selection.

    Falls back to RandomForest if LightGBM unavailable.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n, p).
    y_train : np.ndarray
        Training response, shape (n,).
    feature_names : List[str]
        Feature names.
    random_state : int, default=0
        Random seed.
    n_boot : int, default=50
        Number of bootstrap samples.
    subsample : float, default=0.7
        Subsample fraction.
    max_features : int, optional
        Maximum features to keep.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    sel_idx : List[int]
        Selected feature indices.
    sel_names : List[str]
        Selected feature names.
    """
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
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    random_state: int = 0,
    alphas: Optional[np.ndarray] = None,
    coef_thr: float = FS_COEF_THR,
    max_features: Optional[int] = FS_MAX_FEATURES,
    verbose: bool = True,
) -> Tuple[List[int], List[str]]:
    """
    Select features via LassoCV.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n, p).
    y_train : np.ndarray
        Training response, shape (n,).
    feature_names : List[str]
        Feature names.
    random_state : int, default=0
        Random seed.
    alphas : np.ndarray, optional
        Regularization strengths. Defaults to logspace(-4, 1, 50).
    coef_thr : float, default=1e-8
        Coefficient threshold for nonzero.
    max_features : int, optional
        Maximum features to keep.
    verbose : bool, default=True
        Print progress.

    Returns
    -------
    sel_idx : List[int]
        Selected feature indices.
    sel_names : List[str]
        Selected feature names.
    """
    if alphas is None:
        alphas = FS_ENET_ALPHAS

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
        sel_idx = [sel_idx[i] for i in order[: max_features]]

    sel_names = [feature_names[i] for i in sel_idx]
    if verbose:
        print(f"[FS/LassoCV] kept={len(sel_idx)} -> {sel_names}")

    return sel_idx, sel_names


def select_features(
    X_train: np.ndarray,
    y_train: np.ndarray,
    feature_names: List[str],
    method: str = "stability_lgbm",
    random_state: int = 0,
    verbose: bool = True,
    **kwargs,
) -> Tuple[List[int], List[str]]:
    """
    Dispatcher for feature selection methods.

    Parameters
    ----------
    X_train : np.ndarray
        Training features, shape (n, p).
    y_train : np.ndarray
        Training response, shape (n,).
    feature_names : List[str]
        Feature names.
    method : str, default="stability_lgbm"
        Selection method: "stability_enet", "stability_lgbm", or "lasso".
    random_state : int, default=0
        Random seed.
    verbose : bool, default=True
        Print progress.
    **kwargs
        Additional arguments passed to selection function.

    Returns
    -------
    sel_idx : List[int]
        Selected feature indices.
    sel_names : List[str]
        Selected feature names.
    """
    if method == "stability_enet":
        return _select_stability_elasticnet(
            X_train, y_train, feature_names, random_state=random_state, verbose=verbose, **kwargs
        )
    elif method == "stability_lgbm":
        return _select_stability_lgbm(
            X_train, y_train, feature_names, random_state=random_state, verbose=verbose, **kwargs
        )
    else:
        return _select_lasso(
            X_train, y_train, feature_names, random_state=random_state, verbose=verbose, **kwargs
        )
