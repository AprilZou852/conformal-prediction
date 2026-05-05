"""
Residual regressor training with fallback chain: XGBoost -> LightGBM -> RandomForest.
"""

from typing import Tuple
import os
import numpy as np
import torch

# Check for optional dependencies
try:
    import xgboost as xgb
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    import lightgbm as lgb
    _HAS_LGB = True
except Exception:
    _HAS_LGB = False

from sklearn.ensemble import RandomForestRegressor


# Configuration
XGB_WEAK_LEVEL = "medium"
REQUIRE_XGBOOST_FOR_RESIDUAL = False
RESIDUAL_ALLOW_FALLBACK = True


def train_xgb_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 0,
    weak_level: str = None,
) -> object:
    """
    Train XGBoost regressor with configurable weak level.

    Weak levels control model complexity:
    - "orig": Full capacity (300 estimators, depth 4).
    - "mild": Moderate (200 estimators, depth 3).
    - "medium": Weak (120 estimators, depth 2). [default]
    - "heavy": Very weak (80 estimators, depth 2).

    Args:
        X_train: Training features, shape (n, p).
        y_train: Training targets, shape (n,).
        random_state: Random seed.
        weak_level: Strength preset ("orig", "mild", "medium", "heavy").
                   If None, uses XGB_WEAK_LEVEL global config.

    Returns:
        Fitted XGBoost regressor.

    Raises:
        ImportError: If xgboost is not installed.
    """
    if not _HAS_XGB:
        raise ImportError("xgboost is not installed")

    if weak_level is None:
        weak_level = XGB_WEAK_LEVEL

    # Determine GPU availability
    use_gpu_env = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    use_gpu = torch.cuda.is_available() and use_gpu_env not in ("", "none", "null")
    tree_method = "gpu_hist" if use_gpu else "hist"

    # Presets for different weak levels
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


def train_residual_regressor(
    X_train: np.ndarray,
    y_train: np.ndarray,
    random_state: int = 0,
    weak_level: str = None,
) -> Tuple[object, str]:
    """
    Train residual regressor with automatic fallback chain.

    Tries XGBoost first. If unavailable and fallback enabled, tries:
    LightGBM -> RandomForest.

    Args:
        X_train: Training features, shape (n, p).
        y_train: Training targets, shape (n,).
        random_state: Random seed.
        weak_level: XGBoost weak level (used if XGB available).

    Returns:
        (model, backend_name) where backend_name is one of:
        "xgboost", "lightgbm-fallback", "rf-fallback".

    Raises:
        ImportError: If no suitable backend is available and REQUIRE_XGBOOST_FOR_RESIDUAL=True
                    or RESIDUAL_ALLOW_FALLBACK=False.
    """
    # Try XGBoost first
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

    # Try LightGBM
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

    # Fallback to RandomForest
    model = RandomForestRegressor(
        n_estimators=400,
        max_features="sqrt",
        random_state=random_state,
        n_jobs=1,
    )
    model.fit(X_train, y_train)
    return model, "rf-fallback"
