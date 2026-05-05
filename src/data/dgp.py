"""
Data generating processes (DGP) for simulation studies.

Implements various heteroscedastic and multimodal setups for benchmarking
conformal prediction methods.
"""

from typing import Tuple

import numpy as np


def _standardize_to_unit_var(arr: np.ndarray) -> np.ndarray:
    """Standardize array to unit variance."""
    arr = np.asarray(arr, dtype=float)
    mu = np.mean(arr)
    sd = np.std(arr)
    sd = max(sd, 1e-8)
    return (arr - mu) / sd


def generate_y_for_x(
    X_fixed: np.ndarray, *, setup: int = 1, rng: np.random.Generator, is_train: bool = False
) -> np.ndarray:
    """
    Generate y values from X using specified DGP setup.

    Implements various heteroscedastic and multimodal data generating processes:
    - Setup 0/0b: Skewed mixture (t, Normal, Exponential)
    - Setup 0a: Multimodal with mixture components
    - Setup 0c/0c1/0c2: Variable heteroscedasticity
    - Setup 1-2: Standard heteroscedastic models
    - Setup 3: Exponential (log-normal like)
    - Setup 4-6: Complex heavy-tailed and jump processes

    Parameters
    ----------
    X_fixed : np.ndarray
        Covariate matrix, shape (n, 5).
    setup : int, default=1
        DGP setup identifier.
    rng : np.random.Generator
        NumPy random generator.
    is_train : bool, default=False
        Whether this is training data (affects some shift settings).

    Returns
    -------
    np.ndarray
        Generated y values, shape (n,), dtype float32.
    """
    n = X_fixed.shape[0]
    x1, x2, x3, x4, x5 = [X_fixed[:, i] for i in range(5)]
    f_x = x1**2 + x2 * x3 + x3 * x4 + x5

    setup_str = str(setup).lower()

    # ===== Setup 0 / 0b =====
    if setup_str == "0" or setup_str == "0b":
        tau = 0.005
        g_x = 1.0 / (1.0 + np.exp((x1 - 0.5) / tau))
        dist_sq = 4 * (x1 - 0.5) ** 2
        p1 = dist_sq * g_x
        p3 = dist_sq * (1.0 - g_x)
        p2 = 1.0 - p1 - p3
        eps_t = rng.standard_t(df=3, size=n)
        eps_e = rng.normal(loc=0.0, scale=0.5, size=n)
        eps_ex = np.full(n, 1 * (np.exp(1) - 1))
        u = rng.random(n)
        raw = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_e, eps_ex))
        eps = raw
        y = f_x + eps
        return y.astype("float32")

    # ===== Setup 0a =====
    if str(setup).lower().startswith("0a"):
        a1 = 1.2 - 3.5 * x1
        a2 = 0.6 - 3.0 * (x1 - 0.5) ** 2 * 4.0
        a3 = -0.2 + 3.5 * x1
        logits = np.stack([a1, a2, a3], axis=1)
        logits = logits - logits.max(axis=1, keepdims=True)
        p = np.exp(logits)
        p = p / p.sum(axis=1, keepdims=True)
        p1, p2, p3 = p[:, 0], p[:, 1], p[:, 2]

        mean_1 = -8.0
        eps_1 = mean_1 + 0.6 * rng.standard_t(df=3, size=n)

        mean_2 = 0.0
        eps_2 = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)

        mean_3 = 6.5
        expected_xi3 = np.exp(-0.2 + 0.5**2 / 2.0)
        xi3 = rng.lognormal(mean=-0.2, sigma=0.5, size=n)
        eps_3 = mean_3 + 0.5 * (xi3 - expected_xi3)

        u = rng.random(n)
        raw = np.where(u < p1, eps_1, np.where(u < p1 + p2, eps_2, eps_3))
        mean_raw = p1 * mean_1 + p2 * mean_2 + p3 * mean_3
        eps = raw - mean_raw

        setup_str = str(setup).lower()
        if not is_train:
            if setup_str == "0as_0.01": eps += 0.01
            elif setup_str == "0as_0.05": eps += 0.05
            elif setup_str == "0as_0.075": eps += 0.075
            elif setup_str == "0as_0.1": eps += 0.1
            elif setup_str == "0as_0.5": eps += 0.5
            elif setup_str == "0as_0.75": eps += 0.75
            elif setup_str in ["0as_1", "0as_1.0"]: eps += 1.0

        y = f_x + 0.5 * eps
        return y.astype("float32")

    # ===== Setup 0c =====
    if str(setup).lower() == "0c":
        p1 = 4 * (x1 - 0.5) ** 2 * (x1 < 0.5)
        p3 = 4 * (x1 - 0.5) ** 2 * (x1 >= 0.5)
        p2 = 1.0 - p1 - p3

        eps_t = 0.2 * rng.standard_t(df=3, size=n)
        eps_n = 0.5 * rng.normal(loc=0.0, scale=1.0, size=n)
        eps_e = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)

        u = rng.random(n)
        raw = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_n, eps_e))
        eps = raw
        y = f_x + eps
        return y.astype("float32")

    # ===== Setup 0c1 =====
    if str(setup).lower().startswith("0c1"):
        p1 = 4 * (x1 - 0.5) ** 2 * (x1 < 0.5)
        p3 = 4 * (x1 - 0.5) ** 2 * (x1 >= 0.5)
        p2 = 1.0 - p1 - p3

        eps_t = 0.5 * rng.standard_t(df=3, size=n)
        eps_n = 0.5 * rng.normal(loc=0.0, scale=1.0, size=n)
        eps_e = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)

        u = rng.random(n)
        raw = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_n, eps_e))

        g_x = 0.1 + 3.0 * (x1 - 0.5) ** 2
        eps = g_x * raw

        setup_str = str(setup).lower()
        if not is_train:
            if setup_str == "0c1s": eps += 0.01
            elif setup_str == "0c1s_0.025": eps += 0.025
            elif setup_str == "0c1s_0.05": eps += 0.05
            elif setup_str == "0c1s_0.075": eps += 0.075
            elif setup_str == "0c1s_0.1": eps += 0.1
            elif setup_str == "0c1s_0.25": eps += 0.25
            elif setup_str == "0c1s_0.5": eps += 0.5
            elif setup_str in ["0c1s_1", "0c1s_1.0"]: eps += 1.0

        y = f_x + eps
        return y.astype("float32")

    # ===== Setup 0c2 =====
    if str(setup).lower() == "0c2":
        p1 = 4 * (x1 - 0.5) ** 2 * (x1 < 0.5)
        p3 = 4 * (x1 - 0.5) ** 2 * (x1 >= 0.5)
        p2 = 1.0 - p1 - p3

        eps_t = 2.5 * rng.standard_t(df=3, size=n)
        eps_n = 0.1 * rng.normal(loc=0.0, scale=1.0, size=n)
        eps_e = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)

        u = rng.random(n)
        eps = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_n, eps_e))

        y = f_x + eps
        return y.astype("float32")

    # ===== Setup 2a =====
    if str(setup).lower() == "2a":
        g = x1**2
        x6 = rng.normal(1 + 0.5 * x1, np.sqrt(0.75), n)
        heavy_mask = rng.random(n) < 0.7
        eps_heavy = 0.5 * (x6**2)
        eps_light = rng.normal(-2.0, 1.0, n)
        eps = np.where(heavy_mask, eps_heavy, eps_light)
        y = f_x + g * eps
        return y.astype("float32")

    # ===== Setup 2b =====
    if str(setup).lower() == "2b":
        g = x1**2
        pi = np.clip(0.25 + 0.35 * np.clip(x1, 0, 1), 0.25, 0.6)
        spike_sigma = 0.5
        laplace_b = 0.8
        eps_raw = np.zeros(n, dtype=float)
        for i in range(n):
            if rng.random() < pi[i]:
                eps_raw[i] = rng.laplace(loc=0.0, scale=laplace_b)
            else:
                eps_raw[i] = rng.normal(0.0, spike_sigma)
        mu0, sd0 = np.mean(eps_raw), np.std(eps_raw)
        sd0 = max(sd0, 1e-8)
        eps_raw = np.clip(eps_raw, mu0 - 8 * sd0, mu0 + 8 * sd0)
        eps_std = _standardize_to_unit_var(eps_raw)
        h = 0.08 + 0.22 * (1.0 - np.clip(x1, 0, 1)) ** 1.5
        base = np.where(
            rng.random(n) < 0.5,
            rng.laplace(loc=0.0, scale=0.6, size=n),
            rng.normal(loc=0.0, scale=0.8, size=n),
        )
        p_burst = 0.4 + 0.4 * (1.0 - np.clip(x1, 0, 1))
        burst = np.where(rng.random(n) < p_burst, rng.exponential(scale=0.5, size=n), 0.0)
        epsilon = h * (base + burst)
        y = f_x + g * eps_std + epsilon
        return y.astype("float32")

    # ===== Setup 2c =====
    if str(setup).lower() == "2c":
        x6 = rng.normal(1 + 0.5 * x1, np.sqrt(0.75), n)
        g = x1**2
        heavy_mask = rng.random(n) < 0.7
        eps = np.where(heavy_mask, 0.5 * x6**2, rng.normal(0, 1, n))
        y = f_x + g * eps
        return y.astype("float32")

    # ===== Setup 4 =====
    if setup == 4:
        left = (x1 > 0.0) & (x1 < 1.0 / 3.0)
        mid = (x1 >= 1.0 / 3.0) & (x1 < 2.0 / 3.0)
        right = (x1 >= 2.0 / 3.0) & (x1 < 1.0)
        eps = 1e-8
        xl, xm, xr = np.zeros(n), np.zeros(n), np.zeros(n)
        if np.any(left):
            xl[left] = (x1[left] - 0.0) / (1.0 / 3.0 - 0.0 + eps)
        if np.any(mid):
            xm[mid] = (x1[mid] - 1.0 / 3.0) / (2.0 / 3.0 - 1.0 / 3.0 + eps)
        if np.any(right):
            xr[right] = (x1[right] - 2.0 / 3.0) / (1.0 - 2.0 / 3.0 + eps)
        s = np.zeros(n, dtype=float)
        if np.any(left):
            s[left] = 0.05 + 0.05 * (xl[left] ** 1.25)
        if np.any(mid):
            s[mid] = 0.12 + 0.10 * (xm[mid] ** 1.10)
        if np.any(right):
            s[right] = 0.35 + 0.20 * (xr[right] ** 1.05)
        g = np.zeros(n, dtype=float)
        if np.any(left):
            kL = 0.7
            gamma_r = rng.gamma(shape=kL, scale=1.0, size=np.sum(left))
            exp_jit = rng.exponential(scale=0.4, size=np.sum(left))
            g[left] = s[left] * (gamma_r + 0.5 * exp_jit + 0.05) + 0.05
        if np.any(mid):
            muM = np.log(np.maximum(0.6 * s[mid], 1e-8))
            sigmaM = 0.8
            ln_mid = rng.lognormal(mean=muM, sigma=sigmaM)
            g[mid] = -1.5 * ln_mid
        if np.any(right):
            nr = np.sum(right)
            p_tail = 0.30 + 0.60 * xr[right]
            p_tail = np.clip(p_tail, 0.0, 0.95)
            small = rng.normal(loc=0.0, scale=0.45, size=nr)
            pareto = (1.0 + rng.pareto(a=1.4, size=nr))
            choose = rng.random(nr) < p_tail
            g[right] = s[right] * np.where(choose, pareto, small)
        edge = ~(left | mid | right)
        if np.any(edge):
            g[edge] = s[edge] * rng.exponential(scale=1.0, size=np.sum(edge))
        y = f_x + g
        return y.astype("float32")

    # ===== Setup 5 =====
    if setup == 5:
        s = 0.08 + 0.12 * np.sqrt(np.abs(f_x) + 1e-6)
        g = np.zeros(n, dtype=float)
        left = (x1 > 0.0) & (x1 < 1.0 / 3.0)
        mid = (x1 >= 1.0 / 3.0) & (x1 < 2.0 / 3.0)
        right = (x1 >= 2.0 / 3.0) & (x1 < 1.0)
        if np.any(left):
            mu_ln = np.log(np.maximum(s[left], 1e-6))
            ln = rng.lognormal(mean=mu_ln, sigma=0.7)
            ex = rng.exponential(scale=0.5 * s[left])
            g[left] = ln + 0.3 * ex
        if np.any(mid):
            mu_ln = np.log(np.maximum(s[mid], 1e-6))
            ln = rng.lognormal(mean=mu_ln, sigma=0.6)
            g[mid] = -ln
        if np.any(right):
            m = np.sum(right)
            p = rng.random(m)
            small = rng.normal(loc=0.0, scale=0.4 * s[right], size=m)
            pareto = s[right] * (1.0 + rng.pareto(a=2.0, size=m))
            g[right] = np.where(p < 0.85, small, pareto)
        y = f_x + g
        return y.astype("float32")

    # ===== Setup 6 =====
    if setup == 6:
        s0 = 0.05 + 0.06 * (np.abs(f_x) / (1.0 + np.abs(f_x)))
        baseline = rng.normal(loc=0.0, scale=s0)
        lam = 0.2 + 0.8 * x1 + 0.6 * (np.abs(f_x) / (1.0 + np.abs(f_x)))
        sJ = 0.08 + 0.12 * np.sqrt(np.abs(f_x) + 1e-6)
        jumps = np.zeros(n, dtype=float)
        for i in range(n):
            k = rng.poisson(lam=max(lam[i], 1e-8))
            if k > 0:
                jumps[i] = rng.exponential(scale=sJ[i], size=k).sum()
        y = f_x + baseline + jumps
        return y.astype("float32")

    # ===== Setup 3 =====
    if setup == 3:
        lam = np.abs(0.1 * f_x).astype(float)
        lam = np.maximum(lam, 1e-8)
        g = rng.exponential(scale=lam, size=n)
        y = f_x + g
        return y.astype("float32")

    # ===== Default: Setup 1 or 2 =====
    w = rng.multinomial(1, [0.1, 0.7, 0.2], size=n)
    if setup == 1:
        x6 = rng.normal(1, 1, n)
        g = 1.0
    elif setup == 2:
        x6 = rng.normal(1 + 0.5 * x1, np.sqrt(0.75), n)
        g = x1**2
    else:
        raise ValueError(f"setup {setup} not recognized")

    eps = np.where(
        w[:, 0] == 1,
        rng.normal(-2, 1, n),
        np.where(w[:, 1] == 1, rng.normal(0, 1, n), 0.5 * x6**2),
    )
    y = f_x + g * eps
    return y.astype("float32")


def generate_data(
    n: int, *, setup: int = 1, seed: int = 0, is_train: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate synthetic (X, y) dataset.

    Parameters
    ----------
    n : int
        Sample size.
    setup : int, default=1
        DGP setup identifier.
    seed : int, default=0
        Random seed.
    is_train : bool, default=False
        Whether this is training data.

    Returns
    -------
    X : np.ndarray
        Covariate matrix, shape (n, 5), dtype float32.
    y : np.ndarray
        Response vector, shape (n,), dtype float32.
    """
    rng = np.random.default_rng(seed)
    is_setup_2 = str(setup) == "2"

    if is_setup_2:
        x1 = np.clip(rng.normal(0, 1, n), -3, 3)
        x2 = rng.uniform(0, 1, n)
    else:
        x1 = rng.uniform(0, 1, n)
        x2 = np.clip(rng.normal(0, 1, n), -3, 3)

    x3 = rng.beta(0.5, 0.5, n)
    x4 = rng.binomial(1, 0.5, n)
    x5 = np.clip(rng.poisson(2, n), 0, 5)
    X = np.c_[x1, x2, x3, x4, x5]
    y = generate_y_for_x(X, setup=setup, rng=rng, is_train=is_train)
    return X.astype("float32"), y.astype("float32")


def generate_cal_at_x1(
    n: int, x1_val: float, *, setup: int = 4, seed: int = 0, is_train: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate calibration data conditional on x1 value.

    Parameters
    ----------
    n : int
        Sample size.
    x1_val : float
        Fixed x1 value.
    setup : int, default=4
        DGP setup identifier.
    seed : int, default=0
        Random seed.
    is_train : bool, default=False
        Whether this is training data.

    Returns
    -------
    X : np.ndarray
        Covariate matrix, shape (n, 5), dtype float32.
    y : np.ndarray
        Response vector, shape (n,), dtype float32.
    """
    rng = np.random.default_rng(seed)
    x1 = np.full(n, x1_val)
    is_setup_2 = str(setup) == "2"

    if is_setup_2:
        x2 = rng.uniform(0, 1, n)
    else:
        x2 = np.clip(rng.normal(0, 1, n), -3, 3)

    x3 = rng.beta(0.5, 0.5, n)
    x4 = rng.binomial(1, 0.5, n)
    x5 = np.clip(rng.poisson(2, n), 0, 5)
    X = np.c_[x1, x2, x3, x4, x5]
    y = generate_y_for_x(X, setup=setup, rng=rng, is_train=is_train)
    return X.astype("float32"), y.astype("float32")
