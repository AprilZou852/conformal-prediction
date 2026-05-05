import os
import warnings
import copy
import math
from contextlib import contextmanager
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split
import matplotlib.pyplot as plt
from tqdm import tqdm
warnings.filterwarnings("ignore")
if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")
USE_AMP = True
TORCH_COMPILE = False
NUM_WORKERS = 0
PIN_MEMORY = False
device = "cuda" if torch.cuda.is_available() else "cpu"
BASE_SEED = 0
ALPHA = 0.1
N_TRAIN = 3000
N_TEST_POINTS = 200
N_CAL = 500
N_REP = 3000   
torch.manual_seed(BASE_SEED)
np.random.seed(BASE_SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(BASE_SEED)
METHOD_HAZARD_BASELINE = "HazardNN"
METHOD_RESIDUAL = "Residual"
METHOD_RESCALED = "Rescaled"
METHOD_CQR = "CQR"
METHOD_NOZ_UCENTER = "HazardNN_UCentered_Fixed"   

METHOD_HAZARD_SHORTEST = "CPI-opt-NNCDE"      
METHOD_DCP_OPT = "DCP-opt-NNCDE"

RUN_HAZARD_SHORTEST_C = True
BASELINE_STRENGTH = "medium"
def get_baseline_config(strength: str):
    strength = str(strength).lower().strip()
    if strength not in {"strong", "medium", "weak"}:
        raise ValueError(f"BASELINE_STRENGTH must be one of {{'strong','medium','weak'}}, got {strength}")
    if strength == "strong":
        return dict(
            hidden=64,
            epochs=500,
            patience=10,
            val_frac=0.2,
            batch=128,
            lr=1e-3,
            weight_decay=0.0,
            cqr_hidden=32,
            clamp_logvar=False,
        )
    if strength == "medium":
        return dict(
            hidden=32,
            epochs=180,
            patience=10,
            val_frac=0.3,
            batch=256,
            lr=1e-3,
            weight_decay=1e-23,
            cqr_hidden=16,      
            clamp_logvar=False,  
        )
    return dict(
        hidden=16,
        epochs=120,
        patience=5,
        val_frac=0.35,
        batch=256,
        lr=1e-3,
        weight_decay=1e-2,
        cqr_hidden=0,        
        clamp_logvar=True,
    )
BL_CFG = get_baseline_config(BASELINE_STRENGTH)
Z_GRID_SIZE = 41
Z_EPS_U = 1e-6

def _standardize_to_unit_var(arr):
    arr = np.asarray(arr, dtype=float)
    mu = np.mean(arr)
    sd = np.std(arr)
    sd = max(sd, 1e-8)
    return (arr - mu) / sd
def generate_y_for_x(X_fixed, *, setup=1, rng, is_train=False):
    n = X_fixed.shape[0]
    x1, x2, x3, x4, x5 = [X_fixed[:, i] for i in range(5)]
    f_x = x1**2 + x2 * x3 + x3 * x4 + x5
    setup_str = str(setup).lower()
    if setup_str == "0" or setup_str == "0b":
        tau = 0.005
        g_x = 1.0 / (1.0 + np.exp((x1 - 0.5) / tau))
        dist_sq = 4 * (x1 - 0.5)**2
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
    if str(setup).lower() == "0c":
        # p1(x1) = 4(x1 - 0.5)^2 * I{x1 < 0.5}
        # p3(x1) = 4(x1 - 0.5)^2 * I{x1 >= 0.5}
        # p2(x1) = 1 - p1 - p3
        p1 = 4 * (x1 - 0.5)**2 * (x1 < 0.5)
        p3 = 4 * (x1 - 0.5)**2 * (x1 >= 0.5)
        p2 = 1.0 - p1 - p3
        
        # Component 1: t3
        eps_t = 0.2 * rng.standard_t(df=3, size=n)
        # Component 2: N(0, 0.5^2)
        eps_n = 0.5 * rng.normal(loc=0.0, scale=1.0, size=n)
        # Component 3: (Exp(1) - 1)
        eps_e = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)
        
        u = rng.random(n)
        # if u < p1 -> eps_t
        # if p1 <= u < p1+p2 -> eps_n
        # else -> eps_e
        raw = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_n, eps_e))
        
        eps = raw
        y = f_x + eps
        return y.astype("float32")
    if str(setup).lower().startswith("0c1"):
        p1 = 4 * (x1 - 0.5)**2 * (x1 < 0.5)
        p3 = 4 * (x1 - 0.5)**2 * (x1 >= 0.5)
        p2 = 1.0 - p1 - p3
        
        eps_t = 0.5 * rng.standard_t(df=3, size=n)
        eps_n = 0.5 * rng.normal(loc=0.0, scale=1.0, size=n)
        eps_e = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)
        
        u = rng.random(n)
        raw = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_n, eps_e))
        
        g_x = 0.1 + 3.0 * (x1 - 0.5)**2  
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
    if str(setup).lower() == "0c2":
        p1 = 4 * (x1 - 0.5)**2 * (x1 < 0.5)
        p3 = 4 * (x1 - 0.5)**2 * (x1 >= 0.5)
        p2 = 1.0 - p1 - p3
        
        eps_t = 2.5 * rng.standard_t(df=3, size=n)
        eps_n = 0.1 * rng.normal(loc=0.0, scale=1.0, size=n)
        eps_e = 0.5 * (rng.exponential(scale=1.0, size=n) - 1.0)
        
        u = rng.random(n)
        eps = np.where(u < p1, eps_t, np.where(u < p1 + p2, eps_n, eps_e))
        
        y = f_x + eps
        return y.astype("float32")
    if str(setup).lower() == "2a":
        g = (x1**2)
        x6 = rng.normal(1 + 0.5 * x1, np.sqrt(0.75), n)
        heavy_mask = rng.random(n) < 0.7
        eps_heavy = 0.5 * (x6**2)
        eps_light = rng.normal(-2.0, 1.0, n)
        eps = np.where(heavy_mask, eps_heavy, eps_light)
        y = f_x + g * eps
        return y.astype("float32")
    if str(setup).lower() == "2b":
        g = (x1**2)
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
    if str(setup).lower() == "2c":
        x6 = rng.normal(1 + 0.5 * x1, np.sqrt(0.75), n)
        g = x1**2
        heavy_mask = rng.random(n) < 0.7
        eps = np.where(heavy_mask, 0.5 * x6**2, rng.normal(0, 1, n))
        y = f_x + g * eps
        return y.astype("float32")
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
    if setup == 3:
        lam = np.abs(0.1 * f_x).astype(float)
        lam = np.maximum(lam, 1e-8)
        g = rng.exponential(scale=lam, size=n)
        y = f_x + g
        return y.astype("float32")
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
def generate_data(n, *, setup=1, seed=0, is_train=False):
    """Generates (X, y) with 5 covariates."""
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
def generate_cal_at_x1(n, x1_val, *, setup=4, seed=0, is_train=False):
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
FEATS = ["start", "end", "x1", "x2", "x3", "x4", "x5"]
def expand(X, y):
    uniq = np.sort(np.unique(y))
    t0 = uniq[0] - 1
    rows = []
    for idx, (xi, yi) in enumerate(zip(X, y)):
        s = t0
        for t in uniq[uniq <= yi]:
            rows.append([idx, s, t, int(t == yi), *xi])
            s = t
    df = pd.DataFrame(rows, columns=["id", "start", "end", "delta", "x1", "x2", "x3", "x4", "x5"])
    return df, uniq
class F1Net(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(len(FEATS), hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)
def _make_dataloaders(Xt, Y, val_frac=0.2, batch=128, seed=0):
    ds = TensorDataset(torch.tensor(Xt), torch.tensor(Y))
    n_val = int(len(ds) * val_frac)
    tr_ds, va_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed)
    )
    def _dl(d, shuffle):
        return DataLoader(
            d,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=False,
        )
    return _dl(tr_ds, True), _dl(va_ds, False)
class SimpleNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )
    def forward(self, x):
        return self.net(x)
def _dl_xy(X, y, val_frac=0.2, batch=128, seed=0):
    ds = TensorDataset(torch.tensor(X), torch.tensor(y).unsqueeze(1))
    n_val = int(len(ds) * val_frac)
    tr_ds, va_ds = random_split(
        ds, [len(ds) - n_val, n_val], generator=torch.Generator().manual_seed(seed)
    )
    def _mk(d, shuffle):
        return DataLoader(
            d,
            batch_size=batch,
            shuffle=shuffle,
            num_workers=NUM_WORKERS,
            pin_memory=PIN_MEMORY,
            persistent_workers=False,
        )
    return _mk(tr_ds, True), _mk(va_ds, False)
def train_F1_longtable(df, *, epochs=500, batch=256, val_frac=0.2, patience=40, seed_split=0, 
                       hidden=64, lr=5e-4, weight_decay=0.0):
    X = df[FEATS].values.astype("float32")
    Y = np.c_[
        df["delta"].values.astype("float32"),
        (df["end"] - df["start"]).values.astype("float32"),
    ].astype("float32")
    mu, sg = X.mean(0), X.std(0)
    sg[sg == 0] = 1
    Xt = (X - mu) / sg
    tr_dl, va_dl = _make_dataloaders(Xt, Y, val_frac=val_frac, batch=batch, seed=seed_split)
    net = F1Net(hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        net = torch.compile(net)
    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))
    def loss_fn(y_true, y_pred):
        y_pred = torch.clamp(y_pred, max=10.0)
        delta = y_true[:, :1]
        dt = y_true[:, 1:2]
        return -(delta * y_pred).mean() + (torch.exp(y_pred) * dt).mean()
    best_state, best_val, no_gain = copy.deepcopy(net.state_dict()), float("inf"), 0
    for _ in range(epochs):
        net.train()
        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                out = net(xb)
                loss = loss_fn(yb, out)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        net.eval()
        val_losses = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            for xb, yb in va_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                val_losses.append(loss_fn(yb, net(xb)).item())
        v = float(np.mean(val_losses)) if len(val_losses) else float("inf")
        if v < best_val - 1e-4:
            best_val, best_state, no_gain = v, copy.deepcopy(net.state_dict()), 0
        else:
            no_gain += 1
        if no_gain >= patience:
            break
    print(f"[train_F1_longtable] Stopped at epoch {_ + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    net.load_state_dict(best_state)
    return net, mu.astype("float32"), sg.astype("float32")
def train_simpleNN(
    X,
    y,
    *,
    epochs=500,
    batch=128,
    val_frac=0.2,
    patience=10,
    seed_split=0,
    hidden=64,
    lr=1e-3,
    weight_decay=0.0,
):
    tr_dl, va_dl = _dl_xy(X, y, val_frac=val_frac, batch=batch, seed=seed_split)
    model = SimpleNN(hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))
    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                loss = crit(model(xb), yb)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        losses = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            for xb, yb in va_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                losses.append(crit(model(xb), yb).item())
        v = float(np.mean(losses)) if len(losses) else float("inf")
        if v < best_val - 1e-4:
            best_val, best_state, no_gain = v, copy.deepcopy(model.state_dict()), 0
        else:
            no_gain += 1
        if no_gain >= patience:
            break
    print(f"[train_simpleNN] Stopped at epoch {_ + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model
class MeanVarianceNN(nn.Module):
    def __init__(self, hidden=64):
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(5, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
    def forward(self, x):
        base_out = self.base_net(x)
        return self.mean_head(base_out), self.log_var_head(base_out)
def gaussian_nll_loss(y_true, mean, log_var):
    return (0.5 * log_var + 0.5 * torch.exp(-log_var) * (y_true - mean) ** 2).mean()
def train_mean_variance_NN(
    X,
    y,
    *,
    epochs=500,
    batch=128,
    val_frac=0.2,
    patience=10,
    seed_split=0,
    hidden=64,
    lr=1e-3,
    weight_decay=0.0,
    clamp_logvar=False,
):
    tr_dl, va_dl = _dl_xy(X, y, val_frac=val_frac, batch=batch, seed=seed_split)
    model = MeanVarianceNN(hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))
    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                mean, log_var = model(xb)
                if clamp_logvar:
                    log_var = torch.clamp(log_var, min=-1.0, max=1.0)
                loss = gaussian_nll_loss(yb, mean, log_var)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        losses = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            for xb, yb in va_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                mean, log_var = model(xb)
                if clamp_logvar:
                    log_var = torch.clamp(log_var, min=-1.0, max=1.0)
                losses.append(gaussian_nll_loss(yb, mean, log_var).item())
        v = float(np.mean(losses)) if len(losses) else float("inf")
        if v < best_val - 1e-4:
            best_val, best_state, no_gain = v, copy.deepcopy(model.state_dict()), 0
        else:
            no_gain += 1
        if no_gain >= patience:
            break
    print(f"[train_mean_variance_NN] Stopped at epoch {_ + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model
class QuantileNN(nn.Module):
    def __init__(self, hidden=64):
        """
        hidden == 0 : linear quantile regression baseline (common in literature)
        hidden > 0  : shallow NN quantile regressor
        """
        super().__init__()
        if hidden == 0:
            self.net = nn.Linear(5, 2)
        else:
            self.net = nn.Sequential(
                nn.Linear(5, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )
    def forward(self, x):
        return self.net(x)
def pinball_loss(y_true, y_pred, quantiles):
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[:, i : i + 1]
        losses.append(torch.mean(torch.max((q - 1) * err, q * err)))
    return sum(losses)
def train_quantile_NN(
    X,
    y,
    *,
    quantiles,
    epochs=500,
    batch=128,
    val_frac=0.2,
    patience=10,
    seed_split=0,
    hidden=64,
    lr=1e-3,
    weight_decay=0.0,
):
    tr_dl, va_dl = _dl_xy(X, y, val_frac=val_frac, batch=batch, seed=seed_split)
    model = QuantileNN(hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))
    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), float("inf"), 0
    for _ in range(epochs):
        model.train()
        for xb, yb in tr_dl:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
                loss = pinball_loss(yb, model(xb), quantiles)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        model.eval()
        losses = []
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            for xb, yb in va_dl:
                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)
                losses.append(pinball_loss(yb, model(xb), quantiles).item())
        v = float(np.mean(losses)) if len(losses) else float("inf")
        if v < best_val - 1e-4:
            best_val, best_state, no_gain = v, copy.deepcopy(model.state_dict()), 0
        else:
            no_gain += 1
        if no_gain >= patience:
            break
    print(f"[train_quantile_NN] Stopped at epoch {_ + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model
@torch.no_grad()
def predict_cdf(net, mu, sg, uniq_y, X, *, t_end):
    cov = X.astype("float32")
    t = np.r_[uniq_y[0] - 1, uniq_y[uniq_y <= t_end], t_end].astype("float32")
    t = np.unique(t)
    if len(t) < 2:
        return 0.0
    starts = t[:-1]
    ends = t[1:]
    dt = (ends - starts).reshape(-1, 1)
    cov_rep = np.repeat(cov.reshape(1, -1), len(starts), axis=0)
    rows = np.concatenate([starts.reshape(-1, 1), ends.reshape(-1, 1), cov_rep], axis=1)
    rows_norm = (rows - mu) / sg
    rows_norm_t = torch.tensor(rows_norm, device=device)
    with torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
        h = net(rows_norm_t).squeeze(1)
        h = torch.clamp(h, max=10.0)
        contrib = torch.exp(h) * torch.tensor(dt.squeeze(1), device=device)
        L = torch.sum(contrib).item()
    return 1.0 - math.exp(-L)
def find_quantile(net, mu, sg, uniq_y, x_test, u, y_min, y_max):
    lo, hi = y_min, y_max
    for _ in range(18):
        mid = (lo + hi) / 2
        val = predict_cdf(net, mu, sg, uniq_y, x_test, t_end=mid)
        if val < u:
            lo = mid
        else:
            hi = mid
    return hi
@contextmanager
def preserve_rng_state():
    np_state = np.random.get_state()
    torch_state = torch.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        np.random.set_state(np_state)
        torch.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)
def compute_z_star_gridsearch(
    net, mu, sg, uniq_y, X: np.ndarray, y_min: float, y_max: float, *, alpha: float, grid_size: int, eps_u: float
) -> np.ndarray:
    X = X.astype("float32")
    z_grid = np.linspace(eps_u, alpha - eps_u, grid_size).astype("float32")
    u1_grid = np.clip(z_grid, eps_u, 1 - eps_u)
    u2_grid = np.clip(z_grid + 1.0 - alpha, eps_u, 1 - eps_u)
    target_L1 = -np.log(1.0 - u1_grid)
    target_L2 = -np.log(1.0 - u2_grid)
    
    t_all = np.concatenate([[uniq_y[0] - 1.0], uniq_y[uniq_y <= y_max], [y_max]]).astype("float32")
    t_all = np.unique(t_all)
    starts = t_all[:-1]
    ends = t_all[1:]
    dt = ends - starts
    
    z_star = np.empty(X.shape[0], dtype="float32")
    for i in range(X.shape[0]):
        xi = X[i]
        cov_rep = np.repeat(xi.reshape(1, -1), len(starts), axis=0)
        rows = np.concatenate([starts.reshape(-1, 1), ends.reshape(-1, 1), cov_rep], axis=1)
        rows_norm = (rows - mu) / sg
        rows_norm_t = torch.tensor(rows_norm, device=device, dtype=torch.float32)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            h = net(rows_norm_t).squeeze(1)
            h = torch.clamp(h, max=10.0)
            contrib = torch.exp(h) * torch.tensor(dt, device=device, dtype=torch.float32)
            cum_L = torch.cumsum(contrib, dim=0).cpu().numpy()
            hazard_rates = torch.exp(h).cpu().numpy()
            
        def get_t_for_L(L_targets):
            idx = np.searchsorted(cum_L, L_targets)
            capped = idx >= len(cum_L)
            idx = np.clip(idx, 0, len(cum_L)-1)
            L_prev = np.where(idx == 0, 0.0, cum_L[idx - 1])
            t_prev = starts[idx]
            rate = np.maximum(hazard_rates[idx], 1e-12)
            t_res = t_prev + (L_targets - L_prev) / rate
            t_res[capped] = y_max
            return t_res
            
        q1 = get_t_for_L(target_L1)
        q2 = get_t_for_L(target_L2)
        widths = q2 - q1
        best_idx = np.argmin(widths)
        z_star[i] = z_grid[best_idx]
        
    return z_star
def run_simulation(current_setup):
    print("=" * 70)
    print(f"STARTING SIMULATION FOR SETUP: {current_setup}")
    print(f"Device: {device}, AMP={'ON' if (USE_AMP and device=='cuda') else 'OFF'}")
    print(f"Run learned shortest-c (renamed)? {'YES' if RUN_HAZARD_SHORTEST_C else 'NO'}")
    print(f"Added NO-z methods: {METHOD_NOZ_UCENTER}")
    print(f"Baseline strength: {BASELINE_STRENGTH}")
    print(
        "Baseline config => "
        f"hidden={BL_CFG['hidden']}, epochs={BL_CFG['epochs']}, patience={BL_CFG['patience']}, "
        f"val_frac={BL_CFG['val_frac']}, batch={BL_CFG['batch']}, lr={BL_CFG['lr']}, wd={BL_CFG['weight_decay']}, "
        f"cqr_hidden={BL_CFG['cqr_hidden']}, clamp_logvar={BL_CFG['clamp_logvar']}"
    )
    print("=" * 70)
    print(f"\n[{current_setup}] --- Step 1: Training Models ---")
    X_train, y_train = generate_data(N_TRAIN, setup=current_setup, seed=BASE_SEED, is_train=True)
    df_long, uniq_y_train = expand(X_train, y_train)
    net_fixed, mu_fixed, sg_fixed = train_F1_longtable(df_long, epochs=500, seed_split=BASE_SEED)
    mdl_resid = train_simpleNN(
        X_train,
        y_train,
        epochs=BL_CFG["epochs"],
        batch=BL_CFG["batch"],
        val_frac=BL_CFG["val_frac"],
        patience=BL_CFG["patience"],
        seed_split=BASE_SEED,
        hidden=BL_CFG["hidden"],
        lr=BL_CFG["lr"],
        weight_decay=BL_CFG["weight_decay"],
    )
    mdl_rescaled = train_mean_variance_NN(
        X_train,
        y_train,
        epochs=BL_CFG["epochs"],
        batch=BL_CFG["batch"],
        val_frac=BL_CFG["val_frac"],
        patience=BL_CFG["patience"],
        seed_split=BASE_SEED,
        hidden=BL_CFG["hidden"],
        lr=BL_CFG["lr"],
        weight_decay=BL_CFG["weight_decay"],
        clamp_logvar=BL_CFG["clamp_logvar"],
    )
    mdl_cqr = train_quantile_NN(
        X_train,
        y_train,
        quantiles=[ALPHA / 2, 1 - ALPHA / 2],
        epochs=BL_CFG["epochs"],
        batch=BL_CFG["batch"],
        val_frac=BL_CFG["val_frac"],
        patience=BL_CFG["patience"],
        seed_split=BASE_SEED,
        hidden=BL_CFG["cqr_hidden"],
        lr=BL_CFG["lr"],
        weight_decay=BL_CFG["weight_decay"],
    )
    X_test_fixed, _ = generate_data(N_TEST_POINTS, setup=current_setup, seed=BASE_SEED + 1)
    y_min, y_max = np.quantile(y_train, [0.0, 1.0])
    y_range_extension = (y_max - y_min) * 10
    y_min -= y_range_extension
    y_max += y_range_extension
    z_test_optimal = None
    if RUN_HAZARD_SHORTEST_C:
        print(f"\n[{current_setup}] --- Stage 1-2: Grid Search z(x) on TEST Set for {METHOD_HAZARD_SHORTEST} ---")
        z_test_optimal = compute_z_star_gridsearch(
            net_fixed,
            mu_fixed,
            sg_fixed,
            uniq_y_train,
            X_test_fixed,
            y_min,
            y_max,
            alpha=ALPHA,
            grid_size=Z_GRID_SIZE,
            eps_u=Z_EPS_U,
        )



    print(f"\n[{current_setup}] --- Step 2: Monte Carlo ({N_REP} Reps) ---")
    method_names = [
        METHOD_HAZARD_BASELINE,
        METHOD_NOZ_UCENTER,

        METHOD_RESIDUAL,
        METHOD_RESCALED,
        METHOD_CQR,
    ]
    if RUN_HAZARD_SHORTEST_C:
        method_names.append(METHOD_HAZARD_SHORTEST)
        method_names.append(METHOD_DCP_OPT)

    results = {
        name: {
            "C": np.zeros((N_TEST_POINTS, N_REP), dtype=np.float32),
            "W": np.zeros((N_TEST_POINTS, N_REP), dtype=np.float32),
        }
        for name in method_names
    }
    X_test_fixed_t = torch.tensor(X_test_fixed, device=device)
    for k in tqdm(range(N_REP), desc=f"Sim {current_setup}"):
        rep_seed = BASE_SEED * 1000 + k
        X_cal, y_cal = generate_data(N_CAL, setup=current_setup, seed=rep_seed)
        rng_test = np.random.default_rng(rep_seed + 1)
        y_test = generate_y_for_x(X_test_fixed, setup=current_setup, rng=rng_test)
        q_level = (1 - ALPHA) * (1 + 1 / N_CAL)
        pit_values = np.empty(N_CAL, dtype=np.float32)
        for i in range(N_CAL):
            pit_values[i] = predict_cdf(net_fixed, mu_fixed, sg_fixed, uniq_y_train, X_cal[i], t_end=y_cal[i])
        u_lo, u_hi = np.quantile(pit_values, [ALPHA / 2, 1 - ALPHA / 2])
        q_lo = np.empty(N_TEST_POINTS, dtype=np.float32)
        q_hi = np.empty(N_TEST_POINTS, dtype=np.float32)
        for i in range(N_TEST_POINTS):
            xi = X_test_fixed[i]
            q_lo[i] = find_quantile(net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, float(u_lo), y_min, y_max)
            q_hi[i] = find_quantile(net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, float(u_hi), y_min, y_max)
        results[METHOD_HAZARD_BASELINE]["W"][:, k] = q_hi - q_lo
        results[METHOD_HAZARD_BASELINE]["C"][:, k] = ((y_test >= q_lo) & (y_test <= q_hi))
        c0 = 0.5
        scores_uc = np.abs(pit_values - c0)
        q_hat_uc = np.quantile(scores_uc, q_level)
        u_lo_fixed = float(np.clip(c0 - q_hat_uc, 0.0, 1.0))
        u_hi_fixed = float(np.clip(c0 + q_hat_uc, 0.0, 1.0))
        q_lo_uc = np.empty(N_TEST_POINTS, dtype=np.float32)
        q_hi_uc = np.empty(N_TEST_POINTS, dtype=np.float32)
        for i in range(N_TEST_POINTS):
            xi = X_test_fixed[i]
            q_lo_uc[i] = find_quantile(net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, u_lo_fixed, y_min, y_max)
            q_hi_uc[i] = find_quantile(net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, u_hi_fixed, y_min, y_max)
        results[METHOD_NOZ_UCENTER]["W"][:, k] = q_hi_uc - q_lo_uc
        results[METHOD_NOZ_UCENTER]["C"][:, k] = ((y_test >= q_lo_uc) & (y_test <= q_hi_uc))

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            y_hat_cal = mdl_resid(torch.tensor(X_cal, device=device)).squeeze().float().cpu().numpy()
        scores_resid = np.abs(y_cal - y_hat_cal)
        q_resid = np.quantile(scores_resid, q_level)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            y_hat_test = mdl_resid(X_test_fixed_t).squeeze().float().cpu().numpy()
        lo_r, hi_r = y_hat_test - q_resid, y_hat_test + q_resid
        results[METHOD_RESIDUAL]["W"][:, k] = (hi_r - lo_r)
        results[METHOD_RESIDUAL]["C"][:, k] = ((y_test >= lo_r) & (y_test <= hi_r))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            mean_cal, log_var_cal = mdl_rescaled(torch.tensor(X_cal, device=device))
            if BL_CFG["clamp_logvar"]:
                log_var_cal = torch.clamp(log_var_cal, min=-1.0, max=1.0)
            sigma_cal = torch.exp(0.5 * log_var_cal).squeeze().float().cpu().numpy()
            mean_cal_np = mean_cal.squeeze().float().cpu().numpy()
        sigma_cal = np.maximum(sigma_cal, 1e-8)
        scores_rescaled = np.abs(y_cal - mean_cal_np) / sigma_cal
        q_rescaled = np.quantile(scores_rescaled, q_level)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            mean_test, log_var_test = mdl_rescaled(X_test_fixed_t)
            if BL_CFG["clamp_logvar"]:
                log_var_test = torch.clamp(log_var_test, min=-1.0, max=1.0)
            sigma_test = torch.exp(0.5 * log_var_test).squeeze().float().cpu().numpy()
            mean_test_np = mean_test.squeeze().float().cpu().numpy()
        sigma_test = np.maximum(sigma_test, 1e-8)
        lo_s = mean_test_np - q_rescaled * sigma_test
        hi_s = mean_test_np + q_rescaled * sigma_test
        results[METHOD_RESCALED]["W"][:, k] = (hi_s - lo_s)
        results[METHOD_RESCALED]["C"][:, k] = ((y_test >= lo_s) & (y_test <= hi_s))
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            q_preds_cal = mdl_cqr(torch.tensor(X_cal, device=device)).float().cpu().numpy()
        q_lo_cal, q_hi_cal = q_preds_cal[:, 0], q_preds_cal[:, 1]
        scores_cqr = np.maximum(q_lo_cal - y_cal, y_cal - q_hi_cal)
        q_cqr = np.quantile(scores_cqr, q_level)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            q_preds_test = mdl_cqr(X_test_fixed_t).float().cpu().numpy()
        q_lo_test, q_hi_test = q_preds_test[:, 0], q_preds_test[:, 1]
        lo_c, hi_c = q_lo_test - q_cqr, q_hi_test + q_cqr
        results[METHOD_CQR]["W"][:, k] = (hi_c - lo_c)
        results[METHOD_CQR]["C"][:, k] = ((y_test >= lo_c) & (y_test <= hi_c))
        if RUN_HAZARD_SHORTEST_C:
            assert z_test_optimal is not None
            qvec_lo = np.clip(z_test_optimal.astype("float64"), 0.0, 1.0)
            qvec_hi = np.clip((z_test_optimal + 1.0 - ALPHA).astype("float64"), 0.0, 1.0)
            u_lo_x = np.quantile(pit_values.astype("float64"), qvec_lo).astype("float32")
            u_hi_x = np.quantile(pit_values.astype("float64"), qvec_hi).astype("float32")
            q_lo_s = np.empty(N_TEST_POINTS, dtype=np.float32)
            q_hi_s = np.empty(N_TEST_POINTS, dtype=np.float32)
            for i in range(N_TEST_POINTS):
                xi = X_test_fixed[i]
                q_lo_s[i] = find_quantile(
                    net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, float(u_lo_x[i]), y_min, y_max
                )
                q_hi_s[i] = find_quantile(
                    net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, float(u_hi_x[i]), y_min, y_max
                )
            results[METHOD_HAZARD_SHORTEST]["W"][:, k] = q_hi_s - q_lo_s
            results[METHOD_HAZARD_SHORTEST]["C"][:, k] = ((y_test >= q_lo_s) & (y_test <= q_hi_s))

            z_cal_optimal = compute_z_star_gridsearch(
                net_fixed, mu_fixed, sg_fixed, uniq_y_train, X_cal,
                y_min, y_max, alpha=ALPHA, grid_size=Z_GRID_SIZE, eps_u=Z_EPS_U
            )
            center_cal = z_cal_optimal + (1.0 - ALPHA) / 2.0
            scores_dcp = np.abs(pit_values - center_cal)
            q_hat_dcp = np.quantile(scores_dcp, q_level)
            
            q_lo_dcp = np.empty(N_TEST_POINTS, dtype=np.float32)
            q_hi_dcp = np.empty(N_TEST_POINTS, dtype=np.float32)
            for i in range(N_TEST_POINTS):
                xi = X_test_fixed[i]
                b_test = z_test_optimal[i]
                center_test = b_test + (1.0 - ALPHA) / 2.0
                u_lo_dcp = float(np.clip(center_test - q_hat_dcp, 0.0, 1.0))
                u_hi_dcp = float(np.clip(center_test + q_hat_dcp, 0.0, 1.0))
                q_lo_dcp[i] = find_quantile(net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, u_lo_dcp, y_min, y_max)
                q_hi_dcp[i] = find_quantile(net_fixed, mu_fixed, sg_fixed, uniq_y_train, xi, u_hi_dcp, y_min, y_max)
                
            results[METHOD_DCP_OPT]["W"][:, k] = q_hi_dcp - q_lo_dcp
            results[METHOD_DCP_OPT]["C"][:, k] = ((y_test >= q_lo_dcp) & (y_test <= q_hi_dcp))

    print(f"\n[{current_setup}] --- Step 3: Conditional Diagnostic (Residual Cond. Cal.) ---")
    x1_points_to_test = X_test_fixed[:, 0]
    conditional_widths = []
    conditional_coverages = []
    N_MC = 400
    for i, x1_val in enumerate(tqdm(x1_points_to_test, desc="Cond Cal")):
        X_cal_cond, y_cal_cond = generate_cal_at_x1(N_CAL, x1_val, setup=current_setup, seed=i)
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            y_hat_cal_cond = mdl_resid(torch.tensor(X_cal_cond, device=device)).squeeze().float().cpu().numpy()
        residuals_cond = y_cal_cond - y_hat_cal_cond
        res_q_lo, res_q_hi = np.quantile(residuals_cond, [ALPHA / 2, 1 - ALPHA / 2])
        width = res_q_hi - res_q_lo
        conditional_widths.append(width)
        X_test_point = X_test_fixed[i : i + 1]
        with torch.no_grad(), torch.cuda.amp.autocast(enabled=(USE_AMP and device == "cuda")):
            y_hat_test_point = mdl_resid(torch.tensor(X_test_point, device=device)).squeeze().float().cpu().item()
        lo_i = y_hat_test_point + res_q_lo
        hi_i = y_hat_test_point + res_q_hi
        rng_mc = np.random.default_rng(12345 + i)
        X_rep = np.repeat(X_test_point, N_MC, axis=0)
        y_rep = generate_y_for_x(X_rep, setup=current_setup, rng=rng_mc)
        cov_i = np.mean((y_rep >= lo_i) & (y_rep <= hi_i))
        conditional_coverages.append(cov_i)
    print(f"\n[{current_setup}] --- Step 4: Saving Results ---")
    results_df = pd.DataFrame({"x1": X_test_fixed[:, 0]})
    print(f"\nFinal Aggregated Metrics (Mean over all test points):")
    for name, data in results.items():
        mean_cov = data["C"].mean()
        mean_wid = data["W"].mean()
        print(f"  {name:30s} | Coverage: {mean_cov:.4f} | Width: {mean_wid:.4f}")
        results_df[f"coverage_{name}"] = data["C"].mean(axis=1)
        results_df[f"width_{name}"] = data["W"].mean(axis=1)

    mean_cond_cov = np.mean(conditional_coverages)
    mean_cond_wid = np.mean(conditional_widths)
    print(f"  {'Residual (Cond. Cal.)':30s} | Coverage: {mean_cond_cov:.4f} | Width: {mean_cond_wid:.4f}")

    results_df["width_Residual (Cond. Cal.)"] = conditional_widths
    results_df["coverage_Residual (Cond. Cal.)"] = conditional_coverages
    out_csv = f"simulation_results_setup_{current_setup}_baseline_plus_nozC2C3_plus_shortestC.csv"
    results_df.to_csv(out_csv, index=False)
    print(f"[{current_setup}] Results saved to {out_csv}")
    plt.style.use("seaborn-v0_8-whitegrid")
    sorted_indices = np.argsort(results_df["x1"].values)
    x1_sorted = results_df["x1"].iloc[sorted_indices].values
    all_width_methods = [
        METHOD_HAZARD_BASELINE,
        METHOD_NOZ_UCENTER,

        METHOD_RESIDUAL,
        METHOD_RESCALED,
        METHOD_CQR,
    ]
    if RUN_HAZARD_SHORTEST_C:
        all_width_methods.append(METHOD_HAZARD_SHORTEST)
        all_width_methods.append(METHOD_DCP_OPT)

    all_width_methods.append("Residual (Cond. Cal.)")
    markers = ["o", "s", "D", "^", "x", "v", "P", "*", "h", "+"]
    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for i, name in enumerate(all_width_methods):
        col = f"width_{name}"
        if col not in results_df.columns:
            continue
        width_sorted = results_df[col].iloc[sorted_indices].values
        ax.plot(
            x1_sorted,
            width_sorted,
            marker=markers[i % len(markers)],
            linestyle="-",
            label=name,
            markersize=4,
            alpha=0.9,
        )
    ax.set_title(f"Setup {current_setup}: Mean Interval Width vs. x1")
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plot_width_setup_{current_setup}_baseline_plus_nozC2C3_plus_shortestC.png")
    plt.close(fig)
    cov_methods = [
        METHOD_HAZARD_BASELINE,
        METHOD_NOZ_UCENTER,

        METHOD_RESIDUAL,
        METHOD_RESCALED,
        METHOD_CQR,
    ]
    if RUN_HAZARD_SHORTEST_C:
        cov_methods.append(METHOD_HAZARD_SHORTEST)
        cov_methods.append(METHOD_DCP_OPT)

    fig, ax = plt.subplots(1, 1, figsize=(12, 7))
    for i, name in enumerate(cov_methods):
        col = f"coverage_{name}"
        if col not in results_df.columns:
            continue
        coverage_sorted = results_df[col].iloc[sorted_indices].values
        ax.plot(
            x1_sorted,
            coverage_sorted,
            marker=markers[i % len(markers)],
            linestyle="-",
            label=name,
            markersize=4,
            alpha=0.9,
        )
    ax.axhline(y=1 - ALPHA, color="r", linestyle="--")
    ax.set_title(f"Setup {current_setup}: Coverage vs. x1")
    ax.set_ylim([-0.1, 1.1])
    ax.legend()
    ax.grid(True, linestyle="--", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"plot_coverage_setup_{current_setup}_baseline_plus_nozC2C3_plus_shortestC.png")
    plt.close(fig)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
def main():
    print(f"Device: {device}")
    #TARGET_SETUPS = [
    #    "0a", "0as_1.0", "0c1", "0c1s_0.05", "0c1s_0.1"
    #]
    TARGET_SETUPS = [
        "0a"
    ]
    print(f"Running experiments for Setups: {TARGET_SETUPS}")
    for s in TARGET_SETUPS:
        try:
            run_simulation(s)
        except Exception as e:
            print(f"Error running setup {s}: {e}")
            continue
    print("\nAll requested simulations completed.")
if __name__ == "__main__":
    main()
