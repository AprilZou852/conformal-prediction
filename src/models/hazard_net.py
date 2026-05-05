"""
Hazard Network (F1Net) for survival analysis and conformal prediction.

The hazard network is a 2-layer MLP that models the log-hazard function
in long-table format for event time prediction.
"""

import copy
from typing import Tuple
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, random_split

# Default configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = False
USE_AMP = True
TORCH_COMPILE = False


class F1Net(nn.Module):
    """
    Hazard Network: 2-hidden-layer MLP for log-hazard prediction.

    Takes features [start, end, x1, x2, ...] and outputs a scalar log-hazard
    value (clamped at 10.0 for stability).

    Args:
        in_dim (int): Input feature dimension.
        hidden (int): Number of hidden units per layer (default: 64).
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        """Initialize the hazard network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor of shape (batch_size, in_dim).

        Returns:
            Log-hazard values, shape (batch_size, 1).
        """
        return self.net(x)


def expand_long_table(
    X: np.ndarray,
    y: np.ndarray,
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Expand (X, y) into long-table format for survival training.

    Creates one row per (sample, time point) pair where sample experienced
    event at time y[i] or is censored up to y[i].

    Args:
        X: Features array, shape (n_samples, n_features).
        y: Event times, shape (n_samples,).

    Returns:
        (df, unique_y) where df is long-table DataFrame with columns
        [id, start, end, delta, x1, x2, ...] and unique_y is sorted unique event times.
    """
    uniq = np.sort(np.unique(y))
    t0 = uniq[0] - 1.0
    rows = []
    n_features = X.shape[1]

    for idx, (xi, yi) in enumerate(zip(X, y)):
        s = t0
        for t in uniq[uniq <= yi]:
            delta = int(t == yi)  # 1 if event at t, 0 otherwise
            row = [idx, s, t, delta] + list(xi)
            rows.append(row)
            s = t

    feature_names = [f"x{i+1}" for i in range(n_features)]
    columns = ["id", "start", "end", "delta"] + feature_names
    df = pd.DataFrame(rows, columns=columns)

    return df, uniq


def _make_dataloaders(
    Xt: np.ndarray,
    Y: np.ndarray,
    val_frac: float = 0.2,
    batch: int = 128,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for long-table training.

    Args:
        Xt: Normalized features array.
        Y: Target array [delta, dt] for each row.
        val_frac: Validation fraction.
        batch: Batch size.
        seed: Random seed for reproducibility.

    Returns:
        (train_loader, val_loader).
    """
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


def train_hazard_net(
    df: pd.DataFrame,
    in_dim: int,
    *,
    epochs: int = 500,
    batch: int = 256,
    val_frac: float = 0.2,
    patience: int = 40,
    seed_split: int = 0,
    hidden: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    device: str = DEVICE,
) -> Tuple[F1Net, np.ndarray, np.ndarray]:
    """
    Train hazard network on long-table format.

    The loss is the partial likelihood for continuous-time survival analysis:
        L = -(delta * log_hazard) + exp(log_hazard) * dt

    Early stopping on validation loss with patience.

    Args:
        df: Long-table DataFrame with columns [id, start, end, delta, x1, x2, ...].
        in_dim: Input feature dimension (typically 2 + n_covariates for [start, end, ...]).
        epochs: Maximum training epochs.
        batch: Batch size.
        val_frac: Validation split fraction.
        patience: Early stopping patience.
        seed_split: Seed for train/val split.
        hidden: Number of hidden units per layer.
        lr: Learning rate.
        weight_decay: L2 regularization.
        device: Device to train on ("cuda" or "cpu").

    Returns:
        (model, x_mu, x_sd) where model is trained F1Net, x_mu and x_sd are
        feature normalization statistics.
    """
    # Extract features and targets from long-table
    feature_cols = [f"x{i+1}" for i in range(in_dim - 2)]
    feature_cols = ["start", "end"] + feature_cols

    X = df[feature_cols].values.astype("float32")
    Y = np.c_[
        df["delta"].values.astype("float32"),
        (df["end"] - df["start"]).values.astype("float32"),
    ].astype("float32")

    # Normalize features
    mu, sg = X.mean(0), X.std(0)
    sg[sg == 0] = 1
    Xt = (X - mu) / sg

    # Create dataloaders
    tr_dl, va_dl = _make_dataloaders(Xt, Y, val_frac=val_frac, batch=batch, seed=seed_split)

    # Initialize model
    net = F1Net(in_dim=in_dim, hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        net = torch.compile(net)

    opt = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    def loss_fn(y_true, y_pred):
        """Partial likelihood loss for survival analysis."""
        y_pred = torch.clamp(y_pred, max=10.0)
        delta = y_true[:, :1]
        dt = y_true[:, 1:2]
        return -(delta * y_pred).mean() + (torch.exp(y_pred) * dt).mean()

    best_state, best_val, no_gain = copy.deepcopy(net.state_dict()), float("inf"), 0

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
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

    print(f"[train_hazard_net] Stopped at epoch {epoch + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    net.load_state_dict(best_state)
    return net, mu.astype("float32"), sg.astype("float32")


def train_hazard_ensemble(
    df: pd.DataFrame,
    in_dim: int,
    k: int = 15,
    select_topk: int = 5,
    *,
    epochs: int = 500,
    batch: int = 256,
    val_frac: float = 0.2,
    patience: int = 40,
    hidden: int = 64,
    lr: float = 5e-4,
    weight_decay: float = 0.0,
    device: str = DEVICE,
    verbose: bool = False,
) -> Tuple[list, list, list, list]:
    """
    Train ensemble of K hazard networks, select top-k by validation loss.

    Args:
        df: Long-table DataFrame.
        in_dim: Input dimension.
        k: Number of models to train.
        select_topk: Number of best models to return.
        epochs: Training epochs per model.
        batch: Batch size.
        val_frac: Validation fraction.
        patience: Early stopping patience.
        hidden: Hidden layer size.
        lr: Learning rate.
        weight_decay: L2 regularization.
        device: Training device.
        verbose: Print selection details.

    Returns:
        (models, mus, sgs, val_losses) lists of length select_topk.
    """
    models = []
    mus = []
    sgs = []
    val_losses = []

    for i in range(k):
        model, mu, sg = train_hazard_net(
            df, in_dim,
            epochs=epochs,
            batch=batch,
            val_frac=val_frac,
            patience=patience,
            seed_split=i,
            hidden=hidden,
            lr=lr,
            weight_decay=weight_decay,
            device=device,
        )
        models.append(model)
        mus.append(mu)
        sgs.append(sg)

        # Compute final validation loss
        net = model
        device_used = next(net.parameters()).device

        feature_cols = [f"x{i+1}" for i in range(in_dim - 2)]
        feature_cols = ["start", "end"] + feature_cols
        X = df[feature_cols].values.astype("float32")
        Y = np.c_[
            df["delta"].values.astype("float32"),
            (df["end"] - df["start"]).values.astype("float32"),
        ].astype("float32")

        X_norm = (X - mu) / sg
        _, va_dl = _make_dataloaders(X_norm, Y, val_frac=val_frac, batch=batch, seed=0)

        def loss_fn(y_true, y_pred):
            y_pred = torch.clamp(y_pred, max=10.0)
            delta = y_true[:, :1]
            dt = y_true[:, 1:2]
            return -(delta * y_pred).mean() + (torch.exp(y_pred) * dt).mean()

        net.eval()
        with torch.no_grad():
            losses = []
            for xb, yb in va_dl:
                xb = xb.to(device_used, non_blocking=True)
                yb = yb.to(device_used, non_blocking=True)
                losses.append(loss_fn(yb, net(xb)).item())

        final_val_loss = float(np.mean(losses)) if losses else float("inf")
        val_losses.append(final_val_loss)

    # Select top-k by validation loss
    indices = np.argsort(val_losses)[:select_topk]

    if verbose:
        print(f"[train_hazard_ensemble] Validation losses (top {select_topk}): {[val_losses[i] for i in indices]}")

    return (
        [models[i] for i in indices],
        [mus[i] for i in indices],
        [sgs[i] for i in indices],
        [val_losses[i] for i in indices],
    )
