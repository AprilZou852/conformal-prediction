"""
Baseline neural network models for conformal prediction.

Includes SimpleNN for residual method, MeanVarianceNN for Rescaled,
and QuantileNN for CQR.
"""

import copy
from typing import Tuple
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader, random_split

# Default configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 0
PIN_MEMORY = False
USE_AMP = True
TORCH_COMPILE = False


class SimpleNN(nn.Module):
    """
    Simple 2-layer MLP for point prediction (Residual method).

    Args:
        in_dim (int): Input feature dimension.
        hidden (int): Number of hidden units per layer (default: 64).
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        """Initialize the network."""
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns shape (batch_size, 1)."""
        return self.net(x)


class MeanVarianceNN(nn.Module):
    """
    Neural network predicting both mean and log-variance (Rescaled method).

    Has a shared base network and separate heads for mean and log-variance output.

    Args:
        in_dim (int): Input feature dimension.
        hidden (int): Number of hidden units per layer (default: 64).
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        """Initialize the network."""
        super().__init__()
        self.base_net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Returns:
            (mean, log_var) each of shape (batch_size, 1).
        """
        base_out = self.base_net(x)
        return self.mean_head(base_out), self.log_var_head(base_out)


class QuantileNN(nn.Module):
    """
    Neural network for quantile regression (CQR method).

    Outputs lower and upper quantiles. Can be used as linear quantile regression
    (hidden=0) or as a neural quantile regressor.

    Args:
        in_dim (int): Input feature dimension.
        hidden (int): Hidden layer size. If 0, uses linear quantile regression.
    """

    def __init__(self, in_dim: int, hidden: int = 64):
        """Initialize the network."""
        super().__init__()
        if hidden == 0:
            # Linear quantile regression
            self.net = nn.Linear(in_dim, 2)
        else:
            # Shallow NN quantile regressor
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 2),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass. Returns shape (batch_size, 2) for [q_low, q_high]."""
        return self.net(x)


def gaussian_nll_loss(
    y_true: torch.Tensor,
    mean: torch.Tensor,
    log_var: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian negative log-likelihood loss.

    Loss = 0.5 * log_var + 0.5 * exp(-log_var) * (y - mean)^2

    Args:
        y_true: Target values, shape (batch_size, 1).
        mean: Predicted mean, shape (batch_size, 1).
        log_var: Predicted log-variance, shape (batch_size, 1).

    Returns:
        Scalar loss value.
    """
    return (0.5 * log_var + 0.5 * torch.exp(-log_var) * (y_true - mean) ** 2).mean()


def pinball_loss(
    y_true: torch.Tensor,
    y_pred: torch.Tensor,
    quantiles: list,
) -> torch.Tensor:
    """
    Pinball (quantile) loss.

    For each quantile q:
        loss_q = max((q - 1) * error, q * error)

    Args:
        y_true: Target values, shape (batch_size, 1).
        y_pred: Predicted quantiles, shape (batch_size, len(quantiles)).
        quantiles: List of quantile levels (e.g., [0.05, 0.95]).

    Returns:
        Scalar loss value.
    """
    losses = []
    for i, q in enumerate(quantiles):
        err = y_true - y_pred[:, i : i + 1]
        losses.append(torch.mean(torch.max((q - 1) * err, q * err)))
    return sum(losses)


def _make_xy_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    val_frac: float = 0.2,
    batch: int = 128,
    seed: int = 0,
) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders for (X, y) pairs.

    Args:
        X: Features, shape (n, p).
        y: Targets, shape (n,).
        val_frac: Validation fraction.
        batch: Batch size.
        seed: Random seed.

    Returns:
        (train_loader, val_loader).
    """
    ds = TensorDataset(
        torch.tensor(X, dtype=torch.float32),
        torch.tensor(y, dtype=torch.float32).unsqueeze(1),
    )
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


def train_simple_nn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 500,
    batch: int = 128,
    val_frac: float = 0.2,
    patience: int = 10,
    seed_split: int = 0,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = DEVICE,
) -> SimpleNN:
    """
    Train SimpleNN for point prediction (Residual method).

    Uses MSE loss with early stopping on validation loss.

    Args:
        X: Features, shape (n, p).
        y: Targets, shape (n,).
        epochs: Maximum training epochs.
        batch: Batch size.
        val_frac: Validation split fraction.
        patience: Early stopping patience.
        seed_split: Seed for train/val split.
        hidden: Number of hidden units per layer.
        lr: Learning rate.
        weight_decay: L2 regularization.
        device: Device to train on.

    Returns:
        Trained SimpleNN model (eval mode).
    """
    tr_dl, va_dl = _make_xy_dataloaders(X, y, val_frac=val_frac, batch=batch, seed=seed_split)

    model = SimpleNN(in_dim=X.shape[1], hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    crit = nn.MSELoss()
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), float("inf"), 0

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
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

    print(f"[train_simple_nn] Stopped at epoch {epoch + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_mean_variance_nn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    epochs: int = 500,
    batch: int = 128,
    val_frac: float = 0.2,
    patience: int = 10,
    seed_split: int = 0,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    clamp_logvar: bool = False,
    device: str = DEVICE,
) -> MeanVarianceNN:
    """
    Train MeanVarianceNN for mean + variance prediction (Rescaled method).

    Uses Gaussian NLL loss with early stopping on validation loss.

    Args:
        X: Features, shape (n, p).
        y: Targets, shape (n,).
        epochs: Maximum training epochs.
        batch: Batch size.
        val_frac: Validation split fraction.
        patience: Early stopping patience.
        seed_split: Seed for train/val split.
        hidden: Number of hidden units per layer.
        lr: Learning rate.
        weight_decay: L2 regularization.
        clamp_logvar: If True, clamp log-variance to [-1, 1].
        device: Device to train on.

    Returns:
        Trained MeanVarianceNN model (eval mode).
    """
    tr_dl, va_dl = _make_xy_dataloaders(X, y, val_frac=val_frac, batch=batch, seed=seed_split)

    model = MeanVarianceNN(in_dim=X.shape[1], hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), float("inf"), 0

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
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

    print(f"[train_mean_variance_nn] Stopped at epoch {epoch + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model


def train_quantile_nn(
    X: np.ndarray,
    y: np.ndarray,
    *,
    quantiles: list,
    epochs: int = 500,
    batch: int = 128,
    val_frac: float = 0.2,
    patience: int = 10,
    seed_split: int = 0,
    hidden: int = 64,
    lr: float = 1e-3,
    weight_decay: float = 0.0,
    device: str = DEVICE,
) -> QuantileNN:
    """
    Train QuantileNN for quantile prediction (CQR method).

    Uses pinball loss with early stopping on validation loss.

    Args:
        X: Features, shape (n, p).
        y: Targets, shape (n,).
        quantiles: List of quantile levels (e.g., [0.05, 0.95]).
        epochs: Maximum training epochs.
        batch: Batch size.
        val_frac: Validation split fraction.
        patience: Early stopping patience.
        seed_split: Seed for train/val split.
        hidden: Number of hidden units per layer.
        lr: Learning rate.
        weight_decay: L2 regularization.
        device: Device to train on.

    Returns:
        Trained QuantileNN model (eval mode).
    """
    tr_dl, va_dl = _make_xy_dataloaders(X, y, val_frac=val_frac, batch=batch, seed=seed_split)

    model = QuantileNN(in_dim=X.shape[1], hidden=hidden).to(device)
    if TORCH_COMPILE and hasattr(torch, "compile"):
        model = torch.compile(model)

    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(USE_AMP and device == "cuda"))

    best_state, best_val, no_gain = copy.deepcopy(model.state_dict()), float("inf"), 0

    for epoch in range(epochs):
        # Training phase
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

        # Validation phase
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

    print(f"[train_quantile_nn] Stopped at epoch {epoch + 1}/{epochs}, Best Val Loss: {best_val:.4f}")
    model.load_state_dict(best_state)
    model.eval()
    return model
