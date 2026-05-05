"""Helper functions for reproducibility and configuration."""

from contextlib import contextmanager
import random
import numpy as np
import torch


def set_all_seeds(seed: int) -> None:
    """
    Set all random seeds for reproducibility.

    Sets seeds for:
    - Python's random
    - NumPy
    - PyTorch
    - CUDA (if available)

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def preserve_rng_state():
    """
    Context manager to preserve and restore RNG state.

    Saves the state of NumPy, PyTorch, and CUDA RNGs at entry,
    restores them at exit. Useful for reproducible code blocks
    that should not affect subsequent random generation.

    Example:
        >>> with preserve_rng_state():
        ...     # Random operations here won't affect global RNG state
        ...     x = np.random.randn(10)
        >>> y = np.random.randn(10)  # Independent of operations in block
    """
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
