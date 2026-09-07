"""Seed management for reproducibility."""
from __future__ import annotations

import random
from typing import Any

import numpy as np


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch  # optional
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def derive_seed(base_seed: int, generation: int, member_id: int) -> int:
    """Derive per-member deterministic seed."""
    return (base_seed * 100000 + generation * 1000 + member_id) & 0x7FFFFFFF


def get_rng(seed: int | None = None) -> np.random.Generator:
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(seed)
