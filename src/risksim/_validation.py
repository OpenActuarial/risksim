"""Shared input-validation helpers for risksim."""

from __future__ import annotations

import numpy as np


def as_1d_float_array(losses: np.ndarray | list[float]) -> np.ndarray:
    """Coerce ``losses`` to a 1-D float array, rejecting empty or >1-D input."""
    arr = np.asarray(losses, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError("losses must be a one-dimensional array-like object")

    if arr.size == 0:
        raise ValueError("losses must not be empty")

    return arr


def validate_q(q: float) -> None:
    """Validate that a quantile level ``q`` lies strictly in ``(0, 1)``."""
    if not (0.0 < q < 1.0):
        raise ValueError("q must be strictly between 0 and 1")
