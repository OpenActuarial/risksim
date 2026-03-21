from __future__ import annotations

from typing import Any

import numpy as np


def _as_1d_float_array(losses: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(losses, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError("losses must be a one-dimensional array-like object")

    if arr.size == 0:
        raise ValueError("losses must not be empty")

    return arr


def _validate_q(q: float) -> None:
    if not (0 < q < 1):
        raise ValueError("q must be strictly between 0 and 1")


def mean(losses: np.ndarray | list[float]) -> float:
    arr = _as_1d_float_array(losses)
    return float(np.mean(arr))


def variance(losses: np.ndarray | list[float], ddof: int = 0) -> float:
    arr = _as_1d_float_array(losses)
    return float(np.var(arr, ddof=ddof))


def std(losses: np.ndarray | list[float], ddof: int = 0) -> float:
    arr = _as_1d_float_array(losses)
    return float(np.std(arr, ddof=ddof))


def var(losses: np.ndarray | list[float], q: float) -> float:
    _validate_q(q)
    arr = _as_1d_float_array(losses)
    return float(np.quantile(arr, q))


def tvar(losses: np.ndarray | list[float], q: float) -> float:
    _validate_q(q)
    arr = _as_1d_float_array(losses)

    threshold = var(arr, q)
    tail = arr[arr >= threshold]

    if tail.size == 0:
        return threshold

    return float(np.mean(tail))


def prob_exceeding(losses: np.ndarray | list[float], threshold: float) -> float:
    arr = _as_1d_float_array(losses)
    return float(np.mean(arr > threshold))


def summary(
    losses: np.ndarray | list[float],
    quantiles: tuple[float, ...] = (0.95, 0.99),
) -> dict[str, Any]:
    arr = _as_1d_float_array(losses)

    out: dict[str, Any] = {
        "n_sims": int(arr.size),
        "mean": mean(arr),
        "std": std(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

    for q in quantiles:
        _validate_q(q)
        pct = int(round(q * 100))
        out[f"var_{pct}"] = var(arr, q)
        out[f"tvar_{pct}"] = tvar(arr, q)

    return out