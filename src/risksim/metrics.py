from __future__ import annotations

from typing import Any

import numpy as np

from ._validation import as_1d_float_array, validate_q


def mean(losses: np.ndarray | list[float]) -> float:
    arr = as_1d_float_array(losses)
    return float(np.mean(arr))


def variance(losses: np.ndarray | list[float], ddof: int = 0) -> float:
    arr = as_1d_float_array(losses)
    return float(np.var(arr, ddof=ddof))


def std(losses: np.ndarray | list[float], ddof: int = 0) -> float:
    arr = as_1d_float_array(losses)
    return float(np.std(arr, ddof=ddof))


def var(losses: np.ndarray | list[float], q: float) -> float:
    validate_q(q)
    arr = as_1d_float_array(losses)
    return float(np.quantile(arr, q))


def tvar(losses: np.ndarray | list[float], q: float) -> float:
    validate_q(q)
    arr = as_1d_float_array(losses)

    threshold = var(arr, q)
    tail = arr[arr >= threshold]

    if tail.size == 0:
        return threshold

    return float(np.mean(tail))


def prob_exceeding(losses: np.ndarray | list[float], threshold: float) -> float:
    arr = as_1d_float_array(losses)
    return float(np.mean(arr > threshold))


def _quantile_label(q: float) -> str:
    """Stable, collision-free percentile label.

    ``0.95 -> "95"``, ``0.99 -> "99"``, ``0.995 -> "99.5"``, ``0.999 -> "99.9"``.
    Integer percentiles are unchanged; sub-percent quantiles keep their
    precision instead of rounding to a colliding integer.
    """
    return f"{round(q * 100, 6):g}"


def summary(
    losses: np.ndarray | list[float],
    quantiles: tuple[float, ...] = (0.95, 0.99),
) -> dict[str, Any]:
    arr = as_1d_float_array(losses)

    out: dict[str, Any] = {
        "n_sims": int(arr.size),
        "mean": mean(arr),
        "std": std(arr),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }

    for q in quantiles:
        validate_q(q)
        label = _quantile_label(q)
        out[f"var_{label}"] = var(arr, q)
        out[f"tvar_{label}"] = tvar(arr, q)

    return out
