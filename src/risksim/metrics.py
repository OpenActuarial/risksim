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


def _var_rank(n: int, q: np.ndarray) -> np.ndarray:
    """Rank k of the VaR order statistic: k = ceil(n*q), guarded against
    floating-point error when n*q is an exact integer."""
    nq = n * q
    k = np.where(np.abs(nq - np.round(nq)) < 1e-8, np.round(nq), np.ceil(nq))
    return np.clip(k.astype(np.intp), 1, n)


def var(losses: np.ndarray | list[float], q: float | np.ndarray) -> float | np.ndarray:
    """Empirical Value-at-Risk.

    Uses the actuarial (lower-quantile) definition

        VaR_q(X) = inf{ x : F(x) >= q },

    whose empirical plug-in is the order statistic ``x_(ceil(n*q))``.
    Equivalent to ``np.quantile(losses, q, method="inverted_cdf")``.

    ``q`` may be a scalar (returns ``float``) or array-like
    (returns ``np.ndarray`` of the same length).
    """
    validate_q(q)
    arr = np.sort(as_1d_float_array(losses))
    q_arr = np.asarray(q, dtype=float)
    k = _var_rank(arr.size, q_arr)
    out = arr[k - 1]
    return float(out) if q_arr.ndim == 0 else np.asarray(out, dtype=float)


def tvar(losses: np.ndarray | list[float], q: float | np.ndarray) -> float | np.ndarray:
    """Empirical Tail Value-at-Risk (expected shortfall).

    Uses the average-quantile (coherent) definition

        TVaR_q(X) = (1 / (1 - q)) * integral_q^1 VaR_u(X) du,

    whose empirical plug-in (Acerbi-Tasche) is, with sorted losses
    ``x_(1) <= ... <= x_(n)`` and ``k = ceil(n*q)``,

        TVaR_q = [ sum_{i>k} x_(i) + x_(k) * (k - n*q) ] / (n * (1 - q)).

    This is exact for the empirical distribution (correct with ties/atoms)
    and reduces to the mean of the largest ``n*(1-q)`` observations when
    ``n*q`` is an integer. Always satisfies ``tvar >= var``.

    ``q`` may be a scalar (returns ``float``) or array-like
    (returns ``np.ndarray`` of the same length).
    """
    validate_q(q)
    arr = np.sort(as_1d_float_array(losses))
    n = arr.size
    q_arr = np.asarray(q, dtype=float)
    k = _var_rank(n, q_arr)
    csum = np.concatenate(([0.0], np.cumsum(arr)))
    tail_sum = csum[n] - csum[k]
    var_vals = arr[k - 1]
    nq = n * q_arr
    weight = np.where(np.abs(nq - np.round(nq)) < 1e-8, 0.0, k - nq)
    out = (tail_sum + var_vals * weight) / (n * (1.0 - q_arr))
    # TVaR >= VaR holds as a theorem in exact arithmetic; enforce it so
    # floating-point noise (e.g. a constant tail at a layer limit) can never
    # produce tvar infinitesimally below var.
    out = np.maximum(out, var_vals)
    return float(out) if q_arr.ndim == 0 else np.asarray(out, dtype=float)


def prob_exceeding(losses: np.ndarray | list[float], threshold: float) -> float:
    arr = as_1d_float_array(losses)
    return float(np.mean(arr > threshold))


# Ecosystem-standard name (lossmodels and extremeloss use this spelling).
exceedance_probability = prob_exceeding


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
