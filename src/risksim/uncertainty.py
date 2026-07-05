"""Monte Carlo error quantification for simulation output.

A simulated VaR without an error estimate is a random number with
confidence. Every function here takes the loss vector a simulation
produced and answers "how much of this is signal": normal-theory intervals
for the mean, distribution-free order-statistic intervals for quantiles,
and bootstrap intervals for anything else (TVaR in particular). Each metric
gets the standard tool for that metric -- mixing them in one summary is
deliberate, because pretending one method fits all is how tail estimates
end up with body-sized error bars.

All estimates use the same conventions as :mod:`risksim.metrics` (the
lower-quantile order statistic for VaR), so the point values in
:func:`summary_with_error` match :func:`risksim.metrics.summary` exactly.
"""
from __future__ import annotations

from statistics import NormalDist
from typing import Any, Callable

import numpy as np

from ._validation import as_1d_float_array, validate_q
from .metrics import _quantile_label, _var_rank, mean, std, tvar

__all__ = ["bootstrap_ci", "mean_ci", "quantile_ci", "summary_with_error"]


def _z(confidence: float) -> float:
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    return NormalDist().inv_cdf(0.5 + confidence / 2.0)


def mean_ci(
    losses: np.ndarray | list[float], confidence: float = 0.95
) -> dict[str, float]:
    """Normal-theory confidence interval for the simulated mean.

    The standard error is ``std(losses, ddof=1) / sqrt(n)``; with the
    sample sizes simulations run at, the normal interval is exact for all
    practical purposes.

    Returns
    -------
    dict
        ``estimate``, ``se``, ``ci_low``, ``ci_high``.
    """
    arr = as_1d_float_array(losses)
    if arr.size < 2:
        raise ValueError("at least two simulations are required")
    z = _z(confidence)
    est = mean(arr)
    se = std(arr, ddof=1) / np.sqrt(arr.size)
    return {
        "estimate": est,
        "se": float(se),
        "ci_low": float(est - z * se),
        "ci_high": float(est + z * se),
    }


def quantile_ci(
    losses: np.ndarray | list[float], q: float, confidence: float = 0.95
) -> dict[str, float]:
    """Distribution-free confidence interval for an empirical quantile.

    The number of observations at or below the true ``q``-quantile is
    Binomial(n, q); the interval takes the order statistics at ranks
    ``k -/+ z * sqrt(n q (1-q))`` around the VaR rank ``k = ceil(n q)`` --
    no distributional assumption on the losses at all. Ranks clip to the
    sample: when ``n`` is too small for the requested tail, the bound
    honestly sits at the extreme order statistic rather than extrapolating.

    ``se`` is ``nan`` by design: a quantile has no distribution-free
    standard error (its asymptotic variance involves the unknown density);
    the interval *is* the uncertainty statement.

    Returns
    -------
    dict
        ``estimate``, ``se`` (``nan``), ``ci_low``, ``ci_high``.
    """
    validate_q(q)
    arr = np.sort(as_1d_float_array(losses))
    n = arr.size
    if n < 2:
        raise ValueError("at least two simulations are required")
    z = _z(confidence)
    k = int(_var_rank(n, np.asarray(float(q))))
    half = z * np.sqrt(n * q * (1.0 - q))
    k_lo = int(np.clip(np.floor(k - half), 1, n))
    k_hi = int(np.clip(np.ceil(k + half), 1, n))
    return {
        "estimate": float(arr[k - 1]),
        "se": float("nan"),
        "ci_low": float(arr[k_lo - 1]),
        "ci_high": float(arr[k_hi - 1]),
    }


def bootstrap_ci(
    losses: np.ndarray | list[float],
    statistic: Callable[[np.ndarray], float],
    n_boot: int = 1000,
    confidence: float = 0.95,
    rng: Any = None,
) -> dict[str, float]:
    """Percentile-bootstrap confidence interval for any statistic.

    Resamples the loss vector with replacement ``n_boot`` times and takes
    the empirical quantiles of the replicated statistic. The workhorse for
    statistics with no clean sampling theory -- TVaR above all.

    Parameters
    ----------
    losses : array-like
        Simulated losses.
    statistic : callable
        Maps a 1-d array to a float, e.g. ``lambda a: tvar(a, 0.99)``.
    n_boot : int
        Bootstrap replicates.
    confidence : float
        Interval level.
    rng : optional
        Seed or ``numpy.random.Generator`` for reproducibility.

    Returns
    -------
    dict
        ``estimate`` (the statistic on the full sample -- not the replicate
        mean), ``se`` (replicate standard deviation), ``ci_low``,
        ``ci_high`` (percentile bounds).
    """
    arr = as_1d_float_array(losses)
    if arr.size < 2:
        raise ValueError("at least two simulations are required")
    if n_boot < 2:
        raise ValueError("n_boot must be at least 2")
    if not callable(statistic):
        raise TypeError("statistic must be callable")
    _z(confidence)  # validate early
    gen = np.random.default_rng(rng)
    n = arr.size
    reps = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        reps[b] = float(statistic(arr[gen.integers(0, n, n)]))
    alpha = 1.0 - confidence
    lo, hi = np.quantile(reps, [alpha / 2.0, 1.0 - alpha / 2.0])
    return {
        "estimate": float(statistic(arr)),
        "se": float(np.std(reps, ddof=1)),
        "ci_low": float(lo),
        "ci_high": float(hi),
    }


def summary_with_error(
    losses: np.ndarray | list[float],
    quantiles: tuple[float, ...] = (0.95, 0.99),
    confidence: float = 0.95,
    n_boot: int = 1000,
    rng: Any = None,
) -> dict[str, dict[str, float]]:
    """:func:`risksim.metrics.summary`, with error bars on every metric.

    Point estimates match ``metrics.summary`` exactly; each metric carries
    the interval its sampling theory supports: normal theory for the mean,
    order statistics for VaR (``se`` is ``nan`` there -- see
    :func:`quantile_ci`), bootstrap for TVaR.

    Returns
    -------
    dict of str -> dict
        Keys like ``"mean"``, ``"var_95"``, ``"tvar_99"``; each value has
        ``estimate``, ``se``, ``ci_low``, ``ci_high``.
    """
    arr = as_1d_float_array(losses)
    gen = np.random.default_rng(rng)
    out: dict[str, dict[str, float]] = {"mean": mean_ci(arr, confidence)}
    for q in quantiles:
        validate_q(q)
        label = _quantile_label(q)
        out[f"var_{label}"] = quantile_ci(arr, q, confidence)
        out[f"tvar_{label}"] = bootstrap_ci(
            arr, lambda a, _q=q: tvar(a, _q), n_boot=n_boot,
            confidence=confidence, rng=gen,
        )
    return out
