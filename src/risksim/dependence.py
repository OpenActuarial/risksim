"""Dependence between simulated components, without touching the samplers.

Independence across portfolio components is the classically dangerous
default: it overstates diversification in exactly the tail metrics this
package exists to compute. This module adds dependence by **reordering**
(Iman & Conover, 1982): simulate every component with whatever machinery
already exists, then permute each column of the results so their ranks
follow a target correlation. Marginals are preserved *exactly* -- each
column is a permutation of itself -- so nothing about any component's own
distribution changes; only which scenarios coincide.

Two honest limits, stated loudly. First, the target is a **rank**
correlation: with ``scores="normal"`` the induced Spearman correlation
matches the requested matrix to within the usual ``(6/pi) * asin(rho/2)``
distortion (under 0.02 absolute) -- but rank correlation is *not* tail
dependence, and normal scores produce joint extremes that are
asymptotically independent no matter how high ``rho`` is. If the risk
question is "do the components blow up together", use ``scores="t"`` with
a small ``df``: t scores put genuine mass on joint tail events at the
same rank correlation. Second, this imposes the dependence you assert; it
does not estimate dependence from data.

The portfolio recipe is two lines::

    matrix = np.column_stack([item.sample(n, rng) for item in items])
    total = impose_rank_correlation(matrix, corr, rng).sum(axis=1)
"""
from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["impose_rank_correlation"]


def _psd_factor(corr: np.ndarray, tol: float = 1e-8) -> np.ndarray:
    """A square-root factor ``F`` with ``F @ F.T == corr`` for a PSD ``corr``.

    Uses the symmetric eigendecomposition rather than Cholesky, so it accepts a
    *singular* (rank-deficient) correlation matrix -- e.g. two perfectly
    correlated components -- which is a valid positive-semidefinite correlation
    matrix that Cholesky rejects. Tiny negative eigenvalues from rounding are
    clamped to zero; a genuinely negative eigenvalue (an indefinite matrix)
    raises. Any factor with ``F @ F.T == corr`` imposes the target correlation on
    decorrelated scores, so a non-triangular root works for Iman-Conover.
    """
    w, vecs = np.linalg.eigh(corr)
    scale = max(1.0, float(w.max()))
    if float(w.min()) < -tol * scale:
        raise ValueError(
            "target_corr is not positive semidefinite (it has a negative "
            "eigenvalue); it is not a valid correlation matrix"
        )
    return vecs @ np.diag(np.sqrt(np.clip(w, 0.0, None)))


def impose_rank_correlation(
    samples: np.ndarray,
    target_corr: np.ndarray,
    rng: Any = None,
    scores: str = "normal",
    df: float = 5.0,
) -> np.ndarray:
    """Reorder simulated columns to a target rank correlation (Iman-Conover).

    Parameters
    ----------
    samples : ndarray, shape (n_sims, n_components)
        Independently simulated component outcomes. Not modified.
    target_corr : ndarray, shape (k, k)
        Desired correlation matrix: symmetric, unit diagonal, positive
        semidefinite.
    rng : optional
        Seed or ``numpy.random.Generator`` for the latent scores.
    scores : {"normal", "t"}
        Latent score family. ``"normal"`` gives rank correlation with no
        tail dependence; ``"t"`` adds joint-tail clustering, stronger for
        smaller ``df``.
    df : float
        Degrees of freedom for ``scores="t"``; must exceed 2 (the scores
        need a finite variance for the correlation target to mean
        anything).

    Returns
    -------
    ndarray
        Same shape as ``samples``; each column an exact permutation of the
        corresponding input column.
    """
    x = np.asarray(samples, dtype=float)
    if x.ndim != 2:
        raise ValueError("samples must be 2-D (n_sims, n_components)")
    n, k = x.shape
    if k < 2:
        raise ValueError("dependence needs at least two components")
    if n < k + 2:
        raise ValueError("more simulations than components are required")
    if not np.all(np.isfinite(x)):
        raise ValueError("samples must be finite")
    corr = np.asarray(target_corr, dtype=float)
    if corr.shape != (k, k):
        raise ValueError(f"target_corr must be {k}x{k} to match the samples")
    if not np.allclose(corr, corr.T, atol=1e-10):
        raise ValueError("target_corr must be symmetric")
    if not np.allclose(np.diag(corr), 1.0, atol=1e-10):
        raise ValueError("target_corr must have a unit diagonal")
    try:
        chol_target = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError:
        # Cholesky needs positive *definiteness*; a valid but singular PSD
        # correlation matrix (e.g. two perfectly correlated components) lands
        # here. Fall back to the eigen square root, which accepts PSD and only
        # rejects a genuinely indefinite matrix.
        chol_target = _psd_factor(corr)
    if scores not in ("normal", "t"):
        raise ValueError('scores must be "normal" or "t"')
    if scores == "t" and df <= 2.0:
        raise ValueError("t scores need df > 2 (finite variance)")

    gen = np.random.default_rng(rng)
    if scores == "normal":
        m = gen.standard_normal((n, k))
    else:
        m = gen.standard_t(df, size=(n, k))
    # remove the scores' incidental sample correlation, then impose target
    sample_corr = np.corrcoef(m, rowvar=False)
    chol_sample = np.linalg.cholesky(sample_corr)
    t = m @ np.linalg.inv(chol_sample).T @ chol_target.T

    out = np.empty_like(x)
    for j in range(k):
        ranks = t[:, j].argsort().argsort()
        out[:, j] = np.sort(x[:, j])[ranks]
    return out
