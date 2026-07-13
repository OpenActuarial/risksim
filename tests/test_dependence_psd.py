"""A singular but valid PSD correlation matrix must be accepted, not rejected.

Cholesky requires positive *definiteness*; two perfectly correlated components
give a positive-*semidefinite* correlation matrix that is a valid target. The
eigen fallback accepts it (and still rejects genuinely indefinite matrices).
"""

from __future__ import annotations

import numpy as np
import pytest

from risksim.dependence import _psd_factor, impose_rank_correlation


def _spearman(a, b):
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def test_perfectly_correlated_singular_matrix_accepted():
    rng = np.random.default_rng(0)
    x = rng.exponential(1.0, size=(4000, 2))
    corr = np.array([[1.0, 1.0], [1.0, 1.0]])  # singular, PSD, valid
    out = impose_rank_correlation(x, corr, rng=1)
    # comonotonic result, and each column an exact permutation of its input
    assert _spearman(out[:, 0], out[:, 1]) == pytest.approx(1.0, abs=1e-9)
    assert np.allclose(np.sort(out[:, 0]), np.sort(x[:, 0]))
    assert np.allclose(np.sort(out[:, 1]), np.sort(x[:, 1]))


def test_rank_deficient_three_component_matrix_accepted():
    rng = np.random.default_rng(2)
    x = rng.standard_normal(size=(6000, 3))
    # components 0 and 1 perfectly correlated -> rank-deficient PSD
    corr = np.array([[1.0, 1.0, 0.3], [1.0, 1.0, 0.3], [0.3, 0.3, 1.0]])
    out = impose_rank_correlation(x, corr, rng=3)
    assert _spearman(out[:, 0], out[:, 1]) == pytest.approx(1.0, abs=1e-9)


def test_indefinite_matrix_rejected():
    rng = np.random.default_rng(4)
    x = rng.standard_normal(size=(500, 3))
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99], [-0.99, 0.99, 1.0]])
    with pytest.raises(ValueError, match="not positive semidefinite"):
        impose_rank_correlation(x, bad)


def test_psd_factor_reconstructs_matrix():
    corr = np.array([[1.0, 0.6, 0.2], [0.6, 1.0, 0.4], [0.2, 0.4, 1.0]])
    f = _psd_factor(corr)
    assert np.allclose(f @ f.T, corr, atol=1e-10)
    singular = np.array([[1.0, 1.0], [1.0, 1.0]])
    fs = _psd_factor(singular)
    assert np.allclose(fs @ fs.T, singular, atol=1e-10)
