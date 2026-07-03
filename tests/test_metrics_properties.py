"""Risk-measure properties: coherence-style identities and exact hand cases."""
import numpy as np
import pytest

from risksim import metrics


def test_exact_hand_case():
    losses = np.arange(1.0, 101.0)
    assert metrics.var(losses, 0.9) == 90.0
    assert metrics.tvar(losses, 0.9) == pytest.approx(95.5, rel=1e-12)
    assert metrics.tvar(losses, 0.01) == pytest.approx(51.0, rel=1e-12)


def test_var_is_the_inverted_cdf_quantile():
    losses = np.random.default_rng(3).lognormal(7.0, 1.2, size=997)
    for q in (0.5, 0.9, 0.995):
        assert metrics.var(losses, q) == np.quantile(losses, q, method="inverted_cdf")


@pytest.mark.parametrize("q", [0.5, 0.9, 0.99])
def test_translation_and_homogeneity(q):
    losses = np.random.default_rng(4).gamma(2.0, 800.0, size=5000)
    c, lam = 1234.5, 2.5
    assert metrics.tvar(losses + c, q) == pytest.approx(metrics.tvar(losses, q) + c, rel=1e-12)
    assert metrics.tvar(lam * losses, q) == pytest.approx(lam * metrics.tvar(losses, q), rel=1e-12)
    assert metrics.var(losses + c, q) == pytest.approx(metrics.var(losses, q) + c, rel=1e-12)


def test_tvar_dominates_var_and_is_monotone_in_q():
    losses = np.random.default_rng(5).pareto(2.5, size=4000) * 1000.0
    qs = [0.5, 0.8, 0.9, 0.99]
    tvars = [metrics.tvar(losses, q) for q in qs]
    for q, t in zip(qs, tvars):
        assert t >= metrics.var(losses, q)
    assert all(b >= a for a, b in zip(tvars, tvars[1:]))


def test_summary_statistics_match_numpy():
    losses = np.random.default_rng(6).gamma(3.0, 400.0, size=2000)
    assert metrics.mean(losses) == pytest.approx(losses.mean(), rel=1e-12)
    assert metrics.std(losses) == pytest.approx(losses.std(), rel=1e-6)
    assert metrics.variance(losses) == pytest.approx(losses.var(), rel=1e-6)


def test_exceedance_probability_and_alias():
    losses = np.arange(1.0, 101.0)
    assert metrics.exceedance_probability(losses, 89.5) == pytest.approx(0.11, rel=1e-12)
    assert metrics.prob_exceeding(losses, 89.5) == metrics.exceedance_probability(losses, 89.5)


def test_q_validation():
    losses = np.arange(1.0, 11.0)
    for bad in (0.0, 1.0, -0.5, 2.0):
        with pytest.raises(ValueError):
            metrics.var(losses, bad)
