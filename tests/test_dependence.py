"""Iman-Conover reordering: exact marginals, achieved rank correlation,
and the tail-dependence contrast between normal and t scores."""
import numpy as np
import pytest

from risksim.dependence import impose_rank_correlation


def _spearman(a, b):
    ra = a.argsort().argsort().astype(float)
    rb = b.argsort().argsort().astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


@pytest.fixture()
def components():
    rng = np.random.default_rng(7)
    return np.column_stack([
        rng.lognormal(10.0, 1.0, 4000),
        rng.gamma(2.0, 5_000.0, 4000),
        rng.exponential(20_000.0, 4000),
    ])


def test_marginals_preserved_exactly(components):
    corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    out = impose_rank_correlation(components, corr, rng=1)
    for j in range(3):
        assert np.array_equal(np.sort(out[:, j]), np.sort(components[:, j]))
    # the input itself is untouched
    assert not np.array_equal(out, components)


def test_achieved_rank_correlation(components):
    corr = np.array([[1.0, 0.6, 0.3], [0.6, 1.0, 0.4], [0.3, 0.4, 1.0]])
    out = impose_rank_correlation(components, corr, rng=1)
    for (i, j) in [(0, 1), (0, 2), (1, 2)]:
        assert _spearman(out[:, i], out[:, j]) == pytest.approx(
            corr[i, j], abs=0.03)
    # identity target leaves the columns effectively independent
    indep = impose_rank_correlation(components, np.eye(3), rng=2)
    assert abs(_spearman(indep[:, 0], indep[:, 1])) < 0.05


def test_reproducible(components):
    corr = np.array([[1.0, 0.5, 0.0], [0.5, 1.0, 0.0], [0.0, 0.0, 1.0]])
    a = impose_rank_correlation(components, corr, rng=42)
    b = impose_rank_correlation(components, corr, rng=42)
    assert np.array_equal(a, b)


def test_t_scores_add_joint_tail_mass():
    """Same rank correlation, very different joint tails: t(3) scores put
    real mass on both-components-extreme scenarios, normal scores do not
    -- the documented reason scores='t' exists."""
    rng = np.random.default_rng(3)
    x = np.column_stack([rng.exponential(1.0, 120_000),
                         rng.exponential(1.0, 120_000)])
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])
    q = 0.99
    counts = {}
    for label, kw in (("normal", {"scores": "normal"}),
                      ("t", {"scores": "t", "df": 3.0})):
        out = impose_rank_correlation(x, corr, rng=9, **kw)
        thr0 = np.quantile(out[:, 0], q)
        thr1 = np.quantile(out[:, 1], q)
        counts[label] = int(np.sum((out[:, 0] > thr0) & (out[:, 1] > thr1)))
        # rank correlation is the same either way
        assert _spearman(out[:, 0], out[:, 1]) == pytest.approx(0.6, abs=0.02)
    assert counts["normal"] > 20
    assert counts["t"] > 1.4 * counts["normal"], counts


def test_guards(components):
    good = np.array([[1.0, 0.5], [0.5, 1.0]])
    with pytest.raises(ValueError, match="2-D"):
        impose_rank_correlation(components[:, 0], good)
    with pytest.raises(ValueError, match="match the samples"):
        impose_rank_correlation(components, good)
    bad = np.array([[1.0, 0.99, -0.99], [0.99, 1.0, 0.99],
                    [-0.99, 0.99, 1.0]])
    with pytest.raises(ValueError, match="not positive semidefinite"):
        impose_rank_correlation(components, bad)
    diag = np.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    with pytest.raises(ValueError, match="unit diagonal"):
        impose_rank_correlation(components, diag)
    with pytest.raises(ValueError, match="df > 2"):
        impose_rank_correlation(
            components, np.eye(3), scores="t", df=2.0)
