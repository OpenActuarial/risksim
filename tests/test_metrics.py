import numpy as np
import pytest

from risksim import metrics


def test_mean_variance_std_match_numpy() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0])

    assert metrics.mean(losses) == pytest.approx(np.mean(losses))
    assert metrics.variance(losses) == pytest.approx(np.var(losses))
    assert metrics.std(losses) == pytest.approx(np.std(losses))


def test_var_matches_numpy_inverted_cdf_quantile() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    assert metrics.var(losses, 0.5) == pytest.approx(
        np.quantile(losses, 0.5, method="inverted_cdf")
    )
    assert metrics.var(losses, 0.8) == pytest.approx(
        np.quantile(losses, 0.8, method="inverted_cdf")
    )


def test_tvar_matches_average_quantile_definition() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    # TVaR_q = (1/(1-q)) * integral_q^1 VaR_u du; at q = 0.8 with n = 5 the
    # rank is k = ceil(n*q) = 4, so the estimate is the mean of the top
    # n*(1-q) = 1 observation.
    assert metrics.tvar(losses, 0.8) == pytest.approx(4.0)
    # Non-integer rank: q = 0.7 -> k = 4, VaR = 3.0,
    # TVaR = (4.0 + 3.0 * (4 - 3.5)) / (5 * 0.3) = 11/3.
    assert metrics.tvar(losses, 0.7) == pytest.approx(11.0 / 3.0)


def test_prob_exceeding() -> None:
    losses = np.array([10.0, 20.0, 30.0, 40.0])

    assert metrics.prob_exceeding(losses, 25.0) == pytest.approx(0.5)
    assert metrics.prob_exceeding(losses, 40.0) == pytest.approx(0.0)


def test_summary_contains_expected_keys_and_values() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    out = metrics.summary(losses, quantiles=(0.5, 0.8))

    assert out["n_sims"] == 5
    assert out["mean"] == pytest.approx(np.mean(losses))
    assert out["std"] == pytest.approx(np.std(losses))
    assert out["min"] == pytest.approx(0.0)
    assert out["max"] == pytest.approx(4.0)
    assert out["var_50"] == pytest.approx(
        np.quantile(losses, 0.5, method="inverted_cdf")
    )
    assert out["var_80"] == pytest.approx(
        np.quantile(losses, 0.8, method="inverted_cdf")
    )

    # k = ceil(5 * 0.5) = 3, VaR = 2.0:
    # TVaR = (3 + 4 + 2*(3 - 2.5)) / (5 * 0.5) = 8 / 2.5 = 3.2.
    assert out["tvar_50"] == pytest.approx(3.2)


@pytest.mark.parametrize("q", [0.0, 1.0, -0.1, 1.1])
def test_invalid_q_raises(q: float) -> None:
    losses = np.array([1.0, 2.0, 3.0])

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        metrics.var(losses, q)

    with pytest.raises(ValueError, match="strictly between 0 and 1"):
        metrics.tvar(losses, q)


def test_empty_losses_raise() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        metrics.mean([])