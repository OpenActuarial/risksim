import numpy as np
import pytest

from risksim import metrics


def test_mean_variance_std_match_numpy() -> None:
    losses = np.array([1.0, 2.0, 3.0, 4.0])

    assert metrics.mean(losses) == pytest.approx(np.mean(losses))
    assert metrics.variance(losses) == pytest.approx(np.var(losses))
    assert metrics.std(losses) == pytest.approx(np.std(losses))


def test_var_matches_numpy_quantile() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    assert metrics.var(losses, 0.5) == pytest.approx(np.quantile(losses, 0.5))
    assert metrics.var(losses, 0.8) == pytest.approx(np.quantile(losses, 0.8))


def test_tvar_matches_implemented_tail_definition() -> None:
    losses = np.array([0.0, 1.0, 2.0, 3.0, 4.0])

    q = 0.8
    threshold = np.quantile(losses, q)
    expected = np.mean(losses[losses >= threshold])

    assert metrics.tvar(losses, q) == pytest.approx(expected)


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
    assert out["var_50"] == pytest.approx(np.quantile(losses, 0.5))
    assert out["var_80"] == pytest.approx(np.quantile(losses, 0.8))

    threshold_50 = np.quantile(losses, 0.5)
    expected_tvar_50 = np.mean(losses[losses >= threshold_50])
    assert out["tvar_50"] == pytest.approx(expected_tvar_50)


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