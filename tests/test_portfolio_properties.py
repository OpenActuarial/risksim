"""Portfolio moments, component integrity, and SimulationResult structure."""
import numpy as np
import pytest

from risksim import AggregateLayer, Portfolio, PortfolioItem


class GammaModel:
    def __init__(self, k, theta):
        self.k, self.theta = k, theta

    def mean(self):
        return self.k * self.theta

    def variance(self):
        return self.k * self.theta ** 2

    def sample(self, size=1, rng=None):
        return np.random.default_rng(rng).gamma(self.k, self.theta, size=size)


def _port():
    return Portfolio([PortfolioItem("a", GammaModel(2.0, 500.0)),
                      PortfolioItem("b", GammaModel(3.0, 300.0), weight=0.5)])


def test_analytic_moments_follow_the_weights():
    port = _port()
    assert port.mean() == pytest.approx(2 * 500 + 0.5 * 3 * 300, rel=1e-12)
    assert port.variance() == pytest.approx(2 * 500**2 + 0.25 * 3 * 300**2, rel=1e-12)


def test_sample_moments_match_analytic():
    port = _port()
    s = port.sample(200_000, rng=8)
    assert s.mean() == pytest.approx(port.mean(), rel=0.01)
    assert s.var() == pytest.approx(port.variance(), rel=0.03)


def test_components_sum_to_gross_exactly():
    res = _port().simulate(1000, rng=2)
    assert res.component_losses.shape == (1000, 2)
    assert res.component_names == ["a", "b"]
    assert np.allclose(res.component_losses.sum(axis=1), res.gross_losses, atol=1e-9)


def test_no_contract_means_no_cession_fields():
    res = _port().simulate(50, rng=1)
    assert res.ceded_losses is None
    assert res.retained_losses is None
    assert res.layer_losses is None
    assert res.contract_name is None


def test_single_layer_contract_conserves_without_layer_detail():
    lay = AggregateLayer(1500.0, 1000.0, name="agg_xs")
    res = _port().simulate(2000, contract=lay, rng=3)
    assert np.allclose(res.gross_losses, res.ceded_losses + res.retained_losses, atol=1e-9)
    assert res.contract_name == "agg_xs"
    # per-layer detail is populated only for a ContractProgram
    assert res.layer_losses is None and res.layer_names is None


def test_program_contract_reports_per_layer_detail():
    from risksim import ContractProgram
    program = ContractProgram([AggregateLayer(1000.0, 500.0, name="working"),
                               AggregateLayer(1500.0, None, name="cat")])
    res = _port().simulate(2000, contract=program, rng=3)
    assert np.allclose(res.gross_losses, res.ceded_losses + res.retained_losses, atol=1e-9)
    arr = np.asarray(res.layer_losses)
    assert arr.shape == (2000, 2)
    assert np.allclose(arr.sum(axis=1), res.ceded_losses, atol=1e-9)
    assert list(res.layer_names) == ["working", "cat"]
