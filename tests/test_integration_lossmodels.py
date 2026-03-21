import numpy as np
import pytest

pytest.importorskip("lossmodels")

from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem


def test_portfolio_accepts_lossmodels_collective_models() -> None:
    np.random.seed(12345)

    medical = CollectiveRiskModel(
        Poisson(lam=2.0),
        Lognormal(mu=2.0, sigma=0.3),
    )
    rx = CollectiveRiskModel(
        Poisson(lam=1.0),
        Lognormal(mu=1.5, sigma=0.2),
    )

    portfolio = Portfolio(
        [
            PortfolioItem("medical", medical),
            PortfolioItem("rx", rx),
        ]
    )

    result = portfolio.simulate(size=50_000)

    expected_mean = medical.mean() + rx.mean()

    assert result.n_sims == 50_000
    assert np.all(result.gross_losses >= 0.0)
    assert result.gross_mean() == pytest.approx(expected_mean, rel=0.05)

    component_means = result.component_means()
    assert component_means["medical"] == pytest.approx(medical.mean(), rel=0.08)
    assert component_means["rx"] == pytest.approx(rx.mean(), rel=0.08)


def test_contract_program_works_with_lossmodels_collective_models() -> None:
    np.random.seed(54321)

    model = CollectiveRiskModel(
        Poisson(lam=4.0),
        Lognormal(mu=3.5, sigma=0.7),
    )

    portfolio = Portfolio([PortfolioItem("line", model)])

    program = ContractProgram(
        [
            AggregateLayer(attachment=50.0, limit=50.0, name="l1"),
            AggregateLayer(attachment=100.0, limit=100.0, name="l2"),
        ],
        name="tower",
    )

    result = portfolio.simulate(size=20_000, contract=program)

    assert result.layer_losses is not None
    assert result.layer_losses.shape == (20_000, 2)
    assert result.layer_names == ["l1", "l2"]
    assert result.contract_name == "tower"

    np.testing.assert_allclose(
        result.retained_losses + result.ceded_losses,
        result.gross_losses,
    )
    np.testing.assert_allclose(
        np.sum(result.layer_losses, axis=1),
        result.ceded_losses,
    )

    assert np.all(result.gross_losses >= 0.0)
    assert np.all(result.ceded_losses >= 0.0)
    assert np.all(result.retained_losses >= 0.0)

    layer_means = result.layer_means()
    assert layer_means["l1"] >= 0.0
    assert layer_means["l2"] >= 0.0