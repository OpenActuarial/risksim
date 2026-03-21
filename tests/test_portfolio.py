import numpy as np
import pytest

from risksim.contracts import AggregateLayer, ContractProgram
from risksim.portfolio import Portfolio, PortfolioItem


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(size, self.value, dtype=float)

    def mean(self) -> float:
        return self.value

    def variance(self) -> float:
        return 0.0


class SequenceModel:
    def __init__(self, values: list[float]) -> None:
        self.values = np.asarray(values, dtype=float)

    def sample(self, size: int = 1) -> np.ndarray:
        if size != self.values.size:
            raise ValueError("unexpected size for SequenceModel")
        return self.values.copy()

    def mean(self) -> float:
        return float(np.mean(self.values))

    def variance(self) -> float:
        return float(np.var(self.values))


class AnalyticModel:
    def __init__(self, mean_: float, variance_: float) -> None:
        self._mean = float(mean_)
        self._variance = float(variance_)

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(size, self._mean, dtype=float)

    def mean(self) -> float:
        return self._mean

    def variance(self) -> float:
        return self._variance


class BadShapeModel:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.zeros((size, 1), dtype=float)

    def mean(self) -> float:
        return 0.0

    def variance(self) -> float:
        return 0.0


class WrongSizeModel:
    def sample(self, size: int = 1) -> np.ndarray:
        return np.zeros(size - 1, dtype=float)

    def mean(self) -> float:
        return 0.0

    def variance(self) -> float:
        return 0.0


def test_portfolio_item_applies_weight() -> None:
    item = PortfolioItem(name="test", model=ConstantModel(10.0), weight=2.5)
    samples = item.sample(size=4)

    np.testing.assert_allclose(samples, np.array([25.0, 25.0, 25.0, 25.0]))
    assert item.mean() == pytest.approx(25.0)
    assert item.variance() == pytest.approx(0.0)


def test_portfolio_sample_components_and_total_sample() -> None:
    portfolio = Portfolio(
        [
            PortfolioItem("a", SequenceModel([1.0, 2.0, 3.0])),
            PortfolioItem("b", SequenceModel([10.0, 20.0, 30.0]), weight=0.5),
        ]
    )

    component_losses = portfolio.sample_components(size=3)
    total_losses = portfolio.sample(size=3)

    expected_components = np.array(
        [
            [1.0, 5.0],
            [2.0, 10.0],
            [3.0, 15.0],
        ]
    )
    expected_total = np.array([6.0, 12.0, 18.0])

    np.testing.assert_allclose(component_losses, expected_components)
    np.testing.assert_allclose(total_losses, expected_total)


def test_portfolio_mean_and_variance_under_independence() -> None:
    portfolio = Portfolio(
        [
            PortfolioItem("a", AnalyticModel(mean_=10.0, variance_=4.0), weight=1.0),
            PortfolioItem("b", AnalyticModel(mean_=5.0, variance_=9.0), weight=2.0),
        ]
    )

    assert portfolio.mean() == pytest.approx(20.0)
    assert portfolio.variance() == pytest.approx(40.0)
    assert portfolio.std() == pytest.approx(np.sqrt(40.0))


def test_portfolio_simulate_without_contract() -> None:
    portfolio = Portfolio(
        [
            PortfolioItem("a", SequenceModel([10.0, 20.0, 30.0])),
            PortfolioItem("b", SequenceModel([1.0, 2.0, 3.0])),
        ]
    )

    result = portfolio.simulate(size=3)

    np.testing.assert_allclose(result.gross_losses, np.array([11.0, 22.0, 33.0]))
    np.testing.assert_allclose(result.losses, np.array([11.0, 22.0, 33.0]))
    assert result.component_names == ["a", "b"]


def test_portfolio_simulate_with_single_layer() -> None:
    portfolio = Portfolio([PortfolioItem("gross", ConstantModel(200.0))])
    contract = AggregateLayer(
        attachment=100.0,
        limit=50.0,
        share=1.0,
        name="agg_xol",
    )

    result = portfolio.simulate(size=4, contract=contract)

    np.testing.assert_allclose(result.gross_losses, np.array([200.0, 200.0, 200.0, 200.0]))
    np.testing.assert_allclose(result.ceded_losses, np.array([50.0, 50.0, 50.0, 50.0]))
    np.testing.assert_allclose(result.retained_losses, np.array([150.0, 150.0, 150.0, 150.0]))

    assert result.mean() == pytest.approx(150.0)
    assert result.gross_mean() == pytest.approx(200.0)
    assert result.ceded_mean() == pytest.approx(50.0)
    assert result.retained_mean() == pytest.approx(150.0)
    assert result.contract_name == "agg_xol"


def test_portfolio_simulate_with_contract_program() -> None:
    portfolio = Portfolio([PortfolioItem("gross", ConstantModel(350.0))])
    contract = ContractProgram(
        [
            AggregateLayer(attachment=100.0, limit=100.0, name="l1"),
            AggregateLayer(attachment=200.0, limit=100.0, name="l2"),
        ],
        name="tower",
    )

    result = portfolio.simulate(size=3, contract=contract)

    np.testing.assert_allclose(result.gross_losses, np.array([350.0, 350.0, 350.0]))
    np.testing.assert_allclose(result.ceded_losses, np.array([200.0, 200.0, 200.0]))
    np.testing.assert_allclose(result.retained_losses, np.array([150.0, 150.0, 150.0]))
    np.testing.assert_allclose(
        result.layer_losses,
        np.array(
            [
                [100.0, 100.0],
                [100.0, 100.0],
                [100.0, 100.0],
            ]
        ),
    )
    assert result.layer_names == ["l1", "l2"]
    assert result.layer_means() == {"l1": 100.0, "l2": 100.0}
    assert result.contract_name == "tower"


def test_portfolio_rejects_empty_items() -> None:
    with pytest.raises(ValueError, match="at least one PortfolioItem"):
        Portfolio([])


def test_portfolio_rejects_nonpositive_size() -> None:
    portfolio = Portfolio([PortfolioItem("a", ConstantModel(1.0))])

    with pytest.raises(ValueError, match="size must be positive"):
        portfolio.sample(size=0)


def test_portfolio_rejects_bad_sample_shape() -> None:
    portfolio = Portfolio([PortfolioItem("bad", BadShapeModel())])

    with pytest.raises(ValueError, match="must return a 1D array"):
        portfolio.sample(size=3)


def test_portfolio_rejects_wrong_sample_size() -> None:
    portfolio = Portfolio([PortfolioItem("bad", WrongSizeModel())])

    with pytest.raises(ValueError, match="returned 2 values instead of 3"):
        portfolio.sample(size=3)