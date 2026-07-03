import numpy as np

from risksim import AggregateLayer, Portfolio, PortfolioItem


class _Model:
    def sample(self, size=1, rng=None):
        gen = np.random.default_rng(rng)
        return gen.gamma(2.0, 500.0, size=size)


class _NoRngModel:
    def sample(self, size=1):
        return np.full(size, 3.0)


def test_portfolio_sample_reproducible():
    port = Portfolio([PortfolioItem("a", _Model()), PortfolioItem("b", _Model(), weight=0.5)])
    a = port.sample(1000, rng=11)
    b = port.sample(1000, rng=11)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, port.sample(1000, rng=12))


def test_simulate_threads_one_generator():
    port = Portfolio([PortfolioItem("a", _Model()), PortfolioItem("b", _Model())])
    lay = AggregateLayer(attachment=10_000.0, limit=5_000.0)
    r1 = port.simulate(500, contract=lay, rng=3)
    r2 = port.simulate(500, contract=lay, rng=3)
    assert np.array_equal(r1.gross_losses, r2.gross_losses)
    assert np.array_equal(r1.retained_losses, r2.retained_losses)
    # one generator threaded through -> component draws are distinct streams
    assert not np.array_equal(r1.component_losses[:, 0], r1.component_losses[:, 1])


def test_rngless_model_still_works_when_rng_omitted():
    port = Portfolio([PortfolioItem("c", _NoRngModel())])
    assert port.sample(4).tolist() == [3.0, 3.0, 3.0, 3.0]
