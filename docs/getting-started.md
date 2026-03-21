# Getting started

## Installation

Install `risksim` in editable mode for local development:

```bash
python3 -m pip install -e .
```

If you want to run the examples with `lossmodels` locally:

```bash
python3 -m pip install -e ../lossmodels
```

## Basic workflow

A typical `risksim` workflow is:

1. create one or more loss models
2. wrap them in `PortfolioItem`
3. combine them into a `Portfolio`
4. simulate gross losses
5. optionally apply an aggregate contract
6. inspect the resulting `SimulationResult`

## Minimal example

Any object with a `.sample(size)` method can be used in a `Portfolio`.

```python
import numpy as np

from risksim import AggregateLayer, Portfolio, PortfolioItem


class ConstantModel:
    def __init__(self, value: float) -> None:
        self.value = float(value)

    def sample(self, size: int = 1) -> np.ndarray:
        return np.full(size, self.value, dtype=float)

    def mean(self) -> float:
        return self.value

    def variance(self) -> float:
        return 0.0


medical = ConstantModel(200.0)
rx = ConstantModel(50.0)

portfolio = Portfolio(
    [
        PortfolioItem("medical", medical),
        PortfolioItem("rx", rx),
    ]
)

contract = AggregateLayer(
    attachment=100.0,
    limit=75.0,
    share=1.0,
    name="aggregate_xol",
)

result = portfolio.simulate(size=10_000, contract=contract)

print("net mean:", result.mean())
print("gross mean:", result.gross_mean())
print("ceded mean:", result.ceded_mean())
print("VaR 99%:", result.var(0.99))
print("TVaR 99%:", result.tvar(0.99))
print(result.summary())
```

## Using real models from `lossmodels`

`risksim` is intended to work with models from `lossmodels` that expose `.sample(size)`.

```python
from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import Portfolio, PortfolioItem

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
print(result.summary())
```

## Running tests

```bash
python3 -m pytest -q
```