# risksim

`risksim` is a Python package for portfolio-level loss simulation and aggregate treaty modeling.

It is designed to sit on top of `lossmodels`. In this setup:

- `lossmodels` provides stochastic loss models with a `.sample(size)` method
- `risksim` combines those models into portfolios
- `risksim` applies aggregate annual contracts to simulated losses
- `risksim` summarizes gross, ceded, and retained outcomes

## What `risksim` does

The current version focuses on:

- combining multiple simulated loss components into a portfolio
- applying a single aggregate annual layer
- applying a simple multi-layer aggregate contract program
- computing simulation-based summaries such as mean, VaR, and TVaR

## Example

```python
from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import AggregateLayer, Portfolio, PortfolioItem

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

contract = AggregateLayer(
    attachment=50.0,
    limit=100.0,
    name="agg_xol",
)

result = portfolio.simulate(size=50_000, contract=contract)
print(result.summary())
```

## Current public API

```python
from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem, SimulationResult
```

## Documentation

- [Getting started](getting-started.md)
- [Concepts](concepts.md)
- [Examples](examples.md)
- [API reference](api.md)