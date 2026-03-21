# risksim

`risksim` is a Python package for portfolio-level loss simulation and aggregate treaty modeling.

It is designed to sit on top of stochastic loss models that expose a `.sample(size)` method. In particular, it pairs naturally with `lossmodels`, where `lossmodels` handles loss-model generation and `risksim` handles portfolio composition, aggregate contracts, and simulation summaries.

## Features

- combine multiple simulated loss components into a portfolio
- apply aggregate annual layers
- build simple multi-layer aggregate contract programs
- summarize gross, ceded, and retained loss
- compute simulation-based risk measures such as mean, VaR, and TVaR

## Installation

### From PyPI

```bash
pip install risksim
```

### From source

```bash
git clone https://github.com/michaelabryant/risksim.git
cd risksim
pip install -e .
```

## Optional companion package

`risksim` works with any model that implements:

```python
sample(size: int = 1) -> np.ndarray
```

Many of the examples below use [`lossmodels`](https://github.com/michaelabryant/lossmodels), which provides aggregate actuarial loss models that integrate naturally with `risksim`.

If you want to run the `lossmodels` examples locally, install `lossmodels` as well.

## Quick start

Any object with a `.sample(size)` method can be used inside a `Portfolio`.

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

## Using `lossmodels`

A common workflow is to build aggregate loss models with `lossmodels` and then combine them with `risksim`.

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

## Contract programs

`risksim` also supports simple multi-layer aggregate contract programs.

```python
from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem

line = CollectiveRiskModel(
    Poisson(lam=4.0),
    Lognormal(mu=3.5, sigma=0.7),
)

portfolio = Portfolio([PortfolioItem("line", line)])

program = ContractProgram(
    [
        AggregateLayer(attachment=50.0, limit=50.0, name="layer_1"),
        AggregateLayer(attachment=100.0, limit=100.0, name="layer_2"),
    ],
    name="two_layer_tower",
)

result = portfolio.simulate(size=20_000, contract=program)

print("gross mean:", result.gross_mean())
print("ceded mean:", result.ceded_mean())
print("retained mean:", result.retained_mean())
print("layer means:", result.layer_means())
```

## Public API

```python
from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem, SimulationResult
```

## Core concepts

### `PortfolioItem`
Wraps one simulated component with a name and optional weight.

### `Portfolio`
Combines multiple simulated components into a total portfolio loss.

### `AggregateLayer`
Applies a single aggregate annual contract to simulated gross losses.

For annual gross loss `S`, ceded loss is:

```text
share * min(max(S - attachment, 0), limit)
```

with no cap if `limit=None`.

### `ContractProgram`
Combines multiple aggregate layers applied to the same gross loss.

### `SimulationResult`
Stores simulation outputs and provides convenience methods such as:

- `mean()`
- `variance()`
- `std()`
- `var(q)`
- `tvar(q)`
- `prob_exceeding(threshold)`
- `summary()`

If retained losses are present, the primary `losses` view is net/retained loss. Otherwise it is gross loss.

## Example scripts

The `examples/` directory contains runnable scripts:

- `examples/basic_portfolio.py`
- `examples/aggregate_layer.py`
- `examples/contract_program.py`

Run them with:

```bash
python examples/basic_portfolio.py
python examples/aggregate_layer.py
python examples/contract_program.py
```

## Project scope

The current version focuses on:

- portfolio-level simulation from component loss models
- aggregate annual treaty application
- simple non-overlapping multi-layer programs
- simulation-based portfolio summaries

It does not yet model:

- dependence between components
- occurrence-based contracts requiring claim-level paths
- reinstatements
- capital allocation
- premium or pricing workflows beyond simulated summaries

## Development

Run the test suite with:

```bash
python -m pytest -q
```

## License

MIT