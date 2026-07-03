# API reference

## Public imports

```python
from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem, SimulationResult
```

## `PortfolioItem`

Represents one simulated component in a portfolio.

### Parameters

- `name: str`  
  Component name.

- `model`  
  Any object with a `.sample(size)` method.

- `weight: float = 1.0`  
  Multiplicative weight applied to simulated losses.

### Methods

#### `sample(size: int) -> np.ndarray`
Return weighted simulated losses for the component.

#### `mean() -> float`
Return weighted mean, using the underlying model's `mean()`.

#### `variance() -> float`
Return weighted variance, using the underlying model's `variance()`.

---

## `Portfolio`

Represents a portfolio of simulated aggregate-loss components.

### Parameters

- `items: Sequence[PortfolioItem]`
- `name: str = "portfolio"`

### Methods

#### `component_names() -> list[str]`
Return the component names.

#### `sample_components(size: int = 1) -> np.ndarray`
Return a 2D array with one column per component.

#### `sample(size: int = 1) -> np.ndarray`
Return total gross simulated losses.

#### `mean() -> float`
Return analytic portfolio mean under the current independence assumption.

#### `variance() -> float`
Return analytic portfolio variance under the current independence assumption.

#### `std() -> float`
Return analytic portfolio standard deviation.

#### `simulate(size: int = 100_000, contract=None) -> SimulationResult`
Simulate gross losses and optionally apply an aggregate contract.

Accepted contracts:

- `AggregateLayer`
- `ContractProgram`

---

## `AggregateLayer`

Represents a single aggregate annual contract layer.

### Parameters

- `attachment: float = 0.0`
- `limit: float | None = None`
- `share: float = 1.0`
- `name: str | None = None`

### Methods

#### `ceded(losses) -> np.ndarray`
Return ceded loss under the layer.

#### `retained(losses) -> np.ndarray`
Return retained loss under the layer.

#### `attachment_probability(losses) -> float`
Return the fraction of simulations where the layer attaches.

#### `exhaustion_probability(losses) -> float | None`
Return the fraction of simulations where the layer exhausts, or `None` if unlimited.

---

## `ContractProgram`

Represents multiple aggregate layers applied to the same gross loss.

### Parameters

- `layers: Sequence[AggregateLayer]`
- `name: str = "contract_program"`

### Methods

#### `layer_names() -> list[str]`
Return layer names.

#### `ceded_by_layer(losses) -> np.ndarray`
Return a 2D array of ceded loss by layer.

#### `ceded(losses) -> np.ndarray`
Return total ceded loss across layers.

#### `retained(losses) -> np.ndarray`
Return retained loss after all layers.

---

## `SimulationResult`

Container for simulation output.

### Stored data

A result may contain:

- `gross_losses`
- `ceded_losses`
- `retained_losses`
- `component_losses`
- `component_names`
- `layer_losses`
- `layer_names`
- `contract_name`

### Properties

#### `n_sims`
Number of simulations.

#### `losses`
Primary loss view:
- retained losses if present
- otherwise gross losses

### Methods

#### `mean() -> float`
Mean of the primary loss view.

#### `variance(ddof: int = 0) -> float`
Variance of the primary loss view.

#### `std(ddof: int = 0) -> float`
Standard deviation of the primary loss view.

#### `var(q: float) -> float`
Simulation-based Value at Risk.

#### `tvar(q: float) -> float`
Simulation-based Tail Value at Risk.

#### `prob_exceeding(threshold: float) -> float`
Probability the primary loss view exceeds the threshold.

#### `gross_mean() -> float`
Mean gross loss.

#### `ceded_mean() -> float | None`
Mean ceded loss if present.

#### `retained_mean() -> float | None`
Mean retained loss if present.

#### `component_means() -> dict[str, float]`
Mean loss by portfolio component.

#### `layer_means() -> dict[str, float]`
Mean ceded loss by contract layer.

#### `summary(quantiles: tuple[float, ...] = (0.95, 0.99)) -> dict`
Return a summary dictionary of key metrics.