# Concepts

## Design philosophy

`risksim` is meant to complement `lossmodels`, not replace it.

The division of responsibilities is:

- `lossmodels`: stochastic models and actuarial loss generation
- `risksim`: portfolio composition and aggregate treaty application

That means `risksim` works one level above the individual model.

## Core interface

The minimal interface expected by `risksim` is:

```python
sample(size: int = 1) -> np.ndarray
```

If an object implements `.sample(size)` and returns one simulated loss per simulation, it can be used inside a `Portfolio`.

Analytic methods such as `mean()` and `variance()` are used when available for portfolio summaries.

## Portfolio

A `Portfolio` combines multiple `PortfolioItem` objects.

Each `PortfolioItem` contains:

- a name
- a model
- an optional weight

The portfolio simulates each component separately and adds them together to form total gross loss.

Conceptually:

```text
gross portfolio loss
= component 1 loss
+ component 2 loss
+ ...
+ component n loss
```

## AggregateLayer

An `AggregateLayer` applies a single aggregate annual contract to simulated gross losses.

For gross annual loss `S`, ceded loss is:

```text
C = share * min((S - attachment)+, limit)
```

where:

- `attachment` is the annual deductible
- `limit` is the maximum recoverable amount
- `share` is the participation percentage

Examples:

- if gross loss is below attachment, ceded loss is 0
- if gross loss enters the layer, ceded loss increases
- if gross loss exceeds the top of the layer, ceded loss is capped at the limit

## ContractProgram

A `ContractProgram` combines multiple aggregate layers applied to the same gross loss.

The current implementation is intended for simple non-overlapping towers.

For a two-layer program:

- layer 1 might cover 50 xs 50
- layer 2 might cover 100 xs 100

Total ceded loss is the sum of ceded loss by layer.

## SimulationResult

`SimulationResult` stores outputs from a simulation run.

It can hold:

- `gross_losses`
- `ceded_losses`
- `retained_losses`
- `component_losses`
- `layer_losses`

It also provides convenience methods for:

- `mean()`
- `variance()`
- `std()`
- `var(q)`
- `tvar(q)`
- `prob_exceeding(threshold)`
- `summary()`

If retained losses are present, the primary `losses` view is retained/net loss. Otherwise it is gross loss.

## Scope of the current version

The current version does **not** yet model:

- dependence between components
- overlapping aggregate structures with validation
- occurrence-based contracts requiring claim-path information
- reinstatements
- capital allocation
- premium calculations beyond simulated summaries

The current focus is a clean scaffold for portfolio-level simulation and aggregate annual treaties.