# risksim

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

Portfolio-level loss simulation and aggregate reinsurance modeling.

---

## Overview

**`risksim`** composes one or more stochastic loss models into a portfolio, simulates
the portfolio's aggregate loss, applies aggregate reinsurance structures, and
summarizes the gross, ceded, and retained results. It is intentionally
model-agnostic: anything that exposes a `.sample(size)` method can be a portfolio
component, so **`risksim`** handles *composition, contracts, and simulation summaries*
while the loss generation lives wherever you like.

It pairs naturally with **`lossmodels`** — build a collective-risk model (or any
severity/frequency-based model) there, drop it into a **`risksim`** portfolio here —
but it has no hard dependency on it; its only runtime requirement is **`numpy`**.

## Highlights

- **Portfolios** — combine multiple simulated loss components, each with an
  optional weight, into a single portfolio.
- **Aggregate reinsurance** — aggregate annual layers (attachment, limit, share)
  and multi-layer contract programs.
- **One simulation, every view** — a single run yields gross, ceded, retained,
  per-component, and per-layer losses.
- **Risk measures** — mean, variance / standard deviation, VaR, TVaR, exceedance
  probabilities, and a one-call summary, on any loss vector.
- **Model-agnostic** — any object implementing `.sample(size)` works; components
  that also expose `.mean()` unlock closed-form portfolio moments.

## Installation

```bash
pip install risksim
```

From source:

```bash
pip install -e .
```

Requires Python `>=3.10`; the only runtime dependency is `numpy`. The examples
below use `lossmodels` for the component models — install it too if you want to
run them as written.

## Package structure

```text
risksim/
├── portfolio.py    # Portfolio and PortfolioItem
├── contracts.py    # AggregateLayer, ContractProgram, apply_contract
├── results.py      # SimulationResult (gross / ceded / retained / component / layer views)
├── metrics.py      # mean, variance, std, var, tvar, prob_exceeding, summary
└── protocols.py    # SupportsSample, SupportsMoments
```

## Quick start

```python
from lossmodels import Poisson, Lognormal, CollectiveRiskModel
from risksim import Portfolio, PortfolioItem, AggregateLayer

# two lines of business, each a collective-risk model (any .sample object works)
line_a = CollectiveRiskModel(Poisson(3.0), Lognormal(8.0, 0.9))
line_b = CollectiveRiskModel(Poisson(1.5), Lognormal(9.0, 0.7))

portfolio = Portfolio([
    PortfolioItem("line_a", line_a),
    PortfolioItem("line_b", line_b, weight=1.25),   # optional exposure weight
])

# simulate the portfolio's aggregate annual loss, net of an aggregate layer
layer = AggregateLayer(attachment=250_000, limit=500_000, share=1.0)
result = portfolio.simulate(100_000, contract=layer)

print("gross mean   :", result.gross_mean)
print("ceded mean   :", result.ceded_mean)
print("retained mean:", result.retained_mean)
print("retained 99% VaR :", result.var(0.99))
print(result.summary())
```

## Portfolios

A `PortfolioItem` wraps a sampleable loss model with a name and an optional
`weight` (a multiplier applied to that component's sampled losses). A `Portfolio`
is a sequence of items.

```python
from risksim import Portfolio, PortfolioItem

port = Portfolio([PortfolioItem("a", model_a), PortfolioItem("b", model_b, weight=0.5)])

gross = port.sample(50_000)              # total portfolio loss per simulation
by_component = port.sample_components(50_000)  # per-component loss matrix
print(port.component_names())
```

When every component also exposes `.mean()` / `.variance()` (the `SupportsMoments`
protocol — which `lossmodels` models satisfy), the portfolio reports closed-form
moments without simulating:

```python
port.mean()       # sum of component means (times weights)
port.std()
port.variance()
```

`simulate(size, contract=None)` runs the portfolio and returns a
`SimulationResult`; `sample(size)` returns just the gross loss vector.

## Aggregate reinsurance

An `AggregateLayer` is an aggregate annual structure defined by three numbers:

- `attachment` — the aggregate retention; losses below it are not ceded.
- `limit` — the width of the layer above the attachment (`None` = unlimited).
- `share` — the ceded proportion of the layer (e.g. `0.8` for an 80% placement).

For a gross annual loss `L`, the ceded amount is
`share × min(max(L − attachment, 0), limit)`.

```python
from risksim import AggregateLayer, ContractProgram, apply_contract

# a single layer: 80% of the 500k aggregate layer above a 250k retention
layer = AggregateLayer(attachment=250_000, limit=500_000, share=0.8)

# stack layers into a program (e.g. a working layer plus a higher layer)
program = ContractProgram([
    AggregateLayer(attachment=250_000, limit=500_000, share=1.0, name="working"),
    AggregateLayer(attachment=750_000, limit=1_000_000, share=0.5, name="higher"),
])

# apply either to a loss vector directly
ceded, retained = apply_contract(gross_losses, program)
```

`apply_contract(losses, contract)` returns `(ceded, retained)` arrays aligned with
the input, where `ceded + retained == losses`. Passing a contract to
`Portfolio.simulate(..., contract=program)` does the same inside the simulation and
records the per-layer detail.

## The simulation result

`SimulationResult` holds every view from a single run and computes risk measures on
demand:

| Attribute / method | Meaning |
| --- | --- |
| `gross_losses`, `ceded_losses`, `retained_losses` | aligned loss vectors |
| `component_losses`, `component_names` | per-component loss matrix |
| `layer_losses`, `layer_names` | per-layer ceded amounts (with a contract) |
| `gross_mean`, `ceded_mean`, `retained_mean` | mean of each view |
| `component_means`, `layer_means` | per-component / per-layer means |
| `losses` | the retained vector (the portfolio's net position) |
| `mean()`, `std()`, `variance()` | moments of the net loss |
| `var(q)`, `tvar(q)` | Value-at-Risk and Tail-VaR of the net loss |
| `prob_exceeding(threshold)` | P(net loss > threshold) |
| `summary()` | a dict of headline statistics |
| `n_sims` | the number of simulations |

## Risk measures

The `metrics` module operates on any loss vector (a NumPy array or list), so you
can use it independently of the portfolio machinery:

```python
from risksim import metrics

metrics.mean(losses)
metrics.std(losses)            # metrics.variance(...) also available
metrics.var(losses, 0.99)      # 99% VaR
metrics.tvar(losses, 0.99)     # 99% TVaR
metrics.prob_exceeding(losses, 1_000_000)
metrics.summary(losses, quantiles=(0.95, 0.99, 0.995))
```

## The SupportsSample protocol

Any object implementing `sample(size: int) -> np.ndarray` can be a portfolio
component — a `lossmodels` collective-risk model or severity, a custom class, or a
thin wrapper around a sampler. Components that additionally implement `mean()` /
`variance()` (`SupportsMoments`) enable the closed-form `Portfolio.mean()` /
`.std()` / `.variance()` shortcuts.

## The OpenActuarial ecosystem

**`risksim`** is the portfolio-and-reinsurance layer of a small family of actuarial
packages that interoperate through the `.sample()` / `.mean()` interface:

- **`lossmodels`** — frequency / severity distributions, aggregate (collective-risk)
  loss models, coverage modifications, and model fitting. The natural source of
  **`risksim`** portfolio components.
- **`extremeloss`** — extreme value theory for tail fitting and tail risk measures.
  It can analyze a **`risksim`** simulation directly (e.g. `losses_from_risksim(...)`
  and `tail_summary_from_risksim(...)`).
- **`actuarialpy`** — deterministic, experience-and-data analysis (summaries,
  triangles, trend, credibility) on tabular data.

## Testing

```bash
pytest -q
```

## License

MIT License