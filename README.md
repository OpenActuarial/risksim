# risksim

Portfolio Monte Carlo: aggregate simulation, dependence, reinsurance contracts, and risk measures.

[![CI](https://github.com/OpenActuarial/risksim/actions/workflows/ci.yml/badge.svg)](https://github.com/OpenActuarial/risksim/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/risksim)](https://pypi.org/project/risksim/)
[![Python](https://img.shields.io/pypi/pyversions/risksim)](https://pypi.org/project/risksim/)

## Overview

`risksim` simulates the aggregate loss of a portfolio whose components can
be anything exposing a `.sample(n, rng=...)` method — `lossmodels` aggregate
models, `extremeloss` tails, or your own objects. Components are independent
by default; rank correlation is imposed with the Iman–Conover construction
when you specify a dependence structure.

Aggregate reinsurance is applied as layers or stacked programs, and results
come back with the standard risk measures (VaR, TVaR, exceedance
probabilities) plus bootstrap confidence intervals, so a simulation answer
always carries its simulation error.

## Installation

```bash
pip install risksim
```

Requires Python 3.10 or newer.

## Quick start

The example pairs `risksim` with `lossmodels` (`pip install lossmodels`),
the intended combination for frequency–severity components.

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

# aggregate annual loss, net of an aggregate layer
layer = AggregateLayer(attachment=250_000, limit=500_000, share=1.0)
result = portfolio.simulate(100_000, contract=layer, rng=42)

print("gross mean    :", result.gross_mean)
print("ceded mean    :", result.ceded_mean)
print("retained mean :", result.retained_mean)
print("net (retained) 99% TVaR:", result.tvar(0.99))
```

## What's inside

- **Portfolios** — `Portfolio` / `PortfolioItem` over any objects exposing
  `.sample`, with optional exposure weights.
- **Dependence** — Iman–Conover rank correlation via
  `risksim.dependence.impose_rank_correlation`.
- **Contracts** — `AggregateLayer`, stacked `ContractProgram`s, and
  `apply_contract` on raw loss vectors.
- **Risk measures** — empirical VaR, TVaR, exceedance probabilities, and
  one-call summaries in `risksim.metrics`.
- **Uncertainty** — bootstrap confidence intervals for any statistic in
  `risksim.uncertainty`.

The full API reference and end-to-end worked examples live at
**[openactuarial.org/risksim.html](https://openactuarial.org/risksim.html)**.

## The OpenActuarial ecosystem

`risksim` is one of eight packages that share conventions — tidy tables,
explicit distribution parameterizations, reproducible random-number handling —
and compose across package seams:

| Package | Role |
|---|---|
| [actuarialpy](https://github.com/OpenActuarial/actuarialpy) | Calculation primitives the workflow packages build on |
| [experiencestudies](https://github.com/OpenActuarial/experiencestudies) | Experience reporting, actual-vs-expected, claimant and concentration analysis |
| [projectionmodels](https://github.com/OpenActuarial/projectionmodels) | Claim, premium, and expense projection over a renewal horizon |
| [ratingmodels](https://github.com/OpenActuarial/ratingmodels) | Manual and experience rating, credibility, indication, GLM relativities |
| [reservingmodels](https://github.com/OpenActuarial/reservingmodels) | Claims development and stochastic reserving: chain ladder, BF, Mack, ODP bootstrap |
| [lossmodels](https://github.com/OpenActuarial/lossmodels) | Severity and frequency fitting, aggregate loss distributions |
| [extremeloss](https://github.com/OpenActuarial/extremeloss) | Extreme-value tails: POT/GPD, GEV, return levels, splicing |
| **[risksim](https://github.com/OpenActuarial/risksim)** | Portfolio Monte Carlo, dependence, reinsurance contracts, risk measures |

Install everything at once with `pip install openactuarial`.

## Development

```bash
git clone https://github.com/OpenActuarial/risksim
cd risksim
python -m pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs the same gate on Python 3.10–3.14 across Linux and Windows.

## Versioning and stability

All ecosystem packages are pre-1.0: minor releases may change APIs, and every
release is documented in [CHANGELOG.md](CHANGELOG.md). Current per-package API
stability is tracked at
[openactuarial.org/stability.html](https://openactuarial.org/stability.html).

## License

MIT — see [LICENSE](LICENSE).
