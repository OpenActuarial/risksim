# risksim

`risksim` is a Python package for portfolio-level loss simulation and aggregate treaty modeling.

It is designed to sit on top of `lossmodels`. In this setup, `lossmodels` provides the individual aggregate loss models, and `risksim` combines them into portfolios, applies aggregate contracts, and summarizes the simulated results.

## Features

- combine multiple simulated loss components into a portfolio
- apply aggregate annual layers
- build simple multi-layer aggregate contract programs
- summarize gross, ceded, and retained loss
- compute simulation-based risk measures such as mean, VaR, and TVaR

## Installation

For local development:

```bash
python3 -m pip install -e .