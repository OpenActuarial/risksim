# Examples

The `examples/` directory contains complete runnable scripts.

## `basic_portfolio.py`

This example demonstrates how to:

- create two aggregate loss models with `lossmodels`
- wrap them in `PortfolioItem`
- combine them into a `Portfolio`
- simulate the total gross portfolio loss distribution
- inspect overall and component-level summaries

Run:

```bash
python3 examples/basic_portfolio.py
```

What it shows:

- total gross mean
- portfolio VaR and TVaR
- average loss by component

## `aggregate_layer.py`

This example demonstrates how to:

- simulate a single gross loss component
- apply one aggregate annual layer
- compare gross, ceded, and retained outcomes

Run:

```bash
python3 examples/aggregate_layer.py
```

What it shows:

- gross mean
- ceded mean
- retained mean
- net VaR and TVaR

## `contract_program.py`

This example demonstrates how to:

- build a two-component portfolio
- apply a simple two-layer aggregate contract program
- inspect recoveries by layer

Run:

```bash
python3 examples/contract_program.py
```

What it shows:

- gross, ceded, and retained mean
- average contribution by portfolio component
- average recovery by layer

## Notes on reproducibility

The examples set a NumPy random seed before simulation:

```python
np.random.seed(...)
```

This makes runs reproducible for demonstration purposes.