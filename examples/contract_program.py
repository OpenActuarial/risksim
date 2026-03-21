import numpy as np

from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import AggregateLayer, ContractProgram, Portfolio, PortfolioItem


np.random.seed(2026)

medical = CollectiveRiskModel(
    Poisson(lam=3.0),
    Lognormal(mu=3.0, sigma=0.6),
)
rx = CollectiveRiskModel(
    Poisson(lam=2.0),
    Lognormal(mu=2.3, sigma=0.5),
)

portfolio = Portfolio(
    [
        PortfolioItem("medical", medical),
        PortfolioItem("rx", rx),
    ],
    name="tower_demo_portfolio",
)

program = ContractProgram(
    [
        AggregateLayer(attachment=50.0, limit=50.0, name="layer_1"),
        AggregateLayer(attachment=100.0, limit=100.0, name="layer_2"),
    ],
    name="two_layer_tower",
)

result = portfolio.simulate(size=30_000, contract=program)

print("=== contract program example ===")
print(f"program name: {result.contract_name}")
print(f"gross mean: {result.gross_mean():.4f}")
print(f"ceded mean: {result.ceded_mean():.4f}")
print(f"retained mean: {result.retained_mean():.4f}")
print()
print("component means:")
for name, value in result.component_means().items():
    print(f"  {name}: {value:.4f}")
print()
print("layer means:")
for name, value in result.layer_means().items():
    print(f"  {name}: {value:.4f}")
print()
print("summary:")
print(result.summary())