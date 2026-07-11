import numpy as np
from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import Portfolio, PortfolioItem

np.random.seed(12345)

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
    ],
    name="health_portfolio",
)

result = portfolio.simulate(size=50_000)

print("=== basic portfolio example ===")
print(f"portfolio name: {portfolio.name}")
print(f"number of simulations: {result.n_sims}")
print(f"gross mean: {result.gross_mean():.4f}")
print(f"VaR 95%: {result.var(0.95):.4f}")
print(f"TVaR 95%: {result.tvar(0.95):.4f}")
print()
print("component means:")
for name, value in result.component_means().items():
    print(f"  {name}: {value:.4f}")
print()
print("summary:")
print(result.summary())