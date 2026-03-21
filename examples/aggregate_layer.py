import numpy as np

from lossmodels.aggregate import CollectiveRiskModel
from lossmodels.frequency import Poisson
from lossmodels.severity import Lognormal

from risksim import AggregateLayer, Portfolio, PortfolioItem


np.random.seed(54321)

line = CollectiveRiskModel(
    Poisson(lam=4.0),
    Lognormal(mu=3.5, sigma=0.7),
)

portfolio = Portfolio(
    [PortfolioItem("line", line)],
    name="single_line_portfolio",
)

layer = AggregateLayer(
    attachment=50.0,
    limit=100.0,
    share=1.0,
    name="agg_xol_50_xs_100",
)

gross_result = portfolio.simulate(size=20_000)
net_result = portfolio.simulate(size=20_000, contract=layer)

print("=== aggregate layer example ===")
print(f"contract name: {net_result.contract_name}")
print(f"gross mean: {gross_result.gross_mean():.4f}")
print(f"ceded mean: {net_result.ceded_mean():.4f}")
print(f"retained mean: {net_result.retained_mean():.4f}")
print(f"net VaR 99%: {net_result.var(0.99):.4f}")
print(f"net TVaR 99%: {net_result.tvar(0.99):.4f}")
print()
print("summary:")
print(net_result.summary())