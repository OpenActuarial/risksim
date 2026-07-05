"""How much of a simulated tail metric is signal.

    simulated losses -> summary_with_error: normal theory for the mean,
    order statistics for VaR, bootstrap for TVaR -- each metric with the
    interval its sampling theory supports.

Run with:  python examples/monte_carlo_error.py
"""
from __future__ import annotations

import numpy as np

from risksim import uncertainty


def main() -> None:
    rng = np.random.default_rng(3)

    for n in (2_000, 50_000):
        losses = rng.lognormal(mean=13.0, sigma=1.0, size=n)
        table = uncertainty.summary_with_error(
            losses, quantiles=(0.95, 0.99), rng=7,
        )
        print(f"=== {n:,} simulations ===")
        for name, row in table.items():
            width = row["ci_high"] - row["ci_low"]
            rel = width / row["estimate"]
            print(f"{name:>8}: {row['estimate']:>14,.0f}"
                  f"   ci [{row['ci_low']:>14,.0f}, {row['ci_high']:>14,.0f}]"
                  f"   width {rel:6.1%} of estimate")
        print()
    print("The tail bands shrink like 1/sqrt(n): if the 99% TVaR band is")
    print("too wide, the answer is more simulations -- and now you can see it.")


if __name__ == "__main__":
    main()
