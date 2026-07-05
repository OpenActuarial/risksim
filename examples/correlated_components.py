"""What dependence does to a portfolio tail -- and what kind matters.

    two independent component simulations -> impose_rank_correlation
    (normal scores, then t scores) -> portfolio VaR/TVaR under each

Same marginals in every case, same rank correlation in the last two --
and three different tails. The independence-to-correlation jump is the
diversification you were overstating; the normal-to-t jump at IDENTICAL
rank correlation is tail dependence, the thing a correlation number
does not carry.

Run with:  python examples/correlated_components.py
"""
from __future__ import annotations

import numpy as np

from risksim import metrics
from risksim.dependence import impose_rank_correlation


def main() -> None:
    rng = np.random.default_rng(11)
    n = 200_000
    components = np.column_stack([
        rng.lognormal(mean=13.0, sigma=0.9, size=n),   # property
        rng.lognormal(mean=12.6, sigma=1.1, size=n),   # liability
    ])
    corr = np.array([[1.0, 0.6], [0.6, 1.0]])

    cases = {
        "independent": components,
        "rho=0.6, normal scores": impose_rank_correlation(
            components, corr, rng=1, scores="normal"),
        "rho=0.6, t(3) scores": impose_rank_correlation(
            components, corr, rng=1, scores="t", df=3.0),
    }
    print(f"{'case':>24} {'VaR 99%':>16} {'TVaR 99%':>16}")
    base = None
    for label, mat in cases.items():
        total = mat.sum(axis=1)
        var99 = metrics.var(total, 0.99)
        tvar99 = metrics.tvar(total, 0.99)
        if base is None:
            base = tvar99
        print(f"{label:>24} {var99:>16,.0f} {tvar99:>16,.0f}"
              f"   ({tvar99 / base - 1:+.1%} TVaR vs independent)")
    print()
    print("Marginals are identical in all three rows (reordering only), and")
    print("the last two rows have the SAME rank correlation. The remaining")
    print("gap between them is tail dependence -- decide it explicitly.")


if __name__ == "__main__":
    main()
