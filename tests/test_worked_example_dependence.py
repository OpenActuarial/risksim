"""Regression: the dependence worked-example page numbers stay true.

Pins docs page ``worked-example-dependence.md`` (Example 9) in the
OpenActuarial docs repo. Drop into ``risksim/tests/``. Requires
``lossmodels`` (already a test dependency for the integration suite).
"""
import lossmodels as lm
import numpy as np
import pytest

import risksim as rs
from risksim import metrics, uncertainty
from risksim.dependence import impose_rank_correlation


@pytest.fixture(scope="module")
def worlds():
    medical = lm.CollectiveRiskModel(lm.NegativeBinomial(65.3, 0.269),
                                     lm.Lognormal(9.2, 0.95))
    specialty = lm.CollectiveRiskModel(lm.Poisson(26.0),
                                       lm.Lognormal(10.2, 1.35))
    port = rs.Portfolio([rs.PortfolioItem("medical", medical),
                         rs.PortfolioItem("specialty", specialty)])
    res = port.simulate(200_000, rng=7)
    M = res.component_losses
    corr = np.array([[1.0, 0.5], [0.5, 1.0]])
    Mn = impose_rank_correlation(M, corr, rng=11)
    Mt = impose_rank_correlation(M, corr, rng=11, scores="t", df=4)
    return M, Mn, Mt


def test_dependence_page_numbers(worlds):
    M, Mn, Mt = worlds

    # marginals preserved exactly by the reorder
    assert np.allclose(np.sort(M[:, 0]), np.sort(Mn[:, 0]))
    assert np.allclose(np.sort(M[:, 1]), np.sort(Mt[:, 1]))

    # standalone and combined tail metrics
    assert round(metrics.tvar(M[:, 0], 0.995)) == 4_299_128
    assert round(metrics.tvar(M[:, 1], 0.995)) == 6_745_713
    standalone = metrics.tvar(M[:, 0], 0.995) + metrics.tvar(M[:, 1], 0.995)
    assert round(standalone) == 11_044_841

    totals = {"ind": M.sum(axis=1), "norm": Mn.sum(axis=1),
              "t": Mt.sum(axis=1)}
    tvar995 = {k: metrics.tvar(v, 0.995) for k, v in totals.items()}
    assert round(tvar995["ind"]) == 9_620_716
    assert round(tvar995["norm"]) == 10_316_277
    assert round(tvar995["t"]) == 10_219_875
    assert round(standalone - tvar995["ind"]) == 1_424_125
    assert round(standalone - tvar995["norm"]) == 728_564

    # joint exceedance of own VaR99 — the copula separator
    qm = metrics.var(M[:, 0], 0.99)
    qs = metrics.var(M[:, 1], 0.99)
    counts = {k: int(((X[:, 0] > qm) & (X[:, 1] > qs)).sum())
              for k, X in (("ind", M), ("norm", Mn), ("t", Mt))}
    assert counts == {"ind": 12, "norm": 275, "t": 397}
    # exact probabilities quoted on the page: 0.00006, 0.001375, 0.001985

    program = rs.ContractProgram([
        rs.AggregateLayer(attachment=7_000_000, limit=3_000_000, name="first"),
        rs.AggregateLayer(attachment=10_000_000, limit=5_000_000, share=0.8,
                          name="second")])
    ceded = {k: rs.apply_contract(v, program)[0] for k, v in totals.items()}
    assert round(ceded["ind"].mean()) == 17_859
    assert round(ceded["norm"].mean()) == 29_849
    assert round(ceded["t"].mean()) == 29_475

    _, retained = rs.apply_contract(totals["t"], program)
    s = uncertainty.summary_with_error(retained, quantiles=(0.99, 0.995), rng=7)
    assert s["var_99"]["estimate"] == pytest.approx(7_000_000.0)
    assert s["var_99"]["ci_low"] == pytest.approx(7_000_000.0)
    assert s["var_99"]["ci_high"] == pytest.approx(7_000_000.0)
    assert round(s["tvar_99.5"]["estimate"]) == 7_186_535
    assert round(s["tvar_99.5"]["ci_low"]) == 7_143_683
    assert round(s["tvar_99.5"]["ci_high"]) == 7_242_559

    ci = uncertainty.mean_ci(ceded["t"])
    assert round(ci["estimate"]) == 29_475
    assert round(ci["ci_low"]) == 28_343
    assert round(ci["ci_high"]) == 30_608
