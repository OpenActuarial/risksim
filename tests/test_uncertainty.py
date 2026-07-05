"""Monte Carlo error quantification: hand-checked ranks, coverage, reproducibility."""
import numpy as np
import pytest

from risksim import metrics, uncertainty


def test_mean_ci_normal_theory():
    rng = np.random.default_rng(5)
    arr = rng.normal(100.0, 20.0, 40_000)
    out = uncertainty.mean_ci(arr, confidence=0.95)
    assert out["ci_low"] <= 100.0 <= out["ci_high"]
    assert out["se"] == pytest.approx(20.0 / np.sqrt(40_000), rel=0.05)
    # the interval is exactly estimate +/- z * se
    z = 1.959963984540054
    assert out["ci_high"] - out["estimate"] == pytest.approx(z * out["se"], rel=1e-12)


def test_quantile_ci_hand_checked_ranks():
    arr = np.arange(1.0, 101.0)  # 1..100, so order statistic k has value k
    out = uncertainty.quantile_ci(arr, q=0.90, confidence=0.95)
    # k = ceil(100 * 0.9) = 90; half-width = 1.96 * sqrt(100*0.9*0.1) = 5.88
    # -> ranks floor(84.12) = 84 and ceil(95.88) = 96
    assert out["estimate"] == 90.0
    assert out["ci_low"] == 84.0
    assert out["ci_high"] == 96.0
    assert np.isnan(out["se"])
    assert out["estimate"] == metrics.var(arr, 0.90)  # same rank convention


def test_quantile_ci_clips_at_sample_edge():
    arr = np.arange(1.0, 101.0)
    out = uncertainty.quantile_ci(arr, q=0.999)
    assert out["ci_high"] == 100.0  # honest edge, not extrapolation


def test_quantile_ci_covers_known_quantile():
    rng = np.random.default_rng(11)
    arr = rng.exponential(1.0, 50_000)
    truth = -np.log(0.05)  # exponential q95
    out = uncertainty.quantile_ci(arr, q=0.95)
    assert out["ci_low"] <= truth <= out["ci_high"]


def test_bootstrap_reproducible_and_bracketing():
    rng = np.random.default_rng(23)
    arr = rng.lognormal(10.0, 1.0, 5_000)
    stat = lambda a: metrics.tvar(a, 0.99)  # noqa: E731
    a = uncertainty.bootstrap_ci(arr, stat, n_boot=300, rng=42)
    b = uncertainty.bootstrap_ci(arr, stat, n_boot=300, rng=42)
    assert a == b
    assert a["estimate"] == pytest.approx(stat(arr), rel=1e-12)
    assert a["ci_low"] < a["estimate"] < a["ci_high"]
    assert a["se"] > 0


def test_summary_with_error_matches_point_summary():
    rng = np.random.default_rng(31)
    arr = rng.gamma(2.0, 5_000.0, 20_000)
    table = uncertainty.summary_with_error(arr, quantiles=(0.95, 0.99), rng=7)
    point = metrics.summary(arr, quantiles=(0.95, 0.99))
    assert set(table) == {"mean", "var_95", "tvar_95", "var_99", "tvar_99"}
    for key in ("mean", "var_95", "var_99", "tvar_95", "tvar_99"):
        assert table[key]["estimate"] == pytest.approx(point[key], rel=1e-12)
        assert set(table[key]) == {"estimate", "se", "ci_low", "ci_high"}
    # tail intervals are wider than body intervals for the same metric type
    assert (table["tvar_99"]["ci_high"] - table["tvar_99"]["ci_low"]) > (
        table["tvar_95"]["ci_high"] - table["tvar_95"]["ci_low"]
    )


def test_guards():
    with pytest.raises(ValueError, match="at least two"):
        uncertainty.mean_ci([1.0])
    with pytest.raises(ValueError, match="confidence"):
        uncertainty.mean_ci([1.0, 2.0], confidence=1.5)
    with pytest.raises(TypeError, match="callable"):
        uncertainty.bootstrap_ci([1.0, 2.0], statistic="mean")
    with pytest.raises(ValueError, match="n_boot"):
        uncertainty.bootstrap_ci([1.0, 2.0], statistic=np.mean, n_boot=1)


def test_quantile_ci_lower_tail_hand_ranks():
    arr = np.arange(1.0, 101.0)
    out = uncertainty.quantile_ci(arr, q=0.05, confidence=0.95)
    # k = 5; half = 1.96 * sqrt(100*0.05*0.95) = 4.27 -> ranks 1 (clipped), 10
    assert out["estimate"] == 5.0
    assert out["ci_low"] == 1.0
    assert out["ci_high"] == 10.0


def test_generator_and_seed_paths_agree():
    rng = np.random.default_rng(23)
    arr = rng.lognormal(10.0, 1.0, 2_000)
    stat = lambda a: metrics.tvar(a, 0.95)  # noqa: E731
    via_seed = uncertainty.bootstrap_ci(arr, stat, n_boot=200, rng=99)
    via_gen = uncertainty.bootstrap_ci(arr, stat, n_boot=200,
                                       rng=np.random.default_rng(99))
    assert via_seed == via_gen


def test_interval_coverage_mean_and_quantile():
    """Empirical coverage against known truth: normal theory for the mean
    (should sit at nominal) and the order-statistic quantile interval
    (asymptotic binomial ranks; near-nominal at n = 1000)."""
    rng = np.random.default_rng(8)
    reps, n = 400, 1000
    q, truth_q = 0.90, -np.log(0.10)  # exponential(1) 90th percentile
    mean_hits = q_hits = 0
    for _ in range(reps):
        arr = rng.exponential(1.0, n)
        m = uncertainty.mean_ci(arr, confidence=0.95)
        mean_hits += m["ci_low"] <= 1.0 <= m["ci_high"]
        qi = uncertainty.quantile_ci(arr, q, confidence=0.95)
        q_hits += qi["ci_low"] <= truth_q <= qi["ci_high"]
    assert 0.91 <= mean_hits / reps <= 0.985, mean_hits / reps
    assert 0.90 <= q_hits / reps <= 0.99, q_hits / reps
