import numpy as np
import pytest

import risksim
import risksim.metrics as m
from risksim import AggregateLayer, apply_contract


def test_apply_contract_exported_at_top_level():
    assert hasattr(risksim, "apply_contract")
    assert "apply_contract" in risksim.__all__
    assert apply_contract is risksim.contracts.apply_contract


def test_version_is_consistent():
    # __init__ and pyproject were out of sync (0.1.0 vs 0.1.1); now both 0.2.0
    assert risksim.__version__ == "0.2.0"


@pytest.fixture
def losses():
    return np.random.default_rng(0).lognormal(10, 1.0, 5000)


def test_summary_subpercent_quantiles_distinct_and_labeled(losses):
    s = m.summary(losses, quantiles=(0.995, 0.999))
    # previously both collapsed to the single key var_100 / tvar_100
    assert {"var_99.5", "var_99.9", "tvar_99.5", "tvar_99.9"}.issubset(s)
    assert s["var_99.9"] >= s["var_99.5"]


def test_summary_default_labels_unchanged(losses):
    s = m.summary(losses)  # default (0.95, 0.99)
    assert {"var_95", "var_99", "tvar_95", "tvar_99"}.issubset(s)
    assert "var_100" not in s


def test_attachment_probability_independent_of_share():
    vals = np.array([50.0, 150.0, 250.0, 350.0])  # 3 of 4 exceed attachment=100
    for share in (0.0, 0.5, 1.0):
        layer = AggregateLayer(attachment=100.0, limit=100.0, share=share)
        assert layer.attachment_probability(vals) == pytest.approx(0.75)


def test_attachment_probability_with_zero_limit():
    vals = np.array([50.0, 150.0, 250.0])  # 2 of 3 exceed attachment=100
    layer = AggregateLayer(attachment=100.0, limit=0.0, share=1.0)
    # limit=0 cedes nothing, but the loss still reaches the attachment point
    assert layer.attachment_probability(vals) == pytest.approx(2.0 / 3.0)
