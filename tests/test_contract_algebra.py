"""Contract transforms: the layer clip formula, stacking, and conservation."""
import numpy as np
import pytest

from risksim import AggregateLayer, ContractProgram, apply_contract


def _gross(seed=0, n=2000):
    return np.random.default_rng(seed).gamma(2.0, 900.0, size=n)


@pytest.mark.parametrize("attachment,limit,share", [
    (0.0, 500.0, 1.0),
    (1000.0, 2000.0, 1.0),
    (1000.0, 2000.0, 0.35),
    (2500.0, None, 1.0),
    (2500.0, None, 0.5),
])
def test_layer_is_the_clip_formula(attachment, limit, share):
    gross = _gross()
    ceded, retained = apply_contract(gross, AggregateLayer(attachment, limit, share))
    excess = np.maximum(gross - attachment, 0.0)
    expected = share * (excess if limit is None else np.minimum(excess, limit))
    assert np.allclose(ceded, expected, rtol=0, atol=1e-9)
    assert np.allclose(ceded + retained, gross, rtol=0, atol=1e-9)  # conservation


def test_exhaustion_pins_the_ceded_amount():
    lay = AggregateLayer(100.0, 200.0)
    ceded, _ = apply_contract([50.0, 150.0, 400.0, 5000.0], lay)
    assert np.allclose(ceded, [0.0, 50.0, 200.0, 200.0])


def test_contiguous_program_equals_one_big_layer():
    gross = _gross(1)
    program = ContractProgram([AggregateLayer(100.0, 200.0), AggregateLayer(300.0, 500.0)])
    ceded_p, retained_p = apply_contract(gross, program)
    ceded_b, retained_b = apply_contract(gross, AggregateLayer(100.0, 700.0))
    assert np.allclose(ceded_p, ceded_b, atol=1e-9)
    assert np.allclose(retained_p, retained_b, atol=1e-9)


def test_program_ceded_is_the_sum_of_its_layers():
    gross = _gross(2)
    layers = [AggregateLayer(0.0, 300.0, 0.5), AggregateLayer(1000.0, None, 0.25)]
    ceded, retained = apply_contract(gross, ContractProgram(layers))
    per_layer = [apply_contract(gross, lay)[0] for lay in layers]
    assert np.allclose(ceded, per_layer[0] + per_layer[1], atol=1e-9)
    assert np.allclose(ceded + retained, gross, atol=1e-9)
