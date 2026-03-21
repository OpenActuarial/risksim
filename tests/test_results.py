import numpy as np
import pytest

from risksim.results import SimulationResult


def test_losses_property_prefers_retained_losses() -> None:
    result = SimulationResult(
        gross_losses=np.array([100.0, 200.0]),
        retained_losses=np.array([80.0, 150.0]),
    )

    np.testing.assert_allclose(result.losses, np.array([80.0, 150.0]))


def test_metric_methods_use_primary_losses_view() -> None:
    gross = np.array([10.0, 20.0, 30.0, 40.0])
    result = SimulationResult(gross_losses=gross)

    assert result.mean() == pytest.approx(np.mean(gross))
    assert result.variance() == pytest.approx(np.var(gross))
    assert result.std() == pytest.approx(np.std(gross))
    assert result.var(0.75) == pytest.approx(np.quantile(gross, 0.75))

    threshold = np.quantile(gross, 0.75)
    expected_tvar = np.mean(gross[gross >= threshold])
    assert result.tvar(0.75) == pytest.approx(expected_tvar)


def test_component_means_with_names() -> None:
    component_losses = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    result = SimulationResult(
        gross_losses=np.array([3.0, 7.0, 11.0]),
        component_losses=component_losses,
        component_names=["a", "b"],
    )

    out = result.component_means()

    assert out == {"a": 3.0, "b": 4.0}


def test_layer_means_with_names() -> None:
    result = SimulationResult(
        gross_losses=np.array([100.0, 200.0, 300.0]),
        layer_losses=np.array(
            [
                [10.0, 0.0],
                [20.0, 30.0],
                [30.0, 40.0],
            ]
        ),
        layer_names=["l1", "l2"],
    )

    assert result.layer_means() == {"l1": 20.0, "l2": 23.333333333333332}


def test_summary_includes_optional_fields() -> None:
    result = SimulationResult(
        gross_losses=np.array([100.0, 200.0, 300.0]),
        ceded_losses=np.array([10.0, 20.0, 30.0]),
        retained_losses=np.array([90.0, 180.0, 270.0]),
        component_losses=np.array(
            [
                [40.0, 60.0],
                [80.0, 120.0],
                [120.0, 180.0],
            ]
        ),
        component_names=["x", "y"],
        layer_losses=np.array(
            [
                [10.0],
                [20.0],
                [30.0],
            ]
        ),
        layer_names=["agg_xol"],
        contract_name="tower",
    )

    out = result.summary(quantiles=(0.5,))

    assert out["mean"] == pytest.approx(np.mean([90.0, 180.0, 270.0]))
    assert out["gross_mean"] == pytest.approx(np.mean([100.0, 200.0, 300.0]))
    assert out["ceded_mean"] == pytest.approx(np.mean([10.0, 20.0, 30.0]))
    assert out["retained_mean"] == pytest.approx(np.mean([90.0, 180.0, 270.0]))
    assert out["component_means"] == {"x": 80.0, "y": 120.0}
    assert out["layer_means"] == {"agg_xol": 20.0}
    assert out["contract_name"] == "tower"
    assert "var_50" in out
    assert "tvar_50" in out


def test_ceded_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="ceded_losses must match gross_losses shape"):
        SimulationResult(
            gross_losses=np.array([1.0, 2.0]),
            ceded_losses=np.array([1.0]),
        )


def test_component_names_length_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match="component_names length must match number of component columns",
    ):
        SimulationResult(
            gross_losses=np.array([1.0, 2.0]),
            component_losses=np.array([[1.0, 2.0], [3.0, 4.0]]),
            component_names=["only_one_name"],
        )


def test_layer_names_length_mismatch_raises() -> None:
    with pytest.raises(
        ValueError,
        match="layer_names length must match number of layer columns",
    ):
        SimulationResult(
            gross_losses=np.array([1.0, 2.0]),
            layer_losses=np.array([[1.0, 2.0], [3.0, 4.0]]),
            layer_names=["only_one_name"],
        )