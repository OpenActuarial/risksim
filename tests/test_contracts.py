import numpy as np
import pytest

from risksim.contracts import AggregateLayer, ContractProgram, apply_contract


def test_aggregate_layer_ceded_and_retained_with_limit() -> None:
    losses = np.array([50.0, 100.0, 150.0, 300.0])
    layer = AggregateLayer(attachment=100.0, limit=50.0, share=1.0)

    ceded = layer.ceded(losses)
    retained = layer.retained(losses)

    np.testing.assert_allclose(ceded, np.array([0.0, 0.0, 50.0, 50.0]))
    np.testing.assert_allclose(retained, np.array([50.0, 100.0, 100.0, 250.0]))


def test_aggregate_layer_share_is_applied() -> None:
    losses = np.array([50.0, 100.0, 150.0, 300.0])
    layer = AggregateLayer(attachment=100.0, limit=50.0, share=0.5)

    ceded = layer.ceded(losses)

    np.testing.assert_allclose(ceded, np.array([0.0, 0.0, 25.0, 25.0]))


def test_aggregate_layer_without_limit_has_uncapped_recovery() -> None:
    losses = np.array([50.0, 100.0, 150.0, 300.0])
    layer = AggregateLayer(attachment=100.0, limit=None, share=1.0)

    ceded = layer.ceded(losses)

    np.testing.assert_allclose(ceded, np.array([0.0, 0.0, 50.0, 200.0]))
    assert layer.exhaustion_probability(losses) is None


def test_attachment_and_exhaustion_probabilities() -> None:
    losses = np.array([50.0, 100.0, 150.0, 300.0])
    layer = AggregateLayer(attachment=100.0, limit=50.0, share=1.0)

    assert layer.attachment_probability(losses) == pytest.approx(0.5)
    assert layer.exhaustion_probability(losses) == pytest.approx(0.5)


def test_contract_program_ceded_by_layer_total_and_retained() -> None:
    losses = np.array([50.0, 150.0, 250.0, 350.0])

    layer_1 = AggregateLayer(attachment=100.0, limit=100.0, name="l1")
    layer_2 = AggregateLayer(attachment=200.0, limit=100.0, name="l2")
    program = ContractProgram([layer_1, layer_2], name="tower")

    by_layer = program.ceded_by_layer(losses)
    total_ceded = program.ceded(losses)
    retained = program.retained(losses)

    np.testing.assert_allclose(
        by_layer,
        np.array(
            [
                [0.0, 0.0],
                [50.0, 0.0],
                [100.0, 50.0],
                [100.0, 100.0],
            ]
        ),
    )
    np.testing.assert_allclose(total_ceded, np.array([0.0, 50.0, 150.0, 200.0]))
    np.testing.assert_allclose(retained, np.array([50.0, 100.0, 100.0, 150.0]))
    assert program.layer_names() == ["l1", "l2"]


def test_apply_contract_accepts_contract_program() -> None:
    losses = np.array([50.0, 150.0, 250.0, 350.0])
    program = ContractProgram(
        [
            AggregateLayer(attachment=100.0, limit=100.0, name="l1"),
            AggregateLayer(attachment=200.0, limit=100.0, name="l2"),
        ]
    )

    ceded, retained = apply_contract(losses, program)

    np.testing.assert_allclose(ceded, np.array([0.0, 50.0, 150.0, 200.0]))
    np.testing.assert_allclose(retained, np.array([50.0, 100.0, 100.0, 150.0]))


def test_apply_contract_returns_ceded_and_retained_for_single_layer() -> None:
    losses = np.array([50.0, 100.0, 150.0, 300.0])
    layer = AggregateLayer(attachment=100.0, limit=50.0, share=1.0)

    ceded, retained = apply_contract(losses, layer)

    np.testing.assert_allclose(ceded, np.array([0.0, 0.0, 50.0, 50.0]))
    np.testing.assert_allclose(retained, np.array([50.0, 100.0, 100.0, 250.0]))


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"attachment": -1.0}, "attachment must be nonnegative"),
        ({"attachment": 0.0, "limit": -1.0}, "limit must be nonnegative or None"),
        ({"attachment": 0.0, "share": -0.1}, "share must be between 0 and 1"),
        ({"attachment": 0.0, "share": 1.1}, "share must be between 0 and 1"),
    ],
)
def test_aggregate_layer_validation(kwargs: dict, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        AggregateLayer(**kwargs)


def test_contract_program_requires_at_least_one_layer() -> None:
    with pytest.raises(ValueError, match="at least one AggregateLayer"):
        ContractProgram([])