from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .contracts import AggregateLayer, ContractProgram, apply_contract
from .protocols import SupportsSample
from .results import SimulationResult


def _validate_size(size: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")


def _sample_1d(model: SupportsSample, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
    if rng is None:
        samples = np.asarray(model.sample(size=size), dtype=float)
    else:
        samples = np.asarray(model.sample(size=size, rng=rng), dtype=float)

    if samples.ndim == 0:
        samples = samples.reshape(1)
    elif samples.ndim != 1:
        raise ValueError("model.sample(size) must return a 1D array")

    if samples.size != size:
        raise ValueError(
            f"model.sample(size={size}) returned {samples.size} values instead of {size}"
        )

    return samples


def _model_mean(model: object) -> float:
    if not hasattr(model, "mean") or not callable(model.mean):
        raise TypeError("all portfolio models must implement mean() for this operation")
    return float(model.mean())


def _model_variance(model: object) -> float:
    if not hasattr(model, "variance") or not callable(model.variance):
        raise TypeError(
            "all portfolio models must implement variance() for this operation"
        )
    return float(model.variance())


@dataclass(frozen=True, slots=True)
class PortfolioItem:
    name: str
    model: SupportsSample
    weight: float = 1.0

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("name must be non-empty")

        if self.weight < 0:
            raise ValueError("weight must be nonnegative")

    def sample(self, size: int, rng: np.random.Generator | None = None) -> np.ndarray:
        losses = _sample_1d(self.model, size=size, rng=rng)
        return self.weight * losses

    def mean(self) -> float:
        return self.weight * _model_mean(self.model)

    def variance(self) -> float:
        return (self.weight ** 2) * _model_variance(self.model)


class Portfolio:
    r"""
    Portfolio of aggregate-loss components.

    :meth:`sample` and :meth:`sample_components` draw the components
    **independently** -- the natural default, and the right one when the
    components genuinely do not move together. Independence is not
    imposed on the workflow, only on this sampler: to introduce
    dependence, draw the component matrix here and reorder it with
    :func:`risksim.dependence.impose_rank_correlation`, which preserves
    each component's marginal exactly while imposing a target rank
    correlation (see that function for the rank-vs-tail-dependence
    caveat)::

        from risksim.dependence import impose_rank_correlation

        matrix = portfolio.sample_components(n, rng)
        dependent = impose_rank_correlation(matrix, corr, rng)
        total = dependent.sum(axis=1)

    The analytic :meth:`variance`, :meth:`std`, and :meth:`summary`
    methods assume independence and are unaffected by that
    post-processing; compute dependent risk measures from the reordered
    sample via :mod:`risksim.metrics`.
    """

    def __init__(self, items: Sequence[PortfolioItem], name: str = "portfolio") -> None:
        if not items:
            raise ValueError("items must contain at least one PortfolioItem")

        self.items = tuple(items)
        self.name = name

    def component_names(self) -> list[str]:
        return [item.name for item in self.items]

    def sample_components(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        """Draw a ``(size, n_components)`` matrix, one column per component.

        Components are drawn independently. This matrix is exactly the
        input :func:`risksim.dependence.impose_rank_correlation` expects,
        so imposing dependence is a one-line post-processing step on the
        return value (see the class docstring).
        """
        _validate_size(size)

        gen = None if rng is None else np.random.default_rng(rng)
        columns = [item.sample(size=size, rng=gen) for item in self.items]
        return np.column_stack(columns)

    def sample(self, size: int = 1, rng: np.random.Generator | int | None = None) -> np.ndarray:
        component_losses = self.sample_components(size=size, rng=rng)
        return np.sum(component_losses, axis=1)

    def mean(self) -> float:
        return float(sum(item.mean() for item in self.items))

    def variance(self) -> float:
        """Analytic portfolio variance under the independence assumption.

        This is the closed-form sum of component variances and does **not**
        reflect any dependence imposed downstream via
        :func:`risksim.dependence.impose_rank_correlation`; for a
        dependent portfolio, take the variance of the reordered sample.
        """
        return float(sum(item.variance() for item in self.items))

    def std(self) -> float:
        return float(np.sqrt(self.variance()))

    def simulate(
        self,
        size: int = 100_000,
        contract: AggregateLayer | ContractProgram | None = None,
        rng: np.random.Generator | int | None = None,
    ) -> SimulationResult:
        component_losses = self.sample_components(size=size, rng=rng)
        gross_losses = np.sum(component_losses, axis=1)

        if contract is None:
            return SimulationResult(
                gross_losses=gross_losses,
                component_losses=component_losses,
                component_names=self.component_names(),
            )

        ceded_losses, retained_losses = apply_contract(gross_losses, contract)

        if isinstance(contract, ContractProgram):
            return SimulationResult(
                gross_losses=gross_losses,
                ceded_losses=ceded_losses,
                retained_losses=retained_losses,
                component_losses=component_losses,
                component_names=self.component_names(),
                layer_losses=contract.ceded_by_layer(gross_losses),
                layer_names=contract.layer_names(),
                contract_name=contract.name,
            )

        return SimulationResult(
            gross_losses=gross_losses,
            ceded_losses=ceded_losses,
            retained_losses=retained_losses,
            component_losses=component_losses,
            component_names=self.component_names(),
            contract_name=contract.name or contract.__class__.__name__,
        )
