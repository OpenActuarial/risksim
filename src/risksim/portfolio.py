from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .contracts import AggregateLayer, ContractProgram, apply_contract
from .protocols import SupportsSample
from .results import SimulationResult


def _validate_size(size: int) -> None:
    if size <= 0:
        raise ValueError("size must be positive")


def _sample_1d(model: SupportsSample, size: int) -> np.ndarray:
    samples = np.asarray(model.sample(size=size), dtype=float)

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

    def sample(self, size: int) -> np.ndarray:
        losses = _sample_1d(self.model, size=size)
        return self.weight * losses

    def mean(self) -> float:
        return self.weight * _model_mean(self.model)

    def variance(self) -> float:
        return (self.weight ** 2) * _model_variance(self.model)


class Portfolio:
    """
    Portfolio of aggregate-loss components.

    This first version assumes components are sampled independently.
    """

    def __init__(self, items: Sequence[PortfolioItem], name: str = "portfolio") -> None:
        if not items:
            raise ValueError("items must contain at least one PortfolioItem")

        self.items = tuple(items)
        self.name = name

    def component_names(self) -> list[str]:
        return [item.name for item in self.items]

    def sample_components(self, size: int = 1) -> np.ndarray:
        _validate_size(size)

        columns = [item.sample(size=size) for item in self.items]
        return np.column_stack(columns)

    def sample(self, size: int = 1) -> np.ndarray:
        component_losses = self.sample_components(size=size)
        return np.sum(component_losses, axis=1)

    def mean(self) -> float:
        return float(sum(item.mean() for item in self.items))

    def variance(self) -> float:
        """
        Analytic variance under the independence assumption.
        """
        return float(sum(item.variance() for item in self.items))

    def std(self) -> float:
        return float(np.sqrt(self.variance()))

    def simulate(
        self,
        size: int = 100_000,
        contract: AggregateLayer | ContractProgram | None = None,
    ) -> SimulationResult:
        component_losses = self.sample_components(size=size)
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
