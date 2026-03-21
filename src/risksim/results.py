from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from . import metrics


def _as_1d_float_array(losses: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(losses, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError("losses must be a one-dimensional array-like object")

    if arr.size == 0:
        raise ValueError("losses must not be empty")

    return arr


@dataclass(slots=True)
class SimulationResult:
    """
    Container for portfolio simulation outputs.

    If retained_losses is present, the primary `losses` view is retained/net loss.
    Otherwise, the primary `losses` view is gross loss.
    """

    gross_losses: np.ndarray
    ceded_losses: np.ndarray | None = None
    retained_losses: np.ndarray | None = None
    component_losses: np.ndarray | None = None
    component_names: Sequence[str] | None = None
    contract_name: str | None = None

    def __post_init__(self) -> None:
        self.gross_losses = _as_1d_float_array(self.gross_losses)

        if self.ceded_losses is not None:
            self.ceded_losses = _as_1d_float_array(self.ceded_losses)
            if self.ceded_losses.shape != self.gross_losses.shape:
                raise ValueError("ceded_losses must match gross_losses shape")

        if self.retained_losses is not None:
            self.retained_losses = _as_1d_float_array(self.retained_losses)
            if self.retained_losses.shape != self.gross_losses.shape:
                raise ValueError("retained_losses must match gross_losses shape")

        if self.component_losses is not None:
            self.component_losses = np.asarray(self.component_losses, dtype=float)
            if self.component_losses.ndim != 2:
                raise ValueError("component_losses must be a 2D array")
            if self.component_losses.shape[0] != self.gross_losses.shape[0]:
                raise ValueError("component_losses must have one row per simulation")

            if self.component_names is not None:
                if len(self.component_names) != self.component_losses.shape[1]:
                    raise ValueError(
                        "component_names length must match number of component columns"
                    )

    @property
    def n_sims(self) -> int:
        return int(self.gross_losses.size)

    @property
    def losses(self) -> np.ndarray:
        if self.retained_losses is not None:
            return self.retained_losses
        return self.gross_losses

    def mean(self) -> float:
        return metrics.mean(self.losses)

    def variance(self, ddof: int = 0) -> float:
        return metrics.variance(self.losses, ddof=ddof)

    def std(self, ddof: int = 0) -> float:
        return metrics.std(self.losses, ddof=ddof)

    def var(self, q: float) -> float:
        return metrics.var(self.losses, q)

    def tvar(self, q: float) -> float:
        return metrics.tvar(self.losses, q)

    def prob_exceeding(self, threshold: float) -> float:
        return metrics.prob_exceeding(self.losses, threshold)

    def gross_mean(self) -> float:
        return metrics.mean(self.gross_losses)

    def ceded_mean(self) -> float | None:
        if self.ceded_losses is None:
            return None
        return metrics.mean(self.ceded_losses)

    def retained_mean(self) -> float | None:
        if self.retained_losses is None:
            return None
        return metrics.mean(self.retained_losses)

    def component_means(self) -> dict[str, float]:
        if self.component_losses is None:
            return {}

        means = np.mean(self.component_losses, axis=0)

        if self.component_names is None:
            names = [f"component_{i}" for i in range(self.component_losses.shape[1])]
        else:
            names = list(self.component_names)

        return {name: float(value) for name, value in zip(names, means)}

    def summary(self, quantiles: tuple[float, ...] = (0.95, 0.99)) -> dict[str, Any]:
        out = metrics.summary(self.losses, quantiles=quantiles)
        out["gross_mean"] = self.gross_mean()

        if self.ceded_losses is not None:
            out["ceded_mean"] = self.ceded_mean()

        if self.retained_losses is not None:
            out["retained_mean"] = self.retained_mean()

        component_means = self.component_means()
        if component_means:
            out["component_means"] = component_means

        if self.contract_name is not None:
            out["contract_name"] = self.contract_name

        return out