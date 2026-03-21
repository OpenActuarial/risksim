from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def _as_1d_float_array(losses: np.ndarray | list[float]) -> np.ndarray:
    arr = np.asarray(losses, dtype=float)

    if arr.ndim == 0:
        arr = arr.reshape(1)
    elif arr.ndim != 1:
        raise ValueError("losses must be a one-dimensional array-like object")

    if arr.size == 0:
        raise ValueError("losses must not be empty")

    return arr


@dataclass(frozen=True, slots=True)
class AggregateLayer:
    """
    Aggregate annual layer applied to simulated aggregate losses.

    For aggregate annual loss S, ceded loss is:

        C = share * min((S - attachment)+, limit)

    with no cap if limit is None.
    """

    attachment: float = 0.0
    limit: float | None = None
    share: float = 1.0
    name: str | None = None

    def __post_init__(self) -> None:
        if self.attachment < 0:
            raise ValueError("attachment must be nonnegative")

        if self.limit is not None and self.limit < 0:
            raise ValueError("limit must be nonnegative or None")

        if not (0.0 <= self.share <= 1.0):
            raise ValueError("share must be between 0 and 1")

    def ceded(self, losses: np.ndarray | list[float]) -> np.ndarray:
        gross = _as_1d_float_array(losses)

        recoverable = np.maximum(gross - self.attachment, 0.0)

        if self.limit is not None:
            recoverable = np.minimum(recoverable, self.limit)

        return self.share * recoverable

    def retained(self, losses: np.ndarray | list[float]) -> np.ndarray:
        gross = _as_1d_float_array(losses)
        return gross - self.ceded(gross)

    def attachment_probability(self, losses: np.ndarray | list[float]) -> float:
        ceded = self.ceded(losses)
        return float(np.mean(ceded > 0.0))

    def exhaustion_probability(self, losses: np.ndarray | list[float]) -> float | None:
        if self.limit is None:
            return None

        gross = _as_1d_float_array(losses)
        return float(np.mean(gross >= self.attachment + self.limit))


class ContractProgram:
    """
    Collection of aggregate layers applied to the same gross loss.

    This first version assumes the layers are intended to work together
    without overlap. For a standard non-overlapping tower, total ceded loss
    is the row-wise sum of ceded loss by layer.
    """

    def __init__(
        self,
        layers: Sequence[AggregateLayer],
        name: str = "contract_program",
    ) -> None:
        if not layers:
            raise ValueError("layers must contain at least one AggregateLayer")

        self.layers = tuple(layers)
        self.name = name

    def layer_names(self) -> list[str]:
        names: list[str] = []
        for i, layer in enumerate(self.layers):
            names.append(layer.name or f"layer_{i}")
        return names

    def ceded_by_layer(self, losses: np.ndarray | list[float]) -> np.ndarray:
        gross = _as_1d_float_array(losses)
        cols = [layer.ceded(gross) for layer in self.layers]
        return np.column_stack(cols)

    def ceded(self, losses: np.ndarray | list[float]) -> np.ndarray:
        by_layer = self.ceded_by_layer(losses)
        return np.sum(by_layer, axis=1)

    def retained(self, losses: np.ndarray | list[float]) -> np.ndarray:
        gross = _as_1d_float_array(losses)
        return gross - self.ceded(gross)


def apply_contract(
    losses: np.ndarray | list[float],
    contract: AggregateLayer | ContractProgram,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Return (ceded, retained) arrays for a single aggregate layer
    or a multi-layer contract program.
    """
    gross = _as_1d_float_array(losses)
    ceded = contract.ceded(gross)
    retained = gross - ceded
    return ceded, retained