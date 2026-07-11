from .contracts import AggregateLayer, ContractProgram, apply_contract
from .portfolio import Portfolio, PortfolioItem
from .results import SimulationResult

__all__ = [
    "AggregateLayer",
    "apply_contract",
    "ContractProgram",
    "Portfolio",
    "PortfolioItem",
    "SimulationResult",
]

from importlib.metadata import PackageNotFoundError as _PackageNotFoundError
from importlib.metadata import version as _version

try:
    __version__ = _version("risksim")
except _PackageNotFoundError:  # running from a source tree without an installed distribution
    __version__ = "0.0.0"

del _PackageNotFoundError, _version
