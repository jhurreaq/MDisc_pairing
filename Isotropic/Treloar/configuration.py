"""Per-run settings and state; no mutable module globals or import-time execution."""

from dataclasses import dataclass, field
from typing import Any
from mdisc.configuration import PlotSettings, initialize_plot_palette
import warnings
from sklearn.exceptions import ConvergenceWarning


@dataclass
class CaseContext(PlotSettings):
    """Own the settings and data needed by this case's model and plotting functions."""

    EPS: float = 1e-10
    INCLUDE_EXTENDED_OGDEN: bool = False
    INCLUDE_MOONEY_RIVLIN: bool = True
    INCLUDE_NEOHOOKEAN: bool = True
    INCLUDE_OGDEN: bool = True
    MR_MAX_POWER_I1: int = 3
    MR_MAX_POWER_I2: int = 3
    OGDEN_EXPONENTS: list = field(default_factory=list)
    REFIT_REGRESSION_TYPE: Any = None
    SMALL_STRAIN_THRESHOLD: float = 50.0
    WEIGHT_FACTOR: float = 1.0
    max_total_power: int = 3


def create_context():
    context = CaseContext()
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    initialize_plot_palette(context, broad_fallback=True)
    context.OGDEN_EXPONENTS = []
    return context
