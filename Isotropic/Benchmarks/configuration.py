"""Per-run settings and state; no mutable module globals or import-time execution."""

from dataclasses import dataclass, field
from typing import Any
from mdisc.configuration import PlotSettings, initialize_plot_palette
import warnings


@dataclass
class CaseContext(PlotSettings):
    """Own the settings and data needed by this case's model and plotting functions."""

    COEFF_THRESHOLD: float = 1e-06
    EPS: float = 1e-10
    INCLUDE_EXTENDED_OGDEN: bool = True
    INCLUDE_MOONEY_RIVLIN: bool = True
    INCLUDE_NEOHOOKEAN: bool = True
    INCLUDE_OGDEN: bool = True
    MR_MAX_ORDER: int = 4
    OGDEN_EXPONENTS: list = field(default_factory=list)
    REFIT_REGRESSION_TYPE: Any = None
    RIDGE_ALPHA_REFIT: float = 1e-07


def create_context():
    context = CaseContext()
    warnings.filterwarnings("ignore")
    initialize_plot_palette(context, broad_fallback=False)
    context.OGDEN_EXPONENTS = []
    return context
