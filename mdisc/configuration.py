"""Immutable plot template settings, separate from mechanics and runtime data."""
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class PlotSettings:
    PLOT_TITLE_FONTSIZE: int = 14
    PLOT_LABEL_FONTSIZE: int = 14
    PLOT_TICK_FONTSIZE: int = 14
    PLOT_LEGEND_FONTSIZE: int = 10
    PLOT_LINEWIDTH: float = 1.5
    PLOT_MARKER_SIZE: int = 30
    PLOT_PRED_LINEWIDTH: float = 1.5
    PLOT_HEATMAP_YLABEL_SIZE: int = 9
    SMALL_FIG_LASSO_LARS_OMP: Tuple[float, float] = (3.0, 4.5)
