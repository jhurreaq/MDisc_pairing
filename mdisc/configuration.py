"""Plot defaults shared by the original experiments."""

from dataclasses import dataclass
from typing import Any, Tuple

import matplotlib.pyplot as plt


@dataclass
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
    plasma: Any = None
    TRUNCATED_PLASMA_CMAP: Any = None
    HEATMAP_CMAP: Any = None


def initialize_plot_palette(settings, *, broad_fallback=False):
    """Keep the original plasma/Blues palettes and each case's fallback policy."""
    try:
        settings.plasma = plt.get_cmap("plasma")
        blues = plt.get_cmap("Blues")
    except Exception as error:
        if not broad_fallback and not isinstance(error, ValueError):
            raise
        settings.plasma = plt.cm.plasma
        blues = plt.cm.Blues
    settings.TRUNCATED_PLASMA_CMAP = settings.plasma
    settings.HEATMAP_CMAP = blues
