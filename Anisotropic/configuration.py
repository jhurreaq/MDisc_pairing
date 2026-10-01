"""Per-run settings and state; no mutable module globals or import-time execution."""

from dataclasses import dataclass
from typing import Any
from mdisc.configuration import PlotSettings, initialize_plot_palette
import numpy as np
import warnings
from sklearn.exceptions import ConvergenceWarning


@dataclass
class CaseContext(PlotSettings):
    """Own the settings and data needed by this case's model and plotting functions."""

    BASE_SAVE_DIR: str = 'outputs'
    CV_FOLDS: int = 5
    DATA_START_ROW: int = 4
    EPS: float = 1e-10
    EXCEL_FILE_NAME: str = './input/CANNsHEARTdata_shear05.xlsx'
    NON_LINEAR_OPTIMIZER_METHOD: str = 'L-BFGS-B'
    NON_LINEAR_OPTIMIZER_TOL: float = 1e-06
    RELATIVE_NOISE_LEVELS: Any = None
    SHEET_NAME: str = 'Sheet1'
    WEIGHTING_EPSILON: float = 1e-06
    X_features_stage1: Any = None
    color_noisy_data: Any = None
    color_original: str = 'black'
    color_pdf: Any = None
    color_refined: Any = None
    experimental_data_list_filtered: Any = None
    experimental_plot_series: Any = None
    num_psi_terms: Any = None
    psi_terms_definitions: Any = None


def create_context():
    context = CaseContext()
    warnings.filterwarnings('ignore', category=ConvergenceWarning)
    warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    np.random.seed(42)
    context.RELATIVE_NOISE_LEVELS = [0.0, 0.05, 0.10]
    initialize_plot_palette(context, broad_fallback=False)
    context.color_refined = context.plasma(0.15)
    context.color_pdf = context.plasma(0.55)
    context.color_noisy_data = context.plasma(0.7)
    return context
