"""Validated observations and explicit discovery results."""
from dataclasses import dataclass, field
from typing import Dict, Tuple
import numpy as np

@dataclass(frozen=True)
class Observations:
    F: np.ndarray
    y: np.ndarray
    components: Tuple[Tuple[int, int], ...]
    modes: np.ndarray
    groups: np.ndarray
    measure: str
    unit: str

    def __post_init__(self):
        n = len(self.y)
        if self.y.shape != (n,):
            raise ValueError("Stress targets must be a one-dimensional array")
        if self.F.shape != (n, 3, 3) or len(self.components) != n:
            raise ValueError("Each stress observation requires a 3x3 deformation and component")
        if any(len(component) != 2 or any(not isinstance(index, (int, np.integer)) or
                                          not 0 <= index < 3 for index in component)
               for component in self.components):
            raise ValueError("Stress components require two integer indices between zero and two")
        if self.modes.shape != (n,) or self.groups.shape != (n,):
            raise ValueError("Mode and measurement-group arrays must match observations")
        if n == 0 or not np.all(np.isfinite(self.F)) or not np.all(np.isfinite(self.y)):
            raise ValueError("Observations must be nonempty and finite")
        if not np.allclose(np.linalg.det(self.F), 1.0, rtol=0, atol=1e-8):
            raise ValueError("The incompressible model requires det(F)=1")
        if self.measure not in ("nominal", "cauchy"):
            raise ValueError("Specify nominal or Cauchy stress explicitly")
        if self.unit not in ("Pa", "kPa", "MPa"):
            raise ValueError("Specify stress units as Pa, kPa, or MPa")

    def subset(self, indices):
        indices = np.asarray(indices, dtype=int)
        return Observations(self.F[indices], self.y[indices], tuple(self.components[i] for i in indices),
                            self.modes[indices], self.groups[indices], self.measure, self.unit)

    def with_targets(self, y):
        return Observations(self.F.copy(), np.asarray(y, float).copy(), self.components,
                            self.modes.copy(), self.groups.copy(), self.measure, self.unit)

@dataclass
class ModelFit:
    coefficients: np.ndarray
    prediction: np.ndarray
    effective_df: float
    nonlinear: Dict[int, float] = field(default_factory=dict)
    converged: bool = True
    message: str = "linear nonnegative least squares"
    diagnostics: dict = field(default_factory=dict)

@dataclass
class Candidate:
    parameter: float
    support: Tuple[int, ...]
    fit: ModelFit
    aic: float
    bic: float
    cv_mse: float = float("nan")
    cv_std: float = float("nan")
    path_coefficients: np.ndarray = field(default_factory=lambda: np.array([]))
    mechanics: dict = field(default_factory=dict)

@dataclass
class DiscoveryResult:
    method: str
    candidates: list
    chosen: dict
    folds: list
    cv_predictions: dict
    diagnostics: dict = field(default_factory=dict)
