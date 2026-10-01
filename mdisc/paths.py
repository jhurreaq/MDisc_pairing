"""Actual LAR, positive coordinate-descent LASSO, and nonnegative OMP paths."""
from dataclasses import dataclass
import warnings
import numpy as np
from sklearn.linear_model import lasso_path, lars_path
from sklearn.exceptions import ConvergenceWarning
from .regression import fit_nonnegative

@dataclass(frozen=True)
class PathSettings:
    alpha_min: float = 1e-6
    alpha_max: float = 1.0
    alpha_count: int = 40
    path_count: int = 21
    coefficient_tolerance: float = 1e-8
    max_iter: int = 50000
    solver_tolerance: float = 1e-7

    def __post_init__(self):
        if not np.isfinite([self.alpha_min, self.alpha_max, self.solver_tolerance,
                            self.coefficient_tolerance]).all():
            raise ValueError('Path bounds and tolerances must be finite')
        if any(not isinstance(count, (int, np.integer))
               for count in (self.alpha_count, self.path_count, self.max_iter)):
            raise ValueError('Candidate and iteration counts must be integers')
        if not (0 < self.alpha_min <= self.alpha_max) or self.alpha_count < 2 or self.path_count < 2:
            raise ValueError('Require positive alpha bounds and at least two candidates')
        if self.max_iter < 1 or self.solver_tolerance <= 0 or self.coefficient_tolerance <= 0:
            raise ValueError('Solver iteration counts and tolerances must be positive')

    def parameters(self, method):
        if method == 'Lasso':
            return np.geomspace(self.alpha_max, self.alpha_min, self.alpha_count)
        if method in ('LARS', 'OMP'):
            return np.linspace(0, 1, self.path_count)
        raise ValueError(f"Unknown discovery method: {method}")


def nonnegative_omp_path(X, y):
    """Greedy positive-correlation selection followed by constrained refits.

    This is explicitly nonnegative OMP, not signed classical OMP. Keep every
    iterate, including changed equal-size supports and the zero model.
    """
    p = X.shape[1]
    columns, visited = [np.zeros(p)], set()
    beta = np.zeros(p)
    for _ in range(p):
        correlations = X.T @ (y - X @ beta)
        if visited:
            correlations[list(visited)] = -np.inf
        index = int(np.argmax(correlations))
        if not np.isfinite(correlations[index]) or correlations[index] <= 1e-12 * max(1, np.linalg.norm(y)):
            break
        visited.add(index)
        S = sorted(visited)
        beta = np.zeros(p)
        beta[S] = fit_nonnegative(X[:, S], y).coefficients
        columns.append(beta)
    return np.column_stack(columns)


def discovery_path(X, y, method, settings, *, diagnostics=None):
    diagnostics = {} if diagnostics is None else diagnostics
    diagnostics.update(features=X.shape[1], numerical_rank=int(np.linalg.matrix_rank(X)))
    if method == 'Lasso':
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            _, coefficients, _ = lasso_path(X, y, alphas=settings.parameters(method),
                                               positive=True, max_iter=settings.max_iter,
                                               tol=settings.solver_tolerance)
        if any(issubclass(w.category, ConvergenceWarning) for w in caught):
            raise RuntimeError("LASSO path did not converge; increase max_iter or revise the library")
        return coefficients
    if method == 'LARS':
        # 'lar' is the documented LAR algorithm. Positivity is imposed during
        # constitutive refits, since positive=True is not supported by LAR.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always', ConvergenceWarning)
            _, _, raw = lars_path(X, y, method='lar', positive=False,
                                  max_iter=min(settings.max_iter, diagnostics['numerical_rank']))
        diagnostics['warnings'] = [str(w.message) for w in caught]
        diagnostics['raw_path_steps'] = raw.shape[1]
    elif method == 'OMP':
        raw = nonnegative_omp_path(X, y)
    else:
        raise ValueError(f"Unknown method: {method}")
    indices = np.rint(settings.parameters(method) * (raw.shape[1] - 1)).astype(int)
    return raw[:, indices]
