"""Bounded least squares with explicit intercept, weighting and convergence."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls, lsq_linear
from .preprocessing import validate_weights

@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    intercept: float
    prediction: np.ndarray
    effective_df: float
    solver: str

def fit_nonnegative(X, y, *, weights=None, fit_intercept=False, ridge=0.0):
    """Minimize sum(w*(y-b-X@beta)^2)+ridge*||beta||², beta>=0.

    Constitutive discovery always sets fit_intercept=False. The affine option
    exists for generic regression and uses true weighted centering.
    """
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or y.shape != (len(X),) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite arrays with matching rows")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge penalty must be finite and nonnegative")
    w = validate_weights(weights, len(y))
    mx = np.average(X, axis=0, weights=w) if fit_intercept else np.zeros(X.shape[1])
    my = float(np.average(y, weights=w)) if fit_intercept else 0.0
    A, b = (X - mx) * np.sqrt(w)[:, None], (y - my) * np.sqrt(w)
    original_A = A
    if ridge:
        A = np.vstack([A, np.sqrt(ridge) * np.eye(X.shape[1])])
        b = np.concatenate([b, np.zeros(X.shape[1])])
    solver = "nnls"
    if X.shape[1]:
        try:
            beta, _ = nnls(A, b, maxiter=max(1000, 30 * X.shape[1]))
        except (RuntimeError, np.linalg.LinAlgError):
            result = lsq_linear(A, b, bounds=(0, np.inf), tol=1e-12, max_iter=2000)
            if not result.success:
                raise RuntimeError(f"Bounded least squares did not converge: {result.message}")
            beta, solver = result.x, "lsq_linear"
    else:
        beta = np.zeros(0)
    intercept = my - float(mx @ beta)
    active = beta > max(1.0, np.max(beta, initial=0.0)) * 1e-10
    gram = original_A[:, active].T @ original_A[:, active]
    df = float(np.trace(gram @ np.linalg.pinv(gram + ridge * np.eye(len(gram))))) if len(gram) else 0.0
    return LinearFit(beta, intercept, intercept + X @ beta, df + int(fit_intercept), solver)
