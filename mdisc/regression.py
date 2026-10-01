"""Origin-preserving nonnegative least squares with weighting and convergence."""
from dataclasses import dataclass
import numpy as np
from scipy.optimize import nnls, lsq_linear
from .preprocessing import validate_weights

@dataclass(frozen=True)
class LinearFit:
    coefficients: np.ndarray
    prediction: np.ndarray
    effective_df: float
    solver: str

def fit_nonnegative(X, y, *, weights=None, ridge=0.0):
    """Minimize sum(w*(y-X@beta)^2)+ridge*||beta||², beta>=0, without an intercept."""
    X, y = np.asarray(X, float), np.asarray(y, float)
    if X.ndim != 2 or y.shape != (len(X),) or not np.all(np.isfinite(X)) or not np.all(np.isfinite(y)):
        raise ValueError("X and y must be finite arrays with matching rows")
    if not np.isfinite(ridge) or ridge < 0:
        raise ValueError("Ridge penalty must be finite and nonnegative")
    w = validate_weights(weights, len(y))
    A, b = X * np.sqrt(w)[:, None], y * np.sqrt(w)
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
    active = beta > max(1.0, np.max(beta, initial=0.0)) * 1e-10
    gram = original_A[:, active].T @ original_A[:, active]
    df = float(np.trace(gram @ np.linalg.pinv(gram + ridge * np.eye(len(gram))))) if len(gram) else 0.0
    return LinearFit(beta, X @ beta, df, solver)
