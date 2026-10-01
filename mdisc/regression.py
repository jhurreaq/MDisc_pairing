"""Shared sparse-regression kernels with explicit legacy solver policies.

Preserve centering, intercept recovery, coefficient thresholds, and the distinct
fallbacks used by each original experiment. Case adapters own scenario-specific
path construction, CV accumulation order, and selection/reporting conventions.
"""

import warnings
import numpy as np
from scipy.optimize import nnls
from sklearn.linear_model import LassoCV, Ridge
from sklearn.metrics import mean_squared_error


def fit_lasso_cv(X, y, *, suppress_warnings=False, **solver_options):
    """Fit using the caller's exact grid, intercept, CV, and solver settings."""
    model = LassoCV(**solver_options)
    if suppress_warnings:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model.fit(X, y)
    else:
        model.fit(X, y)
    return model


def fit_ridge(X, y, **solver_options):
    """Refit with the exact regularization, intercept, and solver chosen by the case."""
    model = Ridge(**solver_options)
    model.fit(X, y)
    return model


def lasso_cv_plot_data(model, y, *, eps=1e-10):
    """Original percentile normalization and 20-point sampling of CV curves."""
    alphas = np.asarray(model.alphas_)
    mse_path = np.asarray(model.mse_path_)
    if mse_path.ndim == 3:
        mean = np.nanmean(mse_path, axis=(1, 2))
        std = np.nanstd(mse_path, axis=(1, 2))
    elif mse_path.ndim == 2:
        mean = np.nanmean(mse_path, axis=1)
        std = np.nanstd(mse_path, axis=1)
    else:
        mean = mse_path.ravel()
        std = np.zeros_like(mean)
    y_centered = y - np.mean(y)
    y_var = float(np.var(y_centered))
    if y_var > eps:
        mean = mean / y_var
        std = std / y_var
    p_lo = float(np.nanpercentile(mean, 5))
    p_hi = float(np.nanpercentile(mean, 95))
    den = max(p_hi - p_lo, eps)
    mean = np.clip((mean - p_lo) / den, 0.0, 1.0)
    std = np.clip(std / den, 0.0, 1.0)
    n_plot = min(20, alphas.size)
    if n_plot > 0:
        indices = np.unique(np.linspace(0, alphas.size - 1, n_plot, dtype=int))
        return alphas[indices], mean[indices], std[indices]
    return np.array([]), np.array([]), np.array([])


def refit_support_nnls(Xmat, yvec, S):
    """
    Solve: min_{β_S>=0, b in R} || y - (b + X_S β_S) ||^2
    by centering to remove intercept from the NNLS and then recovering b.
    Returns (full_beta, intercept, y_hat_on_Xmat).
    """
    if len(S) == 0:
        b = float(np.mean(yvec))
        beta_full = np.zeros(Xmat.shape[1])
        yhat = np.full_like(yvec, b, dtype=float)
        return beta_full, b, yhat

    X_S = Xmat[:, S]
    y_mean = float(np.mean(yvec))
    X_means = np.mean(X_S, axis=0)

    y_c = yvec - y_mean
    X_c = X_S - X_means  # column-centering

    # NNLS on centered system
    beta_S, _ = nnls(X_c, y_c)

    # recover intercept
    b = y_mean - float(X_means @ beta_S)

    # assemble full β
    beta_full = np.zeros(Xmat.shape[1])
    beta_full[np.asarray(S, dtype=int)] = beta_S

    # predictions
    yhat = b + X_S @ beta_S
    return beta_full, b, yhat

def projected_gradient_nnls_lars(A, b, max_iter=2000, tol=1e-8, *, eps=1e-10):
    # Simple projected-gradient NNLS fallback: minimize 0.5||Aβ-b||^2 s.t. β>=0
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
    if A.size == 0:
        return np.zeros((A.shape[1] if A.ndim==2 else 0,), dtype=float)
    # Lipschitz constant (spectral norm squared)
    try:
        L = np.linalg.norm(A, 2) ** 2
    except Exception:
        L = np.linalg.norm(A, ord=None) ** 2
    L = max(L, eps)
    beta = np.zeros(A.shape[1], dtype=float)
    for _ in range(max_iter):
        grad = A.T @ (A @ beta - b)
        beta_new = beta - grad / L
        # projection
        beta_new = np.maximum(beta_new, 0.0)
        if np.linalg.norm(beta_new - beta) <= tol * (np.linalg.norm(beta) + tol):
            beta = beta_new
            break
        beta = beta_new
    return beta

def projected_gradient_nnls_omp(A, b, max_iter=2000, tol=1e-8, *, eps=1e-10):
    """Projected-gradient NNLS fallback: min 0.5||Aβ-b||^2 s.t. β>=0"""
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float).ravel()
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros((0,), dtype=float)
    try:
        L = np.linalg.norm(A, 2)**2
    except Exception:
        L = np.linalg.norm(A)**2
    L = max(L, eps)
    beta = np.zeros(A.shape[1], dtype=float)
    for _ in range(max_iter):
        grad = A.T @ (A @ beta - b)
        beta_new = beta - grad / L
        beta_new = np.maximum(beta_new, 0.0)
        if np.linalg.norm(beta_new - beta) <= tol * (np.linalg.norm(beta) + tol):
            beta = beta_new
            break
        beta = beta_new
    return beta

def solve_nnls(A, b, *, eps=1e-10):
    """Prefer scipy.optimize.nnls, else PGD fallback."""
    A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float).ravel()
    if A.size == 0 or A.shape[1] == 0:
        return np.zeros((0,), dtype=float)
    try:
        from scipy.optimize import nnls as _scipy_nnls
        beta, _ = _scipy_nnls(A, b)
    except Exception:
        beta = projected_gradient_nnls_omp(A, b, eps=eps)
    return np.maximum(beta, 0.0)

def refit_nnls_with_intercept(Xmat, yvec, *, variant="lars", eps=1e-10):
    """
    Solve: min_{μ, β>=0} || y - μ - X β ||^2
    by centering X, y on the fitting set, solve NNLS for β on centered data,
    then μ = mean(y) - mean(X,0)·β. Returns (beta, intercept).
    """
    Xmat = np.asarray(Xmat, dtype=float)
    yvec = np.asarray(yvec, dtype=float).ravel()
    if Xmat.size == 0 or Xmat.shape[1] == 0:
        return np.zeros(0, dtype=float), float(np.mean(yvec))
    xmean = Xmat.mean(axis=0)
    ymean = float(np.mean(yvec))
    Xc = Xmat - xmean
    yc = yvec - ymean

    if variant == "lars":
        # The LARS and OMP fallbacks differ in the originals; retain both.
        try:
            beta, _ = nnls(Xc, yc)
        except Exception:
            beta = projected_gradient_nnls_lars(Xc, yc, eps=eps)
        beta = np.maximum(beta, 0.0)
    elif variant == "omp":
        beta = solve_nnls(Xc, yc, eps=eps)
    else:
        raise ValueError(f"Unknown NNLS refit variant: {variant}")
    mu = ymean - float(xmean @ beta)
    return beta, mu

def refit_weighted_nnls_with_intercept(X, y, w=None, *, allow_empty=True, eps=1e-10):
    """
    Solve min ||W^(1/2) (y - (b0 + X b))||_2 subject to b >= 0.
    Weighted by w if provided. Uses SciPy nnls if available,
    otherwise falls back to projected gradient (nonnegative) on centered data.
    Returns (b0, b>=0).
    """
    X = np.asarray(X, float); y = np.asarray(y, float).ravel()
    if allow_empty and X.size == 0:  # no features -> intercept-only model
        if w is None:
            return float(np.mean(y)), np.zeros(0)
        wv = np.asarray(w).ravel()
        return float(np.sum(wv * y) / np.clip(np.sum(wv), eps, None)), np.zeros(0)

    if w is not None:
        sw = np.sqrt(np.clip(np.asarray(w).ravel(), eps, None))
        Xw = X * sw[:, None]
        yw = y * sw
        mu_x = (Xw.sum(axis=0) / np.clip(sw.sum(), eps, None))
        mu_y = (yw.sum()      / np.clip(sw.sum(), eps, None))
        Xc = Xw - mu_x[None, :]
        yc = yw - mu_y
    else:
        mu_x = X.mean(axis=0)
        mu_y = y.mean()
        Xc = X - mu_x[None, :]
        yc = y - mu_y

    beta = None
    try:
        from scipy.optimize import nnls
        beta, _ = nnls(Xc, yc)
    except Exception:
        pass

    if beta is None:  # projected gradient fallback
        beta = np.zeros(Xc.shape[1], float)
        L = max(np.linalg.norm(Xc, ord=2)**2, 1e-6)
        lr = 1.0 / L
        for _ in range(1000):
            grad = -(Xc.T @ (yc - Xc @ beta))
            beta -= lr * grad
            beta = np.maximum(beta, 0.0)

    b0 = float(mu_y - mu_x @ beta)
    return b0, beta

def weighted_mse(y_true, y_pred, w=None, *, eps=1e-10):
    if w is None:
        return mean_squared_error(y_true, y_pred)
    resid2 = (y_true - y_pred) ** 2
    w = np.asarray(w).reshape(-1)
    w_sum = np.sum(w)
    if w_sum <= eps:
        return np.mean(resid2)
    return float(np.sum(w * resid2) / w_sum)

def clip_nonnegative(beta, tol=1e-12):
    b = np.array(beta, dtype=float, copy=True)
    b[b < tol] = 0.0
    return b
