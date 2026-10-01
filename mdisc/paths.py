"""Shared LARS support compression and positive OMP path construction."""

import numpy as np
from scipy.optimize import nnls

from .regression import clip_nonnegative, solve_nnls


def unique_size_supports(coefficients, *, absolute=False, eps=1e-10):
    """Keep the first nonempty support of each size in the original column order."""
    supports, indices = [], []
    seen = set()
    for index in range(coefficients.shape[1]):
        beta = coefficients[:, index]
        values = np.abs(beta) if absolute else np.maximum(beta, 0.0)
        support = np.flatnonzero(values > eps)
        size = int(support.size)
        if size == 0 or size in seen:
            continue
        seen.add(size)
        supports.append(support)
        indices.append(index)
    return supports, indices


def monotone_nonnegative_path(raw_coefficients):
    """Original cardiac LARS cleaning: clip at 1e-12, count above 1e-6."""
    columns, steps = [], []
    last_size = 0
    for index in range(1, raw_coefficients.shape[1]):
        beta = clip_nonnegative(raw_coefficients[:, index])
        size = int(np.sum(beta > 1e-6))
        if size > last_size:
            columns.append(beta)
            steps.append(size)
            last_size = size
    return (np.column_stack(columns) if columns else None), np.array(steps, dtype=int)


def positive_omp_path(X, y, *, variant, eps=1e-10):
    """Build the shared greedy path while retaining each isotropic case's policy.

    Synthetic uses direct SciPy NNLS and updates residuals on the compact support.
    Treloar retains its PGD fallback, nonnegative clipping, and full-matrix residual
    multiplication. The different multiplication orders must not be combined:
    doing so can change floating-point rounding and subsequent support selection.
    """
    if variant not in ("synthetic", "treloar"):
        raise ValueError(f"Unknown OMP path variant: {variant}")
    n_features = X.shape[1]
    y_centered = y - np.mean(y)
    max_k = min(n_features, max(1, np.linalg.matrix_rank(X)))
    active, columns = [], []
    residual = y_centered.copy()
    for k in range(1, max_k + 1):
        correlations = X.T @ residual
        if active:
            correlations[np.array(active, dtype=int)] = -np.inf
        if variant == "treloar":
            correlations[correlations <= 0] = -np.inf
        index = int(np.argmax(correlations))
        if not np.isfinite(correlations[index]) or (variant == "synthetic" and correlations[index] <= 0):
            if variant == "synthetic":
                print(f"  OMP stops at k={k}: no atom with positive correlation remains.")
            else:
                print(f"  OMP stopped at k={k-1}: no positively-correlated atoms remain.")
            break
        if index in active:
            print(f"  OMP early stop at k={k} (duplicate).")
            break
        active.append(index)
        selected_X = X[:, active]
        if variant == "synthetic":
            try:
                selected_beta, _ = nnls(selected_X, y_centered)
            except Exception as error:
                print(f"  WARN: NNLS failed at k={k}: {error}; stopping.")
                break
        else:
            selected_beta = solve_nnls(selected_X, y_centered, eps=eps)
        beta = np.zeros(n_features)
        beta[np.array(active, dtype=int)] = selected_beta
        if variant == "synthetic":
            columns.append(beta)
            residual = y_centered - selected_X @ selected_beta
        else:
            columns.append(np.maximum(beta.copy(), 0.0))
            residual = y_centered - X @ beta
        if np.linalg.norm(residual) < 1e-14:
            print(f"  OMP residual ~0 at k={k}; stopping.")
            break
    if not columns:
        print("ERROR: empty OMP path.")
        return None
    coefficients = np.column_stack(columns)
    return np.maximum(coefficients, 0.0) if variant == "treloar" else coefficients
