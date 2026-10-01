"""Shared criteria operations, preserving the original formulas."""

import numpy as np

def calculate_aic_bic(y_true, y_pred, n_params, n_samples, *, eps=1e-10):
    """Calculates AIC and BIC."""
    if n_samples <= 0: return np.nan, np.nan
    if n_params == 0: print("WARN: Calculating AIC/BIC with n_params=0. Returning NaN."); return np.nan, np.nan
    if n_params >= n_samples: pass # Allow calculation but be aware
    rss = np.sum((y_true - y_pred)**2)
    if rss < eps: log_likelihood_proxy = n_samples * (100)
    else: log_likelihood_proxy = -n_samples / 2.0 * np.log(max(rss / n_samples, eps))
    k_effective = n_params # Assumes no implicit intercept
    aic = -2 * log_likelihood_proxy + 2 * k_effective
    bic = -2 * log_likelihood_proxy + k_effective * np.log(max(n_samples, 1))
    return aic, bic

def calculate_weighted_aic_bic(y_true, y_pred, n_params, n_samples, weights=None, *, eps=1e-10):
    if n_samples <= 0 or n_params >= n_samples or n_params == 0: return np.nan, np.nan
    if weights is None: weights = np.ones_like(y_true)
    if len(weights) != len(y_true): weights = np.ones_like(y_true)
    # Ensure y_pred has same length as y_true and weights
    if len(y_pred) != len(y_true):
        print(f"WARN: y_pred length ({len(y_pred)}) != y_true length ({len(y_true)}) in AIC/BIC calc. Returning NaN.")
        return np.nan, np.nan

    # Filter out NaN predictions before calculating SSR
    valid_mask = ~np.isnan(y_pred)
    if np.sum(valid_mask) < n_params + 1: # Need more points than params
        return np.nan, np.nan

    y_true_f = y_true[valid_mask]
    y_pred_f = y_pred[valid_mask]
    weights_f = weights[valid_mask]
    n_samples_f = len(y_true_f)
    if n_samples_f <= n_params: return np.nan, np.nan # Check again after filtering

    weighted_ssr = np.sum(weights_f * (y_true_f - y_pred_f)**2)
    if weighted_ssr < eps: return -np.inf, -np.inf
    log_likelihood_proxy = -n_samples_f / 2.0 * np.log(max(weighted_ssr / n_samples_f, eps))
    aic = -2 * log_likelihood_proxy + 2 * n_params
    bic = -2 * log_likelihood_proxy + n_params * np.log(max(n_samples_f, 1))
    return aic, bic
