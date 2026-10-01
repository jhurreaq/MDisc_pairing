"""Synthetic discovery; original numerical and presentation choices."""

from mdisc.paths import unique_size_supports, positive_omp_path
from .plotting import plot_lasso_selection, plot_lars_selection, plot_omp_selection
from mdisc.regression import fit_lasso_cv, lasso_cv_plot_data, refit_support_nnls
import numpy as np
import pandas as pd
from sklearn.linear_model import lasso_path, lars_path
from sklearn.metrics import mean_squared_error
import datetime
import traceback
from sklearn.model_selection import KFold
from mdisc.criteria import calculate_aic_bic

def run_lasso_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None, *, context):
    """
    Performs LassoCV (coordinate descent) and Lasso path AIC/BIC selection.
    Plots a single heatmap and criteria plots. Measures time.
    Returns a dictionary of results for 'LassoCV', 'LassoAIC', 'LassoBIC'.
    Enforces non-negative coefficients (positive=True) for physical consistency.
    """
    print(f'\n--- Running LASSO Analysis (CV, AIC, BIC) ({title_prefix}) ---')
    n_samples, n_features = X_scaled.shape
    base_results = { 'coeffs': np.zeros(n_features), 'optimal_k': np.nan, 'n_nonzero': 0, 'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan, 'full_basis_names': feature_names }
    results = { 'LassoCV': {**base_results, 'method': 'LassoCV'}, 'LassoAIC': {**base_results, 'method': 'LassoAIC'}, 'LassoBIC': {**base_results, 'method': 'LassoBIC'} }
    if n_features == 0: print("ERROR: No features."); return results

    # --- 1. Run LassoCV to find the optimal alpha (non-negative) ---
    print("  Running LassoCV to determine optimal alpha...")
    start_time_cv = datetime.datetime.now()
    best_alpha_from_cv = np.nan

    # Define a single source of truth for the alpha range
    alpha_min_log = -8
    alpha_max_log = 0
    alphas_cv_grid = np.logspace(alpha_min_log, alpha_max_log, 1000)  # dense for CV

    try:
        # Use the explicit grid for CV (bounded) and enforce β >= 0
        lasso_cv = fit_lasso_cv(
            X_scaled,
            y,
            suppress_warnings=True,
            cv=cv_folds,
            alphas=alphas_cv_grid,
            max_iter=5000,
            n_jobs=-1,
            random_state=random_seed,
            precompute='auto',
            positive=True,
        )
        best_alpha_from_cv = lasso_cv.alpha_
        print(f'  LassoCV determined optimal alpha: {best_alpha_from_cv:.6f}')

        alphas_cv_plot = np.array([]); cv_nmse_mean_plot = np.array([]); cv_nmse_std_plot = np.array([])
        try:
            alphas_cv_plot, cv_nmse_mean_plot, cv_nmse_std_plot = lasso_cv_plot_data(lasso_cv, y, eps=context.EPS)
        except Exception as e:
            print(f'WARN: Unable to extract/normalize LassoCV CV curve: {e}')

    except Exception as e:
        print(f'ERROR during LassoCV: {e}')
        results['LassoCV'].update({'error': str(e)})

    # --- 2. Create the UNIFIED PATH DataFrame (non-negative) ---
    print("\n  Generating a unified Lasso Path for all criteria...")
    start_time_path = datetime.datetime.now()
    df_path = pd.DataFrame()
    alphas_lasso, coefs_lasso = None, None
    try:
        y_centered = y - np.mean(y)

        # Generate a rich path, ensuring the CV alpha is included for exact matching
        alphas_base = np.logspace(alpha_min_log, alpha_max_log, 20)
        alphas_for_path = np.unique(np.concatenate([alphas_base, [best_alpha_from_cv] if pd.notna(best_alpha_from_cv) else []]))
        alphas_for_path = np.sort(alphas_for_path)[::-1] # Sort descending

        alphas_lasso, coefs_lasso, _ = lasso_path(
            X_scaled, y, alphas=alphas_for_path, max_iter=5000, precompute='auto', positive=True
        )
        # numerical guard
        coefs_lasso = np.maximum(coefs_lasso, 0.0)

        path_results_list = []
        for i in range(coefs_lasso.shape[1]):
            alpha_i = alphas_lasso[i]
            beta_i = np.maximum(coefs_lasso[:, i], 0.0)  # ensure non-negative
            n_params = np.sum(beta_i > context.EPS)

            if n_params == 0 and i < coefs_lasso.shape[1] - 1: continue

            y_pred_train = X_scaled @ beta_i
            mse = mean_squared_error(y_centered, y_pred_train)
            aic, bic = calculate_aic_bic(y, y_pred_train, int(n_params), n_samples, eps=context.EPS)

            path_results_list.append({
                'alpha': alpha_i, 'k': int(n_params), 'coeffs': beta_i,
                'mse_train': mse, 'aic': aic, 'bic': bic
            })

        df_path = pd.DataFrame(path_results_list)
        print(f'  Unified path created with {len(df_path)} unique models.')

    except Exception as e:
        print(f'ERROR during unified path generation: {e}'); traceback.print_exc()
        results['LassoCV'].update({'error': str(e)}); results['LassoAIC'].update({'error': str(e)}); results['LassoBIC'].update({'error': str(e)})
        return results

    # --- 2b. Calculate Normalized Metrics for the Path ---
    y_var = np.var(y_centered)
    if y_var > context.EPS:
        df_path['norm_mse'] = df_path['mse_train'] / y_var
    else:
        df_path['norm_mse'] = df_path['mse_train'] # Avoid division by zero if y is constant

    df_path['l1_norm'] = df_path['coeffs'].apply(lambda x: np.sum(np.abs(x)))
    max_l1_norm = df_path['l1_norm'].max()
    if max_l1_norm > context.EPS:
        df_path['norm_l1'] = df_path['l1_norm'] / max_l1_norm
    else:
        df_path['norm_l1'] = 0.0

    # --- 3. Identify Optimal Models from the UNIFIED DataFrame ---
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    if not df_path.empty and pd.notna(best_alpha_from_cv):
        cv_model_row = df_path.iloc[(df_path['alpha'] - best_alpha_from_cv).abs().argsort()[:1]].iloc[0]
        results['LassoCV'].update({
            'coeffs': cv_model_row['coeffs'], 'alpha': cv_model_row['alpha'], 'n_nonzero': cv_model_row['k'],
            'aic': cv_model_row['aic'], 'bic': cv_model_row['bic'],
            'selected_features': [feature_names[i] for i, c in enumerate(cv_model_row['coeffs']) if c > context.EPS],  # positivity-aware
            'time': duration_cv
        })
        print(f"  LassoCV model (k={cv_model_row['k']}) identified on unified path in {duration_cv:.2f}s.")

    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()
    if not df_path.empty:
        aic_model_row = df_path.loc[df_path['aic'].idxmin()]
        results['LassoAIC'].update({
            'coeffs': aic_model_row['coeffs'], 'alpha': aic_model_row['alpha'], 'n_nonzero': aic_model_row['k'],
            'aic': aic_model_row['aic'], 'bic': aic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(aic_model_row['coeffs']) if c > context.EPS],  # positivity-aware
            'time': duration_path
        })
        print(f"  LassoAIC model (k={aic_model_row['k']}) identified on unified path.")

        bic_model_row = df_path.loc[df_path['bic'].idxmin()]
        results['LassoBIC'].update({
            'coeffs': bic_model_row['coeffs'], 'alpha': bic_model_row['alpha'], 'n_nonzero': bic_model_row['k'],
            'aic': bic_model_row['aic'], 'bic': bic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(bic_model_row['coeffs']) if c > context.EPS],  # positivity-aware
            'time': duration_path
        })
        print(f"  LassoBIC model (k={bic_model_row['k']}) identified on unified path.")

    # --- 4. Plotting from the UNIFIED DataFrame ---
    plot_lasso_selection(
        alphas_cv_plot=alphas_cv_plot,
        alphas_lasso=alphas_lasso,
        best_alpha_from_cv=best_alpha_from_cv,
        coefs_lasso=coefs_lasso,
        context=context,
        cv_nmse_mean_plot=cv_nmse_mean_plot,
        cv_nmse_std_plot=cv_nmse_std_plot,
        df_path=df_path,
        feature_names=feature_names,
        save_dir=save_dir,
        title_prefix=title_prefix,
    )

    return results

def run_lars_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None, *, context):
    """
    LARS with unified path for CV, AIC, and BIC.
    Enforces non-negative coefficients (β >= 0) via NNLS refits on each support.
    CV error is normalized (NMSE) to [0, 1] via min–max scaling across the path.
    """
    from scipy.optimize import nnls
    print(f'\n--- Running LARS Analysis (CV, AIC, BIC; unified path) ({title_prefix}) ---')

    n_samples, n_features = X_scaled.shape
    base_results = {
        'coeffs': np.zeros(n_features),
        'optimal_k': np.nan,
        'n_nonzero': 0,
        'aic': np.nan,
        'bic': np.nan,
        'selected_features': [],
        'time': np.nan,
        'full_basis_names': feature_names
    }
    results = {
        'LarsCV':  {**base_results, 'method': 'LarsCV'},
        'LarsAIC': {**base_results, 'method': 'LarsAIC'},
        'LarsBIC': {**base_results, 'method': 'LarsBIC'},
    }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # ---------- helper: NNLS refit with unconstrained intercept ----------
    _nnls_refit = refit_support_nnls

    # 1) LARS path (support order only)
    start_time_path = datetime.datetime.now()
    y_centered = y - np.mean(y)
    try:
        from sklearn.linear_model import lars_path
        _, _, coefs_path_raw = lars_path(
            X_scaled, y_centered,
            method='lars',
            copy_X=True,
            eps=1e-10,
            max_iter=min(5 * n_features, max(n_features, 100))
        )
        if coefs_path_raw.shape[1] < 2:
            raise ValueError("LARS path contains fewer than 2 steps.")
        coefs_path_lars = coefs_path_raw[:, 1:]  # drop all-zero column
        n_steps = coefs_path_lars.shape[1]
        print(f'  LARS path computed with {n_steps} nonzero steps.')
    except Exception as e:
        print(f'ERROR: LARS path construction failed: {e}')
        traceback.print_exc()
        return results

    # Unique-k compression (based on support size; sign ignored for support detection)
    support_arrays, kept_indices = unique_size_supports(coefs_path_lars, absolute=True, eps=context.EPS)
    supports = list(zip(kept_indices, support_arrays))

    steps_for_plot = np.arange(1, len(supports) + 1)
    print(f'  Unique-k path length: {len(supports)} (k runs from 1 to {len(supports)}).')

    if len(supports) == 0:
        print("ERROR: No supports after unique-k compression.")
        return results

    # Build a β≥0 coefficient matrix for the heatmap using NNLS refits on the full data
    coefs_nnls_cols = []
    for _, S_t in supports:
        beta_full, b_full, _ = _nnls_refit(X_scaled, y, S_t)
        coefs_nnls_cols.append(beta_full)
    coefs_path_nonneg = np.column_stack(coefs_nnls_cols)  # shape (p, len(supports))

    # 2) AIC/BIC on full data (β≥0 via NNLS refit)
    df_path_rows = []
    for idx_in_path, (t, S_t) in enumerate(supports):
        k_t = len(S_t)
        if k_t == 0:
            continue
        try:
            beta_full, b_full, y_hat_full = _nnls_refit(X_scaled, y, S_t)
            # effective parameters: k coefficients + intercept
            aic, bic = calculate_aic_bic(
                y_true=y, y_pred=y_hat_full,
                n_params=k_t + 1, n_samples=n_samples,
                eps=context.EPS,
            )
            # training MSE (non-centered) for record
            mse_train = mean_squared_error(y, y_hat_full)
            df_path_rows.append({
                'step': idx_in_path + 1,
                'k': k_t,
                'coeffs': beta_full,
                'support': S_t,
                'mse_train': mse_train,
                'aic': aic,
                'bic': bic
            })
        except Exception as e_step:
            print(f'  WARN: Full-data NNLS refit failed at step {idx_in_path + 1} (k={k_t}): {e_step}')

    if not df_path_rows:
        print("ERROR: No valid steps for AIC/BIC.")
        return results
    df_path = pd.DataFrame(df_path_rows)

    # 3) CV with the same supports (β≥0 via NNLS in each fold)
    start_time_cv = datetime.datetime.now()
    if isinstance(cv_folds, int):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        kf = cv_folds

    cv_mse_mean = []
    cv_mse_std  = []
    for _, S_t in supports:
        k_t = len(S_t)
        if k_t == 0:
            cv_mse_mean.append(np.nan)
            cv_mse_std.append(np.nan)
            continue

        fold_mse = []
        for tr_idx, va_idx in kf.split(X_scaled, y):
            X_tr = X_scaled[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_scaled[va_idx]
            y_va = y[va_idx]

            try:
                beta_full_tr, b_tr, _ = _nnls_refit(X_tr, y_tr, S_t)
                # predict on validation
                y_va_hat = b_tr + X_va[:, S_t] @ beta_full_tr[S_t]
                fold_mse.append(mean_squared_error(y_va, y_va_hat))
            except Exception:
                fold_mse.append(np.inf)

        cv_mse_mean.append(np.mean(fold_mse))
        cv_mse_std.append(np.std(fold_mse, ddof=1) if len(fold_mse) > 1 else 0.0)

    cv_mse_mean = np.array(cv_mse_mean)
    cv_mse_std  = np.array(cv_mse_std)

    # Robust normalization to [0,1] across the path (display scaling)
    finite_mask = np.isfinite(cv_mse_mean)
    if np.any(finite_mask):
        min_mse = float(np.nanmin(cv_mse_mean))
        max_mse = float(np.nanmax(cv_mse_mean))
        if np.isclose(max_mse, min_mse):
            cv_nmse_mean = np.zeros_like(cv_mse_mean)
            cv_nmse_std  = np.zeros_like(cv_mse_std)
        else:
            rng = max(max_mse - min_mse, context.EPS)
            cv_nmse_mean = (cv_mse_mean - min_mse) / rng
            cv_nmse_std  = cv_mse_std / rng
    else:
        cv_nmse_mean = cv_mse_mean
        cv_nmse_std  = cv_mse_std

    best_idx_cv = int(np.nanargmin(cv_mse_mean))  # choose by true MSE; display uses normalized
    cv_row = df_path.iloc[best_idx_cv]
    best_k_from_cv = int(cv_row['k'])
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    print(f'  CV chose k={best_k_from_cv} on the unified LARS path (NNLS-refit).')

    # 4) Package results (β≥0)
    results['LarsCV'].update({
        'coeffs': cv_row['coeffs'],
        'optimal_k': cv_row['k'],
        'n_nonzero': int(np.sum(cv_row['coeffs'] > context.EPS)),
        'aic': cv_row['aic'],
        'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })

    aic_row = df_path.loc[df_path['aic'].idxmin()]
    results['LarsAIC'].update({
        'coeffs': aic_row['coeffs'],
        'optimal_k': aic_row['k'],
        'n_nonzero': int(np.sum(aic_row['coeffs'] > context.EPS)),
        'aic': aic_row['aic'],
        'bic': aic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > context.EPS],
        'time': (datetime.datetime.now() - start_time_path).total_seconds()
    })

    bic_row = df_path.loc[df_path['bic'].idxmin()]
    results['LarsBIC'].update({
        'coeffs': bic_row['coeffs'],
        'optimal_k': bic_row['k'],
        'n_nonzero': int(np.sum(bic_row['coeffs'] > context.EPS)),
        'aic': bic_row['aic'],
        'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > context.EPS],
        'time': (datetime.datetime.now() - start_time_path).total_seconds()
    })

    # 5) Plotting (use NNLS-refit coefficients for heatmap)
    order_by_k = np.argsort(df_path['k'].values)
    k_sorted = df_path['k'].values[order_by_k].astype(int)
    cv_nmse_mean_sorted = cv_nmse_mean[order_by_k]
    cv_nmse_std_sorted  = cv_nmse_std[order_by_k]

    chosen_values_heatmap = {
        'CV':  {'index': best_idx_cv, 'label_val': int(cv_row['k']),  'color': context.plasma(0.25), 'linestyle': '--'},
        'AIC': {'index': int(np.where(df_path.index == df_path['aic'].idxmin())[0][0]),
                'label_val': int(aic_row['k']), 'color': context.plasma(0.55), 'linestyle': '-'},
        'BIC': {'index': int(np.where(df_path.index == df_path['bic'].idxmin())[0][0]),
                'label_val': int(bic_row['k']), 'color': context.plasma(0.85), 'linestyle': '-.'}
    }
    plot_lars_selection(
        aic_row=aic_row,
        bic_row=bic_row,
        chosen_values_heatmap=chosen_values_heatmap,
        coefs_path_nonneg=coefs_path_nonneg,
        context=context,
        cv_nmse_mean_sorted=cv_nmse_mean_sorted,
        cv_nmse_std_sorted=cv_nmse_std_sorted,
        cv_row=cv_row,
        df_path=df_path,
        feature_names=feature_names,
        k_sorted=k_sorted,
        save_dir=save_dir,
        steps_for_plot=steps_for_plot,
        title_prefix=title_prefix,
    )

    return results

def run_omp_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None, *, context):
    """
    OMP with a SINGLE shared greedy path used for CV, AIC, and BIC.
    Enforces non-negative coefficients (β >= 0) throughout:
      - Path construction: NNLS on centered target (no intercept) and positive-correlation selection.
      - CV/AIC/BIC refits: NNLS with unconstrained intercept recovered analytically.

    Plots and tick behavior follow the same layout and ticks as elsewhere.
    """
    from scipy.optimize import nnls

    print(f'\n--- Running OMP Analysis (CV, AIC, BIC; unified path) ({title_prefix}) ---')
    n_samples, n_features = X_scaled.shape
    base_results = {
        'coeffs': np.zeros(n_features),
        'optimal_k': np.nan,
        'n_nonzero': 0,
        'aic': np.nan,
        'bic': np.nan,
        'selected_features': [],
        'time': np.nan,
        'full_basis_names': feature_names
    }
    results = {
        'OMPCV':  {**base_results, 'method': 'OMPCV'},
        'OMPAIC': {**base_results, 'method': 'OMPAIC'},
        'OMPBIC': {**base_results, 'method': 'OMPBIC'},
    }
    if n_features == 0:
        print("ERROR: No features."); return results

    # Helper: NNLS refit with unconstrained intercept
    _nnls_refit_with_intercept = refit_support_nnls

    # 1) Greedy OMP path on centered y (β ≥ 0 via NNLS; positive-correlation selection)
    X = X_scaled
    coefs_path_nonneg = positive_omp_path(X, y, variant="synthetic", eps=context.EPS)
    if coefs_path_nonneg is None:
        return results
    steps = np.arange(1, coefs_path_nonneg.shape[1] + 1)
    supports = [np.flatnonzero(coefs_path_nonneg[:, i] > context.EPS) for i in range(coefs_path_nonneg.shape[1])]
    K = len(supports)
    print(f'  OMP path computed with K={K} steps (k from 1..{K}).')

    # 2) AIC/BIC on full data (NNLS with free intercept)
    df_path_rows = []
    for i, S in enumerate(supports, start=1):
        try:
            beta_full, b_full, y_hat_full = _nnls_refit_with_intercept(X, y, S)
            aic, bic = calculate_aic_bic(y_true=y, y_pred=y_hat_full,
                                         n_params=len(S) + 1, n_samples=n_samples, eps=context.EPS)
            df_path_rows.append({
                'step': i,
                'k': len(S),
                'coeffs': beta_full,     # nonnegative β (full length)
                'support': S,
                'aic': aic,
                'bic': bic
            })
        except Exception as e_step:
            print(f'  WARN: Full-data NNLS refit failed at k={i}: {e_step}')

    if not df_path_rows:
        print("ERROR: No valid steps for AIC/BIC in OMP."); return results
    df_path = pd.DataFrame(df_path_rows)

    # 3) CV on the same supports (NNLS + free intercept in each fold)
    print("  Cross-validating on unified OMP supports…")
    start_time_cv = datetime.datetime.now()
    if isinstance(cv_folds, int):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        kf = cv_folds

    cv_mse_mean = np.zeros(K)
    cv_mse_std  = np.zeros(K)
    for i, S in enumerate(supports):
        fold_mse = []
        for tr, va in kf.split(X, y):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            try:
                beta_tr, b_tr, _ = _nnls_refit_with_intercept(Xtr, ytr, S)
                yhat = b_tr + Xva[:, S] @ beta_tr[S]
                fold_mse.append(mean_squared_error(yva, yhat))
            except Exception:
                fold_mse.append(np.inf)
        cv_mse_mean[i] = np.mean(fold_mse)
        cv_mse_std[i]  = np.std(fold_mse, ddof=1) if len(fold_mse) > 1 else 0.0

    # Normalize to [0,1] for display (retain true MSE to choose the best)
    mm = float(np.nanmin(cv_mse_mean)); MM = float(np.nanmax(cv_mse_mean))
    if np.isclose(MM, mm):
        cv_nmse_mean = np.zeros_like(cv_mse_mean); cv_nmse_std = np.zeros_like(cv_mse_std)
    else:
        rng = max(MM - mm, context.EPS)
        cv_nmse_mean = (cv_mse_mean - mm) / rng
        cv_nmse_std  = cv_mse_std / rng

    best_idx = int(np.nanargmin(cv_mse_mean))
    cv_row  = df_path.iloc[best_idx]
    aic_row = df_path.iloc[df_path['aic'].idxmin()]
    bic_row = df_path.iloc[df_path['bic'].idxmin()]
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    print(f"  CV chose k={int(cv_row['k'])} on the unified OMP path (NNLS-refit).")

    # 4) Results (β ≥ 0)
    results['OMPCV'].update({
        'coeffs': cv_row['coeffs'],
        'optimal_k': int(cv_row['k']),
        'n_nonzero': int(np.sum(cv_row['coeffs'] > context.EPS)),
        'aic': cv_row['aic'],
        'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })
    results['OMPAIC'].update({
        'coeffs': aic_row['coeffs'],
        'optimal_k': int(aic_row['k']),
        'n_nonzero': int(np.sum(aic_row['coeffs'] > context.EPS)),
        'aic': aic_row['aic'],
        'bic': aic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })
    results['OMPBIC'].update({
        'coeffs': bic_row['coeffs'],
        'optimal_k': int(bic_row['k']),
        'n_nonzero': int(np.sum(bic_row['coeffs'] > context.EPS)),
        'aic': bic_row['aic'],
        'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })

    # 5) Plots (heatmap uses β ≥ 0 from full-data NNLS)
    plot_omp_selection(
        X=X,
        _nnls_refit_with_intercept=_nnls_refit_with_intercept,
        aic_row=aic_row,
        best_idx=best_idx,
        bic_row=bic_row,
        context=context,
        cv_nmse_mean=cv_nmse_mean,
        cv_nmse_std=cv_nmse_std,
        cv_row=cv_row,
        df_path=df_path,
        feature_names=feature_names,
        save_dir=save_dir,
        steps=steps,
        supports=supports,
        title_prefix=title_prefix,
        y=y,
    )

    return results
