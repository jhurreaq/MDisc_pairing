"""Cardiac discovery; original numerical and presentation choices."""

from mdisc.paths import monotone_nonnegative_path
from .plotting import plot_lasso_selection, plot_lars_selection, plot_omp_selection
from functools import partial
from mdisc.regression import fit_lasso_cv, lasso_cv_plot_data, refit_weighted_nnls_with_intercept, weighted_mse, clip_nonnegative
import pandas as pd
import numpy as np
from sklearn.linear_model import lasso_path, lars_path, orthogonal_mp
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
import datetime
import os
import traceback
from mdisc.criteria import calculate_weighted_aic_bic as calculate_aic_bic
from .plotting import plot_activation_heatmap
from mdisc.preprocessing import compute_weighted_stats

def run_lasso_analysis(X_unscaled, y, feature_names, cv_folds, title_prefix, save_dir, sample_weight=None, *, context):
    """
    LASSO with positive coefficients:
      - LassoCV(positive=True, fit_intercept=False)
      - lasso_path(positive=True)
      - Coefficients clipped to >=0 before metrics/AIC/BIC/plots/results.
    """
    print(f'\n--- Running LASSO Analysis (CV, AIC, BIC) ({title_prefix}) ---')
    n_samples, n_features = X_unscaled.shape
    base_results = { 'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
                     'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan, 'k': 0 }
    results = { 'LassoCV': {**base_results, 'method': 'LassoCV'},
                'LassoAIC': {**base_results, 'method': 'LassoAIC'},
                'LassoBIC': {**base_results, 'method': 'LassoBIC'} }
    if n_features == 0:
        print("ERROR: No features."); return results

    # Small helper to ensure strict nonnegativity (guards numerical fuzz)
    _clip_nonneg = clip_nonnegative

    # --- 1) LassoCV (positive) on controlled alpha range ---
    print("  Running LassoCV to determine optimal alpha...")
    start_time_cv = datetime.datetime.now()
    best_alpha_from_cv = np.nan

    alpha_min_log = -8
    alpha_max_log = 0
    alphas_cv_grid = np.logspace(alpha_min_log, alpha_max_log, 1000)

    alphas_cv_plot = np.array([])
    cv_nmse_mean_plot = np.array([])
    cv_nmse_std_plot  = np.array([])

    try:
        lasso_cv = fit_lasso_cv(
            X_unscaled,
            y,
            suppress_warnings=False,
            cv=cv_folds,
            alphas=alphas_cv_grid,
            max_iter=5000,
            n_jobs=-1,
            random_state=42,
            positive=True,
            fit_intercept=False,
        )
        best_alpha_from_cv = lasso_cv.alpha_
        print(f'  LassoCV determined optimal alpha: {best_alpha_from_cv:.6f}')

        # Extract CV MSE and robust-normalize NMSE to [0,1] for plotting
        try:
            alphas_cv_plot, cv_nmse_mean_plot, cv_nmse_std_plot = lasso_cv_plot_data(lasso_cv, y, eps=context.EPS)
        except Exception as e:
            print(f'WARN: Unable to extract/normalize LassoCV CV curve: {e}')
    except Exception as e:
        print(f'ERROR during LassoCV: {e}')
        results['LassoCV'].update({'error': str(e)})

    # --- 2) Unified positive Lasso path (AIC/BIC use the same alpha domain) ---
    print("\n  Generating a unified Lasso Path for all criteria...")
    start_time_path = datetime.datetime.now()
    df_path = pd.DataFrame()
    alphas_lasso, coefs_lasso = None, None

    try:
        alphas_base = np.logspace(alpha_min_log, alpha_max_log, 20)
        alphas_for_path = np.unique(np.concatenate(
            [alphas_base, [best_alpha_from_cv] if pd.notna(best_alpha_from_cv) else []]
        ))
        alphas_for_path = np.sort(alphas_for_path)[::-1]  # descending

        alphas_lasso, coefs_lasso, _ = lasso_path(
            X_unscaled, y, alphas=alphas_for_path,
            positive=True,
            max_iter=5000, tol=1e-4
        )

        # Weighted stats (as in your code)
        w_mse, w_sum, y_mean_w, y_var_w = compute_weighted_stats(y, None)

        path_results_list = []
        for i in range(coefs_lasso.shape[1]):
            alpha_i = alphas_lasso[i]
            beta_i  = _clip_nonneg(coefs_lasso[:, i])
            n_params = int(np.sum(beta_i > 1e-6))
            if n_params == 0 and i < coefs_lasso.shape[1] - 1:
                continue

            y_pred_train = X_unscaled @ beta_i
            residual = y - y_pred_train
            mse_w = np.sum(w_mse * residual**2) / w_sum
            aic, bic = calculate_aic_bic(y, y_pred_train, n_params, X_unscaled.shape[0], weights=None, eps=context.EPS)
            l1_norm = float(np.sum(beta_i))

            path_results_list.append({
                'alpha': alpha_i, 'k': n_params, 'coeffs': beta_i,
                'mse_train': mse_w, 'aic': aic, 'bic': bic, 'l1_norm': l1_norm
            })

        df_path = pd.DataFrame(path_results_list)
        print(f'  Unified path created with {len(df_path)} unique models.')
    except Exception as e:
        print(f'ERROR during unified path generation: {e}')
        traceback.print_exc()
        results['LassoCV'].update({'error': str(e)})
        results['LassoAIC'].update({'error': str(e)})
        results['LassoBIC'].update({'error': str(e)})
        return results

    # --- 2b) Normalized metrics over the path ---
    if 'df_path' in locals() and not df_path.empty:
        if y_var_w > 0:
            df_path['norm_mse'] = df_path['mse_train'] / y_var_w
        else:
            df_path['norm_mse'] = np.nan
        print(f"  DEBUG (Lasso+pos): weighted var={y_var_w:.6e}, NMSE min={df_path['norm_mse'].min():.3f}, max={df_path['norm_mse'].max():.3f}")

    max_l1_norm = df_path['l1_norm'].max()
    df_path['norm_l1'] = (df_path['l1_norm'] / max_l1_norm) if max_l1_norm > context.EPS else 0.0

    # --- 3) Pick models from unified path (use clipped nonnegative coeffs) ---
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    if not df_path.empty and pd.notna(best_alpha_from_cv):
        cv_model_row = df_path.iloc[(np.abs(df_path['alpha'] - best_alpha_from_cv)).argsort()[:1]].iloc[0]
        results['LassoCV'].update({
            'coeffs': cv_model_row['coeffs'], 'alpha': cv_model_row['alpha'],
            'n_nonzero': cv_model_row['k'], 'aic': cv_model_row['aic'], 'bic': cv_model_row['bic'],
            'selected_features': [feature_names[i]['term_label'] for i, c in enumerate(cv_model_row['coeffs']) if c > 1e-6],
            'time': duration_cv, 'k': cv_model_row['k']
        })
        print(f"  LassoCV model (k={cv_model_row['k']}) identified on unified path in {duration_cv:.2f}s.")

    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()
    if not df_path.empty:
        aic_model_row = df_path.loc[df_path['aic'].idxmin()]
        results['LassoAIC'].update({
            'coeffs': aic_model_row['coeffs'], 'alpha': aic_model_row['alpha'], 'n_nonzero': aic_model_row['k'],
            'aic': aic_model_row['aic'], 'bic': aic_model_row['bic'],
            'selected_features': [feature_names[j]['term_label'] for j, c in enumerate(aic_model_row['coeffs']) if c > 1e-6],
            'time': duration_path, 'k': aic_model_row['k']
        })
        print(f"  LassoAIC model (k={aic_model_row['k']}) identified on unified path.")

        bic_model_row = df_path.loc[df_path['bic'].idxmin()]
        results['LassoBIC'].update({
            'coeffs': bic_model_row['coeffs'], 'alpha': bic_model_row['alpha'], 'n_nonzero': bic_model_row['k'],
            'aic': bic_model_row['aic'], 'bic': bic_model_row['bic'],
            'selected_features': [feature_names[j]['term_label'] for j, c in enumerate(bic_model_row['coeffs']) if c > 1e-6],
            'time': duration_path, 'k': bic_model_row['k']
        })
        print(f"  LassoBIC model (k={bic_model_row['k']}) identified on unified path.")

    # --- 4) Plots ---
    plot_lasso_selection(
        _clip_nonneg=_clip_nonneg,
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

    path_data_for_criteria = []
    if not df_path.empty:
        for _, row in df_path.iterrows():
            beta_pos = _clip_nonneg(row['coeffs'])
            y_pred_unweighted = X_unscaled @ beta_pos
            mse_unweighted = mean_squared_error(y, y_pred_unweighted)
            path_data_for_criteria.append({
                'alpha': row['alpha'],
                'k': row['k'],
                'mse_train': row['mse_train'],
                'mse_train_unweighted': mse_unweighted,
                'aic': row['aic'],
                'bic': row['bic'],
                'coeffs': beta_pos
            })

    return results, path_data_for_criteria

def run_lars_analysis(X_unscaled, y, feature_names, cv_folds, title_prefix, save_dir, sample_weight=None, *, context):
    """
    LARS (positive=True) with a unified path used consistently for CV, AIC, and BIC.
    Enforces nonnegative coefficients end-to-end (path, CV refits, full refits).
    Returns ONLY a dict (results) to match the caller's expectation.
    """
    print(f'\n--- Running LARS Analysis (CV, AIC, BIC; unified path, POSITIVE) ({title_prefix}) ---')
    n_samples, n_features = X_unscaled.shape
    base = {'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
            'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan,
            'optimal_k': np.nan, 'k': 0, 'method': None}
    results = {
        'LarsCV':  {**base, 'method': 'LarsCV'},
        'LarsAIC': {**base, 'method': 'LarsAIC'},
        'LarsBIC': {**base, 'method': 'LarsBIC'}
    }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # ---------- helpers ----------
    _clip_nonneg = clip_nonnegative

    _safe_weighted_mse = partial(weighted_mse, eps=context.EPS)

    _fit_nnls_with_intercept = partial(refit_weighted_nnls_with_intercept, allow_empty=False, eps=context.EPS)

    def _annotate_iter(ax, k_val, color):
        y0, y1 = ax.get_ylim()
        y_mid = 0.5 * (y0 + y1)
        x0, x1 = ax.get_xlim()
        dx = max(0.02 * (x1 - x0), 0.1)
        ax.text(k_val + dx, y_mid, f'Iteration = {int(k_val)}',
                rotation=90, va='center', ha='left',
                fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=color)

    # ---------- 1) unified positive LARS path on full data ----------
    start_time_path = datetime.datetime.now()
    try:
        from sklearn.linear_model import lars_path
        _, _, coefs_raw = lars_path(X_unscaled, y, method='lars', positive=True)
        coefs_path, steps_k = monotone_nonnegative_path(coefs_raw)
        if coefs_path is None:
            print("WARN: LARS path is empty after cleaning.")
            return results
    except Exception as e:
        print(f'FATAL ERROR building LARS path: {e}')
        traceback.print_exc()
        return results

    supports = [np.flatnonzero(coefs_path[:, i] > 1e-6) for i in range(coefs_path.shape[1])]

    # ---------- 2) AIC/BIC and train MSE on full data (NNLS refit with intercept) ----------
    path_rows = []
    for i, S in enumerate(supports):
        beta_i = np.zeros(n_features)
        if len(S) == 0:
            # constant-only model (intercept = mean(y) under weights)
            if sample_weight is None:
                b0 = float(np.mean(y))
            else:
                w = np.asarray(sample_weight).ravel()
                b0 = float(np.sum(w * y) / np.clip(np.sum(w), context.EPS, None))
            y_hat = np.full_like(y, b0)
        else:
            b0, beta_S = _fit_nnls_with_intercept(X_unscaled[:, S], y, sample_weight)
            beta_i[S] = beta_S
            y_hat = b0 + X_unscaled[:, S] @ beta_S
        mse_tr = _safe_weighted_mse(y, y_hat, sample_weight)
        # n_params: k + 1 intercept if k>0; if k==0, count 1 parameter (intercept)
        k_eff = len(S) + 1
        aic_i, bic_i = calculate_aic_bic(y, y_hat, k_eff, n_samples, eps=context.EPS)
        path_rows.append({'k': int(steps_k[i]), 'coeffs': beta_i,
                          'mse_train': mse_tr, 'aic': aic_i, 'bic': bic_i})
    df_path = pd.DataFrame(path_rows).drop_duplicates(subset='k').sort_values('k').reset_index(drop=True)

    # ---------- 3) CV on the SAME supports (NNLS refit with intercept in each fold) ----------
    print("  Cross-validating on unified LARS supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_mse_mean, cv_mse_std = [], []
    for i, S in enumerate(supports):
        fold_mse = []
        for tr_idx, va_idx in kf.split(X_unscaled):
            Xtr, Xva = X_unscaled[tr_idx][:, S], X_unscaled[va_idx][:, S]
            ytr, yva = y[tr_idx], y[va_idx]
            wtr = sample_weight[tr_idx] if sample_weight is not None else None
            wva = sample_weight[va_idx] if sample_weight is not None else None
            try:
                if len(S) == 0:
                    # intercept-only in fold
                    if wtr is None:
                        b0 = float(np.mean(ytr))
                    else:
                        sw = np.asarray(wtr).ravel()
                        b0 = float(np.sum(sw * ytr) / np.clip(np.sum(sw), context.EPS, None))
                    yhat_va = np.full_like(yva, b0)
                else:
                    b0, beta_S = _fit_nnls_with_intercept(Xtr, ytr, wtr)
                    yhat_va = b0 + Xva @ beta_S
                mse_va = _safe_weighted_mse(yva, yhat_va, wva)
            except Exception:
                mse_va = np.inf
            fold_mse.append(mse_va)
        fold_mse = np.array(fold_mse, dtype=float)
        cv_mse_mean.append(np.nanmean(fold_mse))
        cv_mse_std.append(np.nanstd(fold_mse))
    cv_mse_mean = np.array(cv_mse_mean); cv_mse_std = np.array(cv_mse_std)

    # robust min–max normalization of CV MSE to [0,1]
    mm_min = np.nanmin(cv_mse_mean); mm_max = np.nanmax(cv_mse_mean)
    scale = (mm_max - mm_min)
    if not np.isfinite(scale) or scale < context.EPS:
        cv_nmse_mean = np.zeros_like(cv_mse_mean)
        cv_nmse_std = np.zeros_like(cv_mse_std)
    else:
        cv_nmse_mean = (cv_mse_mean - mm_min) / scale
        cv_nmse_std = cv_mse_std / scale

    best_idx = int(np.nanargmin(cv_mse_mean))
    best_k = int(steps_k[best_idx])

    # ---------- 4) results (coeffs already nonnegative) ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()

    row_cv = df_path.iloc[best_idx]
    results['LarsCV'].update({
        'coeffs': row_cv['coeffs'], 'n_nonzero': int(row_cv['k']),
        'aic': row_cv['aic'], 'bic': row_cv['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_cv['coeffs'][j] > 1e-6 ],
        'time': duration_cv, 'optimal_k': best_k, 'k': best_k
    })
    idx_aic = int(df_path['aic'].values.argmin())
    idx_bic = int(df_path['bic'].values.argmin())
    row_aic = df_path.iloc[idx_aic]; row_bic = df_path.iloc[idx_bic]
    results['LarsAIC'].update({
        'coeffs': row_aic['coeffs'], 'n_nonzero': int(row_aic['k']),
        'aic': row_aic['aic'], 'bic': row_aic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_aic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_aic['k']), 'k': int(row_aic['k'])
    })
    results['LarsBIC'].update({
        'coeffs': row_bic['coeffs'], 'n_nonzero': int(row_bic['k']),
        'aic': row_bic['aic'], 'bic': row_bic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_bic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_bic['k']), 'k': int(row_bic['k'])
    })

    # ---------- 5) plots ----------
    try:
        chosen = {
            'CV':  {'index': best_idx, 'label_val': best_k,               'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': idx_aic,  'label_val': int(row_aic['k']),    'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': idx_bic,  'label_val': int(row_bic['k']),    'color': context.plasma(0.85), 'linestyle': '-.'}
        }
        # coefs_path already clipped to >=0
        plot_activation_heatmap(
            steps_k, coefs_path, feature_names, chosen,
            title="LARS Activation Path (positive)", x_axis_label="Iteration (k)",
            filename=os.path.join(save_dir, f'{title_prefix}_LARS_ActivationHeatmap.pdf'),
            log_x=False, reverse_x=False, is_omp=False,
            context=context,
        )
    except Exception as e:
        print(f'WARN: LARS heatmap plotting failed: {e}')

    plot_lars_selection(
        _annotate_iter=_annotate_iter,
        best_k=best_k,
        context=context,
        cv_nmse_mean=cv_nmse_mean,
        cv_nmse_std=cv_nmse_std,
        df_path=df_path,
        row_aic=row_aic,
        row_bic=row_bic,
        save_dir=save_dir,
        steps_k=steps_k,
        title_prefix=title_prefix,
    )

    return results

def run_omp_analysis(X_unscaled, y, feature_names, cv_folds, title_prefix, save_dir, sample_weight=None, *, context):
    """
    OMP with a unified path used consistently for CV, AIC, and BIC.
    NOW ENFORCES NONNEGATIVE COEFFICIENTS end-to-end (path, CV refits, full refits).
    Returns ONLY a dict (results) to match the caller's expectation.
    """
    print(f'\n--- Running OMP Analysis (CV, AIC, BIC; unified path, POSITIVE) ({title_prefix}) ---')
    n_samples, n_features = X_unscaled.shape
    base = {'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
            'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan,
            'optimal_k': np.nan, 'k': 0, 'method': None}
    results = {
        'OMPCV':  {**base, 'method': 'OMPCV'},
        'OMPAIC': {**base, 'method': 'OMPAIC'},
        'OMPBIC': {**base, 'method': 'OMPBIC'}
    }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # ---------- helpers ----------
    _clip_nonneg = clip_nonnegative

    _safe_weighted_mse = partial(weighted_mse, eps=context.EPS)

    _fit_nnls_with_intercept = partial(refit_weighted_nnls_with_intercept, allow_empty=True, eps=context.EPS)

    def _annotate_iter(ax, k_val, color):
        y0, y1 = ax.get_ylim(); y_mid = 0.5*(y0+y1)
        x0, x1 = ax.get_xlim(); dx = max(0.02*(x1-x0), 0.1)
        ax.text(k_val + dx, y_mid, f'Iteration = {int(k_val)}',
                rotation=90, va='center', ha='left',
                fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=color)

    # ---------- 1) unified OMP path on full data (then clipped to >=0) ----------
    start_time_path = datetime.datetime.now()
    try:
        Kmax = min(n_features, n_samples - 1) if n_samples > 1 else n_features
        coef_cols, steps_k = [], []
        last_k = 0
        for k in range(1, Kmax + 1):
            b_raw = orthogonal_mp(X_unscaled, y, n_nonzero_coefs=k, precompute=True)
            b_pos = _clip_nonneg(b_raw)  # enforce positivity on path
            k_now = int(np.sum(b_pos > 1e-6))
            if k_now > last_k:
                coef_cols.append(b_pos)
                steps_k.append(k_now)
                last_k = k_now
        if len(coef_cols) == 0:
            print("WARN: OMP path is empty.")
            return results
        coefs_path = np.column_stack(coef_cols)  # (p, n_steps), all >= 0
        steps_k = np.array(steps_k, dtype=int)
    except Exception as e:
        print(f'FATAL ERROR building OMP path: {e}')
        traceback.print_exc()
        return results

    supports = [np.flatnonzero(coefs_path[:, i] > 1e-6) for i in range(coefs_path.shape[1])]

    # ---------- 2) AIC/BIC and train MSE on full data (NNLS refit with intercept) ----------
    path_rows = []
    for i, S in enumerate(supports):
        beta_i = np.zeros(n_features)
        if len(S) == 0:
            # intercept-only
            if sample_weight is None:
                b0 = float(np.mean(y))
            else:
                w = np.asarray(sample_weight).ravel()
                b0 = float(np.sum(w * y) / np.clip(np.sum(w), context.EPS, None))
            y_hat = np.full_like(y, b0)
        else:
            b0, beta_S = _fit_nnls_with_intercept(X_unscaled[:, S], y, sample_weight)
            beta_i[S] = beta_S
            y_hat = b0 + X_unscaled[:, S] @ beta_S
        mse_tr = _safe_weighted_mse(y, y_hat, sample_weight)
        k_eff = len(S) + 1  # include intercept
        aic_i, bic_i = calculate_aic_bic(y, y_hat, k_eff, n_samples, eps=context.EPS)
        path_rows.append({'k': int(steps_k[i]), 'coeffs': beta_i,
                          'mse_train': mse_tr, 'aic': aic_i, 'bic': bic_i})
    df_path = pd.DataFrame(path_rows).drop_duplicates(subset='k').sort_values('k').reset_index(drop=True)

    # ---------- 3) CV on the SAME supports (NNLS refit with intercept in each fold) ----------
    print("  Cross-validating on unified OMP supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_mse_mean, cv_mse_std = [], []
    for i, S in enumerate(supports):
        fold_mse = []
        for tr_idx, va_idx in kf.split(X_unscaled):
            Xtr, Xva = X_unscaled[tr_idx][:, S], X_unscaled[va_idx][:, S]
            ytr, yva = y[tr_idx], y[va_idx]
            wtr = sample_weight[tr_idx] if sample_weight is not None else None
            wva = sample_weight[va_idx] if sample_weight is not None else None
            try:
                if len(S) == 0:
                    # intercept-only in fold
                    if wtr is None:
                        b0 = float(np.mean(ytr))
                    else:
                        sw = np.asarray(wtr).ravel()
                        b0 = float(np.sum(sw * ytr) / np.clip(np.sum(sw), context.EPS, None))
                    yhat_va = np.full_like(yva, b0)
                else:
                    b0, beta_S = _fit_nnls_with_intercept(Xtr, ytr, wtr)
                    yhat_va = b0 + Xva @ beta_S
                mse_va = _safe_weighted_mse(yva, yhat_va, wva)
            except Exception:
                mse_va = np.inf
            fold_mse.append(mse_va)
        fold_mse = np.array(fold_mse, dtype=float)
        cv_mse_mean.append(np.nanmean(fold_mse))
        cv_mse_std.append(np.nanstd(fold_mse))
    cv_mse_mean = np.array(cv_mse_mean); cv_mse_std = np.array(cv_mse_std)

    # robust min–max normalization to [0,1]
    mm_min = np.nanmin(cv_mse_mean); mm_max = np.nanmax(cv_mse_mean)
    scale = (mm_max - mm_min)
    if not np.isfinite(scale) or scale < context.EPS:
        cv_nmse_mean = np.zeros_like(cv_mse_mean)
        cv_nmse_std = np.zeros_like(cv_mse_std)
    else:
        cv_nmse_mean = (cv_mse_mean - mm_min) / scale
        cv_nmse_std = cv_mse_std / scale

    best_idx = int(np.nanargmin(cv_mse_mean))
    best_k = int(steps_k[best_idx])

    # ---------- 4) results (coeffs already nonnegative) ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()

    row_cv = df_path.iloc[best_idx]
    results['OMPCV'].update({
        'coeffs': row_cv['coeffs'], 'n_nonzero': int(row_cv['k']),
        'aic': row_cv['aic'], 'bic': row_cv['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_cv['coeffs'][j] > 1e-6 ],
        'time': duration_cv, 'optimal_k': best_k, 'k': best_k
    })
    idx_aic = int(df_path['aic'].values.argmin())
    idx_bic = int(df_path['bic'].values.argmin())
    row_aic = df_path.iloc[idx_aic]; row_bic = df_path.iloc[idx_bic]
    results['OMPAIC'].update({
        'coeffs': row_aic['coeffs'], 'n_nonzero': int(row_aic['k']),
        'aic': row_aic['aic'], 'bic': row_aic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_aic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_aic['k']), 'k': int(row_aic['k'])
    })
    results['OMPBIC'].update({
        'coeffs': row_bic['coeffs'], 'n_nonzero': int(row_bic['k']),
        'aic': row_bic['aic'], 'bic': row_bic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_bic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_bic['k']), 'k': int(row_bic['k'])
    })

    # ---------- 5) plots ----------
    try:
        chosen = {
            'CV':  {'index': best_idx, 'label_val': best_k,               'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': idx_aic,  'label_val': int(row_aic['k']),    'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': idx_bic,  'label_val': int(row_bic['k']),    'color': context.plasma(0.85), 'linestyle': '-.'}
        }
        plot_activation_heatmap(
            steps_k, coefs_path, feature_names, chosen,
            title="OMP Activation Path (positive)", x_axis_label="Iteration (k)",
            filename=os.path.join(save_dir, f'{title_prefix}_OMP_ActivationHeatmap.pdf'),
            log_x=False, reverse_x=False, is_omp=True,
            context=context,
        )
    except Exception as e:
        print(f'WARN: OMP heatmap plotting failed: {e}')

    plot_omp_selection(
        _annotate_iter=_annotate_iter,
        best_k=best_k,
        context=context,
        cv_nmse_mean=cv_nmse_mean,
        cv_nmse_std=cv_nmse_std,
        df_path=df_path,
        row_aic=row_aic,
        row_bic=row_bic,
        save_dir=save_dir,
        steps_k=steps_k,
        title_prefix=title_prefix,
    )

    return results
