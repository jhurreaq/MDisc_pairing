"""Treloar discovery; original numerical and presentation choices."""

from mdisc.paths import unique_size_supports, positive_omp_path
from .plotting import plot_lasso_selection, plot_lars_selection, plot_omp_selection
from functools import partial
from mdisc.regression import fit_lasso_cv, lasso_cv_plot_data, solve_nnls, refit_nnls_with_intercept
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
    Performs LassoCV (coordinate descent) and Lasso path AIC/BIC selection with non-negative coefficients.
    - CV curve uses LassoCV(positive=True)
    - AIC/BIC path uses lasso_path(positive=True)
    - Coefficients are clipped to >= 0 for robustness
    Plots a single heatmap and criteria plots. Measures time.
    Returns a dictionary of results for 'LassoCV', 'LassoAIC', 'LassoBIC'.
    Updated to match HearthCANN/synthetic CV-NMSE plotting (robustly normalized to [0,1]).
    """
    print(f'\n--- Running LASSO Analysis (CV, AIC, BIC) ({title_prefix}) ---')
    n_samples, n_features = X_scaled.shape
    base_results = { 'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
                     'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan,
                     'full_basis_names': feature_names }
    results = { 'LassoCV': {**base_results, 'method': 'LassoCV'},
                'LassoAIC': {**base_results, 'method': 'LassoAIC'},
                'LassoBIC': {**base_results, 'method': 'LassoBIC'} }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # --- 1. LassoCV on a controlled alpha range (non-negative) ---
    print("  Running LassoCV to determine optimal alpha...")
    start_time_cv = datetime.datetime.now()
    best_alpha_from_cv = np.nan

    alpha_min_log = -6
    alpha_max_log = 0
    alphas_cv_grid = np.logspace(alpha_min_log, alpha_max_log, 1000)  # dense for CV

    # For Row-0 CV curve
    alphas_cv_plot = np.array([])
    cv_nmse_mean_plot = np.array([])
    cv_nmse_std_plot  = np.array([])

    try:
        lasso_cv = fit_lasso_cv(
            X_scaled,
            y,
            suppress_warnings=False,
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

        # ---- Extract CV MSE and robust-normalize NMSE to [0,1] for plotting ----
        try:
            alphas_cv_plot, cv_nmse_mean_plot, cv_nmse_std_plot = lasso_cv_plot_data(lasso_cv, y, eps=context.EPS)

        except Exception as e:
            print(f'WARN: Unable to extract/normalize LassoCV CV curve: {e}')

    except Exception as e:
        print(f'ERROR during LassoCV: {e}')
        results['LassoCV'].update({'error': str(e)})

    # --- 2. Unified Lasso Path (AIC/BIC on same domain; non-negative) ---
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
            X_scaled, y, alphas=alphas_for_path, max_iter=5000, precompute='auto', positive=True  # <<< enforce β >= 0
        )
        # Numerical guard: clip tiny negatives that can appear due to tolerances
        coefs_lasso = np.maximum(coefs_lasso, 0.0)

        path_results_list = []
        for i in range(coefs_lasso.shape[1]):
            alpha_i = alphas_lasso[i]
            beta_i  = np.maximum(coefs_lasso[:, i], 0.0)  # ensure non-negative
            n_params = int(np.sum(beta_i > context.EPS))
            if n_params == 0 and i < coefs_lasso.shape[1] - 1:
                continue

            y_pred_train = X_scaled @ beta_i
            mse = mean_squared_error(y, y_pred_train)
            aic, bic = calculate_aic_bic(y, y_pred_train, n_params, n_samples, eps=context.EPS)
            path_results_list.append({
                'alpha': alpha_i, 'k': n_params, 'coeffs': beta_i,
                'mse_train': mse, 'aic': aic, 'bic': bic
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

    # --- 2b. Normalized metrics on path (unchanged semantics) ---
    y_var = np.var(y)
    if y_var > context.EPS:
        df_path['norm_mse'] = df_path['mse_train'] / y_var
    else:
        df_path['norm_mse'] = df_path['mse_train']

    print(f"  DEBUG: NMSE range: {df_path['norm_mse'].min():.3f} to {df_path['norm_mse'].max():.3f}")
    print(f'  DEBUG: y variance: {y_var:.3f}')

    df_path['l1_norm'] = df_path['coeffs'].apply(lambda x: np.sum(np.abs(x)))
    max_l1_norm = df_path['l1_norm'].max()
    df_path['norm_l1'] = (df_path['l1_norm'] / max_l1_norm) if max_l1_norm > context.EPS else 0.0

    # --- 3. Pick models from unified path (unchanged; positivity respected in coeffs) ---
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    if not df_path.empty and pd.notna(best_alpha_from_cv):
        cv_model_row = df_path.iloc[(df_path['alpha'] - best_alpha_from_cv).abs().argsort()[:1]].iloc[0]
        results['LassoCV'].update({
            'coeffs': cv_model_row['coeffs'], 'alpha': cv_model_row['alpha'], 'n_nonzero': cv_model_row['k'],
            'aic': cv_model_row['aic'], 'bic': cv_model_row['bic'],
            'selected_features': [feature_names[i] for i, c in enumerate(cv_model_row['coeffs']) if c > context.EPS],
            'time': duration_cv
        })
        print(f"  LassoCV model (k={cv_model_row['k']}) identified on unified path in {duration_cv:.2f}s.")

    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()
    if not df_path.empty:
        aic_model_row = df_path.loc[df_path['aic'].idxmin()]
        results['LassoAIC'].update({
            'coeffs': aic_model_row['coeffs'], 'alpha': aic_model_row['alpha'], 'n_nonzero': aic_model_row['k'],
            'aic': aic_model_row['aic'], 'bic': aic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(aic_model_row['coeffs']) if c > context.EPS],
            'time': duration_path
        })
        print(f"  LassoAIC model (k={aic_model_row['k']}) identified on unified path.")

        bic_model_row = df_path.loc[df_path['bic'].idxmin()]
        results['LassoBIC'].update({
            'coeffs': bic_model_row['coeffs'], 'alpha': bic_model_row['alpha'], 'n_nonzero': bic_model_row['k'],
            'aic': bic_model_row['aic'], 'bic': bic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(bic_model_row['coeffs']) if c > context.EPS],
            'time': duration_path
        })
        print(f"  LassoBIC model (k={bic_model_row['k']}) identified on unified path.")

    # --- 4. Plots (unchanged layout; Row 0 uses robustly normalized CV NMSE) ---
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
    LARS with a SINGLE shared path (supports) used for CV, AIC, and BIC.
    Enforces non-negative coefficients (physical consistency) both on the path and on refits.
    """
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
        print("ERROR: No features."); return results

    # ---------- helpers: NNLS with intercept via centering ----------


    _nnls_refit_with_intercept = partial(refit_nnls_with_intercept, variant="lars", eps=context.EPS)

    # ---------- 1) LARS path on centered y, with non-negative coefficients ----------
    y_centered = y - np.mean(y)
    # positive=True enforces non-negative coefficients along the LARS path
    _, _, coefs_raw = lars_path(X_scaled, y_centered, method='lars', positive=True)
    if coefs_raw.shape[1] < 2:
        print("ERROR: LARS path too short."); return results
    coefs_all = coefs_raw[:, 1:]  # remove all-zero start
    T = coefs_all.shape[1]

    # Unique-k compression (keep first occurrence of each support size)
    supports, kept_idx = unique_size_supports(coefs_all, eps=context.EPS)
    coefs_path = np.maximum(coefs_all[:, kept_idx], 0.0)  # (p, n_steps), clipped to >=0
    steps = np.arange(1, len(supports) + 1)               # 1..K

    # ---------- 2) CV on the SAME supports (NNLS refit with intercept) ----------
    print("  Cross-validating on unified LARS supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_mse = np.zeros(len(supports))
    cv_mse_sq = np.zeros(len(supports))

    for tr, va in kf.split(X_scaled):
        Xtr, Xva = X_scaled[tr], X_scaled[va]
        ytr, yva = y[tr], y[va]
        ytr_mean = float(np.mean(ytr))  # used for S=∅ case

        for i, S in enumerate(supports):
            if len(S) == 0:
                # intercept-only model: μ = mean(y_tr)
                yhat = np.full_like(yva, fill_value=ytr_mean)
                mse = float(np.mean((yva - yhat) ** 2))
            else:
                beta_i, mu_i = _nnls_refit_with_intercept(Xtr[:, S], ytr)
                yhat = mu_i + Xva[:, S] @ beta_i
                mse = float(np.mean((yva - yhat) ** 2))
            cv_mse[i] += mse
            cv_mse_sq[i] += mse ** 2

    cv_mse_mean = cv_mse / cv_folds
    cv_mse_std  = np.sqrt(np.maximum(cv_mse_sq / cv_folds - cv_mse_mean ** 2, 0.0))

    # Normalize to [0,1] for plotting (min–max on mean; std scaled by same range)
    mm, MM = float(np.min(cv_mse_mean)), float(np.max(cv_mse_mean))
    rng = max(MM - mm, context.EPS)
    cv_nmse_mean = (cv_mse_mean - mm) / rng
    cv_nmse_std  = cv_mse_std / rng
    best_idx = int(np.argmin(cv_mse_mean))
    best_step = int(steps[best_idx])

    # ---------- 3) AIC/BIC on FULL data (NNLS refit with intercept) ----------
    rows = []
    for i, S in enumerate(supports):
        if len(S) == 0:
            mu_full = float(np.mean(y))
            y_pred = np.full_like(y, fill_value=mu_full)
            k_eff = 1  # intercept only
            coeffs_full = np.zeros(n_features)
        else:
            beta_i, mu_full = _nnls_refit_with_intercept(X_scaled[:, S], y)
            y_pred = mu_full + X_scaled[:, S] @ beta_i
            k_eff = len(S) + 1  # β (nonneg) + intercept
            coeffs_full = np.zeros(n_features); coeffs_full[S] = beta_i
        aic, bic = calculate_aic_bic(y, y_pred, n_params=k_eff, n_samples=n_samples, eps=context.EPS)
        rows.append({
            'step': int(steps[i]),
            'k': len(S),
            'coeffs': coeffs_full,
            'aic': aic,
            'bic': bic
        })
    df_path = pd.DataFrame(rows)

    # ---------- 4) Best rows & results ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    cv_row  = df_path.iloc[best_idx]
    aic_row = df_path.iloc[df_path['aic'].idxmin()]
    bic_row = df_path.iloc[df_path['bic'].idxmin()]

    results['LarsCV'].update({
        'coeffs': cv_row['coeffs'],
        'optimal_k': cv_row['step'],
        'n_nonzero': int(np.sum(cv_row['coeffs'] > context.EPS)),
        'aic': cv_row['aic'],
        'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })
    results['LarsAIC'].update({
    'coeffs': aic_row['coeffs'], 'optimal_k': aic_row['step'],
    'n_nonzero': int(np.sum(aic_row['coeffs'] > context.EPS)),
    'aic': aic_row['aic'], 'bic': aic_row['bic'],
    'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > context.EPS],
    'time': duration_cv
    })
    results['LarsBIC'].update({
        'coeffs': bic_row['coeffs'], 'optimal_k': bic_row['step'],
        'n_nonzero': int(np.sum(bic_row['coeffs'] > context.EPS)),
        'aic': bic_row['aic'], 'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })

    # ---------- 5) Plots (unchanged; now path is non-negative and refits are NNLS) ----------
    plot_lars_selection(
        aic_row=aic_row,
        bic_row=bic_row,
        coefs_path=coefs_path,
        context=context,
        cv_nmse_mean=cv_nmse_mean,
        cv_nmse_std=cv_nmse_std,
        cv_row=cv_row,
        df_path=df_path,
        feature_names=feature_names,
        save_dir=save_dir,
        steps=steps,
        title_prefix=title_prefix,
    )

    return results

def run_omp_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None, *, context):
    """
    OMP with a SINGLE shared greedy path used for CV, AIC, and BIC.
    Enforces non-negative coefficients throughout:
      - Greedy selection uses only positively-correlated atoms.
      - Coefficients at each step are refit via NNLS (β >= 0).
      - CV and AIC/BIC refits use NNLS-with-intercept via centering.
    """

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

    # ---------- helpers: NNLS (fallback PGD) and NNLS-with-intercept ----------


    _nnls = partial(solve_nnls, eps=context.EPS)

    _nnls_refit_with_intercept = partial(refit_nnls_with_intercept, variant="omp", eps=context.EPS)

    # ---------- 1) Greedy OMP path on centered y (NNLS refit, positive-only) ----------
    X = X_scaled
    coefs_path = positive_omp_path(X, y, variant="treloar", eps=context.EPS)
    if coefs_path is None:
        return results
    steps = np.arange(1, coefs_path.shape[1] + 1)
    supports = [np.flatnonzero(coefs_path[:, i] > context.EPS) for i in range(coefs_path.shape[1])]

    # ---------- 2) CV on SAME supports (NNLS-with-intercept refit) ----------
    print("  Cross-validating on unified OMP supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_mse = np.zeros(len(steps))
    cv_mse_sq = np.zeros(len(steps))

    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        ytr_mean = float(np.mean(ytr))
        for i, S in enumerate(supports):
            if len(S) == 0:
                yhat = np.full_like(yva, fill_value=ytr_mean)
                mse = float(np.mean((yva - yhat) ** 2))
            else:
                beta_i, mu_i = _nnls_refit_with_intercept(Xtr[:, S], ytr)
                yhat = mu_i + Xva[:, S] @ beta_i
                mse = float(np.mean((yva - yhat) ** 2))
            cv_mse[i] += mse
            cv_mse_sq[i] += mse**2

    cv_mse_mean = cv_mse / cv_folds
    cv_mse_std  = np.sqrt(np.maximum(cv_mse_sq / cv_folds - cv_mse_mean**2, 0.0))

    # Normalize to [0,1] for plotting (min–max on mean; std scaled by same range)
    mm, MM = float(np.min(cv_mse_mean)), float(np.max(cv_mse_mean))
    rng = max(MM - mm, context.EPS)
    cv_nmse_mean = (cv_mse_mean - mm) / rng
    cv_nmse_std  = cv_mse_std / rng

    best_idx = int(np.argmin(cv_mse_mean))
    best_step = int(steps[best_idx])

    # ---------- 3) AIC/BIC on FULL data (NNLS-with-intercept refit) ----------
    rows = []
    for i, S in enumerate(supports):
        if len(S) == 0:
            mu_full = float(np.mean(y))
            y_pred = np.full_like(y, fill_value=mu_full)
            k_eff = 1
            coeffs_full = np.zeros(n_features)
        else:
            beta_i, mu_full = _nnls_refit_with_intercept(X[:, S], y)
            y_pred = mu_full + X[:, S] @ beta_i
            k_eff = len(S) + 1
            coeffs_full = np.zeros(n_features); coeffs_full[S] = beta_i
        aic, bic = calculate_aic_bic(y, y_pred, n_params=k_eff, n_samples=n_samples, eps=context.EPS)
        rows.append({'step': int(steps[i]), 'k': len(S), 'coeffs': coeffs_full, 'aic': aic, 'bic': bic})
    df_path = pd.DataFrame(rows)

    # ---------- 4) Results ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    cv_row  = df_path.iloc[best_idx]
    aic_row = df_path.iloc[df_path['aic'].idxmin()]
    bic_row = df_path.iloc[df_path['bic'].idxmin()]

    results['OMPCV'].update({
        'coeffs': cv_row['coeffs'], 'optimal_k': cv_row['step'],
        'n_nonzero': int(np.sum(cv_row['coeffs'] > context.EPS)),
        'aic': cv_row['aic'], 'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })
    results['OMPAIC'].update({
    'coeffs': aic_row['coeffs'], 'optimal_k': aic_row['step'],
    'n_nonzero': int(np.sum(aic_row['coeffs'] > context.EPS)),
    'aic': aic_row['aic'], 'bic': aic_row['bic'],
    'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > context.EPS],
    'time': duration_cv
    })
    results['OMPBIC'].update({
        'coeffs': bic_row['coeffs'], 'optimal_k': bic_row['step'],
        'n_nonzero': int(np.sum(bic_row['coeffs'] > context.EPS)),
        'aic': bic_row['aic'], 'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > context.EPS],
        'time': duration_cv
    })

    # ---------- 5) Plots (unchanged layout; positive path) ----------
    plot_omp_selection(
        aic_row=aic_row,
        bic_row=bic_row,
        coefs_path=coefs_path,
        context=context,
        cv_nmse_mean=cv_nmse_mean,
        cv_nmse_std=cv_nmse_std,
        cv_row=cv_row,
        df_path=df_path,
        feature_names=feature_names,
        save_dir=save_dir,
        steps=steps,
        title_prefix=title_prefix,
    )

    return results
