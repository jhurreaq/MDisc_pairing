"""Synthetic plotting; original numerical and presentation choices."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
import matplotlib.ticker
from mdisc.regression import fit_ridge
from sklearn.metrics import mean_squared_error, r2_score
import os
import traceback
from .models import construct_design_matrix
from .models import calculate_dW_dl_vectors
from mdisc.isotropic import format_term_for_latex
from .models import check_physical_consistency

def style_y_axis(ax, kpa_conversion, *, context):
    y_lim = ax.get_ylim()

    ticks_to_set = sorted(list(set([y_lim[0], y_lim[1]])))
    if y_lim[0] < 0 < y_lim[1]:
        ticks_to_set.append(0)
    elif np.abs(y_lim[0]) < 0.05 * (y_lim[1] - y_lim[0]) and y_lim[0] > -context.EPS and y_lim[0] * y_lim[1] >=0 : # Min is positive and close to 0
        ticks_to_set.append(0)
    elif np.abs(y_lim[1]) < 0.05 * (y_lim[1] - y_lim[0]) and y_lim[1] < context.EPS and y_lim[0] * y_lim[1] >=0 : # Max is negative and close to 0
         ticks_to_set.append(0)

    ticks_to_set = sorted(list(set(ticks_to_set)))

    if len(ticks_to_set) > 1:
        processed_ticks = []
        # Heuristic to avoid overly crowded ticks if min/0/max are too close after rounding
        # This simplified logic might need adjustment based on typical data ranges
        if len(ticks_to_set) == 3: # min, 0, max
            if abs(ticks_to_set[0] - ticks_to_set[1]) < 0.05 * abs(y_lim[1]-y_lim[0]) and abs(ticks_to_set[0]) < 0.1: # min and 0 are too close
                processed_ticks = [ticks_to_set[0], ticks_to_set[2]] # Show min and max
            elif abs(ticks_to_set[1] - ticks_to_set[2]) < 0.05 * abs(y_lim[1]-y_lim[0]) and abs(ticks_to_set[2]) < 0.1: # 0 and max are too close
                processed_ticks = [ticks_to_set[0], ticks_to_set[2]] # Show min and max
            else:
                processed_ticks = ticks_to_set
        else: # Should be 2 ticks (min, max) or 1 if min=max
             processed_ticks = ticks_to_set
        ax.set_yticks(processed_ticks)
    else:
        ax.set_yticks(ticks_to_set)

    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)

def plot_synthetic_uniaxial(df, ax, r2_p11_precomputed=np.nan, rmse_p11_precomputed_kpa=np.nan, *, context):
    """Plots synthetic uniaxial data (P11 vs lambda1) and model predictions using TRUNCATED_PLASMA_CMAP and kPa units."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2))
    else:
        fig = ax.get_figure()

    kpa_conversion = 1e3

    df_plot = df[df['mode'] == 'UT'].copy()

    for col in ['P11', 'P11_true']:
        if col in df_plot.columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce') * kpa_conversion

    if 'P11_pred' in df_plot.columns:
         df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')
    else:
         df_plot['P11_pred'] = np.nan

    df_plot['lambda1'] = pd.to_numeric(df_plot['lambda1'], errors='coerce')

    color_synth_data = context.TRUNCATED_PLASMA_CMAP(0.4)
    color_true_curve = context.TRUNCATED_PLASMA_CMAP(0.2)
    color_model_fit  = context.TRUNCATED_PLASMA_CMAP(0.4)

    ax.scatter(df_plot['lambda1'], df_plot['P11'], color=color_synth_data, marker='o',
               s=context.PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3,
               edgecolor='black', linewidth=0.5)

    valid_pred_mask = df_plot['P11_pred'].notna()
    if valid_pred_mask.any():
        df_sorted_pred = df_plot[valid_pred_mask].dropna(subset=['lambda1', 'P11_pred']).sort_values(by='lambda1').drop_duplicates(subset=['lambda1'])
        ax.plot(df_sorted_pred['lambda1'], df_sorted_pred['P11_pred'], color=color_model_fit,
                linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label='Disc. Model')

    if 'P11_true' in df_plot.columns and df_plot['P11_true'].notna().all():
        df_sorted_true = df_plot.dropna(subset=['lambda1', 'P11_true']).sort_values(by='lambda1').drop_duplicates(subset=['lambda1'])
        ax.plot(df_sorted_true['lambda1'], df_sorted_true['P11_true'], color=color_true_curve,
                linestyle=':', linewidth=context.PLOT_PRED_LINEWIDTH*1.2, label='True Model')

    ax.set_xlabel('Stretch $\lambda$ [-]', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis='x', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)

    first_line_title = "Uniaxial $P_{11}$"
    second_line_title_parts = []
    if pd.notna(r2_p11_precomputed):
        second_line_title_parts.append(f'$R^2 \\approx {r2_p11_precomputed:.3f}$')
    if pd.notna(rmse_p11_precomputed_kpa):
        second_line_title_parts.append(f'RMSE $\\approx {rmse_p11_precomputed_kpa:.3f}$ kPa')

    if second_line_title_parts:
        title_str = first_line_title + "\n" + ", ".join(second_line_title_parts)
    else:
        title_str = first_line_title
    # ax.set_title(title_str, fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_title(first_line_title, fontsize=context.PLOT_TITLE_FONTSIZE)

    ax.legend(fontsize=context.PLOT_LEGEND_FONTSIZE, frameon=False) # Legend box turned off

    print("  Adjusting plot Y-limits (kPa)...")
    y_min_list = []
    y_max_list = []
    min_valid_y, max_valid_y = np.nan, np.nan

    if 'P11' in df_plot and df_plot['P11'].notna().any():
        y_min_list.append(df_plot['P11'].min())
        y_max_list.append(df_plot['P11'].max())
    if 'P11_true' in df_plot and df_plot['P11_true'].notna().any():
        y_min_list.append(df_plot['P11_true'].min())
        y_max_list.append(df_plot['P11_true'].max())
    if 'P11_pred' in df_plot and df_plot['P11_pred'].notna().any():
        y_min_list.append(df_plot['P11_pred'].min())
        y_max_list.append(df_plot['P11_pred'].max())

    if y_min_list:
        min_valid_y = np.nanmin(y_min_list)
        max_valid_y = np.nanmax(y_max_list)

    if pd.notna(min_valid_y) and pd.notna(max_valid_y):
        y_range_abs = max_valid_y - min_valid_y
        padding = y_range_abs * 0.075 if y_range_abs > context.EPS else abs(min_valid_y) * 0.1 + (0.01 * kpa_conversion)
        final_y_min = min_valid_y - padding
        final_y_max = max_valid_y + padding
        ax.set_ylim(final_y_min, final_y_max)
        print(f'  Set Y-limits to: ({final_y_min:.1f} kPa, {final_y_max:.1f} kPa)')
        style_y_axis(ax, kpa_conversion, context=context)
    else:
        print("  WARN: Could not determine valid Y range. Using default autoscaling.")
        ax.tick_params(axis='y', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)

    if df_plot['lambda1'].notna().any():
        min_l = df_plot['lambda1'].min()
        max_l = df_plot['lambda1'].max()
        if pd.notna(min_l) and pd.notna(max_l):
            x_padding = (max_l - min_l) * 0.05 if (max_l - min_l) > context.EPS else 0.05
            ax.set_xlim(min_l - x_padding, max_l + x_padding)

    return ax

def plot_model_predictions(
    df_orig,
    sparse_coeffs,
    basis_names,
    method_key,
    scenario_prefix,
    save_dir,
    coeff_threshold=None,
    ridge_alpha=None
, *, context):
    coeff_threshold = coeff_threshold if coeff_threshold is not None else context.COEFF_THRESHOLD
    ridge_alpha = ridge_alpha if ridge_alpha is not None else context.RIDGE_ALPHA_REFIT

    print(f'\n--- Generating Final Model & Predictions ({method_key} / Ridge Refit α={ridge_alpha:.1e} + Check) ---')
    print(f'--- NOTE: Fitting on UT(P11)+PS(P11)+EBT(P11); Evaluating UT(P11), PS(P11), EBT(P11) ---')

    kpa_conversion = 1e3

    current_final_terms, current_final_coeffs_kPa = [], np.array([])
    r2_P11_ut, rmse_P11_ut_kPa, nrmse_P11_ut = np.nan, np.nan, np.nan
    r2_P11_ps, rmse_P11_ps_kPa, nrmse_P11_ps = np.nan, np.nan, np.nan
    r2_P11_ebt, rmse_P11_ebt_kPa, nrmse_P11_ebt = np.nan, np.nan, np.nan
    is_consistent, consistency_notes = False, ["Processing error before completion"]


    selected_indices = np.where(np.abs(sparse_coeffs) > context.EPS)[0]
    if len(selected_indices) == 0:
        print(f'WARN: No features initially selected by {method_key}. Skipping.')
        consistency_notes = ["No features selected"]
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes
    initial_selected_basis_names = [basis_names[i] for i in selected_indices]
    print(f'  Initial selection ({method_key}): {len(initial_selected_basis_names)} features.')

    # Reconstructing WEIGHTED design matrix (UT + PS + EBT data) for refit.
    print("  Reconstructing WEIGHTED design matrix (UT + PS + EBT data) for refit...")
    df_for_refit = df_orig[df_orig['mode'].isin(['UT', 'PS', 'EBT'])].copy()
    if df_for_refit.empty:
        print("ERROR: No Uniaxial (UT), Pure Shear (PS), or Equi-biaxial (EBT) data found in df_orig for refit.")
        consistency_notes = ["No UT, PS, or EBT data for refit"]
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes

    phi_weighted, target_weighted, basis_names_weighted, _, df_subset_weighted = None, None, None, None, None
    try:
        phi_weighted, target_weighted, basis_names_weighted, _, df_subset_weighted = construct_design_matrix(
            df_for_refit,
            context=context,
        )
        if not isinstance(df_subset_weighted, pd.DataFrame): raise TypeError("df_subset_weighted type error")
        if phi_weighted.shape[0] == 0: raise ValueError("Weighted design matrix empty.")
        if basis_names != basis_names_weighted:
             print(f'WARN: Basis name mismatch during weighted recon. Using list from construct_design_matrix ({len(basis_names_weighted)} terms).')
             basis_names = basis_names_weighted
             try:
                 selected_indices = [basis_names.index(term) for term in initial_selected_basis_names if term in basis_names]
                 initial_selected_basis_names = [basis_names[i] for i in selected_indices]
                 print(f'  Re-mapped initial selection: {len(initial_selected_basis_names)} features.')
                 if len(selected_indices)==0: raise ValueError("Remapping indices failed, no initial terms found.")
             except ValueError as ve:
                  print(f'ERROR: Failed to remap selected indices after basis mismatch: {ve}')
                  consistency_notes = ["Basis mismatch & remapping failed"]
                  return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes
        phi_selected_weighted = phi_weighted[:, selected_indices]
    except Exception as e:
        print(f'ERROR: Failed to reconstruct WEIGHTED matrix/target for refit: {e}'); traceback.print_exc()
        consistency_notes = [f'Matrix Recon Error: {e}']
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes

    print(f'  Refitting {len(initial_selected_basis_names)} features on WEIGHTED data (shape: {phi_selected_weighted.shape}) using Ridge...')
    refit_coeffs_weighted_MPa = None
    try:
        ridge_refit = fit_ridge(phi_selected_weighted, target_weighted, alpha=ridge_alpha, fit_intercept=False, solver='auto')
        refit_coeffs_weighted_MPa = ridge_refit.coef_
    except Exception as e:
        print(f'ERROR: Failed during Ridge Regression refit for {method_key}: {e}'); traceback.print_exc()
        consistency_notes = [f'Ridge Refit Error: {e}']
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes

    print(f'  Applying threshold ({coeff_threshold:.1e}) to Ridge coefficients...')
    significant_mask = np.abs(refit_coeffs_weighted_MPa) > coeff_threshold
    final_coeffs_MPa = refit_coeffs_weighted_MPa[significant_mask]
    final_global_indices = selected_indices[significant_mask]
    final_selected_basis_names = [basis_names[i] for i in final_global_indices]
    n_final = len(final_selected_basis_names)
    print(f'  Final selected features after thresholding: {n_final}')
    current_final_terms = final_selected_basis_names
    current_final_coeffs_MPa = final_coeffs_MPa

    if n_final == 0:
        print(f'WARN: All features removed by thresholding for {method_key}.')
        is_consistent, shear_modulus_MPa, consistency_notes = check_physical_consistency([], np.array([]), context=context)
    else:
        current_final_coeffs_display_kPa = current_final_coeffs_MPa * kpa_conversion
        print(f'  Final Model ({method_key}):'); [print(f"     '{name}': {coeff_kpa:.4f} kPa") for name, coeff_kpa in zip(current_final_terms, current_final_coeffs_display_kPa)]
        print("  Checking physical consistency (using MPa coeffs)...");
        is_consistent, shear_modulus_MPa, consistency_notes = check_physical_consistency(current_final_terms, current_final_coeffs_MPa, context=context)
        print("  Consistency Check Results:"); [print(f'     {note}') for note in consistency_notes]
        if not is_consistent: print(f'  WARNING: {method_key} model potentially inconsistent (μ₀={shear_modulus_MPa:.4f} MPa).')
        else: print(f'  INFO: {method_key} model passed basic consistency check (μ₀={shear_modulus_MPa:.4f} MPa).')

    print("  Calculating P11 & P22 predictions for all points using final model...")
    df_plot = df_orig.copy()
    df_plot['P11_pred_MPa'] = np.nan
    df_plot['P22_pred_MPa'] = np.nan
    if n_final > 0:
        final_term_indices_in_full_library = [basis_names.index(term) for term in current_final_terms]
        predictions_calculated = 0
        for idx, row in df_plot.iterrows():
            l1 = float(row['lambda1']) if pd.notna(row['lambda1']) else 1.0
            l2 = float(row['lambda2']) if pd.notna(row['lambda2']) else 1.0
            l3 = float(row['lambda3']) if pd.notna(row['lambda3']) else 1.0
            if not pd.notna(l3) or l3 < context.EPS:
                l1_safe = max(l1, context.EPS); l2_safe = max(l2, context.EPS)
                l3 = max(1.0 / (l1_safe * l2_safe), context.EPS)
            deriv_vecs = calculate_dW_dl_vectors(l1, l2, l3, basis_names, context=context)
            if deriv_vecs is None: continue
            dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs
            if np.any(np.isnan(dW_dl1_all[final_term_indices_in_full_library])) or \
               np.any(np.isnan(dW_dl2_all[final_term_indices_in_full_library])) or \
               np.any(np.isnan(dW_dl3_all[final_term_indices_in_full_library])):
                continue
            dW_pred_dl1_MPa = np.sum(current_final_coeffs_MPa * dW_dl1_all[final_term_indices_in_full_library])
            dW_pred_dl2_MPa = np.sum(current_final_coeffs_MPa * dW_dl2_all[final_term_indices_in_full_library])
            dW_pred_dl3_MPa = np.sum(current_final_coeffs_MPa * dW_dl3_all[final_term_indices_in_full_library])
            p11_pred_MPa = dW_pred_dl1_MPa - (l3 / l1) * dW_pred_dl3_MPa if l1 > context.EPS else 0.0
            p22_pred_MPa = dW_pred_dl2_MPa - (l3 / l2) * dW_pred_dl3_MPa if l2 > context.EPS else 0.0
            if np.isfinite(p11_pred_MPa): df_plot.loc[idx, 'P11_pred_MPa'] = p11_pred_MPa
            if np.isfinite(p22_pred_MPa): df_plot.loc[idx, 'P22_pred_MPa'] = p22_pred_MPa
            predictions_calculated += 1
        print(f'  Calculated P11/P22 predictions for {predictions_calculated} points.')
    else:
        print("  Skipping P11/P22 prediction calculation: n_final is 0.")

    if 'P11_pred_MPa' in df_plot.columns:
        df_plot['P11_pred'] = df_plot['P11_pred_MPa'] * kpa_conversion
    else:
        df_plot['P11_pred'] = np.nan
    if 'P22_pred_MPa' in df_plot.columns:
        df_plot['P22_pred'] = df_plot['P22_pred_MPa'] * kpa_conversion
    else:
         df_plot['P22_pred'] = np.nan

    print("  Calculating metrics (R², RMSE [kPa], NRMSE) for summary...")

    def calculate_component_metrics(mode_filter, y_exp_col_MPa, y_pred_col_MPa, y_true_col_MPa):
        df_mode = df_plot[df_plot['mode'].isin(mode_filter)] if mode_filter else df_plot
        if df_mode.empty: return np.nan, np.nan, np.nan
        y_exp_MPa_vals = df_mode[y_exp_col_MPa]
        y_pred_MPa_vals = df_mode[y_pred_col_MPa]
        y_true_MPa_vals = df_mode[y_true_col_MPa] if y_true_col_MPa in df_mode.columns else None
        valid_mask = pd.notna(y_exp_MPa_vals) & pd.notna(y_pred_MPa_vals)
        if valid_mask.sum() == 0: return np.nan, np.nan, np.nan

        # Use CLEAN truth if available; fall back to noisy measurements
        if y_true_MPa_vals is not None and pd.notna(y_true_MPa_vals[valid_mask]).sum() > 1:
            y_cmp = y_true_MPa_vals[valid_mask]
        else:
            y_cmp = y_exp_MPa_vals[valid_mask]

        r2 = r2_score(y_cmp, y_pred_MPa_vals[valid_mask])
        rmse_MPa_val = np.sqrt(mean_squared_error(y_cmp, y_pred_MPa_vals[valid_mask]))
        rmse_kPa_val = rmse_MPa_val * kpa_conversion
        nrmse = np.nan
        if y_true_MPa_vals is not None:
            y_true_valid_MPa = y_true_MPa_vals[valid_mask]
            if y_true_valid_MPa.notna().sum() > 1:
                y_true_kPa = y_true_valid_MPa * kpa_conversion
                true_range_kPa = np.ptp(y_true_kPa[y_true_kPa.notna()])
                if true_range_kPa > (context.EPS * kpa_conversion): nrmse = rmse_kPa_val / true_range_kPa
                elif rmse_kPa_val < (context.EPS * kpa_conversion): nrmse = 0.0
                else: nrmse = np.inf
        return r2, rmse_kPa_val, nrmse

    r2_P11_ut, rmse_P11_ut_kPa, nrmse_P11_ut = calculate_component_metrics(['UT'], 'P11', 'P11_pred_MPa', 'P11_true')
    r2_P11_ps, rmse_P11_ps_kPa, nrmse_P11_ps = calculate_component_metrics(['PS'], 'P11', 'P11_pred_MPa', 'P11_true')
    r2_P11_ebt, rmse_P11_ebt_kPa, nrmse_P11_ebt = calculate_component_metrics(['EBT'], 'P11', 'P11_pred_MPa', 'P11_true')

    print(f'  Metrics: P11(UT): R²={r2_P11_ut:.4f}, RMSE={rmse_P11_ut_kPa:.3f} kPa, NRMSE={nrmse_P11_ut:.4f}')
    print(f'           P11(PS): R²={r2_P11_ps:.4f}, RMSE={rmse_P11_ps_kPa:.3f} kPa, NRMSE={nrmse_P11_ps:.4f}')
    print(f'           P11(EBT): R²={r2_P11_ebt:.4f}, RMSE={rmse_P11_ebt_kPa:.3f} kPa, NRMSE={nrmse_P11_ebt:.4f}')

    print("  Preparing plots (UT, PS, EBT predictions) using kPa units...")
    fig = None; plot_created = False

    try:
        consistency_status_str = "(Phys OK)" if is_consistent else "(Phys WARN!)"
        title_info = f'{scenario_prefix} ({method_key})\n{context.REFIT_REGRESSION_TYPE} Th={coeff_threshold:.1e} {consistency_status_str}'
        df_ut_plot = df_plot[df_plot['mode'] == 'UT'].copy()
        df_ps_plot = df_plot[df_plot['mode'] == 'PS'].copy()
        df_ebt_plot = df_plot[df_plot['mode'] == 'EBT'].copy()

        if not df_ut_plot.empty or not df_ps_plot.empty or not df_ebt_plot.empty:
            n_plots = (1 if not df_ut_plot.empty else 0) + \
                      (1 if not df_ps_plot.empty else 0) + \
                      (1 if not df_ebt_plot.empty else 0)

            current_figsize = (6, 3) if n_plots == 1 else (10, 3) if n_plots == 2 else (12, 3)
            fig, axes = plt.subplots(1, n_plots, figsize=current_figsize, squeeze=False)
            axes = axes.flatten()
            ax_idx = 0

            color_true_curve = context.TRUNCATED_PLASMA_CMAP(0.3)
            color_data_ps = context.TRUNCATED_PLASMA_CMAP(0.8)
            color_pred_ps = context.TRUNCATED_PLASMA_CMAP(0.7)
            color_data_ebt = context.TRUNCATED_PLASMA_CMAP(0.0)
            color_pred_ebt = context.TRUNCATED_PLASMA_CMAP(0.1)

            if not df_ut_plot.empty:
                plot_synthetic_uniaxial(df_ut_plot, ax=axes[ax_idx],
                                        r2_p11_precomputed=r2_P11_ut,
                                        rmse_p11_precomputed_kpa=rmse_P11_ut_kPa, context=context)
                ax_idx += 1

            if not df_ps_plot.empty:
                ax_ps11 = axes[ax_idx]
                df_ps_plot_kPa = df_ps_plot.copy()
                for col in ['P11', 'P11_true']:
                    if col in df_ps_plot_kPa.columns:
                         df_ps_plot_kPa[col] = pd.to_numeric(df_ps_plot_kPa[col], errors='coerce') * kpa_conversion

                ax_ps11.scatter(df_ps_plot_kPa['lambda1'], df_ps_plot_kPa['P11'], color=color_data_ps, marker='o', s=context.PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3, edgecolor='black', linewidth=0.5)
                valid_ps_p11_pred = df_ps_plot_kPa['P11_pred'].notna()
                if valid_ps_p11_pred.any():
                    ax_ps11.plot(df_ps_plot_kPa.loc[valid_ps_p11_pred, 'lambda1'].sort_values(), df_ps_plot_kPa.loc[valid_ps_p11_pred].sort_values('lambda1')['P11_pred'], color=color_pred_ps, linestyle='-', label='Disc. Model')
                if 'P11_true' in df_ps_plot_kPa.columns and df_ps_plot_kPa['P11_true'].notna().all():
                    ax_ps11.plot(df_ps_plot_kPa['lambda1'].sort_values(), df_ps_plot_kPa.sort_values('lambda1')['P11_true'], color=color_true_curve, linestyle=':', label='True Model')

                title_ps_p11_str = "Pure Shear $P_{11}$"#\n" + f"$R^2 \\approx {r2_P11_ps:.3f}$, RMSE $\\approx {rmse_P11_ps_kPa:.3f}$ kPa"
                ax_ps11.set_title(title_ps_p11_str, fontsize=context.PLOT_TITLE_FONTSIZE)
                ax_ps11.set_xlabel('Stretch $\lambda$ [-]', fontsize=context.PLOT_LABEL_FONTSIZE)
                ax_ps11.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=context.PLOT_LABEL_FONTSIZE)
                ax_ps11.tick_params(axis='both', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)
                if df_ps_plot_kPa['P11_true'].notna().any() or df_ps_plot_kPa['P11_pred'].notna().any(): style_y_axis(ax_ps11, kpa_conversion, context=context)
                ax_ps11.legend(fontsize=context.PLOT_LEGEND_FONTSIZE, frameon=False)
                ax_idx += 1

            if not df_ebt_plot.empty:
                ax_ebt11 = axes[ax_idx]
                df_ebt_plot_kPa = df_ebt_plot.copy()
                for col in ['P11', 'P11_true']:
                    if col in df_ebt_plot_kPa.columns:
                         df_ebt_plot_kPa[col] = pd.to_numeric(df_ebt_plot_kPa[col], errors='coerce') * kpa_conversion

                ax_ebt11.scatter(df_ebt_plot_kPa['lambda1'], df_ebt_plot_kPa['P11'], color=color_data_ebt, marker='o', s=context.PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3, edgecolor='black', linewidth=0.5)
                valid_ebt_p11_pred = df_ebt_plot_kPa['P11_pred'].notna()
                if valid_ebt_p11_pred.any():
                    ax_ebt11.plot(df_ebt_plot_kPa.loc[valid_ebt_p11_pred, 'lambda1'].sort_values(), df_ebt_plot_kPa.loc[valid_ebt_p11_pred].sort_values('lambda1')['P11_pred'], color=color_pred_ebt, linestyle='-', label='Disc. Model')
                if 'P11_true' in df_ebt_plot_kPa.columns and df_ebt_plot_kPa['P11_true'].notna().all():
                    ax_ebt11.plot(df_ebt_plot_kPa['lambda1'].sort_values(), df_ebt_plot_kPa.sort_values('lambda1')['P11_true'], color=color_true_curve, linestyle=':', label='True Model')

                title_ebt_p11_str = "Equi-biaxial $P_{11}$"#\n" + f"$R^2 \\approx {r2_P11_ebt:.3f}$, RMSE $\\approx {rmse_P11_ebt_kPa:.3f}$ kPa"
                ax_ebt11.set_title(title_ebt_p11_str, fontsize=context.PLOT_TITLE_FONTSIZE)
                ax_ebt11.set_xlabel('Stretch $\lambda$ [-]', fontsize=context.PLOT_LABEL_FONTSIZE)
                ax_ebt11.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=context.PLOT_LABEL_FONTSIZE)
                ax_ebt11.tick_params(axis='both', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)
                if df_ebt_plot_kPa['P11_true'].notna().any() or df_ebt_plot_kPa['P11_pred'].notna().any(): style_y_axis(ax_ebt11, kpa_conversion, context=context)
                ax_ebt11.legend(fontsize=context.PLOT_LEGEND_FONTSIZE, frameon=False)
                ax_idx += 1

            plt.tight_layout(rect=[0, 0.03, 1, 0.94])
            plot_created = True
        else:
            print(f'No UT, PS, or EBT data found in df_plot for {method_key}. Skipping prediction plot generation.')

        if save_dir and scenario_prefix and fig and plot_created:
            save_path = os.path.join(save_dir, f'{scenario_prefix}_{method_key}_Predictions_Final_Checked_kPa.pdf')
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
                print(f'Saved final prediction plot (kPa): {save_path}')
            except Exception as e:
                print(f'Error saving final prediction plot {save_path}: {e}')
        elif fig:
            print(f'WARN: Plot not saved for {method_key} (save conditions not met).')

    except Exception as plot_exc:
        print(f'ERROR during plot generation/saving section for {method_key}: {plot_exc}')
        traceback.print_exc()
    finally:
         if fig and plot_created: plt.close(fig)


    current_final_coeffs_kPa = current_final_coeffs_MPa * kpa_conversion
    print(f'--- FINISHING plot_model_predictions for {method_key} ---')
    return (current_final_terms, current_final_coeffs_kPa,
            r2_P11_ut, rmse_P11_ut_kPa, nrmse_P11_ut,
            r2_P11_ps, rmse_P11_ps_kPa, nrmse_P11_ps,
            r2_P11_ebt, rmse_P11_ebt_kPa, nrmse_P11_ebt,
            is_consistent, shear_modulus_MPa, consistency_notes)

def plot_coefficient_path(
    path_variable,
    coef_path,
    basis_names,
    chosen_values_dict,
    title,
    x_axis_label,
    filename=None,
    log_x=True,
    reverse_x=True
, *, context):
    """
    Plots the coefficient paths for Lasso/Lars, showing coefficient magnitude vs. regularization.

    Args:
        path_variable (np.ndarray): The variable for the x-axis (e.g., alphas for Lasso).
        coef_path (np.ndarray): The coefficient values along the path (n_features, n_steps).
        basis_names (list): The names of the features/basis functions.
        chosen_values_dict (dict): Dictionary marking the optimal points selected by different criteria.
        title (str): The title for the plot.
        x_axis_label (str): The label for the x-axis.
        filename (str, optional): Path to save the plot. Defaults to None.
        log_x (bool, optional): Whether to use a logarithmic scale for the x-axis. Defaults to True.
        reverse_x (bool, optional): Whether to reverse the x-axis. Defaults to True.
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # Use a color cycle for the coefficient lines
    colors = plt.cm.viridis(np.linspace(0, 1, coef_path.shape[0]))
    ax.set_prop_cycle(plt.cycler('color', colors))

    plot_x_values = path_variable
    plot_coef_path = coef_path
    if reverse_x:
        # Reverse both the x-axis values and the columns of the coefficient path
        plot_x_values = plot_x_values[::-1]
        plot_coef_path = plot_coef_path[:, ::-1]

    ax.plot(plot_x_values, plot_coef_path.T, linewidth=context.PLOT_LINEWIDTH, alpha=0.7)

    if log_x:
        ax.set_xscale('log')

    ax.set_xlabel(x_axis_label, fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Coefficient Value [MPa]', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=context.PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    # ax.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.5)

    # Add vertical lines for chosen models
    legend_elements = []
    for method, info in chosen_values_dict.items():
        chosen_value = info.get('value')
        if chosen_value is not None and not np.isnan(chosen_value):
            color = info.get('color', 'red')
            linestyle = info.get('linestyle', '--')
            ax.axvline(x=chosen_value, color=color, linestyle=linestyle, linewidth=1.5)
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=f'Selected {method}'))

    if legend_elements:
        ax.legend(handles=legend_elements, fontsize=context.PLOT_LEGEND_FONTSIZE)

    if filename:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            print(f'Saved coefficient path plot: {filename}')
        except Exception as e:
            print(f'Error saving coefficient path plot: {e}')
    plt.close(fig)

def plot_activation_heatmap(
    path_variable,
    coef_path,
    basis_names,
    chosen_values_dict,
    title,
    x_axis_label,
    filename=None,
    log_x=False,
    reverse_x=True,
    activation_thresh=1e-20,
    is_omp=False
, *, context):
    """
    Generalized function to plot feature activation heatmaps with multiple selection criteria.
    Now uses direct indices to avoid coordinate transformation errors.
    """
    n_features, n_steps = coef_path.shape

    if n_steps == 0 or path_variable is None or len(path_variable) != n_steps:
        print(f'Warning: Cannot plot {title}, path or path variable invalid.')
        return

    # --- Create Binary Feature Activation Matrix ---
    feature_status = (np.abs(coef_path) > activation_thresh).astype(float)

    # --- Determine feature entry order (step index) ---
    feature_entry = {}
    for fidx in range(n_features):
        active_indices = np.where(feature_status[fidx, :] > 0)[0]
        feature_entry[fidx] = active_indices[0] if len(active_indices) > 0 else n_steps

    # --- Sort features by entry order ---
    active_feature_indices = [fidx for fidx in range(n_features) if feature_entry[fidx] < n_steps]
    if not active_feature_indices:
         print(f'Warning: No features became active for {title}. Skipping heatmap.')
         return

    sorted_active_features = sorted(active_feature_indices, key=lambda f: feature_entry[f])
    try:
         ordered_basis_names = [format_term_for_latex(basis_names[i]) for i in sorted_active_features]
    except Exception as e:
         print(f'WARN: LaTeX formatting failed for heatmap labels: {e}. Using raw names.')
         ordered_basis_names = [basis_names[i] for i in sorted_active_features]

    sorted_status = feature_status[sorted_active_features, :]
    num_sorted_features = len(sorted_active_features)

    # --- Handle X-axis and Reversal ---
    plot_x_values = path_variable
    plot_status = sorted_status
    if reverse_x:
        plot_status = plot_status[:, ::-1]
        plot_x_values = plot_x_values[::-1]

    # --- Plotting ---
    height_per_feature = 0.4
    fig_height = max(5, num_sorted_features * height_per_feature)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated_heatmap_cmap", context.HEATMAP_CMAP(np.linspace(0, 0.75, 256))
    )

    im = ax.imshow(plot_status, cmap=truncated_cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)

    # --- Y-axis Ticks and Labels ---
    ax.set_yticks(np.arange(num_sorted_features))
    ax.set_yticklabels(ordered_basis_names, fontsize=context.PLOT_HEATMAP_YLABEL_SIZE)
    ax.set_ylabel('Model Terms', fontsize=context.PLOT_LABEL_FONTSIZE)

    # --- X-axis Ticks and Labels ---
    if log_x:
        tick_indices = np.linspace(0, n_steps - 1, min(8, n_steps), dtype=int)
        ax.set_xticks(tick_indices)
        tick_labels = []
        for i in tick_indices:
            log_val = np.log10(plot_x_values[i] + context.EPS)
            if abs(log_val - round(log_val)) < 0.3:
                tick_labels.append(f'10$^{{{int(round(log_val))}}}$')
            else:
                tick_labels.append(f'10$^{{{log_val:.1f}}}$')
        ax.set_xticklabels(tick_labels, fontsize=context.PLOT_TICK_FONTSIZE)
    else:
        tick_indices = np.linspace(0, n_steps - 1, min(10, n_steps), dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{int(plot_x_values[i])}' for i in tick_indices], fontsize=context.PLOT_TICK_FONTSIZE)
    ax.set_xlabel(x_axis_label, fontsize=context.PLOT_LABEL_FONTSIZE)

    # --- Mark the chosen values using DIRECT INDICES ---
    legend_elements = [
        Patch(facecolor=context.HEATMAP_CMAP(0.75), label='Active'),
        Patch(facecolor=context.HEATMAP_CMAP(0.0), edgecolor='darkgray', label='Inactive')
    ]

    for method, info in chosen_values_dict.items():
        chosen_index = info.get('index')  # Now using 'index' instead of 'value'
        label_val_for_display = info.get('label_val', chosen_index)
        color = info.get('color', 'red')
        linestyle = info.get('linestyle', '--')

        if chosen_index is not None and 0 <= chosen_index < n_steps:
            # Direct index mapping with reversal handling
            if reverse_x:
                chosen_x_coord = (n_steps - 1) - chosen_index
            else:
                chosen_x_coord = chosen_index

            ax.axvline(chosen_x_coord, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.9)

            # Format the label for the legend
            if log_x:
                label_val_str = f'10$^{{{np.log10(max(label_val_for_display, context.EPS)):.1f}}}$'
            else:
                label_val_str = f'{int(label_val_for_display)}' if is_omp else f'{label_val_for_display:.2f}'

            vline_label = f'{method}'
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=vline_label))
        else:
            print(f'WARN: Chosen index for {method} is invalid ({chosen_index}), cannot plot vertical line.')

    # --- Grid and Legend ---
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=context.PLOT_LEGEND_FONTSIZE)

    # --- Final Touches ---
    ax.set_title(title, fontsize=context.PLOT_TITLE_FONTSIZE)
    fig.subplots_adjust(bottom=0.25, left=0.3, right=0.95)

    if filename:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            print(f'Saved activation heatmap: {filename}')
        except Exception as e: print(f'Error saving heatmap: {e}')
    plt.close(fig)

def _annotate_iteration(ax, x_iter, color, text=None, *, context):
    """
    Place 'Iteration = X' rotated 90° beside the vertical line.
    """
    if text is None:
        try:
            x_show = int(x_iter)
        except Exception:
            x_show = x_iter
        text = f'Iteration = {x_show}'

    y0, y1 = ax.get_ylim()
    y_mid = 0.5 * (y0 + y1)

    ax.annotate(
        text,
        xy=(x_iter, y_mid), xytext=(4, 0),  # 4pt to the right of line
        textcoords='offset points',
        ha='left', va='center',
        fontsize=context.PLOT_TICK_FONTSIZE - 4,
        color=color,
        clip_on=False,
        rotation=90   # rotate vertical
    )


def plot_lasso_selection(*, alphas_cv_plot, alphas_lasso, best_alpha_from_cv, coefs_lasso, context, cv_nmse_mean_plot, cv_nmse_std_plot, df_path, feature_names, save_dir, title_prefix):
    if not df_path.empty and alphas_lasso is not None and coefs_lasso is not None:
        print("\n  Generating plots from unified path data...")

        # Get the EXACT models from the unified path for consistent marking
        cv_exact_model = None; cv_exact_idx = None
        aic_exact_model = None; aic_exact_idx = None
        bic_exact_model = None; bic_exact_idx = None

        if pd.notna(best_alpha_from_cv):
            cv_exact_idx = (df_path['alpha'] - best_alpha_from_cv).abs().argsort()[0]
            cv_exact_model = df_path.iloc[cv_exact_idx]

        if not df_path.empty:
            aic_exact_idx = df_path['aic'].idxmin()
            aic_exact_model = df_path.loc[aic_exact_idx]
            bic_exact_idx = df_path['bic'].idxmin()
            bic_exact_model = df_path.loc[bic_exact_idx]

        # Find the corresponding indices in the ORIGINAL alphas_lasso array
        def find_alpha_index_in_path(target_alpha, alphas_array):
            return np.argmin(np.abs(alphas_array - target_alpha))

        chosen_values_heatmap = {
            'CV': {
                'index': find_alpha_index_in_path(cv_exact_model['alpha'], alphas_lasso) if cv_exact_model is not None else None,
                'label_val': cv_exact_model['k'] if cv_exact_model is not None else None,
                'color': context.plasma(0.25), 'linestyle': '--'
            },
            'AIC': {
                'index': find_alpha_index_in_path(aic_exact_model['alpha'], alphas_lasso) if aic_exact_model is not None else None,
                'label_val': aic_exact_model['k'] if aic_exact_model is not None else None,
                'color': context.plasma(0.55), 'linestyle': '-'
            },
            'BIC': {
                'index': find_alpha_index_in_path(bic_exact_model['alpha'], alphas_lasso) if bic_exact_model is not None else None,
                'label_val': bic_exact_model['k'] if bic_exact_model is not None else None,
                'color': context.plasma(0.85), 'linestyle': '-.'
            }
        }

        plot_activation_heatmap(alphas_lasso, np.maximum(coefs_lasso, 0.0), feature_names, chosen_values_heatmap,
                                title=f'Lasso Activation Path', x_axis_label=r'Regularization Strength',
                                filename=os.path.join(save_dir, f'{title_prefix}_Lasso_ActivationHeatmap.pdf'),
                                log_x=True, reverse_x=True, is_omp=False, context=context)

        df_plot = df_path.sort_values('alpha', ascending=False)

        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Use the EXACT alpha values from the same models
        alpha_for_cv = cv_exact_model['alpha'] if cv_exact_model is not None else None
        alpha_for_aic = aic_exact_model['alpha'] if aic_exact_model is not None else None
        alpha_for_bic = bic_exact_model['alpha'] if bic_exact_model is not None else None

        # Row 0: CV NMSE (mean ± std) and Norm. L1 on twin axis
        ax0_twin = axs[0].twinx()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0 and np.isfinite(cv_nmse_mean_plot).any():
            axs[0].plot(alphas_cv_plot, cv_nmse_mean_plot, 'o-', ms=3, color=context.plasma(0.25), label='CV NMSE')
            if isinstance(cv_nmse_std_plot, np.ndarray) and cv_nmse_std_plot.size == cv_nmse_mean_plot.size:
                try:
                    axs[0].fill_between(alphas_cv_plot,
                                        cv_nmse_mean_plot - cv_nmse_std_plot,
                                        cv_nmse_mean_plot + cv_nmse_std_plot,
                                        color=context.plasma(0.25), alpha=0.15, linewidth=0)
                except Exception:
                    pass
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes, ha='center', va='center')

        # Norm. L1 on twin axis
        ax0_twin.plot(df_plot['alpha'], df_plot['norm_l1'], 's-', ms=3, color=context.plasma(0.65), alpha=0.6, label='Norm. L1')
        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE, color=context.plasma(0.25), labelpad=-20)
        ax0_twin.set_ylabel('Norm. L1', fontsize=context.PLOT_LABEL_FONTSIZE, color=context.plasma(0.65), labelpad=-10)

        if pd.notna(alpha_for_cv):
            axs[0].axvline(alpha_for_cv, color=chosen_values_heatmap['CV']['color'], ls=chosen_values_heatmap['CV']['linestyle'], lw=1.5)
            axs[0].text(alpha_for_cv * 1.5, 0.5, f'$\\lambda_L={alpha_for_cv:.1e}$', rotation=90, verticalalignment='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=chosen_values_heatmap['CV']['color'])

        # Keep the same y-limits and ticks styling as before
        axs[0].set_ylim(-0.05, 1.05); ax0_twin.set_ylim(-0.05, 1.05)
        axs[0].set_yticks([0, 1.0]); ax0_twin.set_yticks([0, 1.0])
        ax0_twin.tick_params(axis='y', labelcolor=context.plasma(0.65))
        axs[0].tick_params(axis='y', labelcolor=context.plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_plot['alpha'], df_plot['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_aic):
            axs[1].axvline(alpha_for_aic, color=chosen_values_heatmap['AIC']['color'], ls=chosen_values_heatmap['AIC']['linestyle'], lw=1.5)
            axs[1].text(alpha_for_aic * 1.5, np.mean(axs[1].get_ylim()), f'$\\lambda_L={alpha_for_aic:.1e}$', rotation=90, verticalalignment='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=chosen_values_heatmap['AIC']['color'])
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['alpha'], df_plot['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_bic):
            axs[2].axvline(alpha_for_bic, color=chosen_values_heatmap['BIC']['color'], ls=chosen_values_heatmap['BIC']['linestyle'], lw=1.5)
            axs[2].text(alpha_for_bic * 1.5, np.mean(axs[2].get_ylim()), f'$\\lambda_L={alpha_for_bic:.1e}$', rotation=90, verticalalignment='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=chosen_values_heatmap['BIC']['color'])
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])

        # Formatting all axes
        axs[2].set_xlabel('Regularization Strength', fontsize=context.PLOT_LABEL_FONTSIZE)
        x_min = df_plot['alpha'].min(); x_max = df_plot['alpha'].max()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0:
            x_min = min(x_min, np.min(alphas_cv_plot))
            x_max = max(x_max, np.max(alphas_cv_plot))
        for ax in axs:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([x_min, x_max]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        handles = [plt.Line2D([0], [0], color=style['color'], ls=style['linestyle'], label=f'{key}') for key, style in chosen_values_heatmap.items() if pd.notna(style.get('label_val'))]
        # fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fontsize=PLOT_LEGEND_FONTSIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_Lasso_Criteria.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)


def plot_lars_selection(*, aic_row, bic_row, chosen_values_heatmap, coefs_path_nonneg, context, cv_nmse_mean_sorted, cv_nmse_std_sorted, cv_row, df_path, feature_names, k_sorted, save_dir, steps_for_plot, title_prefix):
    try:
        # Heatmap over iterations (columns are β≥0)
        plot_activation_heatmap(
            steps_for_plot, coefs_path_nonneg, feature_names, chosen_values_heatmap,
            title=f'LARS Activation Path', x_axis_label='LARS Iteration',
            filename=os.path.join(save_dir, f'{title_prefix}_Lars_ActivationHeatmap.pdf'),
            log_x=False, reverse_x=False, is_omp=False,
            context=context,
        )

        df_plot = df_path.sort_values('k')
        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std), normalized to [0,1] for display; ticks forced
        if (k_sorted.size > 0 and np.any(np.isfinite(cv_nmse_mean_sorted))):
            lower = np.clip(cv_nmse_mean_sorted - cv_nmse_std_sorted, 0, 1)
            upper = np.clip(cv_nmse_mean_sorted + cv_nmse_std_sorted, 0, 1)
            axs[0].fill_between(k_sorted, lower, upper, alpha=0.2,
                                edgecolor='none', facecolor=context.plasma(0.25))
            axs[0].plot(k_sorted, cv_nmse_mean_sorted, 'o-', ms=3, color=context.plasma(0.25))
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes,
                        ha='center', va='center')

        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE,
                          color=context.plasma(0.25), labelpad=-10)
        axs[0].tick_params(axis='y', labelcolor=context.plasma(0.25))
        axs[0].axvline(int(cv_row['k']), color=context.plasma(0.25), ls='--', lw=1.5)
        _annotate_iteration(axs[0], int(cv_row['k']), color=context.plasma(0.25), context=context)
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([k_sorted.min(), k_sorted.max()])

        # Row 1: AIC
        axs[1].plot(df_plot['k'], df_plot['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['k']), color=context.plasma(0.55), ls='-', lw=1.5)
        _annotate_iteration(axs[1], int(aic_row['k']), color=context.plasma(0.55), context=context)
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])
        axs[1].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['k'], df_plot['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['k']), color=context.plasma(0.85), ls='-.', lw=1.5)
        _annotate_iteration(axs[2], int(bic_row['k']), color=context.plasma(0.85), context=context)
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])
        axs[2].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        axs[2].set_xlabel('LARS iteration', fontsize=context.PLOT_LABEL_FONTSIZE)
        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_Lars_Criteria.pdf'),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)

    except Exception as e_plot:
        print(f'  WARN: LARS plotting failed: {e_plot}')


def plot_omp_selection(*, X, _nnls_refit_with_intercept, aic_row, best_idx, bic_row, context, cv_nmse_mean, cv_nmse_std, cv_row, df_path, feature_names, save_dir, steps, supports, title_prefix, y):
    try:
        # For heatmap we want a (p × steps) matrix of nonnegative coefs at each step.
        # Use FULL-DATA NNLS refits so the heatmap matches AIC/BIC models.
        heat_cols = []
        for S in supports:
            beta_full, _, _ = _nnls_refit_with_intercept(X, y, S)
            heat_cols.append(beta_full)
        heat_coefs = np.column_stack(heat_cols)

        chosen_heat = {
            'CV':  {'index': best_idx,  'label_val': int(cv_row['k']),  'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': int(np.where(df_path.index == df_path['aic'].idxmin())[0][0]),
                    'label_val': int(aic_row['k']), 'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': int(np.where(df_path.index == df_path['bic'].idxmin())[0][0]),
                    'label_val': int(bic_row['k']), 'color': context.plasma(0.85), 'linestyle': '-.'},
        }
        plot_activation_heatmap(
            steps, heat_coefs, feature_names, chosen_heat,
            title="OMP Activation Path", x_axis_label='OMP Iteration',
            filename=os.path.join(save_dir, f'{title_prefix}_OMP_ActivationHeatmap.pdf'),
            log_x=False, reverse_x=False, is_omp=True,
            context=context,
        )

        # 3×1 criteria
        df_plot = df_path.sort_values('k')
        order_by_k = np.argsort(df_path['k'].values)
        k_sorted = df_path['k'].values[order_by_k].astype(int)
        cv_nmse_mean_sorted = cv_nmse_mean[order_by_k]
        cv_nmse_std_sorted  = cv_nmse_std[order_by_k]

        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std), ticks y=[0,1], x=[min(k),max(k)]
        if (k_sorted.size > 0 and np.any(np.isfinite(cv_nmse_mean_sorted))):
            lower = np.clip(cv_nmse_mean_sorted - cv_nmse_std_sorted, 0, 1)
            upper = np.clip(cv_nmse_mean_sorted + cv_nmse_std_sorted, 0, 1)
            axs[0].fill_between(k_sorted, lower, upper, alpha=0.2,
                                edgecolor='none', facecolor=context.plasma(0.25))
            axs[0].plot(k_sorted, cv_nmse_mean_sorted, 'o-', ms=3, color=context.plasma(0.25))
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes,
                        ha='center', va='center')

        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE,
                          color=context.plasma(0.25), labelpad=-10)
        axs[0].tick_params(axis='y', labelcolor=context.plasma(0.25))
        axs[0].axvline(int(cv_row['k']), color=context.plasma(0.25), ls='--', lw=1.5)
        _annotate_iteration(axs[0], int(cv_row['k']), color=context.plasma(0.25), context=context)
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([k_sorted.min(), k_sorted.max()])

        # Row 1: AIC
        axs[1].plot(df_plot['k'], df_plot['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['k']), color=context.plasma(0.55), ls='-', lw=1.5)
        _annotate_iteration(axs[1], int(aic_row['k']), color=context.plasma(0.55), context=context)
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])
        axs[1].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['k'], df_plot['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['k']), color=context.plasma(0.85), ls='-.', lw=1.5)
        _annotate_iteration(axs[2], int(bic_row['k']), color=context.plasma(0.85), context=context)
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])
        axs[2].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        axs[2].set_xlabel('OMP iteration', fontsize=context.PLOT_LABEL_FONTSIZE)
        for ax in axs:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_OMP_Criteria.pdf'),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)

    except Exception as e_plot:
        print(f'  WARN: OMP plotting failed: {e_plot}')
