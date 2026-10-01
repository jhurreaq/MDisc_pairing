"""Treloar plotting; original numerical and presentation choices."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from mdisc.regression import fit_ridge
from sklearn.metrics import mean_squared_error, r2_score
import os
import traceback
from .models import construct_derivatives_matrix
from .models import construct_design_matrix
from mdisc.isotropic import format_term_for_latex
from .models import check_physical_consistency

def plot_activation_heatmap(
    path_variable, # Alphas (Lasso/LARS) or Steps (OMP)
    coef_path, # (n_features, n_steps_or_alphas)
    basis_names,
    chosen_values_dict, # Dict: {'CV': {'value': val, 'color': 'r', 'linestyle': '-'}, ...}
    title,
    x_axis_label,
    filename=None,
    log_x=False,
    reverse_x=True,
    activation_thresh=1e-20,
    is_omp=False # Flag for OMP/LARS specific logic
, *, context):
    """
    Generalized function to plot feature activation heatmaps with multiple selection criteria.
    Handles Lasso (vs log alpha) and LARS/OMP (vs step).
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

    # --- Mark the chosen values ---
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

            label_prefix = 'k=' if is_omp else 'α=' if log_x else ''
            vline_label = f'{method}' #f'Selected {method} ({label_prefix}{label_val_str})'
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=vline_label))
        else:
            print(f'WARN: Chosen value for {method} is None or NaN, cannot plot vertical line.')

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

def plot_model_predictions(
    df_fit_scope,        # Data used ONLY for the refit step
    df_plot_structure,   # Full data structure for prediction calculation & plotting
    sparse_coeffs,       # Coefficients from initial sparse fit (on df_fit_scope)
    basis_names,         # Full list of basis names from initial fit
    method_key,          # Name reflecting selection (e.g., "LassoCV")
    scenario_prefix,     # Includes fitting scenario info
    save_dir,
    weight_factor=1.0,
    small_strain_threshold=50.0,
    coeff_threshold=1e-06,
    ridge_alpha=1e-6
, *, context):
    """
    Performs Ridge refit *only* on df_fit_scope data.
    Calculates predictions and metrics across the full df_plot_structure.
    Plots performance across df_plot_structure (e.g., Treloar & Kawabata).
    Returns 8 values: final_terms, final_coeffs, R2/RMSE metrics (on full structure), consistency info.
    """
    print(f'\n--- Generating Final Model & Predictions ({method_key} / Ridge Refit on FIT SUBSET / Evaluate on FULL SET) ---')
    print(f'    Refit Scope: {len(df_fit_scope)} points. Evaluation Scope: {len(df_plot_structure)} points.')

    # --- Step 1: Identify Initial Selected Features (from sparse fit on subset) ---
    selected_indices = np.where(np.abs(sparse_coeffs) > context.EPS)[0]
    if len(selected_indices) == 0:
        print(f'WARN: No features initially selected by {method_key}. Skipping evaluation.')
        return [], [], np.nan, np.nan, np.nan, np.nan, False, ["No features selected in initial fit"]
    initial_selected_basis_names = [basis_names[i] for i in selected_indices]
    print(f'  Initial selection ({method_key}): {len(initial_selected_basis_names)} features.')

    # --- Step 2: Reconstruct WEIGHTED Design Matrix & Target for Refit (using FIT SUBSET) ---
    print(f'  Reconstructing WEIGHTED design matrix for refit using FIT SUBSET ({len(df_fit_scope)} points)...')
    phi_weighted_fit, target_weighted_fit = None, None
    try:
        phi_weighted_fit, target_weighted_fit, basis_names_refit, _, _ = construct_design_matrix(
            df_fit_scope,
            emphasize_small_strain=True,
            weight_factor=weight_factor,
            small_strain_threshold=small_strain_threshold,
            context=context,
        )
        if phi_weighted_fit.shape[0] == 0: raise ValueError("Weighted design matrix for refit empty.")
        if phi_weighted_fit.shape[0] != len(target_weighted_fit): raise ValueError("Weighted refit matrix/target row mismatch.")
        if basis_names != basis_names_refit: print("WARN: Basis name mismatch during weighted refit matrix recon.")

        phi_selected_weighted_fit = phi_weighted_fit[:, selected_indices]

    except Exception as e:
        print(f'ERROR: Failed to reconstruct WEIGHTED matrix/target for refit using fit subset: {e}')
        traceback.print_exc()
        return [], [], np.nan, np.nan, np.nan, np.nan, False, [f'Refit Matrix Recon Error: {e}']

    # --- Step 3: Refit using Ridge Regression (on FIT SUBSET data) ---
    print(f'  Refitting {len(initial_selected_basis_names)} features on WEIGHTED FIT SUBSET data (shape: {phi_selected_weighted_fit.shape}) using Ridge...')
    final_coeffs = np.zeros(len(initial_selected_basis_names))
    try:
        ridge_refit = fit_ridge(phi_selected_weighted_fit, target_weighted_fit, alpha=ridge_alpha, fit_intercept=False, solver='auto')
        refit_coeffs_weighted = ridge_refit.coef_
    except Exception as e:
        print(f'ERROR: Failed during Ridge Regression refit for {method_key} on fit subset: {e}'); traceback.print_exc()
        return [], [], np.nan, np.nan, np.nan, np.nan, False, [f'Ridge Refit Error: {e}']

    # --- Step 4: Apply Thresholding (to coefficients from refit on subset) ---
    print(f'  Applying threshold ({coeff_threshold:.1e}) to Ridge coefficients...')
    significant_mask = np.abs(refit_coeffs_weighted) > coeff_threshold
    final_coeffs = refit_coeffs_weighted[significant_mask]
    final_global_indices = selected_indices[significant_mask]
    final_selected_basis_names = [basis_names[i] for i in final_global_indices]
    n_final = len(final_selected_basis_names)
    print(f'  Final selected features after thresholding: {n_final}')

    # --- Step 4.5: Check Physical Consistency ---
    is_consistent = False; shear_modulus = np.nan; consistency_notes = ["Thresholding removed all terms"]
    if n_final == 0:
        print(f'WARN: All features removed by thresholding for {method_key}.')
        is_consistent, shear_modulus, consistency_notes = check_physical_consistency([], [], context=context)
    else:
        print(f'  Final Model ({method_key} - based on fit subset refit):'); [print(f"    '{name}': {coeff:.6f}") for name, coeff in zip(final_selected_basis_names, final_coeffs)]
        print("  Checking physical consistency..."); is_consistent, shear_modulus, consistency_notes = check_physical_consistency(final_selected_basis_names, final_coeffs, context=context)
        print("  Consistency Check Results:"); [print(f'    {note}') for note in consistency_notes]
        if not is_consistent: print(f'  WARNING: {method_key} model potentially inconsistent (μ₀={shear_modulus:.4f}).')
        else: print(f'  INFO: {method_key} model passed basic consistency check (μ₀={shear_modulus:.4f}).')

    # --- Step 5: Reconstruct UNWEIGHTED Design Matrix for FULL EVALUATION SET ---
    print(f'\n  Reconstructing UNWEIGHTED design matrix for FULL EVALUATION set ({len(df_plot_structure)} points)...')
    phi_unweighted_full, target_unweighted_full = None, None
    row_to_plot_idx_map = None
    try:
        dW1_mat_uw_full, dW2_mat_uw_full, dW3_mat_uw_full, basis_names_unweighted = construct_derivatives_matrix(df_plot_structure, context=context)
        if basis_names != basis_names_unweighted: raise ValueError("Basis name mismatch!")

        n_samp_full = df_plot_structure.shape[0]; n_basis = len(basis_names)
        n_p11_rows_full = df_plot_structure[pd.notna(df_plot_structure['P11'])].shape[0]
        n_p22_rows_full = df_plot_structure[df_plot_structure['mode'].isin(['EBT', 'BIAX']) & pd.notna(df_plot_structure['P22'])].shape[0]
        total_rows_expected_full = n_p11_rows_full + n_p22_rows_full

        phi_unweighted_full = np.zeros((total_rows_expected_full, n_basis))
        target_unweighted_full = np.zeros(total_rows_expected_full)
        row_to_plot_idx_map = np.full(total_rows_expected_full, -1, dtype=int)

        print(f'    Building unweighted evaluation matrix with expected shape: ({total_rows_expected_full}, {n_basis})')
        current_row_idx = 0
        for idx_in_plot_df in range(n_samp_full):
            row_data = df_plot_structure.iloc[idx_in_plot_df]; l1, l2 = max(row_data['lambda1'], context.EPS), max(row_data['lambda2'], context.EPS)
            l3_inv_prod = l1 * l2; l3 = max(1.0 / l3_inv_prod if l3_inv_prod > context.EPS else 1.0 / context.EPS, context.EPS); mode = row_data['mode']
            dW1_vec_uw = dW1_mat_uw_full[idx_in_plot_df, :]; dW2_vec_uw = dW2_mat_uw_full[idx_in_plot_df, :]; dW3_vec_uw = dW3_mat_uw_full[idx_in_plot_df, :]
            has_p11 = pd.notna(row_data['P11']); has_p22 = pd.notna(row_data['P22']) and mode in ['EBT', 'BIAX']
            if has_p11:
                if current_row_idx < total_rows_expected_full: l3_over_l1 = l3 / l1 if l1 > context.EPS else l3 / context.EPS; phi_unweighted_full[current_row_idx, :] = dW1_vec_uw - l3_over_l1 * dW3_vec_uw; target_unweighted_full[current_row_idx] = row_data['P11']; row_to_plot_idx_map[current_row_idx] = idx_in_plot_df; current_row_idx += 1
            if has_p22:
                if current_row_idx < total_rows_expected_full: l3_over_l2 = l3 / l2 if l2 > context.EPS else l3 / context.EPS; phi_unweighted_full[current_row_idx, :] = dW2_vec_uw - l3_over_l2 * dW3_vec_uw; target_unweighted_full[current_row_idx] = row_data['P22']; row_to_plot_idx_map[current_row_idx] = idx_in_plot_df; current_row_idx += 1

        if current_row_idx != total_rows_expected_full:
            print(f'WARN (Unweighted Full Recon): Row count mismatch ({current_row_idx} vs {total_rows_expected_full}). Resizing.')
            phi_unweighted_full = phi_unweighted_full[:current_row_idx, :]; target_unweighted_full = target_unweighted_full[:current_row_idx]; row_to_plot_idx_map = row_to_plot_idx_map[:current_row_idx]

        print("    Filtering non-finite values from full unweighted matrix/target...")
        finite_mask_uw_full = np.all(np.isfinite(phi_unweighted_full), axis=1) & np.isfinite(target_unweighted_full)
        num_finite_uw_full = np.sum(finite_mask_uw_full)
        if num_finite_uw_full < phi_unweighted_full.shape[0]:
            num_removed = phi_unweighted_full.shape[0] - num_finite_uw_full; print(f'    WARN: Removing {num_removed} non-finite rows from unweighted full matrix/target.')
            phi_unweighted_full = phi_unweighted_full[finite_mask_uw_full, :]; target_unweighted_full = target_unweighted_full[finite_mask_uw_full]; row_to_plot_idx_map = row_to_plot_idx_map[finite_mask_uw_full]

        if phi_unweighted_full.shape[0] == 0:
            raise ValueError("Unweighted full matrix empty after filtering.")
        print(f'    Final unweighted evaluation matrix shape after filtering: {phi_unweighted_full.shape}')

    except Exception as e:
        print(f'ERROR: Failed to reconstruct UNWEIGHTED matrix/target for full evaluation set: {e}')
        traceback.print_exc()
        return final_selected_basis_names, final_coeffs, np.nan, np.nan, np.nan, np.nan, is_consistent, consistency_notes

    # --- Step 6: Calculate UNWEIGHTED Predictions across FULL Evaluation Set ---
    pred_unweighted_full = np.zeros(phi_unweighted_full.shape[0])
    if n_final > 0:
        phi_final_selected_unweighted_full = phi_unweighted_full[:, final_global_indices]
        print(f'  Calculating predictions on FULL evaluation set using matrix shape {phi_final_selected_unweighted_full.shape} and {len(final_coeffs)} coeffs (from subset refit)...')
        try:
            if phi_final_selected_unweighted_full.shape[1] != len(final_coeffs): raise ValueError(f'Matrix columns ({phi_final_selected_unweighted_full.shape[1]}) != Coeffs ({len(final_coeffs)})')
            pred_unweighted_full = phi_final_selected_unweighted_full @ final_coeffs
            if len(pred_unweighted_full) != len(target_unweighted_full): raise ValueError(f'Prediction length ({len(pred_unweighted_full)}) != Target length ({len(target_unweighted_full)})')
        except Exception as e: print(f'ERROR predicting on full set: {e}'); traceback.print_exc(); return final_selected_basis_names, final_coeffs, np.nan, np.nan, np.nan, np.nan, is_consistent, consistency_notes
    else:
        print("  Skipping prediction calculation on full set: n_final is 0.")

    # --- Step 7: Map Predictions back to FULL DataFrame Structure ---
    print("  Mapping predictions back to original FULL DataFrame structure...")
    df_plot = df_plot_structure.copy()
    df_plot['P11_pred'] = np.nan
    df_plot['P22_pred'] = np.nan
    successfully_mapped = 0; mapping_errors = 0

    for i, pred_value in enumerate(pred_unweighted_full):
        matrix_row_index = i
        original_df_idx = row_to_plot_idx_map[matrix_row_index]
        if original_df_idx == -1:
            mapping_errors += 1; continue

        is_p11_row = True
        rows_for_this_original_idx = np.where(row_to_plot_idx_map == original_df_idx)[0]
        if len(rows_for_this_original_idx) > 1 and matrix_row_index == rows_for_this_original_idx[1]:
            is_p11_row = False
        elif len(rows_for_this_original_idx) == 1:
            row_data_check = df_plot_structure.iloc[original_df_idx]; mode_check = row_data_check['mode']
            has_p11_check = pd.notna(row_data_check['P11']); has_p22_check = pd.notna(row_data_check['P22']) and mode_check in ['EBT', 'BIAX']
            if has_p11_check and not has_p22_check: is_p11_row = True
            elif has_p22_check and not has_p11_check: is_p11_row = False
            else: is_p11_row = True

        try:
            if is_p11_row: df_plot.loc[original_df_idx, 'P11_pred'] = pred_value; successfully_mapped += 1
            else: df_plot.loc[original_df_idx, 'P22_pred'] = pred_value; successfully_mapped += 1
        except Exception as map_e:
            mapping_errors += 1; print(f'    ERROR mapping prediction {i} to df index {original_df_idx}: {map_e}')

    if successfully_mapped != len(pred_unweighted_full):
        print(f'WARN: Mapping count mismatch. Expected {len(pred_unweighted_full)}, mapped {successfully_mapped}. Errors: {mapping_errors}')
    else:
        print(f'  Successfully mapped {successfully_mapped} predictions onto the full dataset structure.')

    # --- Step 8: Calculate Overall Metrics (on FULL evaluation set) ---
    print("  Calculating OVERALL metrics on FULL evaluation set predictions...")
    p11_exp_full = df_plot['P11']; p22_exp_full = df_plot['P22']
    p11_pred_final = df_plot['P11_pred']; p22_pred_final = df_plot['P22_pred']

    p11_valid_mask = pd.notna(p11_exp_full) & pd.notna(p11_pred_final)
    p22_valid_mask = pd.notna(p22_exp_full) & pd.notna(p22_pred_final) & df_plot['mode'].isin(['EBT', 'BIAX'])

    r2_P11_overall = r2_score(p11_exp_full[p11_valid_mask], p11_pred_final[p11_valid_mask]) if p11_valid_mask.sum() > 0 else np.nan
    rmse_P11_overall = np.sqrt(mean_squared_error(p11_exp_full[p11_valid_mask], p11_pred_final[p11_valid_mask])) if p11_valid_mask.sum() > 0 else np.nan
    r2_P22_overall = r2_score(p22_exp_full[p22_valid_mask], p22_pred_final[p22_valid_mask]) if p22_valid_mask.sum() > 0 else np.nan
    rmse_P22_overall = np.sqrt(mean_squared_error(p22_exp_full[p22_valid_mask], p22_pred_final[p22_valid_mask])) if p22_valid_mask.sum() > 0 else np.nan

    print(f'  Overall Metrics on Full Set: P11 R²={r2_P11_overall:.4f}, RMSE={rmse_P11_overall:.4f} | P22 R²={r2_P22_overall:.4f}, RMSE={rmse_P22_overall:.4f}')

    mode_specific_metrics = {}
    available_modes = df_plot['mode'].unique()
    print(f'  Calculating mode-specific metrics for modes: {available_modes}')

    for mode in ['UT', 'PS', 'EBT', 'BIAX']:
        if mode in available_modes:
            mode_data = df_plot[df_plot['mode'] == mode]

            # P11 metrics for this mode
            mode_p11_valid = pd.notna(mode_data['P11']) & pd.notna(mode_data['P11_pred'])
            if mode_p11_valid.sum() > 0:
                r2_p11_mode = r2_score(mode_data['P11'][mode_p11_valid], mode_data['P11_pred'][mode_p11_valid])
                rmse_p11_mode = np.sqrt(mean_squared_error(mode_data['P11'][mode_p11_valid], mode_data['P11_pred'][mode_p11_valid]))
                n_points_p11 = mode_p11_valid.sum()
            else:
                r2_p11_mode = np.nan; rmse_p11_mode = np.nan; n_points_p11 = 0

            # P22 metrics for this mode (only for modes that have P22 data)
            if mode in ['EBT', 'BIAX']:
                mode_p22_valid = pd.notna(mode_data['P22']) & pd.notna(mode_data['P22_pred'])
                if mode_p22_valid.sum() > 0:
                    r2_p22_mode = r2_score(mode_data['P22'][mode_p22_valid], mode_data['P22_pred'][mode_p22_valid])
                    rmse_p22_mode = np.sqrt(mean_squared_error(mode_data['P22'][mode_p22_valid], mode_data['P22_pred'][mode_p22_valid]))
                    n_points_p22 = mode_p22_valid.sum()
                else:
                    r2_p22_mode = np.nan; rmse_p22_mode = np.nan; n_points_p22 = 0
            else:
                r2_p22_mode = np.nan; rmse_p22_mode = np.nan; n_points_p22 = 0

            mode_specific_metrics[mode] = {
                'r2_p11': r2_p11_mode, 'rmse_p11': rmse_p11_mode, 'n_points_p11': n_points_p11,
                'r2_p22': r2_p22_mode, 'rmse_p22': rmse_p22_mode, 'n_points_p22': n_points_p22
            }

            # Print mode-specific results
            if n_points_p11 > 0:
                print(f'    {mode} Mode P11: R²={r2_p11_mode:.4f}, RMSE={rmse_p11_mode:.4f} (n={n_points_p11})')
            if n_points_p22 > 0:
                print(f'    {mode} Mode P22: R²={r2_p22_mode:.4f}, RMSE={rmse_p22_mode:.4f} (n={n_points_p22})')
            if n_points_p11 == 0 and n_points_p22 == 0:
                print(f'    {mode} Mode: No valid data points for evaluation')

    # --- Step 9: Create Plots (using FULL df_plot structure) ---
    print("  Preparing plots using full dataset structure...")
    consistency_status_str = "(Phys OK)" if is_consistent else "(Phys WARN!)"
    title_suffix = f'Final Model ({method_key} / Refit on Fit Subset Thresh={coeff_threshold:.1e}) {consistency_status_str}'

    fig = None; plot_created = False
    datasets_in_plot_structure = df_plot['dataset'].unique()
    print(f'  DEBUG: Datasets found in plotting structure: {datasets_in_plot_structure}')
    treloar_present = 'Treloar' in datasets_in_plot_structure
    kawabata_present = 'Kawabata' in datasets_in_plot_structure
    print(f'  DEBUG: Plotting Treloar: {treloar_present}, Plotting Kawabata: {kawabata_present}')

    if treloar_present and kawabata_present:
        print(f'  DEBUG: Entering COMBINED plot block for {method_key}...')
        fig = plt.figure(figsize=(8, 8)); gs = fig.add_gridspec(2, 2)
        ax_treloar = [fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])]
        ax_kawabata = [fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])]
        try:
            # Calculate dataset-specific R2 values for plotting
            treloar_data = df_plot[df_plot['dataset'] == 'Treloar']
            kawabata_data = df_plot[df_plot['dataset'] == 'Kawabata']

            # Treloar-specific R2 for P11
            treloar_p11_valid = pd.notna(treloar_data['P11']) & pd.notna(treloar_data['P11_pred'])
            r2_treloar_p11 = r2_score(treloar_data['P11'][treloar_p11_valid], treloar_data['P11_pred'][treloar_p11_valid]) if treloar_p11_valid.sum() > 0 else np.nan

            # Kawabata-specific R2 for P11 and P22
            kawabata_p11_valid = pd.notna(kawabata_data['P11']) & pd.notna(kawabata_data['P11_pred'])
            kawabata_p22_valid = pd.notna(kawabata_data['P22']) & pd.notna(kawabata_data['P22_pred']) & kawabata_data['mode'].isin(['EBT', 'BIAX'])
            r2_kawabata_p11 = r2_score(kawabata_data['P11'][kawabata_p11_valid], kawabata_data['P11_pred'][kawabata_p11_valid]) if kawabata_p11_valid.sum() > 0 else np.nan
            r2_kawabata_p22 = r2_score(kawabata_data['P22'][kawabata_p22_valid], kawabata_data['P22_pred'][kawabata_p22_valid]) if kawabata_p22_valid.sum() > 0 else np.nan

            plot_treloar_prediction(treloar_data, axes=ax_treloar, r2_overall_precomputed=r2_treloar_p11, context=context)
        except Exception as e: print(f'  ERROR plotting Treloar: {e}'); traceback.print_exc()
        try: plot_Kawabata_prediction(kawabata_data, axes=ax_kawabata, r2_p11_precomputed=r2_kawabata_p11, r2_p22_precomputed=r2_kawabata_p22, context=context)
        except Exception as e: print(f'  ERROR plotting Kawabata: {e}'); traceback.print_exc()
        fig.suptitle(f'{title_suffix} vs. Exp. Data', fontsize=context.PLOT_TITLE_FONTSIZE+1);
        try: plt.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception as e: print(f'  WARN: tight_layout failed: {e}')
        plot_created = True
    elif treloar_present:
        print(f'  DEBUG: Entering TRELOAR ONLY plot block for {method_key}...'); fig, axes = plt.subplots(1, 2, figsize=(8, 5));
        try:
            treloar_data = df_plot[df_plot['dataset'] == 'Treloar']
            treloar_p11_valid = pd.notna(treloar_data['P11']) & pd.notna(treloar_data['P11_pred'])
            r2_treloar_p11 = r2_score(treloar_data['P11'][treloar_p11_valid], treloar_data['P11_pred'][treloar_p11_valid]) if treloar_p11_valid.sum() > 0 else np.nan
            plot_treloar_prediction(treloar_data, axes=axes, r2_overall_precomputed=r2_treloar_p11, context=context); fig.suptitle(f'{title_suffix} vs. Treloar Data', fontsize=context.PLOT_TITLE_FONTSIZE+1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plot_created = True;
        except Exception as e: print(f'  ERROR plotting Treloar: {e}'); traceback.print_exc(); plt.close(fig); fig=None
    elif kawabata_present:
        print(f'  DEBUG: Entering KAWABATA ONLY plot block for {method_key}...'); fig, axes = plt.subplots(1, 2, figsize=(8, 5));
        try:
            kawabata_data = df_plot[df_plot['dataset'] == 'Kawabata']
            kawabata_p11_valid = pd.notna(kawabata_data['P11']) & pd.notna(kawabata_data['P11_pred'])
            kawabata_p22_valid = pd.notna(kawabata_data['P22']) & pd.notna(kawabata_data['P22_pred']) & kawabata_data['mode'].isin(['EBT', 'BIAX'])
            r2_kawabata_p11 = r2_score(kawabata_data['P11'][kawabata_p11_valid], kawabata_data['P11_pred'][kawabata_p11_valid]) if kawabata_p11_valid.sum() > 0 else np.nan
            r2_kawabata_p22 = r2_score(kawabata_data['P22'][kawabata_p22_valid], kawabata_data['P22_pred'][kawabata_p22_valid]) if kawabata_p22_valid.sum() > 0 else np.nan
            plot_Kawabata_prediction(kawabata_data, axes=axes, r2_p11_precomputed=r2_kawabata_p11, r2_p22_precomputed=r2_kawabata_p22, context=context); fig.suptitle(f'{title_suffix} vs. Kawabata Data', fontsize=context.PLOT_TITLE_FONTSIZE+1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plot_created = True;
        except Exception as e: print(f'  ERROR plotting Kawabata: {e}'); traceback.print_exc(); plt.close(fig); fig=None
    else:
        print(f"No recognizable datasets ('Treloar', 'Kawabata') found in df_plot_structure for {method_key}. Skipping prediction plot.")

    # --- Step 10: Save Plot ---
    if save_dir and scenario_prefix and fig and plot_created:
        save_path = os.path.join(save_dir, f'{scenario_prefix}_{method_key}_RefitSubset_EvalFull_Predictions.pdf')
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f'Saved final prediction plot: {save_path}')
        except Exception as e:
            print(f'Error saving final prediction plot {save_path}: {e}')
    elif fig:
        print(f'WARN: Plot not saved for {method_key} (save conditions not met or plot_created={plot_created}).')
    if fig:
        plt.close(fig)

        # --- Step 11: Return Final Model Details & EVALUATION METRICS ---
    return (final_selected_basis_names, final_coeffs, r2_P11_overall, rmse_P11_overall,
            r2_P22_overall, rmse_P22_overall, is_consistent, consistency_notes, mode_specific_metrics)

def plot_treloar_prediction(df, axes, r2_overall_precomputed=np.nan, *, context):
    """Plot Treloar data with model predictions (Helper). Needs axes=[ax_full, ax_zoom]. Uses precomputed R2 for title."""
    ax_full, ax_zoom = axes
    colors = {'UT': context.TRUNCATED_PLASMA_CMAP(0.2), 'PS': context.TRUNCATED_PLASMA_CMAP(0.5), 'EBT': context.TRUNCATED_PLASMA_CMAP(0.8)}
    markers = {'UT': 'o', 'PS': 's', 'EBT': '^'}

    df_plot = df[df['dataset']=='Treloar'].copy()
    df_plot['P11'] = pd.to_numeric(df_plot['P11'], errors='coerce')
    df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')

    r2_overall = r2_overall_precomputed

    max_lambda_full = 0; min_stress_overall = 0; max_stress_overall = 0
    # --- First pass for limits ---
    for mode in ['UT', 'PS', 'EBT']:
        mode_data = df_plot[df_plot['mode'] == mode]
        if not mode_data.empty:
            valid_mask = ~np.isnan(mode_data['P11']) & ~np.isnan(mode_data['P11_pred'])
            if valid_mask.sum() == 0: continue
            lambda1_vals = mode_data['lambda1'][valid_mask]
            exp_stress = mode_data['P11'][valid_mask]; pred_stress = mode_data['P11_pred'][valid_mask]
            if len(lambda1_vals) > 0: max_lambda_full = max(max_lambda_full, np.nanmax(lambda1_vals))
            if len(exp_stress) > 0: min_stress_overall = min(min_stress_overall, np.nanmin(exp_stress)); max_stress_overall = max(max_stress_overall, np.nanmax(exp_stress))
            if len(pred_stress) > 0: min_stress_overall = min(min_stress_overall, np.nanmin(pred_stress)); max_stress_overall = max(max_stress_overall, np.nanmax(pred_stress))

    # --- Second pass for plotting ---
    max_y_zoom_overall = -np.inf; min_y_zoom_overall = np.inf
    for i, ax in enumerate([ax_full, ax_zoom]):
        legend_handles = []
        for mode in ['UT', 'PS', 'EBT']:
            mode_data = df_plot[df_plot['mode'] == mode]
            if not mode_data.empty:
                valid_mask = ~np.isnan(mode_data['P11']) & ~np.isnan(mode_data['P11_pred'])
                if valid_mask.sum() == 0: continue
                lambda1_vals = mode_data['lambda1'][valid_mask]
                exp_stress = mode_data['P11'][valid_mask]; pred_stress = mode_data['P11_pred'][valid_mask]
                sort_idx = np.argsort(lambda1_vals); lambda1_sorted = lambda1_vals.iloc[sort_idx]; exp_sorted = exp_stress.iloc[sort_idx]; pred_sorted = pred_stress.iloc[sort_idx]
                sc = ax.scatter(lambda1_sorted, exp_sorted, color=colors[mode], marker=markers[mode], s=context.PLOT_MARKER_SIZE, label=f'{mode} (Exp. data)' if i==0 else None, alpha=0.7, edgecolor='black', linewidth=0.5)
                pl, = ax.plot(lambda1_sorted, pred_sorted, color=colors[mode], linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'{mode} (Model)' if i==0 else None)
                if i == 0: legend_handles.extend([sc, pl])
                if i == 1:
                    mask_zoom_mode = (lambda1_sorted >= 1.0) & (lambda1_sorted <= 4.5)
                    if mask_zoom_mode.sum() > 0:
                         max_y_zoom_overall = max(max_y_zoom_overall, np.nanmax(exp_sorted[mask_zoom_mode]), np.nanmax(pred_sorted[mask_zoom_mode]))
                         min_y_zoom_overall = min(min_y_zoom_overall, np.nanmin(exp_sorted[mask_zoom_mode]), np.nanmin(pred_sorted[mask_zoom_mode]))

        ax.set_xlabel('Stretch λ', fontsize=context.PLOT_LABEL_FONTSIZE)
        ax.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)

        if i == 0:
            title_r2_str = f'{r2_overall:.3f}' if pd.notna(r2_overall) else "N/A"
            ax.set_title(f'Treloar (Full Range, $R^2 \\approx {title_r2_str}$)', fontsize=context.PLOT_TITLE_FONTSIZE)
            ax.legend(handles=legend_handles, loc='best', fontsize=context.PLOT_LEGEND_FONTSIZE)
            ax.set_xlim(left=1.0, right=max(4.5, max_lambda_full * 1.05))
            ax.set_ylim(bottom=min(0, min_stress_overall - 0.2), top=max(0.2, max_stress_overall * 1.1))
        else:
            ax.set_title(f'Treloar (Zoomed)', fontsize=context.PLOT_TITLE_FONTSIZE)
            ax.set_xlim(1.0, 5.5)
            if not np.isfinite(min_y_zoom_overall) or not np.isfinite(max_y_zoom_overall):
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=min(0, min_y_zoom_overall - 0.1), top=max(0.2, max_y_zoom_overall * 1.1))

def plot_Kawabata_prediction(df, axes, r2_p11_precomputed=np.nan, r2_p22_precomputed=np.nan, *, context):
    """Plot Kawabata data with model predictions (Helper). Uses precomputed R2 values for titles."""
    ax_p11, ax_p22 = axes; df_plot = df[df['dataset']=='Kawabata'].copy()
    df_plot['P11'] = pd.to_numeric(df_plot['P11'], errors='coerce'); df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')
    df_plot['P22'] = pd.to_numeric(df_plot['P22'], errors='coerce'); df_plot['P22_pred'] = pd.to_numeric(df_plot['P22_pred'], errors='coerce')
    lambda1_values = sorted(df_plot['lambda1'].unique()); cmap = context.TRUNCATED_PLASMA_CMAP; colors = cmap(np.linspace(0.1, 0.9, len(lambda1_values)))

    r2_P11 = r2_p11_precomputed
    r2_P22 = r2_p22_precomputed

    valid_p11_plot = ~np.isnan(df_plot['P11']) & ~np.isnan(df_plot['P11_pred'])
    valid_p22_plot = ~np.isnan(df_plot['P22']) & ~np.isnan(df_plot['P22_pred'])

    handles_p11 = []; max_y_p11 = 0
    for i, lambda1 in enumerate(lambda1_values):
        subset = df_plot[df_plot['lambda1'] == lambda1].sort_values(by='lambda2')
        v_p11 = valid_p11_plot[subset.index]
        if v_p11.sum() == 0: continue
        sc = ax_p11.scatter(subset['lambda2'][v_p11], subset['P11'][v_p11], color=colors[i], s=context.PLOT_MARKER_SIZE, alpha=0.7, marker='o', edgecolor='k', linewidth=0.5)
        pl, = ax_p11.plot(subset['lambda2'][v_p11], subset['P11_pred'][v_p11], color=colors[i], linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}')
        if i == 0:
            handles_p11.extend([Patch(color='gray', alpha=0.7, label='Exp'), Patch(color='black', label='Model')])
        handles_p11.append(pl)
        if v_p11.sum() > 0:
            max_y_p11 = max(max_y_p11, np.nanmax(subset['P11'][v_p11]), np.nanmax(subset['P11_pred'][v_p11]))

    ax_p11.set_xlabel('Stretch λ₂', fontsize=context.PLOT_LABEL_FONTSIZE); ax_p11.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    title_r2_p11_str = f'{r2_P11:.3f}' if pd.notna(r2_P11) else "N/A"
    ax_p11.set_title(f'Kawabata P₁₁ ($R^2 \\approx {title_r2_p11_str}$)', fontsize=context.PLOT_TITLE_FONTSIZE)
    ax_p11.legend(handles=handles_p11[:10], fontsize=context.PLOT_LEGEND_FONTSIZE*0.8, loc='best', ncol=2, frameon=False)
    ax_p11.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    ax_p11.set_ylim(bottom= min(0, np.nanmin(df_plot['P11'][valid_p11_plot])-0.1) if valid_p11_plot.sum() > 0 else 0, top=max(0.2, max_y_p11*1.1))

    handles_p22 = []; max_y_p22 = 0
    for i, lambda1 in enumerate(lambda1_values):
        subset = df_plot[df_plot['lambda1'] == lambda1].sort_values(by='lambda2')
        v_p22 = valid_p22_plot[subset.index]
        if v_p22.sum() == 0: continue
        sc = ax_p22.scatter(subset['lambda2'][v_p22], subset['P22'][v_p22], color=colors[i], s=context.PLOT_MARKER_SIZE, alpha=0.7, marker='s', edgecolor='k', linewidth=0.5)
        pl, = ax_p22.plot(subset['lambda2'][v_p22], subset['P22_pred'][v_p22], color=colors[i], linestyle='--', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}')
        if i == 0:
            handles_p22.extend([Patch(color='gray', alpha=0.7, label='Exp'), Patch(color='black', linestyle='--', label='Model')])
        handles_p22.append(pl)
        if v_p22.sum() > 0:
            max_y_p22 = max(max_y_p22, np.nanmax(subset['P22'][v_p22]), np.nanmax(subset['P22_pred'][v_p22]))

    ax_p22.set_xlabel('Stretch λ₂', fontsize=context.PLOT_LABEL_FONTSIZE); ax_p22.set_ylabel('Nominal Stress P₂₂ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    title_r2_p22_str = f'{r2_P22:.3f}' if pd.notna(r2_P22) else "N/A"
    ax_p22.set_title(f'Kawabata P₂₂ ($R^2 \\approx {title_r2_p22_str}$)', fontsize=context.PLOT_TITLE_FONTSIZE)
    ax_p22.legend(handles=handles_p22[:10], fontsize=context.PLOT_LEGEND_FONTSIZE*0.8, loc='best', ncol=2, frameon=False)
    ax_p22.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    ax_p22.set_ylim(bottom= min(0, np.nanmin(df_plot['P22'][valid_p22_plot])-0.1) if valid_p22_plot.sum() > 0 else 0, top=max(0.2, max_y_p22*1.1))

def plot_combined_datasets(df, save_dir=None, filename_prefix=None, *, context):
    """Plot the raw data from all included datasets"""
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    if n_datasets == 0: print("No data to plot."); return
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[0, i]; ds_data = df[df['dataset'] == dataset].copy()
        if dataset == 'Treloar': plot_treloar_data(ds_data, ax=ax, context=context)
        elif dataset == 'Kawabata': plot_Kawabata_data(ds_data, ax=ax, context=context)
        else: print(f"Warning: Unknown dataset type '{dataset}' for plotting."); ax.text(0.5, 0.5, f'Data for\n{dataset}', ha='center', va='center'); ax.set_title(f'{dataset} Data')
    plt.tight_layout()
    if save_dir and filename_prefix:
        save_path = os.path.join(save_dir, f'{filename_prefix}_RawData.pdf')
        try: os.makedirs(save_dir, exist_ok=True); fig.savefig(save_path, format='pdf', bbox_inches='tight'); print(f'Saved raw data plot: {save_path}')
        except Exception as e: print(f'Error saving raw data plot {save_path}: {e}')
    plt.close(fig)

def plot_treloar_data(df, ax=None, *, context):
    """Plot Treloar dataset - Helper for raw data plot"""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'UT': 'blue', 'PS': 'green', 'EBT': 'red'}; markers = {'UT': 'o', 'PS': 's', 'EBT': '^'}
    for mode in ['UT', 'PS', 'EBT']:
        mode_data = df[df['mode'] == mode]; p11_data = pd.to_numeric(mode_data['P11'], errors='coerce'); valid_mask = ~np.isnan(p11_data)
        if valid_mask.sum() > 0:
            strain = (mode_data['lambda1'][valid_mask] - 1) * 100; stress = p11_data[valid_mask]; sort_idx = np.argsort(strain)
            ax.scatter(strain.iloc[sort_idx], stress.iloc[sort_idx], color=colors[mode], marker=markers[mode], label=mode, alpha=0.8, s=context.PLOT_MARKER_SIZE, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Strain (%)', fontsize=context.PLOT_LABEL_FONTSIZE); ax.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_title('Treloar Data', fontsize=context.PLOT_TITLE_FONTSIZE); ax.legend(fontsize=context.PLOT_LEGEND_FONTSIZE); ax.grid(alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    return ax

def plot_Kawabata_data(df, ax=None, *, context):
    """Plot Kawabata biaxial dataset - Helper for raw data plot"""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    lambda1_values = sorted(df['lambda1'].unique()); cmap = context.TRUNCATED_PLASMA_CMAP; colors = cmap(np.linspace(0.1, 0.9, len(lambda1_values)))
    handles = [] # For legend
    for i, lambda1 in enumerate(lambda1_values):
        subset = df[df['lambda1'] == lambda1].copy(); subset['P11'] = pd.to_numeric(subset['P11'], errors='coerce'); subset['P22'] = pd.to_numeric(subset['P22'], errors='coerce')
        subset = subset.sort_values(by='lambda2'); valid_p11 = ~np.isnan(subset['P11']); valid_p22 = ~np.isnan(subset['P22'])
        line_p11, = ax.plot(subset['lambda2'][valid_p11], subset['P11'][valid_p11], marker='o', markersize=4, color=colors[i], linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}') if valid_p11.sum() > 0 else (None,)
        line_p22, = ax.plot(subset['lambda2'][valid_p22], subset['P22'][valid_p22], marker='s', markersize=4, color=colors[i], linestyle='--', linewidth=context.PLOT_PRED_LINEWIDTH) if valid_p22.sum() > 0 else (None,)
        if i==0: handles.extend([Patch(color='gray', label='P₁₁'), Patch(color='gray', linestyle='--', label='P₂₂')]) # General labels
        if line_p11: handles.append(line_p11) # Add specific lambda line handle
    ax.set_xlabel('λ₂', fontsize=context.PLOT_LABEL_FONTSIZE); ax.set_ylabel('Nominal Stress (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_title('Kawabata Biaxial Data', fontsize=context.PLOT_TITLE_FONTSIZE); ax.legend(handles=handles[:12], fontsize=context.PLOT_LEGEND_FONTSIZE*0.9); ax.grid(alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    return ax


def plot_lasso_selection(*, alphas_cv_plot, alphas_lasso, best_alpha_from_cv, coefs_lasso, context, cv_nmse_mean_plot, cv_nmse_std_plot, df_path, feature_names, save_dir, title_prefix):
    if not df_path.empty and alphas_lasso is not None and coefs_lasso is not None:
        print("\n  Generating plots from unified path data...")

        # exact models for marking
        cv_exact_model = None; aic_exact_model = None; bic_exact_model = None
        if pd.notna(best_alpha_from_cv):
            cv_exact_idx = (df_path['alpha'] - best_alpha_from_cv).abs().argsort()[0]
            cv_exact_model = df_path.iloc[cv_exact_idx]
        if not df_path.empty:
            aic_exact_model = df_path.loc[df_path['aic'].idxmin()]
            bic_exact_model = df_path.loc[df_path['bic'].idxmin()]

        def find_alpha_index_in_path(target_alpha, alphas_array):
            return np.argmin(np.abs(alphas_array - target_alpha))

        chosen_values_heatmap = {
            'CV':  {'index': find_alpha_index_in_path(cv_exact_model['alpha'], alphas_lasso) if cv_exact_model is not None else None,
                    'label_val': cv_exact_model['k'] if cv_exact_model is not None else None,
                    'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': find_alpha_index_in_path(aic_exact_model['alpha'], alphas_lasso) if aic_exact_model is not None else None,
                    'label_val': aic_exact_model['k'] if aic_exact_model is not None else None,
                    'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': find_alpha_index_in_path(bic_exact_model['alpha'], alphas_lasso) if bic_exact_model is not None else None,
                    'label_val': bic_exact_model['k'] if bic_exact_model is not None else None,
                    'color': context.plasma(0.85), 'linestyle': '-.'}
        }

        plot_activation_heatmap(
            alphas_lasso, np.maximum(coefs_lasso, 0.0), feature_names, chosen_values_heatmap,  # pass non-neg
            title="Lasso Activation Path", x_axis_label=r'Regularization Strength',
            filename=os.path.join(save_dir, f'{title_prefix}_Lasso_ActivationHeatmap.pdf'),
            log_x=True, reverse_x=True, is_omp=False,
            context=context,
        )

        df_plot = df_path.sort_values('alpha', ascending=False)
        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        alpha_for_cv  = cv_exact_model['alpha']  if cv_exact_model  is not None else None
        alpha_for_aic = aic_exact_model['alpha'] if aic_exact_model is not None else None
        alpha_for_bic = bic_exact_model['alpha'] if aic_exact_model is not None else None

        # Row 0: CV NMSE (mean ± std) vs alpha (robustly normalized to [0,1])
        ax0_twin = axs[0].twinx()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0 and np.isfinite(cv_nmse_mean_plot).any():
            axs[0].plot(alphas_cv_plot, cv_nmse_mean_plot, 'o-', ms=3, color=context.plasma(0.25))
            if isinstance(cv_nmse_std_plot, np.ndarray) and cv_nmse_std_plot.size == cv_nmse_mean_plot.size:
                try:
                    axs[0].fill_between(alphas_cv_plot,
                                        np.clip(cv_nmse_mean_plot - cv_nmse_std_plot, 0.0, 1.0),
                                        np.clip(cv_nmse_mean_plot + cv_nmse_std_plot, 0.0, 1.0),
                                        color=context.plasma(0.25), alpha=0.15, linewidth=0)
                except Exception:
                    pass
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes, ha='center', va='center')

        ax0_twin.plot(df_plot['alpha'], df_plot['norm_l1'], 's-', ms=3, color=context.plasma(0.65), alpha=0.6)
        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE, color=context.plasma(0.25), labelpad=-20)
        ax0_twin.set_ylabel('Norm. L1', fontsize=context.PLOT_LABEL_FONTSIZE, color=context.plasma(0.65), labelpad=-10)

        if pd.notna(alpha_for_cv):
            axs[0].axvline(alpha_for_cv, color=context.plasma(0.25), ls='--', lw=1.5)
            axs[0].text(alpha_for_cv * 1.5, 0.5, f'$\\lambda_L={alpha_for_cv:.1e}$',
                        rotation=90, va='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.25))

        axs[0].set_ylim(-0.05, 1.05); ax0_twin.set_ylim(-0.05, 1.05)
        axs[0].set_yticks([0.0, 1.0]); ax0_twin.set_yticks([0.0, 1.0])
        ax0_twin.tick_params(axis='y', labelcolor=context.plasma(0.65))
        axs[0].tick_params(axis='y', labelcolor=context.plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_plot['alpha'], df_plot['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_aic):
            axs[1].axvline(alpha_for_aic, color=context.plasma(0.55), ls='-', lw=1.5)
            axs[1].text(alpha_for_aic * 1.5, np.mean(axs[1].get_ylim()), f'$\\lambda_L={alpha_for_aic:.1e}$',
                        rotation=90, va='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.55))
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['alpha'], df_plot['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_bic):
            axs[2].axvline(alpha_for_bic, color=context.plasma(0.85), ls='-.', lw=1.5)
            axs[2].text(alpha_for_bic * 1.5, np.mean(axs[2].get_ylim()), f'$\\lambda_L={alpha_for_bic:.1e}$',
                        rotation=90, va='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.85))
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])

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

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_Lasso_Criteria.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)


def plot_lars_selection(*, aic_row, bic_row, coefs_path, context, cv_nmse_mean, cv_nmse_std, cv_row, df_path, feature_names, save_dir, steps, title_prefix):
    try:
        chosen_heat = {
            'CV':  {'index': int(cv_row['step']) - 1,  'label_val': int(cv_row['step']),  'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': int(aic_row['step']) - 1, 'label_val': int(aic_row['step']), 'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': int(bic_row['step']) - 1, 'label_val': int(bic_row['step']), 'color': context.plasma(0.85), 'linestyle': '-.'},
        }
        plot_activation_heatmap(
            steps, coefs_path, feature_names, chosen_heat,
            title="LARS Activation Path", x_axis_label='LARS Iteration',
            filename=os.path.join(save_dir, f'{title_prefix}_Lars_ActivationHeatmap.pdf'),
            log_x=False, reverse_x=False, is_omp=False,
            context=context,
        )

        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std)
        axs[0].plot(steps, cv_nmse_mean, 'o-', ms=3, color=context.plasma(0.25))
        axs[0].fill_between(steps,
                            np.maximum(0, cv_nmse_mean - cv_nmse_std),
                            np.minimum(1, cv_nmse_mean + cv_nmse_std),
                            alpha=0.2, color=context.plasma(0.25))
        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20, color=context.plasma(0.25))
        axs[0].axvline(int(cv_row['step']), color=context.plasma(0.25), ls='--', lw=1.5)
        axs[0].annotate(f"Iteration = {int(cv_row['step'])}",
                        xy=(int(cv_row['step']), 0.5), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.25))
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([int(steps.min()), int(steps.max())])

        # Row 1: AIC
        axs[1].plot(df_path['step'], df_path['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['step']), color=context.plasma(0.55), ls='-', lw=1.5)
        axs[1].annotate(f"Iteration = {int(aic_row['step'])}",
                        xy=(int(aic_row['step']), np.mean(axs[1].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])
        axs[1].set_xticks([int(steps.min()), int(steps.max())])

        # Row 2: BIC
        axs[2].plot(df_path['step'], df_path['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['step']), color=context.plasma(0.85), ls='-.', lw=1.5)
        axs[2].annotate(f"Iteration = {int(bic_row['step'])}",
                        xy=(int(bic_row['step']), np.mean(axs[2].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])
        axs[2].set_xticks([int(steps.min()), int(steps.max())])

        axs[2].set_xlabel('LARS Iteration', fontsize=context.PLOT_LABEL_FONTSIZE)
        for ax in axs:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_Lars_Criteria.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e_plot:
        print(f'  WARN: LARS plotting failed: {e_plot}')


def plot_omp_selection(*, aic_row, bic_row, coefs_path, context, cv_nmse_mean, cv_nmse_std, cv_row, df_path, feature_names, save_dir, steps, title_prefix):
    try:
        chosen_heat = {
            'CV':  {'index': int(cv_row['step'])  - 1, 'label_val': int(cv_row['step']),  'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': int(aic_row['step']) - 1, 'label_val': int(aic_row['step']), 'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': int(bic_row['step']) - 1, 'label_val': int(bic_row['step']), 'color': context.plasma(0.85), 'linestyle': '-.'},
        }
        plot_activation_heatmap(
            steps, coefs_path, feature_names, chosen_heat,
            title="OMP Activation Path", x_axis_label='OMP Iteration',
            filename=os.path.join(save_dir, f'{title_prefix}_OMP_ActivationHeatmap.pdf'),
            log_x=False, reverse_x=False, is_omp=True,
            context=context,
        )

        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std)
        axs[0].plot(steps, cv_nmse_mean, 'o-', ms=3, color=context.plasma(0.25))
        axs[0].fill_between(steps,
                            np.maximum(0, cv_nmse_mean - cv_nmse_std),
                            np.minimum(1, cv_nmse_mean + cv_nmse_std),
                            alpha=0.2, color=context.plasma(0.25))
        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20, color=context.plasma(0.25))
        axs[0].axvline(int(cv_row['step']), color=context.plasma(0.25), ls='--', lw=1.5)
        axs[0].annotate(f"Iteration = {int(cv_row['step'])}",
                        xy=(int(cv_row['step']), 0.5), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.25))
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([int(steps.min()), int(steps.max())])

        # Row 1: AIC
        axs[1].plot(df_path['step'], df_path['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['step']), color=context.plasma(0.55), ls='-', lw=1.5)
        axs[1].annotate(f"Iteration = {int(aic_row['step'])}",
                        xy=(int(aic_row['step']), np.mean(axs[1].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])
        axs[1].set_xticks([int(steps.min()), int(steps.max())])

        # Row 2: BIC
        axs[2].plot(df_path['step'], df_path['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['step']), color=context.plasma(0.85), ls='-.', lw=1.5)
        axs[2].annotate(f"Iteration = {int(bic_row['step'])}",
                        xy=(int(bic_row['step']), np.mean(axs[2].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])
        axs[2].set_xticks([int(steps.min()), int(steps.max())])

        axs[2].set_xlabel('OMP Iteration', fontsize=context.PLOT_LABEL_FONTSIZE)
        for ax in axs:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_OMP_Criteria.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e_plot:
        print(f'  WARN: OMP plotting failed: {e_plot}')
