"""Cardiac experiment orchestration with the original scenario order."""

from functools import partial
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import datetime
import os
import traceback
from .models import get_invariants as _get_invariants
from .models import get_psi_deriv_for_term as _get_psi_deriv_for_term
from .models import calculate_stress_contribution_linearized as _calculate_stress_contribution_linearized
from .models import calculate_predictions as _calculate_predictions
from mdisc.preprocessing import build_stress_magnitude_weights as _build_stress_magnitude_weights
from .refinement import refine_model as _refine_model
from .plotting import generate_stress_strain_plot_refined as _generate_stress_strain_plot_refined
from .discovery import run_lasso_analysis as _run_lasso_analysis
from .discovery import run_lars_analysis as _run_lars_analysis
from .discovery import run_omp_analysis as _run_omp_analysis
from mdisc.preprocessing import prepare_weighted_normalized_design as _prepare_weighted_normalized_design
from .configuration import create_context
from .data import load_experimental_data
from .models import make_term_library


def run(case_dir=None):
    """Run every original scenario, writing the original log and plot templates."""
    case_dir = Path(case_dir) if case_dir is not None else Path(__file__).resolve().parent
    context = create_context()
    get_invariants = _get_invariants
    get_psi_deriv_for_term = _get_psi_deriv_for_term
    calculate_stress_contribution_linearized = _calculate_stress_contribution_linearized
    calculate_predictions = partial(_calculate_predictions, context=context)
    build_stress_magnitude_weights = partial(_build_stress_magnitude_weights, eps=context.EPS)
    refine_model = partial(_refine_model, context=context)
    generate_stress_strain_plot_refined = partial(_generate_stress_strain_plot_refined, context=context)
    run_lasso_analysis = partial(_run_lasso_analysis, context=context)
    run_lars_analysis = partial(_run_lars_analysis, context=context)
    run_omp_analysis = partial(_run_omp_analysis, context=context)
    prepare_weighted_normalized_design = _prepare_weighted_normalized_design

    experimental_data_list, y_target_original = load_experimental_data(context=context)
    context.psi_terms_definitions = make_term_library()
    context.num_psi_terms = len(context.psi_terms_definitions)
    num_exp_terms = sum(1 for t in context.psi_terms_definitions if t['is_exp'])
    pdf_model_active_indices = [6, 11, 19, 23]
    pdf_model_coeffs_w2 = np.zeros(context.num_psi_terms)
    pdf_model_w1_map = {}
    pdf_model_coeffs_w2[6] = 5.162
    pdf_model_coeffs_w2[11] = 0.081
    pdf_model_w1_map[11] = 21.151
    pdf_model_coeffs_w2[19] = 0.315
    pdf_model_w1_map[19] = 4.371
    pdf_model_coeffs_w2[23] = 0.486
    pdf_model_w1_map[23] = 0.508
    overall_start_time = datetime.datetime.now()
    print(f"\nStarting Anisotropic Model Discovery at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    os.makedirs(context.BASE_SAVE_DIR, exist_ok=True)
    all_results_by_noise = {}
    num_data_points_orig = len(experimental_data_list)
    context.X_features_stage1 = np.zeros((num_data_points_orig, context.num_psi_terms))
    y_target_orig = np.zeros(num_data_points_orig)
    data_point_test_names_orig = []
    valid_rows = np.ones(num_data_points_orig, dtype=bool)
    context.experimental_data_list_filtered = experimental_data_list
    n_samples_clean = num_data_points_orig
    print(f'Building Stage 1 feature matrix (fixed w1=1.0) using original data...')
    for i, data_point in enumerate(experimental_data_list):
        y_target_orig[i] = data_point['target_stress']
        data_point_test_names_orig.append(data_point['test_name'])
        invariants = get_invariants(data_point['F'])
        for j, term_def in enumerate(context.psi_terms_definitions):
            psi_derivs = get_psi_deriv_for_term(term_def, invariants, w1=1.0)
            context.X_features_stage1[i, j] = calculate_stress_contribution_linearized(data_point, psi_derivs, term_def)
    valid_rows = ~np.isnan(context.X_features_stage1).any(axis=1) & ~np.isinf(context.X_features_stage1).any(axis=1) & ~np.isnan(y_target_orig) & ~np.isinf(y_target_orig)
    if np.sum(~valid_rows) > 0:
        print(f'Warning: Removing {np.sum(~valid_rows)} rows with NaN/inf values from original X_features/y_target.')
        context.X_features_stage1 = context.X_features_stage1[valid_rows]
        y_target_original_clean = y_target_orig[valid_rows] # Store the cleaned original target
        data_point_test_names = [data_point_test_names_orig[i] for i, v_row in enumerate(valid_rows) if v_row]
        context.experimental_data_list_filtered = [dp for i, dp in enumerate(experimental_data_list) if valid_rows[i]]
    else:
        y_target_original_clean = y_target_orig
        data_point_test_names = data_point_test_names_orig
        context.experimental_data_list_filtered = experimental_data_list
    n_samples_clean = context.X_features_stage1.shape[0]
    if n_samples_clean == 0:
        exit("CRITICAL ERROR: No valid data points remaining after cleaning feature matrix. Cannot proceed.")
    print(f'Stage 1 Feature matrix built. Cleaned data points: {n_samples_clean}')
    all_invariants_filtered = [get_invariants(dp['F']) for dp in context.experimental_data_list_filtered]
    pdf_model_predictions = calculate_predictions(pdf_model_coeffs_w2, pdf_model_w1_map, all_invariants_filtered, context.experimental_data_list_filtered)
    for noise_level in context.RELATIVE_NOISE_LEVELS:
        noise_prefix = f'RelNoise{noise_level * 100:.0f}pct'
        print("\n" + "#"*80)
        print(f'### PROCESSING NOISE LEVEL: {noise_prefix} ###')
        print("#"*80)

        # Apply noise to the CLEANED original target data
        if noise_level == 0.0:
            y_target_current = y_target_original_clean.copy()
        else:
            noise_scale = noise_level * np.abs(y_target_original_clean)
            noise = np.random.normal(0, noise_scale, size=y_target_original_clean.shape)
            noise[np.abs(y_target_original_clean) < context.WEIGHTING_EPSILON] = 0.0
            y_target_current = y_target_original_clean + noise

        # Update target in filtered list (for Stage 2 objective)
        for i, dp in enumerate(context.experimental_data_list_filtered):
            if i < len(y_target_current):
                dp['target_stress'] = y_target_current[i]

        current_save_dir = os.path.join(context.BASE_SAVE_DIR, noise_prefix)
        os.makedirs(current_save_dir, exist_ok=True)

        stress_based_weights = build_stress_magnitude_weights(context.experimental_data_list_filtered)
        sample_weights_current = stress_based_weights.copy()
        print(f'[{noise_prefix}] Using inverse stress magnitude sample weights.')

        # Build weighted-normalized design used by ALL Stage-1 methods
        X_stage1_weighted, y_stage1_weighted, col_scales = prepare_weighted_normalized_design(
            context.X_features_stage1, y_target_current, sample_weights_current
        )

        # --- Run Stage 1 Analysis on weighted-normalized design (no sample_weight) ---
        scenario_initial_results = {}
        model_selection_results_path = []
        analysis_functions = {'Lasso': run_lasso_analysis, 'LARS': run_lars_analysis, 'OMP': run_omp_analysis}

        for method_base_name, analysis_func in analysis_functions.items():
            results_dict = None
            print("\n" + "="*30 + f' Running {method_base_name} Variants for {noise_prefix} ' + "="*30)
            try:
                if method_base_name == 'Lasso':
                    # Lasso returns results + path data
                    results_dict, lasso_path_data = analysis_func(
                        X_stage1_weighted, y_stage1_weighted, context.psi_terms_definitions, context.CV_FOLDS,
                        noise_prefix, current_save_dir, sample_weight=None
                    )
                    model_selection_results_path = lasso_path_data
                else:
                    results_dict = analysis_func(
                        X_stage1_weighted, y_stage1_weighted, context.psi_terms_definitions, context.CV_FOLDS,
                        noise_prefix, current_save_dir, sample_weight=None
                    )

                # Back-transform coefficients to original X scale for downstream (Stage 2, plotting)
                for k_res, v in results_dict.items():
                    if isinstance(v, dict) and ('coeffs' in v) and (v['coeffs'] is not None):
                        beta_wnorm = v['coeffs']
                        if isinstance(beta_wnorm, np.ndarray) and beta_wnorm.shape[0] == col_scales.shape[0]:
                            v['coeffs'] = beta_wnorm / col_scales
                scenario_initial_results.update(results_dict)

            except Exception as outer_e:
                print(f'FATAL ERROR during {method_base_name} Analysis block: {outer_e}')
                traceback.print_exc()

        # --- Refine and Plot for this noise level ---
        process_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
        noise_level_refined_results = {}

        for method_key in process_order:
            if method_key in scenario_initial_results:
                stage1_result = scenario_initial_results[method_key]
                if 'error' not in stage1_result and stage1_result['coeffs'] is not None:
                    current_coeffs_for_refinement = stage1_result['coeffs'].copy()
                    is_lars_or_omp = any(keyword in method_key for keyword in ['Lars', 'OMP'])

                    if is_lars_or_omp:
                        print(f'  INFO: Applying NNLS pre-refinement for {method_key} coefficients.')
                        selected_feature_indices_stage1 = np.where(np.abs(current_coeffs_for_refinement) > 1e-9)[0]
                        if len(selected_feature_indices_stage1) > 0:
                            # Use ORIGINAL (unweighted) X for NNLS prior to Stage 2
                            X_selected_for_nnls = context.X_features_stage1[:, selected_feature_indices_stage1]
                            try:
                                nnls_model = LinearRegression(positive=True, fit_intercept=False)
                                nnls_model.fit(X_selected_for_nnls, y_target_current, sample_weight=sample_weights_current)
                                nnls_coeffs_full_vector = np.zeros_like(current_coeffs_for_refinement)
                                nnls_coeffs_full_vector[selected_feature_indices_stage1] = nnls_model.coef_
                                nnls_coeffs_full_vector = np.maximum(nnls_coeffs_full_vector, 0)
                                current_coeffs_for_refinement = nnls_coeffs_full_vector
                                n_positive_after_nnls = np.sum(current_coeffs_for_refinement > 1e-6)
                                print(f'    NNLS for {method_key}: {len(selected_feature_indices_stage1)} -> {n_positive_after_nnls} positive coefficients.')
                            except Exception as e:
                                print(f'    WARNING: NNLS pre-refinement failed for {method_key}: {e}. Falling back to clamping.')
                                current_coeffs_for_refinement = np.maximum(current_coeffs_for_refinement, 0)
                        else:
                            print(f'    No features originally selected by {method_key} in Stage 1. Coefficients will be zero for refinement.')
                            current_coeffs_for_refinement = np.zeros_like(current_coeffs_for_refinement)

                    temp_stage1_dict_for_refinement = stage1_result.copy()
                    temp_stage1_dict_for_refinement['coeffs'] = current_coeffs_for_refinement

                    refined_model = refine_model(
                        temp_stage1_dict_for_refinement,
                        f'{noise_prefix} {method_key}',
                        y_target_current,
                        sample_weights_current,   # weights used inside Stage 2 loss
                        all_invariants_filtered
                    )
                    noise_level_refined_results[method_key] = refined_model

                    if refined_model.get('success', False):
                        generate_stress_strain_plot_refined(
                            refined_model['coeffs'], refined_model['w1_map'],
                            pdf_model_predictions, y_target_current,
                            f'{method_key}', save_dir=current_save_dir,
                            file_prefix=f'{noise_prefix}_{method_key}',
                            all_invariants_current=all_invariants_filtered
                        )
                    else:
                        print(f'Skipping plot for {method_key} due to refinement failure or no terms selected.')
                elif 'error' in stage1_result:
                    print(f"Skipping refinement for {method_key} due to Stage 1 error: {stage1_result['error']}")
                    noise_level_refined_results[method_key] = {'success': False, 'message': f"Stage 1 Error: {stage1_result['error']}"}
                else:
                    print(f'Skipping refinement for {method_key} due to missing coefficients in Stage 1 result.')
                    noise_level_refined_results[method_key] = {'success': False, 'message': 'Missing coefficients in Stage 1 result'}
            else:
                print(f'No Stage 1 results found for {method_key} in scenario_initial_results.')
                noise_level_refined_results[method_key] = {'success': False, 'message': 'No Stage 1 results'}

        all_results_by_noise[noise_prefix] = {'initial': scenario_initial_results, 'refined': noise_level_refined_results}
    print("\n\n" + "="*80)
    print("### FINAL DETAILED PERFORMANCE SUMMARY ACROSS NOISE LEVELS ###")
    print("="*80)
    summary_table_data = []
    y_true_eval_for_r2 = y_target_original_clean.copy()
    pdf_metrics_overall = {}
    pdf_metrics_per_mode = {}
    valid_pdf_preds = ~np.isnan(pdf_model_predictions)
    if np.sum(valid_pdf_preds) > 1:
        y_true_valid = y_target_original_clean[valid_pdf_preds] # Compare PDF to original clean data
        y_pred_valid = pdf_model_predictions[valid_pdf_preds]
        try:
            pdf_metrics_overall['R2'] = r2_score(y_true_valid, y_pred_valid)
            pdf_metrics_overall['RMSE'] = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
        except ValueError: pass
    temp_plot_series_pdf = {k: v.copy() for k, v in context.experimental_plot_series.items()}
    pred_map_pdf_only = {}
    for i, dp_filtered in enumerate(context.experimental_data_list_filtered):
        key = (dp_filtered['test_name'], dp_filtered['original_x'])
        if i < len(pdf_model_predictions): pred_map_pdf_only[key] = pdf_model_predictions[i]
    for test_key, series_data in temp_plot_series_pdf.items():
        series_data['y_pred_pdf'] = np.full_like(series_data['x'], np.nan, dtype=float)
        for i_plot_pt in range(len(series_data['x'])):
            map_key = (test_key, series_data['x'][i_plot_pt])
            if map_key in pred_map_pdf_only: series_data['y_pred_pdf'][i_plot_pt] = pred_map_pdf_only[map_key]
        r2, rmse = np.nan, np.nan
        if len(series_data['y_exp']) > 1 and not np.all(np.isnan(series_data['y_pred_pdf'])):
            valid_pred_indices = ~np.isnan(series_data['y_pred_pdf'])
            if np.sum(valid_pred_indices) > 1:
                 try:
                     r2 = r2_score(series_data['y_exp'][valid_pred_indices], series_data['y_pred_pdf'][valid_pred_indices])
                     rmse = np.sqrt(mean_squared_error(series_data['y_exp'][valid_pred_indices], series_data['y_pred_pdf'][valid_pred_indices]))
                 except ValueError: pass
        pdf_metrics_per_mode[test_key] = {'R2': r2, 'RMSE': rmse}
    summary_table_data.append({
        'Noise Level': 'N/A (Reference)', 'Method': 'PDF Model', 'Selection': '-', 'N Terms': len(pdf_model_active_indices),
        'Overall R2': pdf_metrics_overall.get('R2', np.nan), 'Overall RMSE': pdf_metrics_overall.get('RMSE', np.nan),
        **{f'{mode} R2': pdf_metrics_per_mode.get(mode,{}).get('R2',np.nan) for mode in context.experimental_plot_series.keys()},
        **{f'{mode} RMSE': pdf_metrics_per_mode.get(mode,{}).get('RMSE',np.nan) for mode in context.experimental_plot_series.keys()}
    })
    for noise_key, noise_results in all_results_by_noise.items():
        # Use CLEAN target for overall metrics
        y_target_eval = y_true_eval_for_r2

        for method_key, refined_result in noise_results['refined'].items():
            overall_r2, overall_rmse = np.nan, np.nan
            refined_metrics_per_mode = {}
            n_terms_refined = 'FAIL'
            if refined_result.get('success', False):
                n_terms_refined = refined_result.get('k_w2', 0)
                y_pred_refined = calculate_predictions(
                    refined_result['coeffs'], refined_result['w1_map'],
                    all_invariants_filtered, context.experimental_data_list_filtered
                )
                valid_preds = ~np.isnan(y_pred_refined)
                if np.sum(valid_preds) > 1:
                    # Compare to clean target
                    y_true_valid = y_target_eval[valid_preds]
                    y_pred_valid = y_pred_refined[valid_preds]
                    try:
                        overall_r2 = r2_score(y_true_valid, y_pred_valid)
                        overall_rmse = np.sqrt(mean_squared_error(y_true_valid, y_pred_valid))
                    except ValueError:
                        pass

                # Calculate per-mode metrics for refined model (comparing to ORIGINAL y_exp)
                temp_plot_series_ref = {k: v.copy() for k, v in context.experimental_plot_series.items()}
                pred_map_ref = {}
                for i, dp_filtered in enumerate(context.experimental_data_list_filtered):
                    key = (dp_filtered['test_name'], dp_filtered['original_x'])
                    if i < len(y_pred_refined): pred_map_ref[key] = y_pred_refined[i]
                for test_key, series_data in temp_plot_series_ref.items():
                    series_data['y_pred_ref'] = np.full_like(series_data['x'], np.nan, dtype=float)
                    for i_plot_pt in range(len(series_data['x'])):
                        map_key = (test_key, series_data['x'][i_plot_pt])
                        if map_key in pred_map_ref: series_data['y_pred_ref'][i_plot_pt] = pred_map_ref[map_key]
                    r2, rmse = np.nan, np.nan
                    if len(series_data['y_exp']) > 1 and not np.all(np.isnan(series_data['y_pred_ref'])):
                        valid_pred_indices = ~np.isnan(series_data['y_pred_ref'])
                        if np.sum(valid_pred_indices) > 1:
                             try:
                                 r2 = r2_score(series_data['y_exp'][valid_pred_indices], series_data['y_pred_ref'][valid_pred_indices])
                                 rmse = np.sqrt(mean_squared_error(series_data['y_exp'][valid_pred_indices], series_data['y_pred_ref'][valid_pred_indices]))
                             except ValueError: pass
                    refined_metrics_per_mode[test_key] = {'R2': r2, 'RMSE': rmse}

            summary_table_data.append({
                'Noise Level': noise_key,
                'Method': method_key.replace('Best','').replace('Lasso','').replace('Lars','').replace('OMP',''),
                'Selection': method_key.replace(method_key.replace('Best','').replace('Lasso','').replace('Lars','').replace('OMP',''),''),
                'N Terms': n_terms_refined,
                'Overall R2': overall_r2,
                'Overall RMSE': overall_rmse,
                **{f'{mode} R2': refined_metrics_per_mode.get(mode,{}).get('R2',np.nan) for mode in context.experimental_plot_series.keys()},
                **{f'{mode} RMSE': refined_metrics_per_mode.get(mode,{}).get('RMSE',np.nan) for mode in context.experimental_plot_series.keys()}
            })
    summary_df_final = pd.DataFrame(summary_table_data)
    overall_cols = ['Noise Level', 'Method', 'Selection', 'N Terms', 'Overall R2', 'Overall RMSE']
    mode_cols_r2 = sorted([f'{mode} R2' for mode in context.experimental_plot_series.keys()])
    mode_cols_rmse = sorted([f'{mode} RMSE' for mode in context.experimental_plot_series.keys()])
    final_cols = overall_cols + mode_cols_r2 + mode_cols_rmse
    for col in final_cols:
        if col not in summary_df_final.columns:
            summary_df_final[col] = np.nan
    summary_df_final = summary_df_final[final_cols]
    print("\n\n--- FINAL DETAILED PERFORMANCE SUMMARY TABLE ---")
    with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000):
         print(summary_df_final.to_string(index=False, float_format="%.3f", na_rep="N/A"))
    print(f"\nCompleted all scenarios at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Total analysis duration: {datetime.datetime.now() - overall_start_time}')
    return {"noise_results": all_results_by_noise, "summary": summary_table_data}
