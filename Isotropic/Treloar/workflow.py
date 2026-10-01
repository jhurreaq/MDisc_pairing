"""Treloar experiment orchestration with the original scenario order."""

from functools import partial
from pathlib import Path
import numpy as np
import pandas as pd
from mdisc.preprocessing import fit_standard_scaler
import datetime
import os
import traceback
from .models import construct_design_matrix as _construct_design_matrix
from .plotting import plot_model_predictions as _plot_model_predictions
from .plotting import plot_combined_datasets as _plot_combined_datasets
from .discovery import run_lasso_analysis as _run_lasso_analysis
from .discovery import run_lars_analysis as _run_lars_analysis
from .discovery import run_omp_analysis as _run_omp_analysis
from .data import combine_datasets as _combine_datasets
from .configuration import create_context


def run(case_dir=None):
    """Run every original scenario, writing the original log and plot templates."""
    case_dir = Path(case_dir) if case_dir is not None else Path(__file__).resolve().parent
    context = create_context()
    construct_design_matrix = partial(_construct_design_matrix, context=context)
    plot_model_predictions = partial(_plot_model_predictions, context=context)
    plot_combined_datasets = partial(_plot_combined_datasets, context=context)
    run_lasso_analysis = partial(_run_lasso_analysis, context=context)
    run_lars_analysis = partial(_run_lars_analysis, context=context)
    run_omp_analysis = partial(_run_omp_analysis, context=context)
    combine_datasets = partial(_combine_datasets, context=context)

    KAWABATA_FILE_PATH = 'input/Kawabata.csv'
    EMPHASIZE_SMALL_STRAIN = False
    CV_FOLDS = 5
    BASE_SAVE_DIR = "outputs"
    COEFF_THRESHOLD = 1e-6
    RIDGE_ALPHA_REFIT = 1e-6
    context.WEIGHT_FACTOR = 2.0
    context.SMALL_STRAIN_THRESHOLD = 50.0
    fitting_scenarios = [
        ("Fit_Treloar_Only", True, False)
    ]
    MODEL_CONFIG_NAME = "MR3_OgdenExt"
    context.INCLUDE_NEOHOOKEAN = False
    context.INCLUDE_MOONEY_RIVLIN = True
    INCLUDE_YEOH = False
    INCLUDE_ARRUDA_BOYCE = False
    context.INCLUDE_OGDEN = True
    context.INCLUDE_EXTENDED_OGDEN = True
    context.OGDEN_EXPONENTS = []
    INCLUDE_GENT = False
    INCLUDE_DATA_DRIVEN = False
    INCLUDE_FRACTIONAL_POWERS = False
    INCLUDE_LOW_STRAIN_TERMS = False
    INCLUDE_ULTRA_SMALL_STRAIN = False
    context.REFIT_REGRESSION_TYPE = f'Ridge(α={RIDGE_ALPHA_REFIT:.1e})' if RIDGE_ALPHA_REFIT > 0 else "OLS"
    overall_start_time = datetime.datetime.now()
    print(f"Starting Sparse Regression Analysis (Fit Subset / Evaluate Full) at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Using Model Library: {MODEL_CONFIG_NAME}')
    print(f'Base Save Directory: {BASE_SAVE_DIR}')
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    print("\n--- Loading Full Combined Dataset (for evaluation structure) ---")
    treloar_ut_exists = os.path.exists("input/TreloarDataUT.csv")
    treloar_ps_exists = os.path.exists("input/TreloarDataPS.csv")
    treloar_ebt_exists = os.path.exists("input/TreloarDataEBT.csv")
    kawabata_exists = os.path.exists(KAWABATA_FILE_PATH)
    can_load_treloar = treloar_ut_exists or treloar_ps_exists or treloar_ebt_exists
    can_load_kawabata = kawabata_exists
    if not can_load_treloar and not can_load_kawabata:
         print("FATAL ERROR: Neither Treloar nor Kawabata data files found. Cannot proceed.")
         exit()
    try:
        df_full_combined = combine_datasets(
            include_treloar=can_load_treloar,
            include_Kawabata=can_load_kawabata,
            Kawabata_path=KAWABATA_FILE_PATH
        )
        if df_full_combined.empty:
             raise ValueError("Full combined dataset is empty despite file existence checks.")
        print("--- Full Combined Dataset Loaded Successfully ---")
        # Plot the combined raw data once for reference
        plot_combined_datasets(df_full_combined, BASE_SAVE_DIR, "Combined_RawData_Reference")

    except Exception as e:
        print(f'FATAL ERROR: Could not load the full combined dataset: {e}')
        print("Cannot proceed without the full dataset structure for evaluation.")
        traceback.print_exc()
        exit()
    all_initial_results = {}
    all_refit_summaries = {}
    for fitting_name, use_treloar_fit, use_kawabata_fit in fitting_scenarios:

        print("\n" + "#"*80)
        print(f'### PROCESSING FITTING SCENARIO: {fitting_name} ###')
        print(f'### Model Library: {MODEL_CONFIG_NAME} ###')
        print(f'### Refit Method (on subset): {context.REFIT_REGRESSION_TYPE}, Threshold={COEFF_THRESHOLD:.1e} ###')
        print(f'### Weight Factor: {context.WEIGHT_FACTOR}, Small Strain Thresh: {context.SMALL_STRAIN_THRESHOLD}% ###')
        print("#"*80 + "\n")

        current_save_dir = os.path.join(BASE_SAVE_DIR, fitting_name)
        os.makedirs(current_save_dir, exist_ok=True)
        # Prefix includes both fitting scenario and model config
        scenario_prefix = f'{fitting_name}_{MODEL_CONFIG_NAME}'

        # --- Create Fitting Subset Data ---
        print(f'--- Creating data subset for fitting: Treloar={use_treloar_fit}, Kawabata={use_kawabata_fit} ---')
        df_fitting_subset = pd.DataFrame() # Initialize empty DataFrame

        # Check if data exists in the loaded full set before trying to include it
        treloar_available = 'Treloar' in df_full_combined['dataset'].unique()
        kawabata_available = 'Kawabata' in df_full_combined['dataset'].unique()

        if use_treloar_fit and treloar_available:
             df_fitting_subset = pd.concat([df_fitting_subset, df_full_combined[df_full_combined['dataset'] == 'Treloar']], ignore_index=True)
             print("  Included Treloar data for fitting.")
        elif use_treloar_fit and not treloar_available:
             print("  WARN: Requested Treloar for fit, but it wasn't loaded/available in full dataset.")

        if use_kawabata_fit and kawabata_available:
             df_fitting_subset = pd.concat([df_fitting_subset, df_full_combined[df_full_combined['dataset'] == 'Kawabata']], ignore_index=True)
             print("  Included Kawabata data for fitting.")
        elif use_kawabata_fit and not kawabata_available:
             print("  WARN: Requested Kawabata for fit, but it wasn't loaded/available in full dataset.")


        if df_fitting_subset.empty:
            print(f"ERROR: Fitting subset for scenario '{fitting_name}' is empty. Skipping this scenario.")
            all_initial_results[fitting_name] = {} # Mark as empty/failed
            all_refit_summaries[fitting_name] = {}
            continue # Skip to the next fitting scenario

        print(f'  Total data points in fitting subset: {len(df_fitting_subset)}')

        # --- Construct Matrix, Scale (using ONLY fitting subset) ---
        basis_names = [] # Ensure scope
        try:
            print(f'\nGenerating Model Library & Constructing Design Matrix from Fitting Subset...')
            # Generate basis functions based on global config (done within construct_design_matrix)
            # The global flags determine the library structure used here.
            design_matrix_weighted_fit, target_vector_weighted_fit, basis_names, _, df_subset_info = construct_design_matrix(
                 df_fitting_subset, # Use the fitting subset here
                 emphasize_small_strain=EMPHASIZE_SMALL_STRAIN,
                 weight_factor=context.WEIGHT_FACTOR,
                 small_strain_threshold=context.SMALL_STRAIN_THRESHOLD
            )

            if design_matrix_weighted_fit.shape[0] == 0:
                raise ValueError("Weighted design matrix for fitting is empty.")
            if not basis_names:
                 raise ValueError("Basis names list is empty after matrix construction.")
            if len(basis_names) != design_matrix_weighted_fit.shape[1]:
                raise ValueError(f'Basis ({len(basis_names)}) / Matrix column ({design_matrix_weighted_fit.shape[1]}) mismatch for fitting matrix.')

            print(f'Weighted design matrix for FITTING created with shape: {design_matrix_weighted_fit.shape}')

            # Scale the fitting data - SCALE BOTH X AND Y
            print("Scaling fitting data (both X and y)...")
            scaler_X = fit_standard_scaler(design_matrix_weighted_fit)
            X_weighted_scaled_fit = scaler_X.transform(design_matrix_weighted_fit)
            scaler_y = fit_standard_scaler(target_vector_weighted_fit.reshape(-1, 1))
            target_vector_scaled_fit = scaler_y.transform(target_vector_weighted_fit.reshape(-1, 1)).flatten()

            if not np.all(np.isfinite(X_weighted_scaled_fit)):
                print("WARN: Non-finite values after scaling fitting data. Applying nan_to_num.")
                X_weighted_scaled_fit = np.nan_to_num(X_weighted_scaled_fit)

            print(f'Fitting data prepared and scaled successfully. X shape: {X_weighted_scaled_fit.shape}, y scaled variance: {np.var(target_vector_scaled_fit):.4f}')


        except Exception as e:
            print(f"ERROR in data preparation for fitting scenario '{fitting_name}': {e}")
            traceback.print_exc()
            all_initial_results[fitting_name] = {} # Mark as failed
            all_refit_summaries[fitting_name] = {}
            continue # Skip to the next scenario

        # --- Run Analyses (on fitting subset data) ---
        scenario_initial_results = {}
        scenario_refit_summary = {} # Initialize summary dict for this scenario

        analysis_functions = {
            'Lasso': run_lasso_analysis,
            'LARS': run_lars_analysis,
            'OMP': run_omp_analysis
        }

        for method_base_name, analysis_func in analysis_functions.items():
            results_dict = None
            print("\n" + "="*30 + f" Running {method_base_name} Variants on '{fitting_name}' Data " + "="*30)
            try:
                # Pass the SCALED FITTING data to the analysis functions
                results_dict = analysis_func(
                    X_weighted_scaled_fit,      # Scaled matrix from fitting subset
                    # target_vector_weighted_fit, # Target vector from fitting subset
                    target_vector_scaled_fit,
                    basis_names,                # Full basis names list
                    CV_FOLDS,
                    scenario_prefix,            # Prefix includes fitting scenario
                    current_save_dir
                )
                scenario_initial_results.update(results_dict) # Add results from this method (CV, AIC, BIC)
            except Exception as outer_e:
                print(f"FATAL ERROR during {method_base_name} Analysis block for '{fitting_name}': {outer_e}")
                traceback.print_exc()
                # Mark corresponding results as errored
                keys_to_add = []
                if method_base_name == 'Lasso': keys_to_add = ['LassoCV', 'LassoAIC', 'LassoBIC']
                elif method_base_name == 'LARS': keys_to_add = ['LarsCV', 'LarsAIC', 'LarsBIC']
                elif method_base_name == 'OMP': keys_to_add = ['OMPCV', 'OMPAIC', 'OMPBIC']
                for key in keys_to_add:
                    if key not in scenario_initial_results: scenario_initial_results[key] = {'error': str(outer_e)}
                    # Also mark refit summary as failed upfront
                    if key not in scenario_refit_summary: scenario_refit_summary[key] = {'consistent': None, 'notes': [f'Fatal Error in initial fit: {outer_e}']}

        # --- Post-process ALL results for this fitting scenario ---
        # Refit on subset, Evaluate/Plot on full combined set
        print("\n" + "="*40 + f' Post-processing All Models for Fitting Scenario: {fitting_name} ' + "="*40)
        print(f'--- Refitting on FIT SUBSET, Evaluating on FULL Combined Dataset ---')

        process_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
        for method_key in process_order:
            if method_key in scenario_initial_results:
                result_data = scenario_initial_results[method_key]
                print(f"\n--- Post-processing Result for: {method_key} (fitted on '{fitting_name}') ---")
                # Initialize refit summary entry if not already added by error handling above
                if method_key not in scenario_refit_summary:
                     scenario_refit_summary[method_key] = {'consistent': None, 'notes': ['Post-processing not performed']}

                # Proceed only if initial fit was successful
                if 'error' not in result_data and 'coeffs' in result_data:
                    n_nonzero_initial = result_data.get('n_nonzero', np.sum(np.abs(result_data['coeffs']) > context.EPS))
                    if n_nonzero_initial > 0:
                        try:
                            (refit_terms, refit_coeffs, r2_p11, rmse_p11, r2_p22, rmse_p22,
                             is_consistent, consistency_notes, mode_metrics) = plot_model_predictions(
                                 df_fitting_subset,     # FIT SUBSET for refit scope
                                 df_full_combined,      # FULL SET for prediction/plot scope
                                 result_data['coeffs'], # Initial coeffs from sparse fit on subset
                                 basis_names,           # Full list of basis names
                                 method_key,            # Identifier (e.g., "LassoCV")
                                 scenario_prefix,       # File prefix (includes fitting scenario)
                                 current_save_dir,      # Save directory
                                 weight_factor=context.WEIGHT_FACTOR,
                                 small_strain_threshold=context.SMALL_STRAIN_THRESHOLD,
                                 coeff_threshold=COEFF_THRESHOLD,
                                 ridge_alpha=RIDGE_ALPHA_REFIT
                              )
                            # Store the evaluation results (metrics are now based on full set)
                            scenario_refit_summary[method_key] = {
                                'terms': refit_terms, 'coeffs': refit_coeffs, # Coeffs from subset refit
                                'r2_p11': r2_p11, 'rmse_p11': rmse_p11,       # Metrics on full set
                                'r2_p22': r2_p22, 'rmse_p22': rmse_p22,       # Metrics on full set
                                'consistent': is_consistent, 'notes': consistency_notes,
                                'mode_metrics': mode_metrics  # NEW: Store mode-specific metrics
                             }
                        except Exception as plot_e:
                            print(f"ERROR during evaluation/plotting for {method_key} (fitted on '{fitting_name}'): {plot_e}")
                            traceback.print_exc()
                            # Store error info in refit summary
                            scenario_refit_summary[method_key]['error_postprocess'] = str(plot_e)
                            scenario_refit_summary[method_key]['notes'] = [f'Plotting/Evaluation Error: {plot_e}']
                            # Keep consistent=None to indicate failure
                    else:
                        print(f"Skipping evaluation/plot for {method_key}: No features initially selected during fit on '{fitting_name}'.")
                        scenario_refit_summary[method_key] = {'consistent': None, 'notes': ['No features selected in initial fit']}
                else:
                    # If there was an initial fit error, ensure summary reflects it
                    print(f"Skipping evaluation/plot for {method_key}: Initial fit on '{fitting_name}' failed.")
                    err_msg = result_data.get('error', 'Unknown initial fit error')
                    # Ensure the summary entry exists and contains the error note
                    if method_key not in scenario_refit_summary: scenario_refit_summary[method_key] = {}
                    scenario_refit_summary[method_key]['consistent'] = None
                    scenario_refit_summary[method_key]['notes'] = [f'Initial fit error: {err_msg}']


        # --- Store results for this fitting scenario ---
        all_initial_results[fitting_name] = scenario_initial_results
        all_refit_summaries[fitting_name] = scenario_refit_summary
        print(f'\n### Finished All Analyses & Post-processing for Fitting Scenario: {fitting_name} ###\n')
    print("\n" + "="*80)
    print("### INITIAL SPARSE SELECTION SUMMARY (Fit on Scenario Subset, Weighted) ###")
    print("="*80)
    for fitting_scenario_name, methods_results in all_initial_results.items():
        print(f'\n--- Fitting Scenario Used: {fitting_scenario_name} ---')
        if not methods_results:
             print("| No successful initial fits performed for this scenario. |")
             continue
        print("| Method      | Optimal Param    | Non-Zero | AIC       | BIC       | Time (s) |")
        print("|-------------|------------------|----------|-----------|-----------|----------|")
        print_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
        for method_key in print_order:
            if method_key in methods_results:
                res = methods_results[method_key]
                method_print_key = method_key
                if 'error' in res:
                    time_str = f"{res.get('time', np.nan):<8.2f}" if pd.notna(res.get('time', np.nan)) else "N/A"
                    error_msg_short = str(res.get('error', 'Unknown Error'))[:15].strip()
                    print(f'| {method_print_key:<11} | ERROR ({error_msg_short}...) | ---      | ---       | ---       | {time_str:<8} |')
                else:
                    n_nonzero = res.get('n_nonzero', 'N/A'); aic_str = f"{res.get('aic', np.nan):<9.2e}" if pd.notna(res.get('aic', np.nan)) else 'N/A'; bic_str = f"{res.get('bic', np.nan):<9.2e}" if pd.notna(res.get('bic', np.nan)) else 'N/A'; time_str = f"{res.get('time', np.nan):<8.4f}" if pd.notna(res.get('time', np.nan)) else 'N/A'
                    opt_param_details = [];
                    if 'CV' in method_key: opt_param_details.append("CV")
                    if 'AIC' in method_key: opt_param_details.append("AIC")
                    if 'BIC' in method_key: opt_param_details.append("BIC")
                    if 'alpha' in res and pd.notna(res['alpha']): opt_param_details.append(f"α≈{res['alpha']:.1e}")
                    if 'optimal_k' in res and pd.notna(res['optimal_k']): opt_param_details.append(f"k={int(res['optimal_k'])}")
                    elif 'n_features_opt' in res and pd.notna(res['n_features_opt']): opt_param_details.append(f"k={int(res['n_features_opt'])}")
                    opt_param_str = " ".join(opt_param_details) if opt_param_details else "N/A"
                    print(f'| {method_print_key:<11} | {opt_param_str:<16} | {n_nonzero:<8} | {aic_str:<9} | {bic_str:<9} | {time_str:<8} |')
    print("\n" + "="*80)
    print(f'### FINAL MODEL EVALUATION SUMMARY (Refit: {context.REFIT_REGRESSION_TYPE} on Fit Subset, Threshold={COEFF_THRESHOLD:.1e}) ###')
    print("### Metrics Calculated on FULL UNWEIGHTED Combined Data | Consistency Check (μ₀ > 0) ###")
    print("="*80)
    for fitting_scenario_name, methods_summary in all_refit_summaries.items():
        print(f'\n--- Fitting Scenario Used: {fitting_scenario_name} ---')
        if not methods_summary:
            print("| No successful evaluations or refits performed for this scenario. |")
            continue
        print("| Method      | N Terms | R² (P11) | RMSE (P11) | R² (P22) | RMSE (P22) | Consistent |")
        print("|-------------|---------|----------|------------|----------|------------|------------|")
        print_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
        for method_key in print_order:
             if method_key in methods_summary:
                 summary = methods_summary[method_key]
                 method_print_key = method_key
                 if 'consistent' not in summary or summary['consistent'] is None:
                     # Handle cases where evaluation didn't happen
                     consistent_str = "N/A"
                     n_terms = "---"
                     r2_p11_str, rmse_p11_str = "---", "---"
                     r2_p22_str, rmse_p22_str = "---", "---"
                     notes = summary.get('notes', ['Evaluation not performed or failed'])
                     note_summary = notes[0][:70] # Truncate long notes
                     print(f'| {method_print_key:<11} | {n_terms:<7} | {r2_p11_str:<8} | {rmse_p11_str:<10} | {r2_p22_str:<8} | {rmse_p22_str:<10} | {consistent_str:<10} |')
                     print(f'|           -> Note: {note_summary}')
                     print("|" + "-"*78 + "|")
                 else:
                     # Normal case - evaluation was successful
                     n_terms = len(summary.get('terms', []))
                     r2_p11_str = f"{summary.get('r2_p11', np.nan):.4f}" if pd.notna(summary.get('r2_p11', np.nan)) else "N/A"
                     rmse_p11_str = f"{summary.get('rmse_p11', np.nan):.4f}" if pd.notna(summary.get('rmse_p11', np.nan)) else "N/A"
                     r2_p22_str = f"{summary.get('r2_p22', np.nan):.4f}" if pd.notna(summary.get('r2_p22', np.nan)) else "N/A"
                     rmse_p22_str = f"{summary.get('rmse_p22', np.nan):.4f}" if pd.notna(summary.get('rmse_p22', np.nan)) else "N/A"
                     consistent_flag = summary.get('consistent', False)
                     consistent_str = "OK" if consistent_flag is True else "FAIL"
                     print(f'| {method_print_key:<11} | {n_terms:<7} | {r2_p11_str:<8} | {rmse_p11_str:<10} | {r2_p22_str:<8} | {rmse_p22_str:<10} | {consistent_str:<10} |')
                     if summary.get('terms'):
                         print(f'|  -> Final Model Terms & Coefficients (from subset refit):')
                         max_terms_to_print = 10 # Show fewer terms in summary table
                         for i, (term, coeff) in enumerate(zip(summary['terms'], summary['coeffs'])):
                             if i < max_terms_to_print:
                                 # Use format_term_for_latex for better display if needed, or keep raw
                                 term_display = term # format_term_for_latex(term).strip('$') # Optional LaTeX formatting
                                 print(f'|      - {term_display:<40}: {coeff:.6f}')
                             elif i == max_terms_to_print:
                                 print(f"|      ... (remaining {len(summary['terms']) - max_terms_to_print} terms not shown)")
                         notes = summary.get('notes', ['N/A'])
                         mu0_note = next((note for note in notes if 'μ₀ ≈' in note), 'μ₀ calculation note N/A')
                         consistency_note = next((note for note in notes if 'CONSISTENCY' in note), 'Consistency note N/A')
                         print(f'|  -> {mu0_note.strip()}')
                         print(f'|  -> {consistency_note.strip()}')
                     else:
                         print("|  -> (No terms remaining after thresholding/refit)")
                     print("|" + "-"*78 + "|")
    print("\n" + "="*80)
    print("### MODE-SPECIFIC R² EVALUATION SUMMARY ###")
    print("="*80)
    for fitting_scenario_name, methods_summary in all_refit_summaries.items():
        print(f'\n--- Fitting Scenario Used: {fitting_scenario_name} ---')
        if not methods_summary:
            print("| No mode-specific evaluations available for this scenario. |")
            continue

        print("| Method      | UT R²(P11) | PS R²(P11) | EBT R²(P11) | EBT R²(P22) | BIAX R²(P11) | BIAX R²(P22) |")
        print("|-------------|------------|------------|-------------|-------------|--------------|--------------|")

        print_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
        for method_key in print_order:
            if method_key in methods_summary and 'mode_metrics' in methods_summary[method_key]:
                mode_metrics = methods_summary[method_key]['mode_metrics']
                method_print_key = method_key

                # Extract R² values for each mode
                ut_r2 = f"{mode_metrics.get('UT', {}).get('r2_p11', np.nan):.3f}" if pd.notna(mode_metrics.get('UT', {}).get('r2_p11', np.nan)) else "N/A"
                ps_r2 = f"{mode_metrics.get('PS', {}).get('r2_p11', np.nan):.3f}" if pd.notna(mode_metrics.get('PS', {}).get('r2_p11', np.nan)) else "N/A"
                ebt_r2_p11 = f"{mode_metrics.get('EBT', {}).get('r2_p11', np.nan):.3f}" if pd.notna(mode_metrics.get('EBT', {}).get('r2_p11', np.nan)) else "N/A"
                ebt_r2_p22 = f"{mode_metrics.get('EBT', {}).get('r2_p22', np.nan):.3f}" if pd.notna(mode_metrics.get('EBT', {}).get('r2_p22', np.nan)) else "N/A"
                biax_r2_p11 = f"{mode_metrics.get('BIAX', {}).get('r2_p11', np.nan):.3f}" if pd.notna(mode_metrics.get('BIAX', {}).get('r2_p11', np.nan)) else "N/A"
                biax_r2_p22 = f"{mode_metrics.get('BIAX', {}).get('r2_p22', np.nan):.3f}" if pd.notna(mode_metrics.get('BIAX', {}).get('r2_p22', np.nan)) else "N/A"

                print(f'| {method_print_key:<11} | {ut_r2:<10} | {ps_r2:<10} | {ebt_r2_p11:<11} | {ebt_r2_p22:<11} | {biax_r2_p11:<12} | {biax_r2_p22:<12} |')

        # Add sample size information
        print("\n| Sample Sizes by Mode:")
        for method_key in ['LassoCV']:  # Just pick one method to show sample sizes (they're the same for all)
            if method_key in methods_summary and 'mode_metrics' in methods_summary[method_key]:
                mode_metrics = methods_summary[method_key]['mode_metrics']
                for mode in ['UT', 'PS', 'EBT', 'BIAX']:
                    if mode in mode_metrics:
                        n_p11 = mode_metrics[mode]['n_points_p11']
                        n_p22 = mode_metrics[mode]['n_points_p22']
                        if n_p11 > 0 or n_p22 > 0:
                            p11_str = f'P11: n={n_p11}' if n_p11 > 0 else ""
                            p22_str = f'P22: n={n_p22}' if n_p22 > 0 else ""
                            both_str = f'{p11_str}, {p22_str}' if p11_str and p22_str else (p11_str or p22_str)
                            print(f'|   {mode}: {both_str}')
                break
    print(f"\nCompleted all scenarios at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f'Total analysis duration: {datetime.datetime.now() - overall_start_time}')
    print("="*80)
    return {"initial_results": all_initial_results, "refit_summaries": all_refit_summaries}
