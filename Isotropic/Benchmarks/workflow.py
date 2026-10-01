"""Synthetic experiment orchestration with the original scenario order."""

from functools import partial
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mdisc.preprocessing import fit_standard_scaler
import datetime
import os
import traceback
from .models import generate_model_library as _generate_model_library
from .models import construct_design_matrix as _construct_design_matrix
from .data import generate_synth_data_via_framework as _generate_synth_data_via_framework
from .plotting import plot_model_predictions as _plot_model_predictions
from .discovery import run_lasso_analysis as _run_lasso_analysis
from .discovery import run_lars_analysis as _run_lars_analysis
from .discovery import run_omp_analysis as _run_omp_analysis
from .configuration import create_context


def run(case_dir=None):
    """Run every original scenario, writing the original log and plot templates."""
    case_dir = Path(case_dir) if case_dir is not None else Path(__file__).resolve().parent
    context = create_context()
    generate_model_library = partial(_generate_model_library, context=context)
    construct_design_matrix = partial(_construct_design_matrix, context=context)
    generate_synth_data_via_framework = partial(_generate_synth_data_via_framework, context=context)
    plot_model_predictions = partial(_plot_model_predictions, context=context)
    run_lasso_analysis = partial(_run_lasso_analysis, context=context)
    run_lars_analysis = partial(_run_lars_analysis, context=context)
    run_omp_analysis = partial(_run_omp_analysis, context=context)

    script_dir = str(case_dir)
    outputs_root = os.path.join(script_dir, "outputs")
    BASE_SAVE_DIR = outputs_root
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)
    log_file_name = os.path.join(
        BASE_SAVE_DIR,
        "log_clean.txt"
    )
    original_stdout = sys.stdout
    log_file_handle = None
    try:
        # Redirect stdout to the log file
        log_file_handle = open(log_file_name, 'w')
        sys.stdout = log_file_handle

        # Configuration
        RANDOM_SEED = 42 # Master seed for all random operations
        np.random.seed(RANDOM_SEED) # Seed NumPy's random number generator
        print(f'INFO: Using global random seed: {RANDOM_SEED}')

        CV_FOLDS = 5
        # BASE_SAVE_DIR is already defined relative to the script directory above
        context.COEFF_THRESHOLD = 1e-6
        context.RIDGE_ALPHA_REFIT = 1e-7

        SYNTH_N_POINTS = 60
        SYNTH_LAMBDA_RANGE = (0.6, 5.0)
        SYNTH_GAMMA_RANGE = (-1.0, 1.0)

        SYNTH_RELATIVE_NOISE_LEVELS = [0.0, 0.05, 0.10]

        # MR_MAX_ORDER = 3
        TARGET_OGDEN_EXPONENTS_INT = [-10, -5, 5, 1, -1]
        STANDARD_OGDEN_EXPONENTS = [-9, -8, -7, -6, -4, -3, 3, 4, 5, 6, 7, 8, 9, 10]
        FULL_OGDEN_DISCOVERY_SET = sorted(list(set(STANDARD_OGDEN_EXPONENTS + TARGET_OGDEN_EXPONENTS_INT)))

        print(f'Potential Ogden Exponents for Discovery Libraries: {FULL_OGDEN_DISCOVERY_SET}')

        # synthetic_model_definitions = {
        #     "O1": { "Ogden: λ^-10": 2.0 },
        #     "O2": { "Ogden: λ^-5": 16.0, "Ogden: λ^5": 8.0 },
        #     "MR2": { "MR: (I₁-3)": 40.0, "MR: (I₂-3)": 20.0 },
        #     "MR1O1": { "MR: (I₂-3)^2": 30.0, "Ogden: λ^-10": 2.0 }
        # }

        synthetic_model_definitions = {
            "O2": { "Ogden: λ^-3": 16.0, "Ogden: λ^3": 8.0 },
            "MR2": { "MR: (I₁-3)": 40.0, "MR: (I₂-3)": 20.0 },
            "MR1O1": { "MR: (I₂-3)": 40.0, "Ogden: λ^-3": 8.0 },
            "MR2O2": { "MR: (I₁-3)": 40.0, "MR: (I₂-3)": 20.0, "Ogden: λ^-3": 16.0, "Ogden: λ^1": 800.0 } #"Ogden: λ^-3": 10, "Ogden: λ^1": 500 }
        }

        context.REFIT_REGRESSION_TYPE = f'Ridge(α={context.RIDGE_ALPHA_REFIT:.1e})' if context.RIDGE_ALPHA_REFIT > 0 else "OLS"

        overall_start_time = datetime.datetime.now()

        print(f"\nStarting Synthetic Data Rediscovery (UT+PS+EBT P11 Fit, Relative Noise) at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)

        all_initial_results = {}
        all_refit_summaries = {}

        for synth_model_name, term_coeff_map_Pa in synthetic_model_definitions.items():
            for rel_noise in SYNTH_RELATIVE_NOISE_LEVELS:
                scenario_name = f'{synth_model_name}_RelNoise{rel_noise * 100:.0f}pct'
                print("\n" + "#"*80)
                print(f'### PROCESSING SCENARIO: {scenario_name} (Fit UT+PS+EBT P11) ###')

                print(f'--- Configuring Discovery Library for Ground Truth: {synth_model_name} ---')
                context.INCLUDE_NEOHOOKEAN=False; context.INCLUDE_MOONEY_RIVLIN=False; INCLUDE_YEOH=False;
                INCLUDE_ARRUDA_BOYCE=False; context.INCLUDE_OGDEN=False; INCLUDE_GENT=False;
                INCLUDE_DATA_DRIVEN=False; INCLUDE_FRACTIONAL_POWERS=False;
                INCLUDE_LOW_STRAIN_TERMS=False; INCLUDE_ULTRA_SMALL_STRAIN=False;
                context.OGDEN_EXPONENTS = []

                if synth_model_name in ["MR2", "MR3"]:
                    MODEL_CONFIG_NAME = "MR_Lib_Only"
                    context.INCLUDE_MOONEY_RIVLIN = True; #MR_MAX_ORDER = 4
                elif synth_model_name in ["O1", "O2"]:
                    MODEL_CONFIG_NAME = "Ogden_Lib_Only"
                    context.INCLUDE_OGDEN = True; context.OGDEN_EXPONENTS = FULL_OGDEN_DISCOVERY_SET
                elif synth_model_name in ["MR1O1", "MR2O1", "MR2O2"]:
                    MODEL_CONFIG_NAME = "MR_Ogden_Lib"
                    context.INCLUDE_MOONEY_RIVLIN = True; #MR_MAX_ORDER = 4
                    context.INCLUDE_OGDEN = True; context.OGDEN_EXPONENTS = FULL_OGDEN_DISCOVERY_SET
                else:
                    print(f"ERROR: Unknown synth_model_name '{synth_model_name}'. Skipping."); continue

                print(f'Discovery Library Configured: {MODEL_CONFIG_NAME}')
                CURRENT_BASIS_NAMES = generate_model_library()

                if not CURRENT_BASIS_NAMES: print(f'ERROR: Failed to generate library for {scenario_name}. Skipping.'); continue
                print(f'Generated {len(CURRENT_BASIS_NAMES)} terms for discovery.')

                all_terms_found = True; print("Checking if synthetic model terms exist in the generated library...")
                for term in term_coeff_map_Pa.keys():
                    if term not in CURRENT_BASIS_NAMES:
                        print(f"ERROR: Term '{term}' needed for model '{synth_model_name}' is MISSING from library '{MODEL_CONFIG_NAME}'!"); all_terms_found = False
                    else: print(f"  OK: Term '{term}' found.")
                if not all_terms_found: print("\nCannot proceed."); continue

                # Collapse path: only outputs/<scenario>, no extra subfolder per library
                current_save_dir = os.path.join(BASE_SAVE_DIR, scenario_name)
                os.makedirs(current_save_dir, exist_ok=True)
                file_prefix = f'{scenario_name}'

                df_generated_full = None
                try:
                    df_generated_full = generate_synth_data_via_framework(
                        synth_model_name, term_coeff_map_Pa, CURRENT_BASIS_NAMES,
                        [rel_noise],
                        SYNTH_LAMBDA_RANGE, SYNTH_GAMMA_RANGE, SYNTH_N_POINTS
                    )
                    if df_generated_full is None or df_generated_full.empty: raise ValueError("Data generation failed")

                    fig_raw, axes_raw = plt.subplots(1, 3, figsize=(18, 5), sharey=False)
                    df_raw_ut = df_generated_full[df_generated_full['mode'] == 'UT']
                    df_raw_ps = df_generated_full[df_generated_full['mode'] == 'PS']
                    df_raw_ebt = df_generated_full[df_generated_full['mode'] == 'EBT']
                    if not df_raw_ut.empty: axes_raw[0].scatter(df_raw_ut['lambda1'], df_raw_ut['P11'], label=f'Noisy P11 (RelNoise {rel_noise * 100:.0f}%)', s=15, alpha=0.7);
                    else: axes_raw[0].text(0.5, 0.5, "No UT Data", ha='center', va='center')
                    if 'P11_true' in df_raw_ut.columns: axes_raw[0].plot(df_raw_ut['lambda1'].sort_values(), df_raw_ut.sort_values('lambda1')['P11_true'], 'k--', label='True P11')
                    axes_raw[0].set_xlabel('$λ$'); axes_raw[0].set_ylabel('$P_{11}$ (MPa)'); axes_raw[0].set_title('Generated UT Data P11'); axes_raw[0].legend(frameon=False);
                    if not df_raw_ps.empty: axes_raw[1].scatter(df_raw_ps['lambda1'], df_raw_ps['P11'], label=f'Noisy P11 (RelNoise {rel_noise * 100:.0f}%)', s=15, alpha=0.7, color='darkorange');
                    else: axes_raw[1].text(0.5, 0.5, "No PS Data (P11)", ha='center', va='center')
                    if 'P11_true' in df_raw_ps.columns: axes_raw[1].plot(df_raw_ps['lambda1'].sort_values(), df_raw_ps.sort_values('lambda1')['P11_true'], 'k:', label='True P11')
                    axes_raw[1].set_xlabel('$λ$'); axes_raw[1].set_ylabel('$P_{11}$ (MPa)'); axes_raw[1].set_title('Generated PS Data P11'); axes_raw[1].legend(frameon=False);
                    if not df_raw_ebt.empty: axes_raw[2].scatter(df_raw_ebt['lambda1'], df_raw_ebt['P11'], label=f'Noisy P11 (RelNoise {rel_noise * 100:.0f}%)', s=15, alpha=0.7, color='green');
                    else: axes_raw[2].text(0.5, 0.5, "No EBT Data (P11)", ha='center', va='center')
                    if 'P11_true' in df_raw_ebt.columns: axes_raw[2].plot(df_raw_ebt['lambda1'].sort_values(), df_raw_ebt.sort_values('lambda1')['P11_true'], 'k:', label='True P11')
                    axes_raw[2].set_xlabel('$λ$'); axes_raw[2].set_ylabel('$P_{11}$ (MPa)'); axes_raw[2].set_title('Generated EBT Data P11'); axes_raw[2].legend(frameon=False);
                    fig_raw.suptitle(f'Raw Synthetic Data (Framework Logic): {scenario_name}')
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    raw_data_save_path = os.path.join(current_save_dir, f'{file_prefix}_RawSynthData_P11_MPa.pdf')
                    try: fig_raw.savefig(raw_data_save_path, format='pdf', bbox_inches='tight'); print(f'Saved raw synthetic data plot: {raw_data_save_path}')
                    except Exception as e: print(f'Error saving raw data plot: {e}')
                    plt.close(fig_raw)
                except Exception as e: print(f'ERROR during data generation/plotting: {e}'); traceback.print_exc(); continue

                # Preparing data for model discovery (UT P11 + PS P11 + EBT P11)
                print("\n--- Preparing data for model discovery (UT P11 + PS P11 + EBT P11) ---")
                df_for_discovery_all_modes = df_generated_full[df_generated_full['mode'].isin(['UT', 'PS', 'EBT'])].copy()

                # Create a dataframe that only contains the P11 data points we want to use for discovery
                df_p11_ut = df_for_discovery_all_modes[df_for_discovery_all_modes['mode'] == 'UT'][['lambda1', 'lambda2', 'lambda3', 'P11', 'mode', 'strain_pct']]
                df_p11_ps = df_for_discovery_all_modes[df_for_discovery_all_modes['mode'] == 'PS'][['lambda1', 'lambda2', 'lambda3', 'P11', 'mode', 'strain_pct']]
                df_p11_ebt = df_for_discovery_all_modes[df_for_discovery_all_modes['mode'] == 'EBT'][['lambda1', 'lambda2', 'lambda3', 'P11', 'mode', 'strain_pct']]

                df_for_discovery = pd.concat([df_p11_ut, df_p11_ps, df_p11_ebt], ignore_index=True)

                if df_for_discovery.empty:
                    print("ERROR: No UT, PS, or EBT P11 data in generated DF for discovery. Skipping.")
                    continue

                X_weighted_scaled = None; target_vector_weighted = None; basis_names_from_matrix = None
                try:
                    print(f'Model Library for Discovery: {MODEL_CONFIG_NAME}')

                    design_matrix_unweighted, _, basis_names_from_matrix, _, _ = construct_design_matrix(
                        df_for_discovery
                    )
                    if design_matrix_unweighted.shape[0] == 0: raise ValueError("Unweighted matrix for scaling is empty.")

                    scaler = fit_standard_scaler(design_matrix_unweighted)

                    design_matrix_final, target_vector_weighted, _, _, _ = construct_design_matrix(
                        df_for_discovery
                    )
                    if design_matrix_final.shape[0] == 0: raise ValueError("Final design matrix is empty.")

                    X_weighted_scaled = scaler.transform(design_matrix_final)

                    if CURRENT_BASIS_NAMES != basis_names_from_matrix: print(f'WARN: Basis name mismatch; using list from matrix construction.')
                    basis_names_for_analysis = basis_names_from_matrix
                    if len(basis_names_for_analysis) != X_weighted_scaled.shape[1]: raise ValueError("Basis/matrix column mismatch.")

                    if not np.all(np.isfinite(X_weighted_scaled)):
                        print("WARN: Non-finite values after scaling. Replacing with 0.")
                        X_weighted_scaled = np.nan_to_num(X_weighted_scaled)

                except Exception as e: print(f'ERROR in data prep: {e}'); traceback.print_exc(); continue

                scenario_initial_results = {}; scenario_refit_summary = {}

                analysis_functions = {'Lasso': run_lasso_analysis, 'LARS': run_lars_analysis, 'OMP': run_omp_analysis}
                for method_base_name, analysis_func in analysis_functions.items():
                    results_dict = None; print("\n" + "="*30 + f' Running {method_base_name} Variants for {scenario_name} ({MODEL_CONFIG_NAME}) ' + "="*30)
                    try:
                        if method_base_name == 'Lasso':
                            results_dict = analysis_func(X_weighted_scaled, target_vector_weighted, basis_names_for_analysis, CV_FOLDS, file_prefix, current_save_dir, random_seed=RANDOM_SEED)
                        else:
                            results_dict = analysis_func(X_weighted_scaled, target_vector_weighted, basis_names_for_analysis, CV_FOLDS, file_prefix, current_save_dir)
                        scenario_initial_results.update(results_dict)
                    except Exception as outer_e: print(f'FATAL ERROR during {method_base_name} Analysis block: {outer_e}'); traceback.print_exc();


                print("\n" + "="*40 + f' Post-processing All Models for {scenario_name} ({MODEL_CONFIG_NAME}) ' + "="*40)

                process_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']

                for method_key in process_order:
                    if method_key in scenario_initial_results:
                        result_data = scenario_initial_results[method_key]
                        print(f'\n--- Post-processing Result for: {method_key} ---')
                        scenario_refit_summary[method_key] = {'consistent': None, 'notes': ['Post-processing start']}

                        if 'error' not in result_data and 'coeffs' in result_data:
                            initial_coeffs_MPa = result_data['coeffs']
                            n_nonzero_initial = result_data.get('n_nonzero', np.sum(np.abs(initial_coeffs_MPa) > context.EPS))

                            if n_nonzero_initial > 0:
                                try:
                                    # Collect refit results
                                    (refit_terms, refit_coeffs_kPa,
                                     r2_p11_ut, rmse_p11_ut_kPa, nrmse_p11_ut,
                                     r2_p11_ps, rmse_p11_ps_kPa, nrmse_p11_ps,
                                     r2_p11_ebt, rmse_p11_ebt_kPa, nrmse_p11_ebt,
                                     is_consistent, shear_modulus_MPa, consistency_notes
                                     ) = plot_model_predictions(
                                         df_generated_full, initial_coeffs_MPa, basis_names_for_analysis,
                                         method_key, file_prefix, current_save_dir,
                                         coeff_threshold=context.COEFF_THRESHOLD, ridge_alpha=context.RIDGE_ALPHA_REFIT
                                     )

                                    # Store summary (using Pa/NRMSE)
                                    scenario_refit_summary[method_key] = {
                                        'terms': refit_terms, 'coeffs': refit_coeffs_kPa,
                                        'r2_p11_ut': r2_p11_ut, 'rmse_p11_ut': rmse_p11_ut_kPa, 'nrmse_p11_ut': nrmse_p11_ut,
                                        'r2_p11_ps': r2_p11_ps, 'rmse_p11_ps': rmse_p11_ps_kPa, 'nrmse_p11_ps': nrmse_p11_ps,
                                        'r2_p11_ebt': r2_p11_ebt, 'rmse_p11_ebt': rmse_p11_ebt_kPa, 'nrmse_p11_ebt': nrmse_p11_ebt,
                                        'consistent': is_consistent, 'notes': consistency_notes
                                    }

                                except ValueError as ve:
                                    print(f'ERROR caught during post-processing call for {method_key}: UNPACKING/VALUE ERROR - {ve}')
                                    scenario_refit_summary[method_key]['error_postprocess'] = f'Unpacking/Value Error: {ve}'
                                    scenario_refit_summary[method_key]['notes'] = [f'Unpacking/Value Error: {ve}']
                                    traceback.print_exc()
                                except Exception as plot_e:
                                    print(f'ERROR caught during post-processing call for {method_key}: {plot_e}'); traceback.print_exc()
                                    scenario_refit_summary[method_key]['error_postprocess'] = str(plot_e)
                                    scenario_refit_summary[method_key]['notes'] = [f'Plotting/PostProc Error: {plot_e}']
                            else:
                                scenario_refit_summary[method_key]['notes'] = ['No features initially selected']
                                scenario_refit_summary[method_key]['consistent'] = None
                        else:
                            err_msg = result_data.get('error', 'Unknown initial fit error')
                            scenario_refit_summary[method_key]['notes'] = [f'Initial fit error: {err_msg}']
                            scenario_refit_summary[method_key]['consistent'] = None
                    else:
                        scenario_refit_summary[method_key] = {'consistent': None, 'notes': [f'{method_key} not run or no results']}

                all_initial_results[scenario_name] = scenario_initial_results
                all_refit_summaries[scenario_name] = scenario_refit_summary
                print(f'\n### Finished All Analyses & Post-processing for Scenario: {scenario_name} ###\n')

        # Print overall summaries & verification table
        print("\n" + "="*80); print("### OVERALL INITIAL SPARSE SELECTION SUMMARY ###"); print("="*80)
        for scenario_name_sum, methods_results_sum in all_initial_results.items():
            print(f'\n--- Scenario: {scenario_name_sum} ---')
            print("| Method      | Optimal Param    | Non-Zero | AIC       | BIC       | Time (s) |")
            print("|-------------|------------------|----------|-----------|-----------|----------|")
            process_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
            for method_key_sum in process_order:
                if method_key_sum in methods_results_sum:
                    res = methods_results_sum[method_key_sum] ; method_print_key = method_key_sum
                    if 'error' in res: time_str = f"{res.get('time', np.nan):<8.2f}" if pd.notna(res.get('time', np.nan)) else "N/A     " ; error_msg_short = str(res.get('error', 'Unknown Error'))[:15] + "..." ; print(f'| {method_print_key:<11} | ERROR             | ---      | ---       | ---       | {time_str} | {error_msg_short}')
                    else: n_nonzero = res.get('n_nonzero', 'N/A'); aic_str = f"{res.get('aic', np.nan):<9.2e}" if pd.notna(res.get('aic', np.nan)) else 'N/A        '; bic_str = f"{res.get('bic', np.nan):<9.2e}" if pd.notna(res.get('bic', np.nan)) else 'N/A        '; time_str = f"{res.get('time', np.nan):<8.4f}" if pd.notna(res.get('time', np.nan)) else 'N/A     '; opt_param_details = [];
                    if 'CV' in method_key_sum: opt_param_details.append("CV")
                    if 'AIC' in method_key_sum: opt_param_details.append("AIC")
                    if 'BIC' in method_key_sum: opt_param_details.append("BIC")
                    if 'alpha' in res and pd.notna(res['alpha']): opt_param_details.append(f"α≈{res['alpha']:.2g}")
                    if 'optimal_k' in res and pd.notna(res['optimal_k']): opt_param_details.append(f"k={int(res['optimal_k'])}")
                    elif 'n_features_opt' in res and pd.notna(res['n_features_opt']): opt_param_details.append(f"k={int(res['n_features_opt'])}")
                    opt_param_str = " ".join(opt_param_details) if opt_param_details else "N/A"; print(f'| {method_print_key:<11} | {opt_param_str:<16} | {n_nonzero:<8} | {aic_str} | {bic_str} | {time_str} |')

        header_width = 170
        print("\n" + "="*header_width); print(f'### OVERALL FINAL REFIT MODEL SUMMARY (Refit: {context.REFIT_REGRESSION_TYPE}, Thresh={context.COEFF_THRESHOLD:.1e}) ###'.center(header_width)); print("### Metrics: UT(P11), PS(P11), EBT(P11) on UNWEIGHTED Data | Consistency Check ###".center(header_width))
        print(f"| {'Method':<11} | {'N Terms':<7} | {'R²(UT)':<8} | {'RMSE(UT)[kPa]':<13} | {'NRMSE(UT)':<9} | {'R²(PS11)':<9} | {'RMSE(PS11)[kPa]':<15} | {'NRMSE(PS11)':<11} | {'R²(EBT11)':<10} | {'RMSE(EBT11)[kPa]':<16} | {'NRMSE(EBT11)':<12} | {'Consist':<7} | {'Notes':<30} |")
        print(f"|{'-' * 13}|{'-' * 9}|{'-' * 10}|{'-' * 15}|{'-' * 11}|{'-' * 11}|{'-' * 17}|{'-' * 13}|{'-' * 12}|{'-' * 18}|{'-' * 14}|{'-' * 9}|{'-' * 32}|")

        for scenario_name_sum, methods_summary_sum in all_refit_summaries.items():
            print(f'\n--- Scenario: {scenario_name_sum} ---')
            for method_key_sum in process_order:
                if method_key_sum in methods_summary_sum:
                    summary = methods_summary_sum.get(method_key_sum, {})
                    n_terms = len(summary.get('terms', [])) if isinstance(summary.get('terms', []), list) else 0
                    r2_ut = f"{summary.get('r2_p11_ut', np.nan):.4f}" if pd.notna(summary.get('r2_p11_ut', np.nan)) else " N/A  "
                    rmse_ut = f"{summary.get('rmse_p11_ut', np.nan):.1f}" if pd.notna(summary.get('rmse_p11_ut', np.nan)) else " N/A"
                    nrmse_ut = f"{summary.get('nrmse_p11_ut', np.nan):.4f}" if pd.notna(summary.get('nrmse_p11_ut', np.nan)) else " N/A"
                    r2_ps11 = f"{summary.get('r2_p11_ps', np.nan):.4f}" if pd.notna(summary.get('r2_p11_ps', np.nan)) else " N/A  "
                    rmse_ps11 = f"{summary.get('rmse_p11_ps', np.nan):.1f}" if pd.notna(summary.get('rmse_p11_ps', np.nan)) else " N/A"
                    nrmse_ps11 = f"{summary.get('nrmse_p11_ps', np.nan):.4f}" if pd.notna(summary.get('nrmse_p11_ps', np.nan)) else " N/A"
                    r2_ebt11 = f"{summary.get('r2_p11_ebt', np.nan):.4f}" if pd.notna(summary.get('r2_p11_ebt', np.nan)) else " N/A  "
                    rmse_ebt11 = f"{summary.get('rmse_p11_ebt', np.nan):.1f}" if pd.notna(summary.get('rmse_p11_ebt', np.nan)) else " N/A"
                    nrmse_ebt11 = f"{summary.get('nrmse_p11_ebt', np.nan):.4f}" if pd.notna(summary.get('nrmse_p11_ebt', np.nan)) else " N/A"
                    consistent = "OK" if summary.get('consistent') is True else "FAIL" if summary.get('consistent') is False else "N/A"
                    notes_list = summary.get('notes', ["N/A"])
                    notes_str = (notes_list[0][:27] + '...') if notes_list and len(notes_list[0]) > 30 else (notes_list[0] if notes_list else "N/A")
                    print(f'| {method_key_sum:<11} | {n_terms:<7} | {r2_ut:<8} | {rmse_ut:<13} | {nrmse_ut:<9} | {r2_ps11:<9} | {rmse_ps11:<15} | {nrmse_ps11:<11} | {r2_ebt11:<10} | {rmse_ebt11:<16} | {nrmse_ebt11:<12} | {consistent:<7} | {notes_str:<30} |')

        print("\n" + "="*150); print("### VERIFICATION SUMMARY: Discovered vs. Ground Truth Terms ###"); print("="*150)
        verification_summary_list = []

        if not all_refit_summaries: print("ERROR: 'all_refit_summaries' empty. Cannot generate verification table.")
        else:
            for scenario_name_sum, methods_summary_sum in all_refit_summaries.items():
                base_model_name = None
                for model_key in synthetic_model_definitions.keys():
                    if scenario_name_sum.startswith(model_key + "_RelNoise"): base_model_name = model_key; break
                if base_model_name is None: continue
                try:
                    ground_truth_map_Pa_local = synthetic_model_definitions[base_model_name]
                    ground_truth_map_MPa_local = {term: coeff / 1e6 for term, coeff in ground_truth_map_Pa_local.items()}
                    ground_truth_terms_set_local = set(ground_truth_map_MPa_local.keys()); n_ground_truth = len(ground_truth_terms_set_local)
                except KeyError: continue

                for method_key_sum in process_order:
                    summary_row = { "Scenario": scenario_name_sum, "Method": method_key_sum, "N_Ground_Truth": n_ground_truth, "N_Discovered": 0, "N_Correct": 0, "N_Missed": n_ground_truth, "N_Spurious": 0, "Avg_Coeff_Rel_Error(%)": np.nan, "Correct_Terms": "-", "Missed_Terms": "-", "Spurious_Terms": "-"}
                    if method_key_sum in methods_summary_sum:
                        summary = methods_summary_sum.get(method_key_sum, {})

                        if ('terms' in summary and isinstance(summary['terms'], list)):
                            final_coeffs_map_MPa = {}
                            if 'coeffs' in summary and len(summary['terms']) == len(summary['coeffs']):
                                final_coeffs_kPa = summary['coeffs']
                                final_terms = summary['terms']
                                final_coeffs_map_MPa = {term: coeff / 1000.0 for term, coeff in zip(final_terms, final_coeffs_kPa)}

                            discovered_terms_after_refit = summary['terms']
                            discovered_terms_set = set(discovered_terms_after_refit)

                            summary_row["N_Discovered"] = len(discovered_terms_set)

                            correctly_found_terms = ground_truth_terms_set_local.intersection(discovered_terms_set)
                            summary_row["N_Correct"] = len(correctly_found_terms)
                            summary_row["N_Missed"] = len(ground_truth_terms_set_local.difference(discovered_terms_set))
                            summary_row["N_Spurious"] = len(discovered_terms_set.difference(ground_truth_terms_set_local))
                            summary_row["Correct_Terms"] = ", ".join(sorted(list(correctly_found_terms))) if correctly_found_terms else "-"
                            summary_row["Missed_Terms"] = ", ".join(sorted(list(ground_truth_terms_set_local.difference(discovered_terms_set)))) if ground_truth_terms_set_local.difference(discovered_terms_set) else "-"
                            summary_row["Spurious_Terms"] = ", ".join(sorted(list(discovered_terms_set.difference(ground_truth_terms_set_local)))) if discovered_terms_set.difference(ground_truth_terms_set_local) else "-"

                            coeff_rel_errors = []
                            for term in correctly_found_terms:
                                if term in final_coeffs_map_MPa and term in ground_truth_map_MPa_local:
                                    gt_coeff_MPa = ground_truth_map_MPa_local[term]
                                    disc_coeff_MPa = final_coeffs_map_MPa[term]
                                    if abs(gt_coeff_MPa) > 1e-12:
                                        coeff_rel_errors.append(abs(disc_coeff_MPa - gt_coeff_MPa) / abs(gt_coeff_MPa) * 100)
                            if coeff_rel_errors: summary_row["Avg_Coeff_Rel_Error(%)"] = np.nanmean(coeff_rel_errors)
                        elif 'notes' in summary:
                            summary_row["Correct_Terms"] = f"({summary['notes'][0]})"; summary_row["Missed_Terms"] = ", ".join(sorted(list(ground_truth_terms_set_local))); summary_row["Spurious_Terms"] = "-"
                    verification_summary_list.append(summary_row)

            if verification_summary_list:
                verification_df = pd.DataFrame(verification_summary_list)
                cols_ordered = ["Scenario", "Method", "N_Ground_Truth", "N_Discovered", "N_Correct", "N_Missed", "N_Spurious", "Avg_Coeff_Rel_Error(%)", "Correct_Terms", "Missed_Terms", "Spurious_Terms"]
                for col in cols_ordered:
                    if col not in verification_df.columns: verification_df[col] = np.nan
                verification_df = verification_df.reindex(columns=cols_ordered)
                print("\nVerification Summary Table:");
                with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 200): print(verification_df.to_string(index=False, float_format="%.1f", na_rep="-"))
            else: print("No verification results collected for this scenario to print.")
        print("\n" + "="*150); print("### END Verification Summary Table ###"); print("="*150)

    # Restore stdout and close file
    finally:
        if log_file_handle is not None: # Check if file was successfully opened
            sys.stdout = original_stdout # Reset redirect.
            log_file_handle.close()
            # This print will go to console
            print(f'\nLog output saved to {log_file_name}')
        else:
            sys.stdout = original_stdout # Still ensure stdout is reset
            print(f'\nLogging to file failed. Outputting to console.')

        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f'\nCompleted all scenarios at: {current_time_str}')
        if 'overall_start_time' in locals() and overall_start_time is not None:
            total_duration = datetime.datetime.now() - overall_start_time
            print(f'Total analysis duration: {total_duration}')
        print("="*170)
    return {"initial_results": all_initial_results, "refit_summaries": all_refit_summaries}
