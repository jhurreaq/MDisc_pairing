"""Cardiac refinement; original numerical and presentation choices."""

from functools import partial
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import time
from .models import calculate_single_prediction_full_tensor

def objective_function(params, active_term_indices, num_active_terms_in_subset, data_list_indices, weights, y_target_for_obj, all_invariants_list, *, context):
    """ Objective function for Stage 2 non-linear refinement. Uses the full tensor calculation. """
    w2_params = params[:num_active_terms_in_subset]
    w1_params = params[num_active_terms_in_subset:]

    # Build the full coefficient vectors for the prediction function
    coeffs_w2_full = np.zeros(context.num_psi_terms)
    coeffs_w2_full[active_term_indices] = w2_params
    w1_map_full = {}
    w1_counter_local = 0
    for term_idx_global in active_term_indices:
        if context.psi_terms_definitions[term_idx_global]['is_exp']:
            if w1_counter_local < len(w1_params):
                w1_map_full[term_idx_global] = w1_params[w1_counter_local]
                w1_counter_local += 1

    predictions = np.zeros(len(data_list_indices))
    for i_dp_idx, data_idx in enumerate(data_list_indices):
        data_point = context.experimental_data_list_filtered[data_idx]
        invariants = all_invariants_list[data_idx]
        pred = calculate_single_prediction_full_tensor(data_point, invariants, coeffs_w2_full, w1_map_full, context=context)
        predictions[i_dp_idx] = pred if not np.isnan(pred) else np.inf

    y_target_subset = y_target_for_obj[data_list_indices]
    weights_subset = weights[data_list_indices]
    if np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)): return float('inf')
    total_weighted_ssr = np.sum(weights_subset * (y_target_subset - predictions)**2)
    return total_weighted_ssr

def refine_model(stage1_model_dict, model_name, y_target_current, sample_weights_current, all_invariants_current, *, context):
    print(f'\n--- Stage 2: Refining {model_name} Model ---')

    # Handle potential negative coefficients from LARS/OMP by applying NNLS pre-processing
    active_coeffs_s1 = stage1_model_dict['coeffs'].copy()

    # Check if this is a LARS or OMP method that might have negative coefficients
    is_lars_or_omp = any(keyword in model_name for keyword in ['Lars', 'OMP'])

    if is_lars_or_omp:
        print(f'  INFO: Applying NNLS pre-refinement for {model_name} coefficients.')
        # Identify features selected by LARS/OMP in Stage 1
        selected_feature_indices_stage1 = np.where(np.abs(active_coeffs_s1) > 1e-9)[0]

        if len(selected_feature_indices_stage1) > 0:
            # Build design matrix for selected features
            X_selected_for_nnls = context.X_features_stage1[:, selected_feature_indices_stage1]

            try:
                # Perform Non-Negative Least Squares on selected features
                nnls_model = LinearRegression(positive=True, fit_intercept=False)
                nnls_model.fit(X_selected_for_nnls, y_target_current, sample_weight=sample_weights_current)

                # Create new full coefficient vector
                nnls_coeffs_full_vector = np.zeros_like(active_coeffs_s1)
                nnls_coeffs_full_vector[selected_feature_indices_stage1] = nnls_model.coef_
                nnls_coeffs_full_vector = np.maximum(nnls_coeffs_full_vector, 0)  # Ensure positivity

                active_coeffs_s1 = nnls_coeffs_full_vector
                n_positive_after_nnls = np.sum(active_coeffs_s1 > 1e-6)
                print(f'    NNLS for {model_name}: {len(selected_feature_indices_stage1)} initially selected -> {n_positive_after_nnls} positive coefficients.')
            except Exception as e:
                print(f'    WARNING: NNLS pre-refinement failed for {model_name}: {e}. Falling back to clamping.')
                active_coeffs_s1 = np.maximum(active_coeffs_s1, 0)
        else:
            print(f'    No features originally selected by {model_name} in Stage 1.')
            active_coeffs_s1 = np.zeros_like(active_coeffs_s1)

    # Select terms based on coefficients being meaningfully positive
    active_term_indices = np.where(active_coeffs_s1 > 1e-6)[0]

    if len(active_term_indices) == 0:
        print(f'  INFO: {model_name} (after Stage 1 pre-processing) has 0 active terms with coeff > 1e-6. Skipping refinement.')
        return {'coeffs': np.zeros(context.num_psi_terms), 'w1_map': {}, 'success': False, 'message': 'No positive terms selected/retained from Stage 1 for refinement', 'time':0.0}

    active_terms_defs = [context.psi_terms_definitions[i] for i in active_term_indices]
    active_w2_coeffs_initial = active_coeffs_s1[active_term_indices] # Initial guess for w2 parameters

    active_exp_term_indices_in_subset = [i for i, term_idx in enumerate(active_term_indices) if context.psi_terms_definitions[term_idx]['is_exp']]
    num_active_exp_terms = len(active_exp_term_indices_in_subset)
    num_active_terms_subset = len(active_term_indices)

    # Initial guess: [initial w2 coeffs for active terms, initial w1 coeffs (all 1.0) for active exponential terms]
    initial_guess_w1 = np.full(num_active_exp_terms, 1.0) # Initial guess for w1 parameters
    initial_guess = np.concatenate([active_w2_coeffs_initial, initial_guess_w1])

    num_params_to_optimize = len(initial_guess)

    # Bounds: w2 params are positive, w1 params also positive (or other appropriate bounds)
    bounds_w2 = [(1e-9, None)] * num_active_terms_subset
    bounds_w1 = [(1e-9, None)] * num_active_exp_terms # Assuming w1 should also be positive
    bounds = bounds_w2 + bounds_w1

    print(f'  Optimizing {num_params_to_optimize} parameters ({num_active_terms_subset} w2, {num_active_exp_terms} w1) for {len(active_term_indices)} active terms...')
    opt_start_time = time.time()

    n_samples_for_obj = len(y_target_current)

    optimization_result = minimize(
        partial(objective_function, context=context), initial_guess,
        args=(active_term_indices, num_active_terms_subset, np.arange(n_samples_for_obj), sample_weights_current, y_target_current, all_invariants_current),
        method=context.NON_LINEAR_OPTIMIZER_METHOD, bounds=bounds,
        options={'maxiter': 1000, 'ftol': context.NON_LINEAR_OPTIMIZER_TOL}
    )
    opt_end_time = time.time()
    print(f'  Optimization finished in {opt_end_time - opt_start_time:.2f} seconds. Success: {optimization_result.success}')

    final_coeffs_full = np.zeros(context.num_psi_terms)
    optimized_w1_map = {}
    refined_model_summary = {'success': optimization_result.success, 'message': optimization_result.message, 'time': opt_end_time - opt_start_time}

    if optimization_result.success:
        optimized_params = optimization_result.x
        final_w2_coeffs = optimized_params[:num_active_terms_subset]
        final_w1_coeffs = optimized_params[num_active_terms_subset:]

        final_coeffs_full[active_term_indices] = final_w2_coeffs # Assign refined w2

        w1_counter = 0
        print(f'\n  --- Refined {model_name} Model Summary ---')
        print(f'  Number of active terms: {len(active_term_indices)}')
        refined_model_summary['coeffs'] = final_coeffs_full
        refined_model_summary['k_w2'] = np.sum(final_coeffs_full > 1e-6) # Count actual non-zero after refinement
        refined_model_summary['k_w1'] = num_active_exp_terms # Number of w1 params optimized
        refined_model_summary['w1_map'] = {}

        for i_term_subset, term_idx_global in enumerate(active_term_indices):
            term_def = context.psi_terms_definitions[term_idx_global]
            w2_val = final_w2_coeffs[i_term_subset]
            print(f"    Term {term_idx_global + 1} ({term_def['term_label']}): w2 = {w2_val:.4f}", end="")
            if term_def['is_exp']:
                if w1_counter < len(final_w1_coeffs):
                    w1_val = final_w1_coeffs[w1_counter]
                    optimized_w1_map[term_idx_global] = w1_val
                    refined_model_summary['w1_map'][term_idx_global] = w1_val
                    print(f', w1 = {w1_val:.4f}')
                    w1_counter += 1
                else: # Should not happen if logic is correct
                    print(", w1 = N/A (error in indexing)")
            else:
                print("")
        # Recalculate k_w2 based on refined coefficients being > threshold
        refined_model_summary['k_w2'] = np.sum(final_coeffs_full[active_term_indices] > 1e-6)

    else:
        print(f'  Non-linear optimization failed for {model_name}. Using Stage 1 (pre-processed) coefficients as initial guess for reporting.')
        # Report based on the initial guess that went into the failed optimization
        final_coeffs_full[active_term_indices] = active_w2_coeffs_initial
        refined_model_summary['coeffs'] = final_coeffs_full
        refined_model_summary['w1_map'] = {} # No w1 values optimized
        # For w1_map in failure, could fill with initial 1.0s for active exp terms for consistency if needed by downstream
        w1_counter_fail = 0
        for i_term_subset, term_idx_global in enumerate(active_term_indices):
            term_def = context.psi_terms_definitions[term_idx_global]
            if term_def['is_exp']:
                refined_model_summary['w1_map'][term_idx_global] = initial_guess_w1[w1_counter_fail] # Report initial guess
                w1_counter_fail +=1

        refined_model_summary['k_w2'] = np.sum(active_w2_coeffs_initial > 1e-6)
        refined_model_summary['k_w1'] = num_active_exp_terms if refined_model_summary['k_w2'] > 0 else 0

    return refined_model_summary
