"""Synthetic data; original numerical and presentation choices."""

import numpy as np
import pandas as pd
from .models import calculate_dW_dl_vectors

def generate_synth_data_via_framework(model_name, term_coeff_map, basis_names,
                                      relative_noise_levels,
                                      lambda_range, gamma_range, n_points, *, context):
    """
    Generates synthetic UT(P11) and SS(P11, P22) data using the discovery
    framework's logic (dW/dl_i summation) for stress calculation.
    Noise is added relative to the true stress value.

    Args:
        model_name (str): Name for the dataset column (e.g., "MR2").
        term_coeff_map (dict): Map {'basis_term_string': coefficient_Pa}. Coefficients should be in Pa.
        basis_names (list): Full list of basis names from generate_model_library.
        relative_noise_levels (list): List of relative noise levels (e.g., [0.0, 0.05 for 5%]).
        lambda_range (tuple): Min/max lambda for UT.
        gamma_range (tuple): Min/max gamma for SS.
        n_points (int): Number of points per mode.

    Returns:
        pd.DataFrame or None: DataFrame with synthetic data, or None if required terms missing.
    """
    all_synth_data = []
    lambdas = np.linspace(lambda_range[0], lambda_range[1], n_points)
    gammas = np.linspace(gamma_range[0], gamma_range[1], n_points)
    pa_to_mpa = 1e-6 # Conversion factor

    # Find indices of the required terms in the full basis list
    term_indices = {}
    missing_terms = []
    for term in term_coeff_map.keys():
        try:
            term_indices[term] = basis_names.index(term)
        except ValueError:
            missing_terms.append(term)

    if missing_terms:
        print(f"ERROR: Cannot generate data for '{model_name}'. Required terms missing from library:")
        for term in missing_terms: print(f' - {term}')
        print("Please adjust `generate_model_library` flags (e.g., Ogden exponents, MR powers).")
        return None

    print(f'--- Generating Synthetic UT, PS & EBT Data [Framework Logic] for: {model_name} ---')

    # --- Uniaxial Tension/Compression ---
    print(f'  Generating Uniaxial (UT) data (λ = {lambda_range}, {n_points} points)...')
    for lam_val in lambdas:
        l1 = max(float(lam_val), context.EPS)
        l2 = max(l1**(-0.5), context.EPS)
        l3 = l2 # Incompressible UT

        deriv_vecs = calculate_dW_dl_vectors(l1, l2, l3, basis_names, context=context)
        if deriv_vecs is None: continue

        dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs

        dW_total_dl1_Pa = 0.0; dW_total_dl2_Pa = 0.0; dW_total_dl3_Pa = 0.0
        # term_coeff_map provides coefficients in Pa
        for term, coeff_Pa in term_coeff_map.items():
            idx = term_indices[term]
            dW_total_dl1_Pa += coeff_Pa * dW_dl1_all[idx]
            dW_total_dl2_Pa += coeff_Pa * dW_dl2_all[idx]
            dW_total_dl3_Pa += coeff_Pa * dW_dl3_all[idx]

        p11_true_Pa = dW_total_dl1_Pa - (l3 / l1) * dW_total_dl3_Pa
        p11_true_MPa = p11_true_Pa * pa_to_mpa

        for rel_noise in relative_noise_levels:
            # Calculate absolute noise standard deviation based on relative level and true stress
            noise_std_dev_Pa = rel_noise * abs(p11_true_Pa)
            # Add a very small constant to std_dev if p11_true_Pa is zero to avoid issues,
            # or ensure np.random.normal handles scale=0 gracefully (it does, returns mean)
            if abs(p11_true_Pa) < context.EPS and rel_noise > 0: # Only add if true stress is zero but noise is requested
                noise_std_dev_Pa = rel_noise * (context.EPS * 1000) # Add noise relative to a small nominal stress

            actual_noise_Pa = np.random.normal(0, noise_std_dev_Pa)
            P11_noisy_Pa = p11_true_Pa + actual_noise_Pa
            P11_noisy_MPa = P11_noisy_Pa * pa_to_mpa

            dataset_name = f'{model_name}_RelNoise{rel_noise * 100:.0f}pct'

            data_point = {
                'lambda1': l1, 'lambda2': l2, 'lambda3': l3,
                'gamma': np.nan,
                'P11': P11_noisy_MPa, 'P22': 0.0, 'P12': np.nan,
                'P11_true': p11_true_MPa, 'P12_true': np.nan, 'P22_true': 0.0,
                'mode': 'UT', 'dataset': dataset_name,
                'noise_sigma': rel_noise, # Storing the relative noise level
                'strain_pct': (l1 - 1.0) * 100.0
            }
            all_synth_data.append(data_point)

    # --- Pure Shear ---
    print(f'  Generating Pure Shear (PS) data (λ = {lambda_range}, {n_points} points)...')
    for lam_val in lambdas:
        l1_ps = max(float(lam_val), context.EPS)
        l2_ps = 1.0
        l3_ps = max(1.0 / l1_ps, context.EPS)

        deriv_vecs = calculate_dW_dl_vectors(l1_ps, l2_ps, l3_ps, basis_names, context=context)
        if deriv_vecs is None: continue

        dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs

        dW_total_dl1_Pa = 0.0; dW_total_dl2_Pa = 0.0; dW_total_dl3_Pa = 0.0
        for term, coeff_Pa in term_coeff_map.items():
            idx = term_indices[term]
            dW_total_dl1_Pa += coeff_Pa * dW_dl1_all[idx]
            dW_total_dl2_Pa += coeff_Pa * dW_dl2_all[idx]
            dW_total_dl3_Pa += coeff_Pa * dW_dl3_all[idx]

        # PK1 with incompressibility
        p11_true_Pa = dW_total_dl1_Pa - (l3_ps / l1_ps) * dW_total_dl3_Pa if l1_ps > context.EPS else 0.0
        p22_true_Pa = dW_total_dl2_Pa - (l3_ps / l2_ps) * dW_total_dl3_Pa if l2_ps > context.EPS else 0.0
        p11_true_MPa = p11_true_Pa * pa_to_mpa
        p22_true_MPa = p22_true_Pa * pa_to_mpa

        for rel_noise in relative_noise_levels:
            noise11_std_dev_Pa = rel_noise * abs(p11_true_Pa)
            noise22_std_dev_Pa = rel_noise * abs(p22_true_Pa)

            if abs(p11_true_Pa) < context.EPS and rel_noise > 0:
                noise11_std_dev_Pa = rel_noise * (context.EPS * 1000)
            if abs(p22_true_Pa) < context.EPS and rel_noise > 0:
                noise22_std_dev_Pa = rel_noise * (context.EPS * 1000)

            noise11_Pa = np.random.normal(0, noise11_std_dev_Pa)
            noise22_Pa = np.random.normal(0, noise22_std_dev_Pa)
            P11_noisy_Pa = p11_true_Pa + noise11_Pa
            P22_noisy_Pa = p22_true_Pa + noise22_Pa
            P11_noisy_MPa = P11_noisy_Pa * pa_to_mpa
            P22_noisy_MPa = P22_noisy_Pa * pa_to_mpa

            dataset_name = f'{model_name}_RelNoise{rel_noise * 100:.0f}pct'

            data_point = {
                'lambda1': l1_ps, 'lambda2': l2_ps, 'lambda3': l3_ps,
                'gamma': np.nan,
                'P11': P11_noisy_MPa, 'P22': P22_noisy_MPa, 'P12': np.nan,
                'P11_true': p11_true_MPa, 'P12_true': np.nan, 'P22_true': p22_true_MPa,
                'mode': 'PS', 'dataset': dataset_name,
                'noise_sigma': rel_noise,
                'strain_pct': abs(l1_ps - 1.0) * 100.0
            }
            all_synth_data.append(data_point)

    # --- Equi-biaxial Tension ---
    print(f'  Generating Equi-biaxial (EBT) data (λ = {lambda_range}, {n_points} points)...')
    for lam_val in lambdas:
        l1_ebt = max(float(lam_val), context.EPS)
        l2_ebt = l1_ebt  # Stretch is equal in both directions
        l3_ebt = max(l1_ebt**(-2.0), context.EPS)  # Incompressible: l1 * l2 * l3 = 1

        deriv_vecs = calculate_dW_dl_vectors(l1_ebt, l2_ebt, l3_ebt, basis_names, context=context)
        if deriv_vecs is None: continue

        dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs

        dW_total_dl1_Pa = 0.0; dW_total_dl2_Pa = 0.0; dW_total_dl3_Pa = 0.0
        for term, coeff_Pa in term_coeff_map.items():
            idx = term_indices[term]
            dW_total_dl1_Pa += coeff_Pa * dW_dl1_all[idx]
            dW_total_dl2_Pa += coeff_Pa * dW_dl2_all[idx]
            dW_total_dl3_Pa += coeff_Pa * dW_dl3_all[idx]

        p11_true_Pa = dW_total_dl1_Pa - (l3_ebt / l1_ebt) * dW_total_dl3_Pa if l1_ebt > context.EPS else 0.0
        p22_true_Pa = dW_total_dl2_Pa - (l3_ebt / l2_ebt) * dW_total_dl3_Pa if l2_ebt > context.EPS else 0.0
        p11_true_MPa = p11_true_Pa * pa_to_mpa
        p22_true_MPa = p22_true_Pa * pa_to_mpa

        for rel_noise in relative_noise_levels:
            noise11_std_dev_Pa = rel_noise * abs(p11_true_Pa)
            noise22_std_dev_Pa = rel_noise * abs(p22_true_Pa)

            if abs(p11_true_Pa) < context.EPS and rel_noise > 0: noise11_std_dev_Pa = rel_noise * (context.EPS * 1000)
            if abs(p22_true_Pa) < context.EPS and rel_noise > 0: noise22_std_dev_Pa = rel_noise * (context.EPS * 1000)

            noise11_Pa = np.random.normal(0, noise11_std_dev_Pa)
            noise22_Pa = np.random.normal(0, noise22_std_dev_Pa)
            P11_noisy_MPa = p11_true_Pa + noise11_Pa
            P22_noisy_MPa = p22_true_Pa + noise22_Pa
            P11_noisy_MPa = P11_noisy_MPa * pa_to_mpa
            P22_noisy_MPa = P22_noisy_MPa * pa_to_mpa

            dataset_name = f'{model_name}_RelNoise{rel_noise * 100:.0f}pct'

            data_point = {
                'lambda1': l1_ebt, 'lambda2': l2_ebt, 'lambda3': l3_ebt,
                'gamma': np.nan,
                'P11': P11_noisy_MPa, 'P22': P22_noisy_MPa, 'P12': np.nan,
                'P11_true': p11_true_MPa, 'P12_true': np.nan, 'P22_true': p22_true_MPa,
                'mode': 'EBT', 'dataset': dataset_name,
                'noise_sigma': rel_noise,
                'strain_pct': abs(l1_ebt - 1.0) * 100.0
            }
            all_synth_data.append(data_point)

    print(f'--- Finished generating data [Framework Logic] for: {model_name} ---')
    df = pd.DataFrame(all_synth_data)
    num_cols = ['lambda1', 'lambda2', 'lambda3', 'gamma', 'P11', 'P22', 'P12',
                'P11_true', 'P12_true', 'P22_true', 'strain_pct', 'noise_sigma']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['P11'])
    return df
