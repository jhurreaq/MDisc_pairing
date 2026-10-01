"""Treloar models; original numerical and presentation choices."""

import numpy as np
import pandas as pd
import re
from mdisc.isotropic import mooney_rivlin_terms
from mdisc.isotropic import compute_invariants
from mdisc.isotropic import compute_invariant_derivatives
from mdisc.numerics import clipped_safe_power as safe_power

def generate_model_library(*, context):
    """Generates a list of basis function strings based on configuration flags."""
    if context.INCLUDE_OGDEN:
        if context.INCLUDE_EXTENDED_OGDEN:
            STANDARD_OGDEN_EXPONENTS_local = [-4, -3, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 3, 4]
            SMALL_STRAIN_OGDEN_EXPONENTS_local = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]
            context.OGDEN_EXPONENTS = [-4, -3, -1, 1, 3, 4] #STANDARD_OGDEN_EXPONENTS_local + SMALL_STRAIN_OGDEN_EXPONENTS_local if INCLUDE_ULTRA_SMALL_STRAIN else STANDARD_OGDEN_EXPONENTS_local
        else:
            context.OGDEN_EXPONENTS = [-4, -3, -1, 1, 3, 4] #[-4, -3, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 3, 4]
    else:
        context.OGDEN_EXPONENTS = []

    basis_functions = []
    seen_terms = set()
    def add_term(term_string):
        if term_string not in seen_terms:
            basis_functions.append(term_string)
            seen_terms.add(term_string)
    if context.INCLUDE_NEOHOOKEAN and not context.INCLUDE_MOONEY_RIVLIN: add_term("NH: (I₁-3)")
    if context.INCLUDE_MOONEY_RIVLIN:
        # max_total_power = MR_MAX_POWER_I1 + MR_MAX_POWER_I2
        for term in mooney_rivlin_terms(context.max_total_power, max_i1=context.MR_MAX_POWER_I1, max_i2=context.MR_MAX_POWER_I2):
            add_term(term)
    if context.INCLUDE_OGDEN:
        for a in context.OGDEN_EXPONENTS:
             if a != 0: add_term(f'Ogden: λ^{a}')
    print(f'Generated {len(basis_functions)} unique basis terms for this scenario.')
    return basis_functions

def construct_derivatives_matrix(df, *, context):
    num_samples = len(df)
    basis_names = generate_model_library(context=context)
    num_basis = len(basis_names)
    dW_dl1 = np.zeros((num_samples, num_basis)); dW_dl2 = np.zeros((num_samples, num_basis)); dW_dl3 = np.zeros((num_samples, num_basis))
    i1_power_pattern = re.compile(r'\(I₁-3\)(?:\^([\d\.-]+))?'); i2_power_pattern = re.compile(r'\(I₂-3\)(?:\^([\d\.-]+))?')
    ogden_power_pattern = re.compile(r'λ\^([\-\+\d\.]*\d)')
    for idx in range(num_samples):
        row = df.iloc[idx]; l1, l2 = max(row['lambda1'], context.EPS), max(row['lambda2'], context.EPS)
        l3_inv_prod = l1 * l2; l3 = max(1.0 / l3_inv_prod if l3_inv_prod > context.EPS else 1.0 / context.EPS, context.EPS)
        I1, I2 = compute_invariants(l1, l2, l3)
        (dI1dl1, dI1dl2, dI1dl3), (dI2dl1, dI2dl2, dI2dl3) = compute_invariant_derivatives(l1, l2, l3)
        i1m3 = I1 - 3.0; i2m3 = I2 - 3.0; i1m3_safe = max(i1m3, 0.0); i2m3_safe = max(i2m3, 0.0)
        for basis_idx, term_name in enumerate(basis_names):
            dW_dI1_term = 0.0; dW_dI2_term = 0.0; dWterm_dl1 = 0.0; dWterm_dl2 = 0.0; dWterm_dl3 = 0.0
            if term_name.startswith("MR:"):
                math_part = term_name.split(":", 1)[1].strip(); i_match = i1_power_pattern.search(math_part); j_match = i2_power_pattern.search(math_part)
                i = float(i_match.group(1)) if i_match and i_match.group(1) else (1 if i_match else 0)
                j = float(j_match.group(1)) if j_match and j_match.group(1) else (1 if j_match else 0)
                if i > 0: dW_dI1_term = i * safe_power(i1m3_safe, i - 1.0, eps=context.EPS) * safe_power(i2m3_safe, j, eps=context.EPS)
                if j > 0: dW_dI2_term = safe_power(i1m3_safe, i, eps=context.EPS) * j * safe_power(i2m3_safe, j - 1.0, eps=context.EPS)
            elif term_name.startswith("NH:"):
                match = i1_power_pattern.search(term_name); n = float(match.group(1)) if match and match.group(1) else (1 if match else 0)
                if n > 0: dW_dI1_term = n * safe_power(i1m3_safe, n - 1.0, eps=context.EPS)
            elif term_name.startswith("Ogden:"):
                match = ogden_power_pattern.search(term_name); alpha = float(match.group(1)) if match else 0.0
                if alpha != 0.0:
                    dWterm_dl1 = alpha * safe_power(l1, alpha - 1.0, eps=context.EPS); dWterm_dl2 = alpha * safe_power(l2, alpha - 1.0, eps=context.EPS); dWterm_dl3 = alpha * safe_power(l3, alpha - 1.0, eps=context.EPS)
                    dW_dI1_term = 0.0; dW_dI2_term = 0.0
            if not term_name.startswith("Ogden:"):
                dWterm_dl1 = dW_dI1_term * dI1dl1 + dW_dI2_term * dI2dl1; dWterm_dl2 = dW_dI1_term * dI1dl2 + dW_dI2_term * dI2dl2; dWterm_dl3 = dW_dI1_term * dI1dl3 + dW_dI2_term * dI2dl3
            dW_dl1[idx, basis_idx] = dWterm_dl1; dW_dl2[idx, basis_idx] = dWterm_dl2; dW_dl3[idx, basis_idx] = dWterm_dl3
    dW_dl1 = np.nan_to_num(dW_dl1, nan=0.0, posinf=1e12, neginf=-1e12); dW_dl2 = np.nan_to_num(dW_dl2, nan=0.0, posinf=1e12, neginf=-1e12); dW_dl3 = np.nan_to_num(dW_dl3, nan=0.0, posinf=1e12, neginf=-1e12)
    return dW_dl1, dW_dl2, dW_dl3, basis_names

def weighting_function(strain_pct, emphasize_small_strain=True, weight_factor=1.0, small_strain_threshold=50.0):
    """Applies weights to data points, potentially emphasizing small strains."""
    weights = np.ones_like(strain_pct, dtype=float)
    if not emphasize_small_strain:
        print("INFO: Weighting disabled.")
        return weights
    small_strain_mask = strain_pct <= small_strain_threshold
    num_weighted = np.sum(small_strain_mask)
    if num_weighted > 0:
        weights[small_strain_mask] = weight_factor
        print(f'INFO: Applied weight factor {weight_factor} to {num_weighted} points <= {small_strain_threshold}% strain.')
    else:
        print("INFO: Weighting enabled, but no points met the small strain threshold.")
    return weights

def construct_design_matrix(df, use_small_strain_only=False, emphasize_small_strain=True, weight_factor=1.0, small_strain_threshold=50.0, *, context):
    """
    Constructs the design matrix Phi and target vector y for regression.
    Design matrix rows correspond to PK1 (Nominal) stress predictions
    assuming plane stress (sigma_3 = 0), even for plane strain data (like Treloar PS).
    This is a common approximation. Weighting is applied if enabled.

    Args:
        df (pd.DataFrame): Input dataframe with lambda1, lambda2, P11, P22, mode, strain_pct.
        use_small_strain_only (bool): If True, use only data below the threshold.
        emphasize_small_strain (bool): If True, apply weighting to small strains.
        weight_factor (float): Weight factor for small strains.
        small_strain_threshold (float): Strain threshold for weighting.

    Returns:
        tuple: (design_matrix, target_vector, basis_names, row_weights_vector, df_subset)
               Returns empty/zero arrays on error.
    """

    if use_small_strain_only:
        df_subset = df[df['strain_pct'] <= small_strain_threshold].copy()
        print(f'Using small strain data subset: {len(df_subset)} points.')
        if len(df_subset) < 5:
            print(f'WARN: Not enough small strain points ({len(df_subset)}). Using full dataset instead.')
            df_subset = df.copy()
    else:
        df_subset = df.copy()

    df_subset = df_subset.reset_index(drop=True) # Ensure index is sequential for mapping later

    # Calculate weights based on the subset being used
    point_weights = weighting_function(df_subset['strain_pct'].values, emphasize_small_strain, weight_factor, small_strain_threshold)

    dW1_mat, dW2_mat, dW3_mat, basis_names = construct_derivatives_matrix(df_subset, context=context)

    n_samp, n_basis = dW1_mat.shape
    if n_basis == 0:
        print("ERROR: Model library empty.")
        return np.zeros((0,0)), np.zeros((0,)), [], np.zeros((0,)), pd.DataFrame()

    # Determine the required size of the design matrix and target vector
    n_p11_rows = df_subset[pd.notna(df_subset['P11'])].shape[0]
    n_p22_rows = df_subset[df_subset['mode'].isin(['EBT', 'BIAX']) & pd.notna(df_subset['P22'])].shape[0]
    total_rows = n_p11_rows + n_p22_rows

    design_matrix = np.zeros((total_rows, n_basis))
    target_vector = np.zeros(total_rows)
    row_weights_vector = np.zeros(total_rows) # Store weight associated with each row of Phi/y
    original_indices_map = [] # Track which original df index corresponds to which row range

    current_row_idx = 0
    processed_indices = set() # Track original df indices processed

    for idx in range(n_samp): # Iterate through the subset dataframe
        row_data = df_subset.iloc[idx]
        l1, l2 = max(row_data['lambda1'], context.EPS), max(row_data['lambda2'], context.EPS)
        l3_inv_prod = l1 * l2
        l3 = max(1.0 / l3_inv_prod if l3_inv_prod > context.EPS else 1.0 / context.EPS, context.EPS)
        mode = row_data['mode']
        p_weight = point_weights[idx] # Weight for this specific lambda point

        dW1_vec = dW1_mat[idx, :]
        dW2_vec = dW2_mat[idx, :]
        dW3_vec = dW3_mat[idx, :]

        has_p11 = pd.notna(row_data['P11'])
        has_p22 = pd.notna(row_data['P22']) and mode in ['EBT', 'BIAX']

        start_row_for_idx = current_row_idx # Store starting row for this df index

        if has_p11:
            if current_row_idx < total_rows:
                l3_over_l1 = l3 / l1 if l1 > context.EPS else l3 / context.EPS
                phi_row_p11 = dW1_vec - l3_over_l1 * dW3_vec
                # Apply weight to BOTH phi_row and target
                design_matrix[current_row_idx, :] = p_weight * phi_row_p11
                target_vector[current_row_idx] = p_weight * row_data['P11']
                row_weights_vector[current_row_idx] = p_weight # Store the weight used for this row
                current_row_idx += 1
                processed_indices.add(idx)
            else:
                print(f'WARN: Row index {current_row_idx} exceeds expected total {total_rows} adding P11 idx {idx}.')

        if has_p22:
            if current_row_idx < total_rows:
                l3_over_l2 = l3 / l2 if l2 > context.EPS else l3 / context.EPS
                phi_row_p22 = dW2_vec - l3_over_l2 * dW3_vec
                # Apply weight to BOTH phi_row and target
                design_matrix[current_row_idx, :] = p_weight * phi_row_p22
                target_vector[current_row_idx] = p_weight * row_data['P22']
                row_weights_vector[current_row_idx] = p_weight # Store the weight used for this row
                current_row_idx += 1
                processed_indices.add(idx)
            else:
                print(f'WARN: Row index {current_row_idx} exceeds expected total {total_rows} adding P22 idx {idx}.')

        end_row_for_idx = current_row_idx # Store ending row (exclusive)
        if end_row_for_idx > start_row_for_idx: # If rows were added for this index
             original_indices_map.append({
                 'original_df_index': df_subset.index[idx], # Get original index BEFORE reset_index
                 'matrix_row_start': start_row_for_idx,
                 'matrix_row_end': end_row_for_idx
             })


    if current_row_idx != total_rows:
        print(f'WARN: Final row index ({current_row_idx}) != Expected total ({total_rows}). Resizing arrays.')
        design_matrix = design_matrix[:current_row_idx, :]
        target_vector = target_vector[:current_row_idx]
        row_weights_vector = row_weights_vector[:current_row_idx]

    # Check for non-finite values (can happen with extreme derivatives/powers)
    finite_mask = np.all(np.isfinite(design_matrix), axis=1) & np.isfinite(target_vector)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        print(f'WARN: Removing {num_removed} rows from design matrix/target due to non-finite values.')
        design_matrix = design_matrix[finite_mask, :]
        target_vector = target_vector[finite_mask]
        row_weights_vector = row_weights_vector[finite_mask]

    if design_matrix.shape[0] == 0:
        print("ERROR: Design matrix empty after filtering.")
        return np.zeros((0, n_basis)), np.zeros((0,)), basis_names, np.zeros((0,)), df_subset

    print(f"Constructed design matrix ({('Weighted' if emphasize_small_strain else 'Unweighted')}): {design_matrix.shape}")
    # Pass df_subset back, as it corresponds to the rows used (before filtering non-finite)
    # The mapping back to original df needs care in the plotting function.
    return design_matrix, target_vector, basis_names, row_weights_vector, df_subset

def check_physical_consistency(final_terms, final_coeffs, *, context):
    """
    Performs basic physical consistency checks on the final model coefficients.
    Focuses on initial shear modulus (mu_0 > 0).
    CORRECTED: Assumes Ogden coefficient 'coeff' corresponds to C_i = mu_i / alpha_i
               in the standard SEF W = sum(C_i * (lambda_sum**alpha_i - 3)),
               and applies the correct mu_0 formula.

    Args:
        final_terms (list): List of final selected basis function names.
        final_coeffs (np.ndarray): Array of corresponding final coefficients.

    Returns:
        tuple: (is_consistent, shear_modulus, consistency_notes)
               is_consistent (bool): True if mu_0 > EPS.
               shear_modulus (float): Calculated initial shear modulus mu_0.
               consistency_notes (list): List of strings detailing checks.
    """

    if not isinstance(final_terms, list) or not isinstance(final_coeffs, np.ndarray):
         return False, np.nan, ["Error: Invalid input types for terms/coeffs."]

    if len(final_terms) != len(final_coeffs):
         if len(final_terms) == 0 and len(final_coeffs) == 0:
              return False, 0.0, ["No terms remaining after thresholding."]
         else:
              return False, np.nan, [f'Error: Mismatched length of terms ({len(final_terms)}) and coeffs ({len(final_coeffs)}).']

    if len(final_terms) == 0:
        return False, 0.0, ["No terms remaining after thresholding."]


    term_coeff_map = {term: coeff for term, coeff in zip(final_terms, final_coeffs)}
    notes = []
    # Initialize contributions separately for clarity
    mr_contrib_sum = 0.0  # Will store (C10 + C01)
    ogden_contrib_sum_mu_alpha = 0.0 # Will store sum(mu_i * alpha_i)
    ogden_term_found = False
    mr_c10_term = 'MR: (I₁-3)'
    mr_c01_term = 'MR: (I₂-3)'

    # --- Contribution from MR terms ---
    c10_coeff = term_coeff_map.get(mr_c10_term, 0.0)
    c01_coeff = term_coeff_map.get(mr_c01_term, 0.0)

    if mr_c10_term in term_coeff_map:
        notes.append(f"  MR C10 Term ('{mr_c10_term}') Coeff (C10) = {c10_coeff:.4f}")
        if c10_coeff <= context.EPS:
            notes.append("    WARN: C10 component is non-positive.")

    if mr_c01_term in term_coeff_map:
        notes.append(f"  MR C01 Term ('{mr_c01_term}') Coeff (C01) = {c01_coeff:.4f}")
        if c01_coeff < -context.EPS:
             notes.append("    INFO: C01 component is negative.")

    # Store sum for MR contribution part: (C10 + C01)
    mr_contrib_sum = c10_coeff + c01_coeff

    # --- Contribution from Ogden terms ---
    ogden_pattern = re.compile(r'Ogden: λ\^([\-\+\d\.]+)$') # Slightly stricter regex
    ogden_coeffs_details = [] # Keep for detailed notes if needed

    for term, coeff in term_coeff_map.items():
        match = ogden_pattern.search(term)
        if match:
            try:
                alpha_i = float(match.group(1))

                if abs(alpha_i) < context.EPS:
                    mu_i = 0.0
                    notes.append(f"  WARN: Ogden term '{term}' has alpha near zero. mu_i set to 0.")
                else:
                    mu_i = coeff * alpha_i

                mu_alpha_product = mu_i * alpha_i
                ogden_contrib_sum_mu_alpha += mu_alpha_product # Sum (mu_i * alpha_i)

                ogden_coeffs_details.append({'coeff_Ci': coeff, 'mu_i': mu_i, 'alpha': alpha_i, 'mu_alpha': mu_alpha_product})
                notes.append(f'  Ogden Term: Coeff (C_i={coeff:.4f}), α={alpha_i:.4f} => Implied μ_i={mu_i:.4f}. (μ_i*α_i = {mu_alpha_product:.4f})')
                ogden_term_found = True
            except ValueError:
                notes.append(f"  WARN: Could not parse Ogden exponent in '{term}'.")
            except Exception as e:
                 notes.append(f"  ERROR processing Ogden term '{term}': {e}")

    # --- Calculate Final Shear Modulus using correct formula ---
    # mu_0 = 2 * (C10 + C01) + 0.5 * sum(mu_i * alpha_i)
    shear_modulus = 2.0 * mr_contrib_sum + 0.5 * ogden_contrib_sum_mu_alpha

    # --- Final Shear Modulus Check ---
    notes.insert(0, f'Calculated initial shear modulus μ₀ ≈ {shear_modulus:.4f} (using μ₀ = 2(C10+C01) + 0.5*Σ(μᵢαᵢ))')
    if ogden_term_found:
        notes.append(f'  MR contribution (2*(C10+C01)) = {2.0 * mr_contrib_sum:.4f}')
        notes.append(f'  Ogden contribution (0.5*Σ(μᵢαᵢ)) = {0.5 * ogden_contrib_sum_mu_alpha:.4f}')
        notes.append("  NOTE: Ogden consistency assumes discovered 'coeff' C_i relates to standard μ_i via μ_i = C_i * α_i.")


    is_consistent = True
    if shear_modulus <= context.EPS: # Use EPS for safety margin around zero
        notes.append("  CONSISTENCY FAIL: Initial shear modulus μ₀ is non-positive.")
        is_consistent = False
    else:
        notes.append("  CONSISTENCY PASS: Initial shear modulus μ₀ is positive.")


    return is_consistent, shear_modulus, notes
