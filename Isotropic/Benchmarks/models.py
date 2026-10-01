"""Synthetic models; original numerical and presentation choices."""

from mdisc.isotropic import mooney_rivlin_terms
import numpy as np
import pandas as pd
import re
from mdisc.isotropic import compute_invariant_derivatives
from mdisc.numerics import exact_safe_power as safe_power

def generate_model_library(*, context):
    """
    Generates a list of basis function strings based on configuration flags.
    Uses the OGDEN_EXPONENTS list already defined globally in the main block.
    """
    # OGDEN_EXPONENTS should already be defined globally and populated correctly
    # in the __main__ block before this function is called.
    # Access global variables needed.

    # --- Determine which Ogden exponents list to use ---
    # Check if OGDEN_EXPONENTS was properly set globally via main block
    if context.INCLUDE_OGDEN and (isinstance(context.OGDEN_EXPONENTS, list) and context.OGDEN_EXPONENTS):
         # Use the list set globally in main if it exists and is not empty
         ogden_exponents_to_use = context.OGDEN_EXPONENTS
         print(f'  generate_model_library: Using globally set OGDEN_EXPONENTS = {ogden_exponents_to_use}')
    elif context.INCLUDE_OGDEN:
         # Fallback if not set globally
         print("WARN: Global OGDEN_EXPONENTS not set. Falling back based on INCLUDE_EXTENDED_OGDEN.")
         if context.INCLUDE_EXTENDED_OGDEN:
             STANDARD_OGDEN_EXPONENTS_local = [-4, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 2, 4]
             ogden_exponents_to_use = STANDARD_OGDEN_EXPONENTS_local
         else:
             ogden_exponents_to_use = [-2, -1, -0.5, 0.5, 1, 2, 3]
         print(f'  generate_model_library: Fallback OGDEN_EXPONENTS = {ogden_exponents_to_use}')
    else:
         ogden_exponents_to_use = [] # No Ogden terms included


    basis_functions = []
    seen_terms = set()
    def add_term(term_string):
        if term_string not in seen_terms:
            basis_functions.append(term_string)
            seen_terms.add(term_string)

    # --- Term generation logic (uses the correct ogden_exponents_to_use) ---
    if context.INCLUDE_NEOHOOKEAN and not context.INCLUDE_MOONEY_RIVLIN: add_term("NH: (I₁-3)")

    if context.INCLUDE_MOONEY_RIVLIN:
        # Generate MR terms with total order i+j <= MR_MAX_ORDER
        for term in mooney_rivlin_terms(context.MR_MAX_ORDER):
            add_term(term)
    if context.INCLUDE_OGDEN:
        for a in ogden_exponents_to_use:
            if a != 0: add_term(f'Ogden: λ^{a}') # Format Ogden terms
    print(f'Generated {len(basis_functions)} unique basis terms for this scenario (using determined OGDEN_EXPONENTS).')
    return basis_functions

def construct_design_matrix(df, normalize_by_stress_magnitude=True, *, context):
    """
    Constructs the design matrix Phi and target vector y for regression.
    Includes automatic weight normalization based on stress magnitudes.
    """
    df_subset = df.copy()
    df_subset = df_subset.reset_index(drop=True)

    dW1_mat, dW2_mat, dW3_mat, basis_names = construct_derivatives_matrix(df_subset, context=context)
    n_samp, n_basis = dW1_mat.shape

    if n_basis == 0:
        print("ERROR: Model library empty.")
        return np.zeros((0,0)), np.zeros((0,)), [], np.zeros((0,)), pd.DataFrame()

    # Calculate stress magnitude normalization weights
    if normalize_by_stress_magnitude:
        print("  Computing automatic stress magnitude normalization weights...")

        # Collect stress magnitudes by mode for normalization
        stress_stats = {}
        for mode in ['UT', 'PS', 'EBT', 'BIAX']:
            mode_data = df_subset[df_subset['mode'] == mode]
            if not mode_data.empty:
                # Collect all stress values for this mode
                all_stresses = []
                # Check if P11 column exists before accessing it.
                if 'P11' in mode_data.columns and pd.notna(mode_data['P11']).any():
                    all_stresses.extend(mode_data['P11'].dropna().abs().values)
                # Check if P22 column exists before accessing it.
                if 'P22' in mode_data.columns and mode in ['PS', 'EBT', 'BIAX'] and pd.notna(mode_data['P22']).any():
                    all_stresses.extend(mode_data['P22'].dropna().abs().values)

                if all_stresses:
                    stress_stats[mode] = {
                        'rms': np.sqrt(np.mean(np.array(all_stresses)**2)),
                        'max': np.max(all_stresses),
                        'std': np.std(all_stresses),
                        'n_points': len(all_stresses)
                    }
                    print(f"    {mode} mode: RMS={stress_stats[mode]['rms']:.3f} MPa, Max={stress_stats[mode]['max']:.3f} MPa, n={stress_stats[mode]['n_points']}")

        # Calculate global normalization factor (use RMS across all modes)
        all_rms_values = [stats['rms'] for stats in stress_stats.values()]
        if all_rms_values:
            global_rms = np.sqrt(np.mean(np.array(all_rms_values)**2))
            print(f'    Global RMS stress: {global_rms:.3f} MPa')

            # Calculate per-mode normalization weights (inverse of RMS to equalize contributions)
            mode_weights = {}
            for mode, stats in stress_stats.items():
                mode_weights[mode] = global_rms / stats['rms'] if stats['rms'] > context.EPS else 1.0
                print(f'    {mode} mode normalization weight: {mode_weights[mode]:.3f}')
        else:
            print("    WARN: No valid stress data found for normalization. Using equal weights.")
            mode_weights = {}
    else:
        mode_weights = {}
        print("  Stress magnitude normalization disabled.")

    # Determine matrix size
    # Check if P11 column exists before calculating rows.
    n_p11_rows = df_subset[pd.notna(df_subset['P11'])].shape[0] if 'P11' in df_subset.columns else 0
    # Check if P22 column exists before calculating rows.
    n_p22_rows = 0
    if 'P22' in df_subset.columns:
        n_p22_rows = df_subset[df_subset['mode'].isin(['EBT', 'BIAX', 'PS']) & pd.notna(df_subset['P22'])].shape[0]
    total_rows = n_p11_rows + n_p22_rows

    design_matrix = np.zeros((total_rows, n_basis))
    target_vector = np.zeros(total_rows)
    row_weights_vector = np.zeros(total_rows)
    original_indices_map = []

    current_row_idx = 0
    processed_indices = set()

    for idx in range(n_samp):
        row_data = df_subset.iloc[idx]
        l1, l2 = max(row_data['lambda1'], context.EPS), max(row_data['lambda2'], context.EPS)
        l3_inv_prod = l1 * l2
        l3 = max(1.0 / l3_inv_prod if l3_inv_prod > context.EPS else 1.0 / context.EPS, context.EPS)
        mode = row_data['mode']

        # Apply mode-specific normalization weight (uniform base weights)
        mode_norm_weight = mode_weights.get(mode, 1.0)
        final_weight = mode_norm_weight

        dW1_vec = dW1_mat[idx, :]
        dW2_vec = dW2_mat[idx, :]
        dW3_vec = dW3_mat[idx, :]

        # Check for column existence before checking for NaN.
        has_p11 = 'P11' in row_data and pd.notna(row_data['P11'])
        has_p22 = 'P22' in row_data and pd.notna(row_data['P22']) and mode in ['EBT', 'BIAX', 'PS']

        start_row_for_idx = current_row_idx

        if has_p11:
            if current_row_idx < total_rows:
                l3_over_l1 = l3 / l1 if l1 > context.EPS else l3 / context.EPS
                phi_row_p11 = dW1_vec - l3_over_l1 * dW3_vec
                design_matrix[current_row_idx, :] = final_weight * phi_row_p11
                target_vector[current_row_idx] = final_weight * row_data['P11']
                row_weights_vector[current_row_idx] = final_weight
                current_row_idx += 1

        if has_p22:
            if current_row_idx < total_rows:
                l3_over_l2 = l3 / l2 if l2 > context.EPS else l3 / context.EPS
                phi_row_p22 = dW2_vec - l3_over_l2 * dW3_vec
                design_matrix[current_row_idx, :] = final_weight * phi_row_p22
                target_vector[current_row_idx] = final_weight * row_data['P22']
                row_weights_vector[current_row_idx] = final_weight
                current_row_idx += 1

        end_row_for_idx = current_row_idx
        if end_row_for_idx > start_row_for_idx:
             original_indices_map.append({
                 'original_df_index': df_subset.index[idx],
                 'matrix_row_start': start_row_for_idx,
                 'matrix_row_end': end_row_for_idx
             })

    # Handle size mismatches and non-finite values
    if current_row_idx != total_rows:
        print(f'WARN: Final row index ({current_row_idx}) != Expected total ({total_rows}). Resizing arrays.')
        design_matrix = design_matrix[:current_row_idx, :]
        target_vector = target_vector[:current_row_idx]
        row_weights_vector = row_weights_vector[:current_row_idx]

    finite_mask = np.all(np.isfinite(design_matrix), axis=1) & np.isfinite(target_vector)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        print(f'WARN: Removing {num_removed} rows due to non-finite values.')
        design_matrix = design_matrix[finite_mask, :]
        target_vector = target_vector[finite_mask]
        row_weights_vector = row_weights_vector[finite_mask]

    if design_matrix.shape[0] == 0:
        print("ERROR: Design matrix empty after filtering.")
        return np.zeros((0, n_basis)), np.zeros((0,)), basis_names, np.zeros((0,)), df_subset

    print(f'Constructed design matrix (Stress-Normalized: {normalize_by_stress_magnitude}): {design_matrix.shape}')
    return design_matrix, target_vector, basis_names, row_weights_vector, df_subset

def calculate_dW_dl_vectors(l1, l2, l3, basis_names, *, context):
    """
    Calculates the derivative vectors dW/dl1, dW/dl2, dW/dl3 for each
    basis function, with checks for stability. Returns None on critical error.
    """
    num_basis = len(basis_names)
    if num_basis == 0: return None

    # Ensure inputs are floats and positive, using original numpy for safety here
    l1, l2, l3 = max(float(l1), context.EPS), max(float(l2), context.EPS), max(float(l3), context.EPS)

    # Use original numpy for array initialization
    dW_dl1_vec = np.zeros(num_basis)
    dW_dl2_vec = np.zeros(num_basis)
    dW_dl3_vec = np.zeros(num_basis)

    # Pre-calculate invariants and their derivatives using standard numpy helpers
    try:
        # Assuming compute_invariant_derivatives uses standard numpy
        (dI1dl1, dI1dl2, dI1dl3), (dI2dl1, dI2dl2, dI2dl3) = compute_invariant_derivatives(l1, l2, l3)
        I1 = l1**2 + l2**2 + l3**2 # Calculate I1 based on current stretches
        I2 = (l1*l2)**2 + (l2*l3)**2 + (l1*l3)**2 # Calculate I2
        i1m3 = I1 - 3.0
        i2m3 = I2 - 3.0

        # Check invariant derivatives are finite
        inv_derivs = [dI1dl1, dI1dl2, dI1dl3, dI2dl1, dI2dl2, dI2dl3]
        if not np.all(np.isfinite(inv_derivs)):
            print(f'WARN: Non-finite invariant derivatives at l1,l2,l3 = {l1:.3f},{l2:.3f},{l3:.3f}. Setting to 0.')
            dI1dl1, dI1dl2, dI1dl3, dI2dl1, dI2dl2, dI2dl3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    except Exception as e:
        print(f'ERROR calculating invariants/derivs at l1,l2,l3 = {l1:.3f},{l2:.3f},{l3:.3f}: {e}')
        return None # Cannot proceed if this fails

    # Pre-compile regex patterns for active model types only
    i1_power_pattern = re.compile(r'\(I₁-3\)(?:\^([\d\.-]+))?')
    i2_power_pattern = re.compile(r'\(I₂-3\)(?:\^([\d\.-]+))?')
    ogden_power_pattern = re.compile(r'Ogden: λ\^([\-\+\d\.]+)$')

    FINITE_LIMIT_DERIV = 1e12 # Limit for individual derivative components

    for basis_idx, term_name in enumerate(basis_names):
        dW_dI1_term = 0.0; dW_dI2_term = 0.0
        dWterm_dl1 = 0.0; dWterm_dl2 = 0.0; dWterm_dl3 = 0.0
        term_is_finite = True # Assume finite initially

        try:
            # --- Calculate dW/dIi or direct dW/dli for this term ---
            # --- Mooney-Rivlin & Neo-Hookean terms ---
            if term_name.startswith(("MR:", "NH:")):
                math_part = term_name.split(":", 1)[1].strip(); i_match = i1_power_pattern.search(math_part); j_match = i2_power_pattern.search(math_part)
                i = float(i_match.group(1)) if i_match and i_match.group(1) else (1 if i_match else 0)
                j = float(j_match.group(1)) if j_match and j_match.group(1) else (1 if j_match else 0)
                if not j_match and term_name.startswith("NH:"): j = 0.0
                elif not i_match and term_name.startswith("MR:") and j_match: i = 0.0
                # Use safe_power, check results before calculating derivatives
                pow_i1 = safe_power(i1m3, i - 1.0) if i > 0 else (safe_power(i1m3, i) if j > 0 else 1.0)
                pow_i2 = safe_power(i2m3, j) if i > 0 else (safe_power(i2m3, j - 1.0) if j > 0 else 1.0)
                pow_i1_full = safe_power(i1m3, i) if j > 0 else 1.0
                pow_i2_full = safe_power(i2m3, j) if i > 0 else 1.0

                if not (np.isfinite(pow_i1) and np.isfinite(pow_i2) and np.isfinite(pow_i1_full) and np.isfinite(pow_i2_full)):
                    term_is_finite = False
                else:
                    if i > 0: dW_dI1_term = i * pow_i1 * pow_i2_full
                    if j > 0: dW_dI2_term = j * pow_i1_full * pow_i2
            # --- Ogden ---
            elif term_name.startswith("Ogden:"):
                match = ogden_power_pattern.search(term_name)
                if match:
                    try:
                        alpha = float(match.group(1))
                        if alpha != 0.0:
                            pow_l1 = safe_power(l1, alpha - 1.0)
                            pow_l2 = safe_power(l2, alpha - 1.0)
                            pow_l3 = safe_power(l3, alpha - 1.0)
                            dWterm_dl1 = alpha * pow_l1
                            dWterm_dl2 = alpha * pow_l2
                            dWterm_dl3 = alpha * pow_l3
                            if not np.all(np.isfinite([dWterm_dl1, dWterm_dl2, dWterm_dl3])):
                                term_is_finite = False
                    except (ValueError, TypeError):
                         print(f"WARN: Could not parse exponent for Ogden term '{term_name}'")
                         term_is_finite = False
                else:
                     print(f"WARN: Could not match exponent pattern for Ogden term '{term_name}'")
                     term_is_finite = False
                dW_dI1_term = 0.0; dW_dI2_term = 0.0 # Ensure these are zero

            # --- Check finiteness of dW/dIi before chain rule ---
            if not term_name.startswith("Ogden:") and not np.all(np.isfinite([dW_dI1_term, dW_dI2_term])):
                 term_is_finite = False

            # --- Combine using Chain Rule (if not Ogden and term hasn't failed) ---
            if not term_name.startswith("Ogden:") and term_is_finite:
                 if not np.all(np.isfinite([dI1dl1, dI1dl2, dI1dl3, dI2dl1, dI2dl2, dI2dl3])):
                     term_is_finite = False # Invariant derivs failed
                 else:
                    dWterm_dl1 = dW_dI1_term * dI1dl1 + dW_dI2_term * dI2dl1
                    dWterm_dl2 = dW_dI1_term * dI1dl2 + dW_dI2_term * dI2dl2
                    dWterm_dl3 = dW_dI1_term * dI1dl3 + dW_dI2_term * dI2dl3
                    # Final check on combined values
                    if not np.all(np.isfinite([dWterm_dl1, dWterm_dl2, dWterm_dl3])):
                         term_is_finite = False

            # Store NaN if term failed, otherwise store clipped value
            if term_is_finite:
                dW_dl1_vec[basis_idx] = np.clip(dWterm_dl1, -FINITE_LIMIT_DERIV, FINITE_LIMIT_DERIV)
                dW_dl2_vec[basis_idx] = np.clip(dWterm_dl2, -FINITE_LIMIT_DERIV, FINITE_LIMIT_DERIV)
                dW_dl3_vec[basis_idx] = np.clip(dWterm_dl3, -FINITE_LIMIT_DERIV, FINITE_LIMIT_DERIV)
            else:
                dW_dl1_vec[basis_idx] = np.nan
                dW_dl2_vec[basis_idx] = np.nan
                dW_dl3_vec[basis_idx] = np.nan

        except Exception as e:
             print(f"ERROR calculating derivatives for term '{term_name}' at l1={l1:.3f}, l2={l2:.3f}, l3={l3:.3f}: {e}")
             dW_dl1_vec[basis_idx] = np.nan # Use NaN on error
             dW_dl2_vec[basis_idx] = np.nan
             dW_dl3_vec[basis_idx] = np.nan

    # Return vectors possibly containing NaNs - let calling function handle them
    return dW_dl1_vec, dW_dl2_vec, dW_dl3_vec

def construct_derivatives_matrix(df, *, context):
    """
    Constructs matrices of dW/dl1, dW/dl2, dW/dl3 for all data points
    by calling the helper function for each point.
    """
    num_samples = len(df)
    # Generate library once - IMPORTANT: ensure global flags match discovery intent
    basis_names = generate_model_library(context=context)
    num_basis = len(basis_names)

    if num_basis == 0:
        print("ERROR: Model library is empty in construct_derivatives_matrix.")
        return np.zeros((num_samples, 0)), np.zeros((num_samples, 0)), np.zeros((num_samples, 0)), []

    # Initialize matrices
    dW_dl1_mat = np.zeros((num_samples, num_basis))
    dW_dl2_mat = np.zeros((num_samples, num_basis))
    dW_dl3_mat = np.zeros((num_samples, num_basis))

    # Calculate derivatives for each data point
    for idx in range(num_samples):
        row = df.iloc[idx]
        # Ensure stretches are valid floats
        l1 = float(row['lambda1']) if pd.notna(row['lambda1']) else 1.0
        l2 = float(row['lambda2']) if pd.notna(row['lambda2']) else 1.0
        l3 = float(row['lambda3']) if pd.notna(row['lambda3']) else 1.0

        # Ensure l3 is consistent if needed (e.g., from l1*l2*l3=1)
        # If l3 is NaN or invalid, recalculate assuming incompressibility
        if not pd.notna(l3) or l3 < context.EPS:
             l1_safe = max(l1, context.EPS); l2_safe = max(l2, context.EPS)
             l3 = max(1.0 / (l1_safe * l2_safe), context.EPS)

        # Call the helper function
        deriv_vecs = calculate_dW_dl_vectors(l1, l2, l3, basis_names, context=context)
        if deriv_vecs:
             dW_dl1_mat[idx, :] = deriv_vecs[0]
             dW_dl2_mat[idx, :] = deriv_vecs[1]
             dW_dl3_mat[idx, :] = deriv_vecs[2]
        # Else: error handled within helper, rows remain zero

    return dW_dl1_mat, dW_dl2_mat, dW_dl3_mat, basis_names

def check_physical_consistency(final_terms, final_coeffs, *, context):
    """
    Performs basic physical consistency checks on the final model coefficients.
    Focuses on initial shear modulus (mu_0 > 0).
    Assumes Ogden coefficient 'coeff' corresponds to C_i = mu_i / alpha_i.
    Reports implied coefficients in Pa.

    Args:
        final_terms (list): List of final selected basis function names.
        final_coeffs (np.ndarray): Array of corresponding final coefficients (in MPa).

    Returns:
        tuple: (is_consistent, shear_modulus, consistency_notes)
               is_consistent (bool): True if mu_0 > EPS.
               shear_modulus (float): Calculated initial shear modulus mu_0 (in MPa).
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

    # Coefficients from regression are in MPa, convert to Pa for reporting consistency if desired
    term_coeff_map_MPa = {term: coeff for term, coeff in zip(final_terms, final_coeffs)}
    term_coeff_map_Pa = {term: coeff * 1e6 for term, coeff in term_coeff_map_MPa.items()} # For notes

    notes = []
    shear_modulus_contrib_mr = 0.0 # Keep calculation in MPa for consistency with inputs
    shear_modulus_contrib_ogden = 0.0 # Keep calculation in MPa
    ogden_term_found = False
    mr_c10_term = 'MR: (I₁-3)'
    mr_c01_term = 'MR: (I₂-3)'

    # --- Contribution from MR terms ---
    c10_coeff_MPa = term_coeff_map_MPa.get(mr_c10_term, 0.0)
    c01_coeff_MPa = term_coeff_map_MPa.get(mr_c01_term, 0.0)
    c10_coeff_Pa = term_coeff_map_Pa.get(mr_c10_term, 0.0) # For notes
    c01_coeff_Pa = term_coeff_map_Pa.get(mr_c01_term, 0.0) # For notes


    if mr_c10_term in term_coeff_map_MPa:
        # Note: Report C10 in Pa, but use MPa value for mu0 calc consistency below
        notes.append(f"  MR C10 Term ('{mr_c10_term}') Coeff (C10) = {c10_coeff_Pa:.1f} Pa")
        if c10_coeff_MPa <= context.EPS: # Check MPa value against EPS
            notes.append("    WARN: C10 component is non-positive.")

    if mr_c01_term in term_coeff_map_MPa:
        # Note: Report C01 in Pa, but use MPa value for mu0 calc consistency below
        notes.append(f"  MR C01 Term ('{mr_c01_term}') Coeff (C01) = {c01_coeff_Pa:.1f} Pa")
        if c01_coeff_MPa < -context.EPS: # Check MPa value against EPS
             notes.append("    INFO: C01 component is negative.")

    # Calculate MR contribution to mu0/2 using MPa coefficients
    shear_modulus_contrib_mr = c10_coeff_MPa + c01_coeff_MPa

    # --- Contribution from Ogden terms ---
    ogden_pattern = re.compile(r'Ogden: λ\^([\-\+\d\.]+)$')
    sum_mu_alpha_ogden_MPa = 0.0 # Calculate this sum in MPa

    for term, coeff_MPa in term_coeff_map_MPa.items(): # Iterate using MPa coefficients
        match = ogden_pattern.search(term)
        if match:
            try:
                alpha_i = float(match.group(1))
                coeff_Pa = coeff_MPa * 1e6 # For notes

                # Calculate implied mu_i in MPa for consistency check
                if abs(alpha_i) < context.EPS:
                    mu_i_MPa = 0.0
                    notes.append(f"  WARN: Ogden term '{term}' has alpha near zero. Contribution to mu0 is zero.")
                else:
                    # mu_i = C_i * alpha_i, where C_i is coeff_MPa
                    mu_i_MPa = coeff_MPa * alpha_i
                # End per-term calculation

                mu_alpha_product_MPa = mu_i_MPa * alpha_i
                sum_mu_alpha_ogden_MPa += mu_alpha_product_MPa

                # Report coefficients in Pa for clarity in notes
                mu_i_Pa = mu_i_MPa * 1e6
                mu_alpha_product_Pa = mu_alpha_product_MPa * 1e6
                notes.append(f'  Ogden Term: Coeff (C_i={coeff_Pa:.1f} Pa), α={alpha_i:.4f} => Implied μ_i={mu_i_Pa:.1f} Pa. (μ_i*α_i = {mu_alpha_product_Pa:.1f} Pa)')
                ogden_term_found = True
            except ValueError:
                notes.append(f"  WARN: Could not parse Ogden exponent in '{term}'.")
            except Exception as e:
                 notes.append(f"  ERROR processing Ogden term '{term}': {e}")

    if ogden_term_found:
        notes.append(f'  Sum(μᵢ * αᵢ) for Ogden terms = {sum_mu_alpha_ogden_MPa * 1000000.0:.1f} Pa') # Report sum in Pa
        # Contribution to shear modulus = 0.5 * sum(mu*alpha) in MPa
        shear_modulus_contrib_ogden = 0.5 * sum_mu_alpha_ogden_MPa

    # --- Calculate Final Shear Modulus (in MPa) ---
    shear_modulus_MPa = 2.0 * shear_modulus_contrib_mr + shear_modulus_contrib_ogden

    # --- Final Shear Modulus Check ---
    # Report mu0 in both MPa and Pa for context
    notes.insert(0, f'Calculated initial shear modulus μ₀ ≈ {shear_modulus_MPa:.4f} MPa ({shear_modulus_MPa * 1000000.0:.1f} Pa)')

    is_consistent = True
    if shear_modulus_MPa <= context.EPS: # Check consistency using MPa value (or Pa, doesn't matter near zero)
        notes.append("  CONSISTENCY FAIL: Initial shear modulus μ₀ is non-positive.")
        is_consistent = False
    else:
        notes.append("  CONSISTENCY PASS: Initial shear modulus μ₀ is positive.")

    # Add note about interpretation
    if ogden_term_found:
        notes.append("  NOTE: Ogden consistency assumes discovered 'coeff' C_i relates to standard μ_i via μ_i = C_i * α_i.")

    # Return mu0 in MPa as it's derived from MPa coeffs/targets
    return is_consistent, shear_modulus_MPa, notes
