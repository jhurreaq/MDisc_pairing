# %%
# Imports
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch
import matplotlib.ticker

from numpy.linalg import eigh, LinAlgError
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LassoCV, lasso_path,
    LarsCV, lars_path,
    OrthogonalMatchingPursuitCV, orthogonal_mp,
    Ridge,
    LinearRegression
)
from sklearn.metrics import mean_squared_error, r2_score
import re
import datetime
import os
from scipy.special import erf
import warnings
from sklearn.exceptions import ConvergenceWarning
import traceback

from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

# Plotting defaults
PLOT_TITLE_FONTSIZE = 14
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 14
PLOT_LEGEND_FONTSIZE = 10
PLOT_LINEWIDTH = 1.5
PLOT_PATH_ALPHA = 0.7 # Transparency for path lines
PLOT_MARKER_SIZE = 30 # For scatter plots in predictions
PLOT_PRED_LINEWIDTH = 1.5 # For prediction lines
PLOT_HEATMAP_YLABEL_SIZE = 9 # Specific size for potentially long feature lists

SMALL_FIG_LASSO_LARS_OMP = (3.0, 4.5)

# Colormap configuration
try:
    plasma = plt.get_cmap('plasma')
    blues_cmap = plt.get_cmap('Blues')
except ValueError:
    plasma = plt.cm.plasma
    blues_cmap = plt.cm.Blues

# Truncated plasma colormap used in plotting
TRUNCATED_PLASMA_CMAP = plasma

# Colormap for heatmaps (binary activation)
HEATMAP_CMAP = blues_cmap

# Model configuration
# Standard model components (Will be overridden in main loop)
INCLUDE_NEOHOOKEAN = True
INCLUDE_MOONEY_RIVLIN = True
INCLUDE_OGDEN = True

# Optional components for S-shaped response
INCLUDE_EXTENDED_OGDEN = True # Often used with Ogden

# Maximum powers for various expansions
MR_MAX_ORDER = 4 

# Fractional-powers configuration (optional)

# Ogden configuration (exponents set during library generation)
OGDEN_EXPONENTS = []

# Numerical stability
EPS = 1e-10

# Coefficient threshold and ridge alpha used in refits
COEFF_THRESHOLD = 1e-6
RIDGE_ALPHA_REFIT = 1e-7

# Invariant/derivative helpers
def compute_invariants(l1, l2, l3):
    I1 = l1**2 + l2**2 + l3**2
    I2 = (l1*l2)**2 + (l2*l3)**2 + (l1*l3)**2
    return I1, I2

def compute_invariant_derivatives(l1, l2, l3):
    dI1dl1, dI1dl2, dI1dl3 = 2*l1, 2*l2, 2*l3
    dI2dl1 = 2*l1*(l2**2 + l3**2)
    dI2dl2 = 2*l2*(l1**2 + l3**2)
    dI2dl3 = 2*l3*(l1**2 + l2**2)
    return (dI1dl1, dI1dl2, dI1dl3), (dI2dl1, dI2dl2, dI2dl3)

def safe_power(base, exponent, *, tol_int=1e-12, limit=1e18):
    """
    Compute base**exponent with:
      - exact integer-power behavior (including negative base for odd integers),
      - real-valued output; if base < 0 and exponent is non-integer, return 0.0,
      - clipping of large magnitudes to avoid infs.

    Works with scalars or numpy arrays; broadcasting follows numpy.
    """
    base_arr = np.asarray(base, dtype=float)
    exp_arr  = np.asarray(exponent, dtype=float)

    # helper: is exponent (elementwise) integer within tolerance?
    def is_int(x):
        return np.isfinite(x) & (np.abs(x - np.round(x)) <= tol_int)

    # Allocate result
    out = np.zeros(np.broadcast(base_arr, exp_arr).shape, dtype=float)

    # Masks
    m_neg_base = base_arr < 0.0
    m_pos_base = base_arr >= 0.0
    m_exp_int  = is_int(exp_arr)

    # 1) Non-integer exponent with negative base → return 0.0 (stay real)
    mask_bad = m_neg_base & (~m_exp_int)
    # out[mask_bad] = 0.0  # already zero

    # 2) Integer exponents (handle exactly, keep sign)
    mask_int = m_exp_int
    if np.any(mask_int):
        exp_int = np.round(exp_arr[mask_int]).astype(int)
        out[mask_int] = np.sign(base_arr[mask_int])**(np.abs(exp_int) % 2) * np.power(np.abs(base_arr[mask_int]), exp_int, dtype=float)

    # 3) Positive base with any exponent (standard real power)
    mask_real = m_pos_base & (~mask_int)
    if np.any(mask_real):
        with np.errstate(over='ignore', invalid='ignore'):
            out[mask_real] = np.power(base_arr[mask_real], exp_arr[mask_real], dtype=float)

    # 4) Clip and clean
    out = np.nan_to_num(out, nan=0.0, posinf=limit, neginf=-limit)
    out = np.clip(out, -limit, limit)

    # Handle exact special cases (scalar bases with scalar exponents)
    if np.isscalar(base) and np.isscalar(exponent):
        return float(out)
    return out

def safe_log(x):
    if isinstance(x, (np.ndarray)):
        res = np.full_like(x, -700.0, dtype=float) # Use large negative for log(small)
        mask = x > EPS
        res[mask] = np.log(x[mask])
        return res
    else: # Scalar case
        return np.log(x) if x > EPS else -700.0

# Model library generation
def generate_model_library():
    """
    Generates a list of basis function strings based on configuration flags.
    Uses the OGDEN_EXPONENTS list already defined globally in the main block.
    """
    # OGDEN_EXPONENTS should already be defined globally and populated correctly
    # in the __main__ block before this function is called.
    # Access global variables needed.
    global OGDEN_EXPONENTS
    global MR_MAX_ORDER
    global INCLUDE_NEOHOOKEAN, INCLUDE_MOONEY_RIVLIN
    global INCLUDE_OGDEN
    global INCLUDE_EXTENDED_OGDEN

    # --- Determine which Ogden exponents list to use ---
    # Check if OGDEN_EXPONENTS was properly set globally via main block
    if INCLUDE_OGDEN and ('OGDEN_EXPONENTS' in globals() and isinstance(OGDEN_EXPONENTS, list) and OGDEN_EXPONENTS):
         # Use the list set globally in main if it exists and is not empty
         ogden_exponents_to_use = OGDEN_EXPONENTS
         print(f"  generate_model_library: Using globally set OGDEN_EXPONENTS = {ogden_exponents_to_use}")
    elif INCLUDE_OGDEN:
         # Fallback if not set globally
         print("WARN: Global OGDEN_EXPONENTS not set. Falling back based on INCLUDE_EXTENDED_OGDEN.")
         if INCLUDE_EXTENDED_OGDEN:
             STANDARD_OGDEN_EXPONENTS_local = [-4, -2, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 2, 4]
             ogden_exponents_to_use = STANDARD_OGDEN_EXPONENTS_local
         else:
             ogden_exponents_to_use = [-2, -1, -0.5, 0.5, 1, 2, 3]
         print(f"  generate_model_library: Fallback OGDEN_EXPONENTS = {ogden_exponents_to_use}")
    else:
         ogden_exponents_to_use = [] # No Ogden terms included


    basis_functions = []
    seen_terms = set()
    def add_term(term_string):
        if term_string not in seen_terms:
            basis_functions.append(term_string)
            seen_terms.add(term_string)

    # --- Term generation logic (uses the correct ogden_exponents_to_use) ---
    if INCLUDE_NEOHOOKEAN and not INCLUDE_MOONEY_RIVLIN: add_term("NH: (I₁-3)")

    if INCLUDE_MOONEY_RIVLIN:
        # Generate MR terms with total order i+j <= MR_MAX_ORDER
        for total_power in range(1, MR_MAX_ORDER + 1):
            for i in range(total_power + 1):
                j = total_power - i  # ensures i + j == total_power
                term_i1 = f"(I₁-3)^{i}" if i > 1 else ("(I₁-3)" if i == 1 else "")
                term_i2 = f"(I₂-3)^{j}" if j > 1 else ("(I₂-3)" if j == 1 else "")
                if term_i1 and term_i2:
                    term_str = f"MR: {term_i1}{term_i2}"
                elif term_i1:
                    term_str = f"MR: {term_i1}"
                else:
                    term_str = f"MR: {term_i2}"
                add_term(term_str)
    if INCLUDE_OGDEN:
        for a in ogden_exponents_to_use:
            if a != 0: add_term(f"Ogden: λ^{a}") # Format Ogden terms
    print(f"Generated {len(basis_functions)} unique basis terms for this scenario (using determined OGDEN_EXPONENTS).")
    return basis_functions


# ==============================================
# Construct design matrix (Φ) and target vector (y)
# PK1 STRESS Formulation
# ==============================================
def construct_design_matrix(df, normalize_by_stress_magnitude=True):
    """
    Constructs the design matrix Phi and target vector y for regression.
    Includes automatic weight normalization based on stress magnitudes.
    """
    df_subset = df.copy()
    df_subset = df_subset.reset_index(drop=True)

    dW1_mat, dW2_mat, dW3_mat, basis_names = construct_derivatives_matrix(df_subset)
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
            print(f"    Global RMS stress: {global_rms:.3f} MPa")
            
            # Calculate per-mode normalization weights (inverse of RMS to equalize contributions)
            mode_weights = {}
            for mode, stats in stress_stats.items():
                mode_weights[mode] = global_rms / stats['rms'] if stats['rms'] > EPS else 1.0
                print(f"    {mode} mode normalization weight: {mode_weights[mode]:.3f}")
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
        l1, l2 = max(row_data['lambda1'], EPS), max(row_data['lambda2'], EPS)
        l3_inv_prod = l1 * l2
        l3 = max(1.0 / l3_inv_prod if l3_inv_prod > EPS else 1.0 / EPS, EPS)
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
                l3_over_l1 = l3 / l1 if l1 > EPS else l3 / EPS
                phi_row_p11 = dW1_vec - l3_over_l1 * dW3_vec
                design_matrix[current_row_idx, :] = final_weight * phi_row_p11
                target_vector[current_row_idx] = final_weight * row_data['P11']   
                row_weights_vector[current_row_idx] = final_weight
                current_row_idx += 1

        if has_p22:
            if current_row_idx < total_rows:
                l3_over_l2 = l3 / l2 if l2 > EPS else l3 / EPS
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
        print(f"WARN: Final row index ({current_row_idx}) != Expected total ({total_rows}). Resizing arrays.")
        design_matrix = design_matrix[:current_row_idx, :]
        target_vector = target_vector[:current_row_idx]
        row_weights_vector = row_weights_vector[:current_row_idx]

    finite_mask = np.all(np.isfinite(design_matrix), axis=1) & np.isfinite(target_vector)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        print(f"WARN: Removing {num_removed} rows due to non-finite values.")
        design_matrix = design_matrix[finite_mask, :]
        target_vector = target_vector[finite_mask]
        row_weights_vector = row_weights_vector[finite_mask]

    if design_matrix.shape[0] == 0:
        print("ERROR: Design matrix empty after filtering.")
        return np.zeros((0, n_basis)), np.zeros((0,)), basis_names, np.zeros((0,)), df_subset

    print(f"Constructed design matrix (Stress-Normalized: {normalize_by_stress_magnitude}): {design_matrix.shape}")
    return design_matrix, target_vector, basis_names, row_weights_vector, df_subset

# Helper to calculate dW/dl vectors
def calculate_dW_dl_vectors(l1, l2, l3, basis_names):
    """
    Calculates the derivative vectors dW/dl1, dW/dl2, dW/dl3 for each
    basis function, with checks for stability. Returns None on critical error.
    """
    num_basis = len(basis_names)
    if num_basis == 0: return None

    # Ensure inputs are floats and positive, using original numpy for safety here
    l1, l2, l3 = max(float(l1), EPS), max(float(l2), EPS), max(float(l3), EPS)

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
            print(f"WARN: Non-finite invariant derivatives at l1,l2,l3 = {l1:.3f},{l2:.3f},{l3:.3f}. Setting to 0.")
            dI1dl1, dI1dl2, dI1dl3, dI2dl1, dI2dl2, dI2dl3 = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

    except Exception as e:
        print(f"ERROR calculating invariants/derivs at l1,l2,l3 = {l1:.3f},{l2:.3f},{l3:.3f}: {e}")
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

# Derivatives matrix builder
def construct_derivatives_matrix(df):
    """
    Constructs matrices of dW/dl1, dW/dl2, dW/dl3 for all data points
    by calling the helper function for each point.
    """
    num_samples = len(df)
    # Generate library once - IMPORTANT: ensure global flags match discovery intent
    basis_names = generate_model_library()
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
        if not pd.notna(l3) or l3 < EPS:
             l1_safe = max(l1, EPS); l2_safe = max(l2, EPS)
             l3 = max(1.0 / (l1_safe * l2_safe), EPS)

        # Call the helper function
        deriv_vecs = calculate_dW_dl_vectors(l1, l2, l3, basis_names)
        if deriv_vecs:
             dW_dl1_mat[idx, :] = deriv_vecs[0]
             dW_dl2_mat[idx, :] = deriv_vecs[1]
             dW_dl3_mat[idx, :] = deriv_vecs[2]
        # Else: error handled within helper, rows remain zero

    return dW_dl1_mat, dW_dl2_mat, dW_dl3_mat, basis_names

# Synthetic data generation (relative noise)

def generate_synth_data_via_framework(model_name, term_coeff_map, basis_names, 
                                      relative_noise_levels,
                                      lambda_range, gamma_range, n_points):
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
        for term in missing_terms: print(f" - {term}")
        print("Please adjust `generate_model_library` flags (e.g., Ogden exponents, MR powers).")
        return None

    print(f"--- Generating Synthetic UT, PS & EBT Data [Framework Logic] for: {model_name} ---")

    # --- Uniaxial Tension/Compression ---
    print(f"  Generating Uniaxial (UT) data (λ = {lambda_range}, {n_points} points)...")
    for lam_val in lambdas:
        l1 = max(float(lam_val), EPS)
        l2 = max(l1**(-0.5), EPS)
        l3 = l2 # Incompressible UT

        deriv_vecs = calculate_dW_dl_vectors(l1, l2, l3, basis_names)
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
            if abs(p11_true_Pa) < EPS and rel_noise > 0: # Only add if true stress is zero but noise is requested
                noise_std_dev_Pa = rel_noise * (EPS * 1000) # Add noise relative to a small nominal stress
                
            actual_noise_Pa = np.random.normal(0, noise_std_dev_Pa)
            P11_noisy_Pa = p11_true_Pa + actual_noise_Pa
            P11_noisy_MPa = P11_noisy_Pa * pa_to_mpa
            
            dataset_name = f"{model_name}_RelNoise{rel_noise*100:.0f}pct"

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
    print(f"  Generating Pure Shear (PS) data (λ = {lambda_range}, {n_points} points)...")
    for lam_val in lambdas:
        l1_ps = max(float(lam_val), EPS)
        l2_ps = 1.0
        l3_ps = max(1.0 / l1_ps, EPS)

        deriv_vecs = calculate_dW_dl_vectors(l1_ps, l2_ps, l3_ps, basis_names)
        if deriv_vecs is None: continue

        dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs

        dW_total_dl1_Pa = 0.0; dW_total_dl2_Pa = 0.0; dW_total_dl3_Pa = 0.0
        for term, coeff_Pa in term_coeff_map.items():
            idx = term_indices[term]
            dW_total_dl1_Pa += coeff_Pa * dW_dl1_all[idx]
            dW_total_dl2_Pa += coeff_Pa * dW_dl2_all[idx]
            dW_total_dl3_Pa += coeff_Pa * dW_dl3_all[idx]

        # PK1 with incompressibility
        p11_true_Pa = dW_total_dl1_Pa - (l3_ps / l1_ps) * dW_total_dl3_Pa if l1_ps > EPS else 0.0
        p22_true_Pa = dW_total_dl2_Pa - (l3_ps / l2_ps) * dW_total_dl3_Pa if l2_ps > EPS else 0.0
        p11_true_MPa = p11_true_Pa * pa_to_mpa
        p22_true_MPa = p22_true_Pa * pa_to_mpa

        for rel_noise in relative_noise_levels:
            noise11_std_dev_Pa = rel_noise * abs(p11_true_Pa)
            noise22_std_dev_Pa = rel_noise * abs(p22_true_Pa)
            
            if abs(p11_true_Pa) < EPS and rel_noise > 0:
                noise11_std_dev_Pa = rel_noise * (EPS * 1000) 
            if abs(p22_true_Pa) < EPS and rel_noise > 0:
                noise22_std_dev_Pa = rel_noise * (EPS * 1000)

            noise11_Pa = np.random.normal(0, noise11_std_dev_Pa)
            noise22_Pa = np.random.normal(0, noise22_std_dev_Pa)
            P11_noisy_Pa = p11_true_Pa + noise11_Pa
            P22_noisy_Pa = p22_true_Pa + noise22_Pa
            P11_noisy_MPa = P11_noisy_Pa * pa_to_mpa
            P22_noisy_MPa = P22_noisy_Pa * pa_to_mpa
            
            dataset_name = f"{model_name}_RelNoise{rel_noise*100:.0f}pct"

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
    print(f"  Generating Equi-biaxial (EBT) data (λ = {lambda_range}, {n_points} points)...")
    for lam_val in lambdas:
        l1_ebt = max(float(lam_val), EPS)
        l2_ebt = l1_ebt  # Stretch is equal in both directions
        l3_ebt = max(l1_ebt**(-2.0), EPS)  # Incompressible: l1 * l2 * l3 = 1
        
        deriv_vecs = calculate_dW_dl_vectors(l1_ebt, l2_ebt, l3_ebt, basis_names)
        if deriv_vecs is None: continue

        dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs

        dW_total_dl1_Pa = 0.0; dW_total_dl2_Pa = 0.0; dW_total_dl3_Pa = 0.0
        for term, coeff_Pa in term_coeff_map.items():
            idx = term_indices[term]
            dW_total_dl1_Pa += coeff_Pa * dW_dl1_all[idx]
            dW_total_dl2_Pa += coeff_Pa * dW_dl2_all[idx]
            dW_total_dl3_Pa += coeff_Pa * dW_dl3_all[idx]

        p11_true_Pa = dW_total_dl1_Pa - (l3_ebt / l1_ebt) * dW_total_dl3_Pa if l1_ebt > EPS else 0.0
        p22_true_Pa = dW_total_dl2_Pa - (l3_ebt / l2_ebt) * dW_total_dl3_Pa if l2_ebt > EPS else 0.0
        p11_true_MPa = p11_true_Pa * pa_to_mpa
        p22_true_MPa = p22_true_Pa * pa_to_mpa

        for rel_noise in relative_noise_levels:
            noise11_std_dev_Pa = rel_noise * abs(p11_true_Pa)
            noise22_std_dev_Pa = rel_noise * abs(p22_true_Pa)
            
            if abs(p11_true_Pa) < EPS and rel_noise > 0: noise11_std_dev_Pa = rel_noise * (EPS * 1000) 
            if abs(p22_true_Pa) < EPS and rel_noise > 0: noise22_std_dev_Pa = rel_noise * (EPS * 1000)

            noise11_Pa = np.random.normal(0, noise11_std_dev_Pa)
            noise22_Pa = np.random.normal(0, noise22_std_dev_Pa)
            P11_noisy_MPa = p11_true_Pa + noise11_Pa
            P22_noisy_MPa = p22_true_Pa + noise22_Pa
            P11_noisy_MPa = P11_noisy_MPa * pa_to_mpa
            P22_noisy_MPa = P22_noisy_MPa * pa_to_mpa
            
            dataset_name = f"{model_name}_RelNoise{rel_noise*100:.0f}pct"

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
    
    print(f"--- Finished generating data [Framework Logic] for: {model_name} ---")
    df = pd.DataFrame(all_synth_data)
    num_cols = ['lambda1', 'lambda2', 'lambda3', 'gamma', 'P11', 'P22', 'P12',
                'P11_true', 'P12_true', 'P22_true', 'strain_pct', 'noise_sigma']
    for col in num_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    df = df.dropna(subset=['P11'])
    return df

# ==============================================
# Helper Function for LaTeX Formatting of Terms
# ==============================================
def format_term_for_latex(term_name):
    if ':' in term_name: prefix, math_part = term_name.split(':', 1); term = math_part.strip()
    else: term = term_name
    term = term.replace('I₁', 'I_{1}'); term = term.replace('I₂', 'I_{2}'); term = term.replace('λ', r'\lambda')
    def exponent_replacer(match):
        base = match.group(1); exponent = match.group(2); base_formatted = base
        try:
            exp_val = float(exponent)
            if exp_val == 1.0: return base_formatted
            exponent_formatted = f"{int(exp_val)}" if exp_val == int(exp_val) else f"{exp_val:.3g}".rstrip('0').rstrip('.')
        except ValueError: exponent_formatted = exponent
        return f"{base_formatted}^{{{exponent_formatted}}}"
    term = re.sub(r'(\(I_{\d}-3\))\^([\-\+]?\d+\.?\d*)', exponent_replacer, term)
    term = re.sub(r'(\\lambda)\^([\-\+]?\d+\.?\d*)', exponent_replacer, term)
    term = term.replace('^2', '^{2}'); term = term.replace('^3', '^{3}')
    term = term.replace('ln', r'\ln'); term = term.replace('exp', r'\exp'); term = term.replace('tanh', r'\tanh')
    term = term.replace('sigmoid', r'\text{sigmoid}'); term = term.replace('erf', r'\text{erf}'); term = term.replace('arctan', r'\arctan')
    term = term.replace('*', r'')
    term = re.sub(r'\((I_\{1\}-3)\)\/(0\.\d+)\+\(I_\{1\}-3\)', r'\\frac{\1}{\2+\1}', term) # Basic rational format
    return f"${term}$"

# ==============================================
# AIC / BIC Calculation Helper
# ==============================================
def calculate_aic_bic(y_true, y_pred, n_params, n_samples):
    """Calculates AIC and BIC."""
    if n_samples <= 0: return np.nan, np.nan
    if n_params == 0: print("WARN: Calculating AIC/BIC with n_params=0. Returning NaN."); return np.nan, np.nan
    if n_params >= n_samples: pass # Allow calculation but be aware
    rss = np.sum((y_true - y_pred)**2)
    if rss < EPS: log_likelihood_proxy = n_samples * (100)
    else: log_likelihood_proxy = -n_samples / 2.0 * np.log(max(rss / n_samples, EPS))
    k_effective = n_params # Assumes no implicit intercept
    aic = -2 * log_likelihood_proxy + 2 * k_effective
    bic = -2 * log_likelihood_proxy + k_effective * np.log(max(n_samples, 1))
    return aic, bic

# Physical consistency check helper
def check_physical_consistency(final_terms, final_coeffs):
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
    global EPS # Make sure EPS is accessible

    if not isinstance(final_terms, list) or not isinstance(final_coeffs, np.ndarray):
         return False, np.nan, ["Error: Invalid input types for terms/coeffs."]

    if len(final_terms) != len(final_coeffs):
         if len(final_terms) == 0 and len(final_coeffs) == 0:
              return False, 0.0, ["No terms remaining after thresholding."]
         else:
              return False, np.nan, [f"Error: Mismatched length of terms ({len(final_terms)}) and coeffs ({len(final_coeffs)})."]

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
        if c10_coeff_MPa <= EPS: # Check MPa value against EPS
            notes.append("    WARN: C10 component is non-positive.")

    if mr_c01_term in term_coeff_map_MPa:
        # Note: Report C01 in Pa, but use MPa value for mu0 calc consistency below
        notes.append(f"  MR C01 Term ('{mr_c01_term}') Coeff (C01) = {c01_coeff_Pa:.1f} Pa")
        if c01_coeff_MPa < -EPS: # Check MPa value against EPS
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
                if abs(alpha_i) < EPS:
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
                notes.append(f"  Ogden Term: Coeff (C_i={coeff_Pa:.1f} Pa), α={alpha_i:.4f} => Implied μ_i={mu_i_Pa:.1f} Pa. (μ_i*α_i = {mu_alpha_product_Pa:.1f} Pa)")
                ogden_term_found = True
            except ValueError:
                notes.append(f"  WARN: Could not parse Ogden exponent in '{term}'.")
            except Exception as e:
                 notes.append(f"  ERROR processing Ogden term '{term}': {e}")

    if ogden_term_found:
        notes.append(f"  Sum(μᵢ * αᵢ) for Ogden terms = {sum_mu_alpha_ogden_MPa * 1e6:.1f} Pa") # Report sum in Pa
        # Contribution to shear modulus = 0.5 * sum(mu*alpha) in MPa
        shear_modulus_contrib_ogden = 0.5 * sum_mu_alpha_ogden_MPa

    # --- Calculate Final Shear Modulus (in MPa) ---
    shear_modulus_MPa = 2.0 * shear_modulus_contrib_mr + shear_modulus_contrib_ogden

    # --- Final Shear Modulus Check ---
    # Report mu0 in both MPa and Pa for context
    notes.insert(0, f"Calculated initial shear modulus μ₀ ≈ {shear_modulus_MPa:.4f} MPa ({shear_modulus_MPa*1e6:.1f} Pa)")

    is_consistent = True
    if shear_modulus_MPa <= EPS: # Check consistency using MPa value (or Pa, doesn't matter near zero)
        notes.append("  CONSISTENCY FAIL: Initial shear modulus μ₀ is non-positive.")
        is_consistent = False
    else:
        notes.append("  CONSISTENCY PASS: Initial shear modulus μ₀ is positive.")

    # Add note about interpretation
    if ogden_term_found:
        notes.append("  NOTE: Ogden consistency assumes discovered 'coeff' C_i relates to standard μ_i via μ_i = C_i * α_i.")

    # Return mu0 in MPa as it's derived from MPa coeffs/targets
    return is_consistent, shear_modulus_MPa, notes
# Helper function for Y-axis styling (ticks and precision)
def style_y_axis(ax, kpa_conversion):
    y_lim = ax.get_ylim()
    
    ticks_to_set = sorted(list(set([y_lim[0], y_lim[1]]))) 
    if y_lim[0] < 0 < y_lim[1]: 
        ticks_to_set.append(0)
    elif np.abs(y_lim[0]) < 0.05 * (y_lim[1] - y_lim[0]) and y_lim[0] > -EPS and y_lim[0] * y_lim[1] >=0 : # Min is positive and close to 0
        ticks_to_set.append(0)
    elif np.abs(y_lim[1]) < 0.05 * (y_lim[1] - y_lim[0]) and y_lim[1] < EPS and y_lim[0] * y_lim[1] >=0 : # Max is negative and close to 0
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
    ax.tick_params(axis='y', which='major', labelsize=PLOT_LABEL_FONTSIZE)


# Plotting helper for synthetic uniaxial data (kPa)
def plot_synthetic_uniaxial(df, ax, r2_p11_precomputed=np.nan, rmse_p11_precomputed_kpa=np.nan):
    """Plots synthetic uniaxial data (P11 vs lambda1) and model predictions using TRUNCATED_PLASMA_CMAP and kPa units."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 2))
    else:
        fig = ax.get_figure()

    global TRUNCATED_PLASMA_CMAP
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

    color_synth_data = TRUNCATED_PLASMA_CMAP(0.4)
    color_true_curve = TRUNCATED_PLASMA_CMAP(0.2)
    color_model_fit  = TRUNCATED_PLASMA_CMAP(0.4)

    ax.scatter(df_plot['lambda1'], df_plot['P11'], color=color_synth_data, marker='o',
               s=PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3,
               edgecolor='black', linewidth=0.5)    

    valid_pred_mask = df_plot['P11_pred'].notna()
    if valid_pred_mask.any():
        df_sorted_pred = df_plot[valid_pred_mask].dropna(subset=['lambda1', 'P11_pred']).sort_values(by='lambda1').drop_duplicates(subset=['lambda1'])
        ax.plot(df_sorted_pred['lambda1'], df_sorted_pred['P11_pred'], color=color_model_fit,
                linestyle='-', linewidth=PLOT_PRED_LINEWIDTH, label='Disc. Model')
        
    if 'P11_true' in df_plot.columns and df_plot['P11_true'].notna().all():
        df_sorted_true = df_plot.dropna(subset=['lambda1', 'P11_true']).sort_values(by='lambda1').drop_duplicates(subset=['lambda1'])
        ax.plot(df_sorted_true['lambda1'], df_sorted_true['P11_true'], color=color_true_curve,
                linestyle=':', linewidth=PLOT_PRED_LINEWIDTH*1.2, label='True Model')

    ax.set_xlabel('Stretch $\lambda$ [-]', fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=PLOT_LABEL_FONTSIZE) 
    ax.tick_params(axis='x', which='major', labelsize=PLOT_LABEL_FONTSIZE) 

    first_line_title = "Uniaxial $P_{11}$"
    second_line_title_parts = []
    if pd.notna(r2_p11_precomputed):
        second_line_title_parts.append(f"$R^2 \\approx {r2_p11_precomputed:.3f}$")
    if pd.notna(rmse_p11_precomputed_kpa):
        second_line_title_parts.append(f"RMSE $\\approx {rmse_p11_precomputed_kpa:.3f}$ kPa") 
    
    if second_line_title_parts:
        title_str = first_line_title + "\n" + ", ".join(second_line_title_parts)
    else:
        title_str = first_line_title
    # ax.set_title(title_str, fontsize=PLOT_TITLE_FONTSIZE)
    ax.set_title(first_line_title, fontsize=PLOT_TITLE_FONTSIZE) 

    ax.legend(fontsize=PLOT_LEGEND_FONTSIZE, frameon=False) # Legend box turned off

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
        padding = y_range_abs * 0.075 if y_range_abs > EPS else abs(min_valid_y) * 0.1 + (0.01 * kpa_conversion)
        final_y_min = min_valid_y - padding
        final_y_max = max_valid_y + padding
        ax.set_ylim(final_y_min, final_y_max)
        print(f"  Set Y-limits to: ({final_y_min:.1f} kPa, {final_y_max:.1f} kPa)")
        style_y_axis(ax, kpa_conversion) 
    else:
        print("  WARN: Could not determine valid Y range. Using default autoscaling.")
        ax.tick_params(axis='y', which='major', labelsize=PLOT_LABEL_FONTSIZE)

    if df_plot['lambda1'].notna().any():
        min_l = df_plot['lambda1'].min()
        max_l = df_plot['lambda1'].max()
        if pd.notna(min_l) and pd.notna(max_l):
            x_padding = (max_l - min_l) * 0.05 if (max_l - min_l) > EPS else 0.05
            ax.set_xlim(min_l - x_padding, max_l + x_padding)

    return ax

# Plotting helper for model predictions (kPa, RMSE in title)
def plot_model_predictions(
    df_orig, 
    sparse_coeffs, 
    basis_names, 
    method_key, 
    scenario_prefix, 
    save_dir,
    coeff_threshold=None, 
    ridge_alpha=None 
):
    coeff_threshold = coeff_threshold if coeff_threshold is not None else COEFF_THRESHOLD
    ridge_alpha = ridge_alpha if ridge_alpha is not None else RIDGE_ALPHA_REFIT

    print(f"\n--- Generating Final Model & Predictions ({method_key} / Ridge Refit α={ridge_alpha:.1e} + Check) ---")
    print(f"--- NOTE: Fitting on UT(P11)+PS(P11)+EBT(P11); Evaluating UT(P11), PS(P11), EBT(P11) ---")

    kpa_conversion = 1e3 

    current_final_terms, current_final_coeffs_kPa = [], np.array([])
    r2_P11_ut, rmse_P11_ut_kPa, nrmse_P11_ut = np.nan, np.nan, np.nan
    r2_P11_ps, rmse_P11_ps_kPa, nrmse_P11_ps = np.nan, np.nan, np.nan
    r2_P11_ebt, rmse_P11_ebt_kPa, nrmse_P11_ebt = np.nan, np.nan, np.nan
    is_consistent, consistency_notes = False, ["Processing error before completion"]

    global TRUNCATED_PLASMA_CMAP

    selected_indices = np.where(np.abs(sparse_coeffs) > EPS)[0]
    if len(selected_indices) == 0:
        print(f"WARN: No features initially selected by {method_key}. Skipping.")
        consistency_notes = ["No features selected"]
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes
    initial_selected_basis_names = [basis_names[i] for i in selected_indices]
    print(f"  Initial selection ({method_key}): {len(initial_selected_basis_names)} features.")

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
            df_for_refit
        )
        if not isinstance(df_subset_weighted, pd.DataFrame): raise TypeError("df_subset_weighted type error")
        if phi_weighted.shape[0] == 0: raise ValueError("Weighted design matrix empty.")
        if basis_names != basis_names_weighted:
             print(f"WARN: Basis name mismatch during weighted recon. Using list from construct_design_matrix ({len(basis_names_weighted)} terms).")
             basis_names = basis_names_weighted
             try:
                 selected_indices = [basis_names.index(term) for term in initial_selected_basis_names if term in basis_names]
                 initial_selected_basis_names = [basis_names[i] for i in selected_indices]
                 print(f"  Re-mapped initial selection: {len(initial_selected_basis_names)} features.")
                 if len(selected_indices)==0: raise ValueError("Remapping indices failed, no initial terms found.")
             except ValueError as ve:
                  print(f"ERROR: Failed to remap selected indices after basis mismatch: {ve}")
                  consistency_notes = ["Basis mismatch & remapping failed"]
                  return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes
        phi_selected_weighted = phi_weighted[:, selected_indices]
    except Exception as e:
        print(f"ERROR: Failed to reconstruct WEIGHTED matrix/target for refit: {e}"); traceback.print_exc()
        consistency_notes = [f"Matrix Recon Error: {e}"]
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes

    print(f"  Refitting {len(initial_selected_basis_names)} features on WEIGHTED data (shape: {phi_selected_weighted.shape}) using Ridge...")
    refit_coeffs_weighted_MPa = None
    try:
        ridge_refit = Ridge(alpha=ridge_alpha, fit_intercept=False, solver='auto')
        ridge_refit.fit(phi_selected_weighted, target_weighted) 
        refit_coeffs_weighted_MPa = ridge_refit.coef_
    except Exception as e:
        print(f"ERROR: Failed during Ridge Regression refit for {method_key}: {e}"); traceback.print_exc()
        consistency_notes = [f"Ridge Refit Error: {e}"]
        return [], [], np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, False, consistency_notes

    print(f"  Applying threshold ({coeff_threshold:.1e}) to Ridge coefficients...")
    significant_mask = np.abs(refit_coeffs_weighted_MPa) > coeff_threshold
    final_coeffs_MPa = refit_coeffs_weighted_MPa[significant_mask]
    final_global_indices = selected_indices[significant_mask]
    final_selected_basis_names = [basis_names[i] for i in final_global_indices]
    n_final = len(final_selected_basis_names)
    print(f"  Final selected features after thresholding: {n_final}")
    current_final_terms = final_selected_basis_names
    current_final_coeffs_MPa = final_coeffs_MPa 

    if n_final == 0:
        print(f"WARN: All features removed by thresholding for {method_key}.")
        is_consistent, shear_modulus_MPa, consistency_notes = check_physical_consistency([], np.array([]))
    else:
        current_final_coeffs_display_kPa = current_final_coeffs_MPa * kpa_conversion 
        print(f"  Final Model ({method_key}):"); [print(f"     '{name}': {coeff_kpa:.4f} kPa") for name, coeff_kpa in zip(current_final_terms, current_final_coeffs_display_kPa)]
        print("  Checking physical consistency (using MPa coeffs)...");
        is_consistent, shear_modulus_MPa, consistency_notes = check_physical_consistency(current_final_terms, current_final_coeffs_MPa)
        print("  Consistency Check Results:"); [print(f"     {note}") for note in consistency_notes]
        if not is_consistent: print(f"  WARNING: {method_key} model potentially inconsistent (μ₀={shear_modulus_MPa:.4f} MPa).")
        else: print(f"  INFO: {method_key} model passed basic consistency check (μ₀={shear_modulus_MPa:.4f} MPa).")

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
            if not pd.notna(l3) or l3 < EPS:
                l1_safe = max(l1, EPS); l2_safe = max(l2, EPS)
                l3 = max(1.0 / (l1_safe * l2_safe), EPS)
            deriv_vecs = calculate_dW_dl_vectors(l1, l2, l3, basis_names)
            if deriv_vecs is None: continue
            dW_dl1_all, dW_dl2_all, dW_dl3_all = deriv_vecs
            if np.any(np.isnan(dW_dl1_all[final_term_indices_in_full_library])) or \
               np.any(np.isnan(dW_dl2_all[final_term_indices_in_full_library])) or \
               np.any(np.isnan(dW_dl3_all[final_term_indices_in_full_library])):
                continue
            dW_pred_dl1_MPa = np.sum(current_final_coeffs_MPa * dW_dl1_all[final_term_indices_in_full_library])
            dW_pred_dl2_MPa = np.sum(current_final_coeffs_MPa * dW_dl2_all[final_term_indices_in_full_library])
            dW_pred_dl3_MPa = np.sum(current_final_coeffs_MPa * dW_dl3_all[final_term_indices_in_full_library])
            p11_pred_MPa = dW_pred_dl1_MPa - (l3 / l1) * dW_pred_dl3_MPa if l1 > EPS else 0.0
            p22_pred_MPa = dW_pred_dl2_MPa - (l3 / l2) * dW_pred_dl3_MPa if l2 > EPS else 0.0
            if np.isfinite(p11_pred_MPa): df_plot.loc[idx, 'P11_pred_MPa'] = p11_pred_MPa
            if np.isfinite(p22_pred_MPa): df_plot.loc[idx, 'P22_pred_MPa'] = p22_pred_MPa
            predictions_calculated += 1
        print(f"  Calculated P11/P22 predictions for {predictions_calculated} points.")
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
                if true_range_kPa > (EPS * kpa_conversion): nrmse = rmse_kPa_val / true_range_kPa
                elif rmse_kPa_val < (EPS * kpa_conversion): nrmse = 0.0
                else: nrmse = np.inf
        return r2, rmse_kPa_val, nrmse

    r2_P11_ut, rmse_P11_ut_kPa, nrmse_P11_ut = calculate_component_metrics(['UT'], 'P11', 'P11_pred_MPa', 'P11_true')
    r2_P11_ps, rmse_P11_ps_kPa, nrmse_P11_ps = calculate_component_metrics(['PS'], 'P11', 'P11_pred_MPa', 'P11_true')
    r2_P11_ebt, rmse_P11_ebt_kPa, nrmse_P11_ebt = calculate_component_metrics(['EBT'], 'P11', 'P11_pred_MPa', 'P11_true')

    print(f"  Metrics: P11(UT): R²={r2_P11_ut:.4f}, RMSE={rmse_P11_ut_kPa:.3f} kPa, NRMSE={nrmse_P11_ut:.4f}")
    print(f"           P11(PS): R²={r2_P11_ps:.4f}, RMSE={rmse_P11_ps_kPa:.3f} kPa, NRMSE={nrmse_P11_ps:.4f}")
    print(f"           P11(EBT): R²={r2_P11_ebt:.4f}, RMSE={rmse_P11_ebt_kPa:.3f} kPa, NRMSE={nrmse_P11_ebt:.4f}")

    print("  Preparing plots (UT, PS, EBT predictions) using kPa units...")
    fig = None; plot_created = False
    try:
        global REFIT_REGRESSION_TYPE 
    except NameError:
        REFIT_REGRESSION_TYPE = "Ridge Refit" 

    try:
        consistency_status_str = "(Phys OK)" if is_consistent else "(Phys WARN!)"
        title_info = f"{scenario_prefix} ({method_key})\n{REFIT_REGRESSION_TYPE} Th={coeff_threshold:.1e} {consistency_status_str}"
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

            color_true_curve = TRUNCATED_PLASMA_CMAP(0.3)
            color_data_ps = TRUNCATED_PLASMA_CMAP(0.8)
            color_pred_ps = TRUNCATED_PLASMA_CMAP(0.7)
            color_data_ebt = TRUNCATED_PLASMA_CMAP(0.0) 
            color_pred_ebt = TRUNCATED_PLASMA_CMAP(0.1)

            if not df_ut_plot.empty:
                plot_synthetic_uniaxial(df_ut_plot, ax=axes[ax_idx], 
                                        r2_p11_precomputed=r2_P11_ut, 
                                        rmse_p11_precomputed_kpa=rmse_P11_ut_kPa)
                ax_idx += 1

            if not df_ps_plot.empty:
                ax_ps11 = axes[ax_idx]
                df_ps_plot_kPa = df_ps_plot.copy()
                for col in ['P11', 'P11_true']:
                    if col in df_ps_plot_kPa.columns:
                         df_ps_plot_kPa[col] = pd.to_numeric(df_ps_plot_kPa[col], errors='coerce') * kpa_conversion

                ax_ps11.scatter(df_ps_plot_kPa['lambda1'], df_ps_plot_kPa['P11'], color=color_data_ps, marker='o', s=PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3, edgecolor='black', linewidth=0.5)
                valid_ps_p11_pred = df_ps_plot_kPa['P11_pred'].notna()
                if valid_ps_p11_pred.any(): 
                    ax_ps11.plot(df_ps_plot_kPa.loc[valid_ps_p11_pred, 'lambda1'].sort_values(), df_ps_plot_kPa.loc[valid_ps_p11_pred].sort_values('lambda1')['P11_pred'], color=color_pred_ps, linestyle='-', label='Disc. Model')
                if 'P11_true' in df_ps_plot_kPa.columns and df_ps_plot_kPa['P11_true'].notna().all(): 
                    ax_ps11.plot(df_ps_plot_kPa['lambda1'].sort_values(), df_ps_plot_kPa.sort_values('lambda1')['P11_true'], color=color_true_curve, linestyle=':', label='True Model')
                
                title_ps_p11_str = "Pure Shear $P_{11}$"#\n" + f"$R^2 \\approx {r2_P11_ps:.3f}$, RMSE $\\approx {rmse_P11_ps_kPa:.3f}$ kPa"
                ax_ps11.set_title(title_ps_p11_str, fontsize=PLOT_TITLE_FONTSIZE)
                ax_ps11.set_xlabel('Stretch $\lambda$ [-]', fontsize=PLOT_LABEL_FONTSIZE)
                ax_ps11.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=PLOT_LABEL_FONTSIZE)
                ax_ps11.tick_params(axis='both', which='major', labelsize=PLOT_LABEL_FONTSIZE)
                if df_ps_plot_kPa['P11_true'].notna().any() or df_ps_plot_kPa['P11_pred'].notna().any(): style_y_axis(ax_ps11, kpa_conversion)
                ax_ps11.legend(fontsize=PLOT_LEGEND_FONTSIZE, frameon=False)
                ax_idx += 1

            if not df_ebt_plot.empty:
                ax_ebt11 = axes[ax_idx]
                df_ebt_plot_kPa = df_ebt_plot.copy()
                for col in ['P11', 'P11_true']:
                    if col in df_ebt_plot_kPa.columns:
                         df_ebt_plot_kPa[col] = pd.to_numeric(df_ebt_plot_kPa[col], errors='coerce') * kpa_conversion

                ax_ebt11.scatter(df_ebt_plot_kPa['lambda1'], df_ebt_plot_kPa['P11'], color=color_data_ebt, marker='o', s=PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3, edgecolor='black', linewidth=0.5)
                valid_ebt_p11_pred = df_ebt_plot_kPa['P11_pred'].notna()
                if valid_ebt_p11_pred.any(): 
                    ax_ebt11.plot(df_ebt_plot_kPa.loc[valid_ebt_p11_pred, 'lambda1'].sort_values(), df_ebt_plot_kPa.loc[valid_ebt_p11_pred].sort_values('lambda1')['P11_pred'], color=color_pred_ebt, linestyle='-', label='Disc. Model')
                if 'P11_true' in df_ebt_plot_kPa.columns and df_ebt_plot_kPa['P11_true'].notna().all(): 
                    ax_ebt11.plot(df_ebt_plot_kPa['lambda1'].sort_values(), df_ebt_plot_kPa.sort_values('lambda1')['P11_true'], color=color_true_curve, linestyle=':', label='True Model')

                title_ebt_p11_str = "Equi-biaxial $P_{11}$"#\n" + f"$R^2 \\approx {r2_P11_ebt:.3f}$, RMSE $\\approx {rmse_P11_ebt_kPa:.3f}$ kPa"
                ax_ebt11.set_title(title_ebt_p11_str, fontsize=PLOT_TITLE_FONTSIZE)
                ax_ebt11.set_xlabel('Stretch $\lambda$ [-]', fontsize=PLOT_LABEL_FONTSIZE)
                ax_ebt11.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=PLOT_LABEL_FONTSIZE)
                ax_ebt11.tick_params(axis='both', which='major', labelsize=PLOT_LABEL_FONTSIZE)
                if df_ebt_plot_kPa['P11_true'].notna().any() or df_ebt_plot_kPa['P11_pred'].notna().any(): style_y_axis(ax_ebt11, kpa_conversion)
                ax_ebt11.legend(fontsize=PLOT_LEGEND_FONTSIZE, frameon=False)
                ax_idx += 1

            plt.tight_layout(rect=[0, 0.03, 1, 0.94]) 
            plot_created = True
        else:
            print(f"No UT, PS, or EBT data found in df_plot for {method_key}. Skipping prediction plot generation.")

        if save_dir and scenario_prefix and fig and plot_created:
            save_path = os.path.join(save_dir, f"{scenario_prefix}_{method_key}_Predictions_Final_Checked_kPa.pdf") 
            try:
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                fig.savefig(save_path, format='pdf', bbox_inches='tight')
                print(f"Saved final prediction plot (kPa): {save_path}")
            except Exception as e:
                print(f"Error saving final prediction plot {save_path}: {e}")
        elif fig:
            print(f"WARN: Plot not saved for {method_key} (save conditions not met).")

    except Exception as plot_exc:
        print(f"ERROR during plot generation/saving section for {method_key}: {plot_exc}")
        traceback.print_exc()
    finally: 
         if fig and plot_created: plt.close(fig) 


    current_final_coeffs_kPa = current_final_coeffs_MPa * kpa_conversion
    print(f"--- FINISHING plot_model_predictions for {method_key} ---")
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
):
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

    ax.plot(plot_x_values, plot_coef_path.T, linewidth=PLOT_LINEWIDTH, alpha=0.7)

    if log_x:
        ax.set_xscale('log')

    ax.set_xlabel(x_axis_label, fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Coefficient Value [MPa]', fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
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
        ax.legend(handles=legend_elements, fontsize=PLOT_LEGEND_FONTSIZE)

    if filename:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            print(f"Saved coefficient path plot: {filename}")
        except Exception as e:
            print(f"Error saving coefficient path plot: {e}")
    plt.close(fig)
    
# --- Heatmap Plotters ---
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
):
    """
    Generalized function to plot feature activation heatmaps with multiple selection criteria.
    Now uses direct indices to avoid coordinate transformation errors.
    """
    n_features, n_steps = coef_path.shape

    if n_steps == 0 or path_variable is None or len(path_variable) != n_steps:
        print(f"Warning: Cannot plot {title}, path or path variable invalid.")
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
         print(f"Warning: No features became active for {title}. Skipping heatmap.")
         return

    sorted_active_features = sorted(active_feature_indices, key=lambda f: feature_entry[f])
    try:
         ordered_basis_names = [format_term_for_latex(basis_names[i]) for i in sorted_active_features]
    except Exception as e:
         print(f"WARN: LaTeX formatting failed for heatmap labels: {e}. Using raw names.")
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
        "truncated_heatmap_cmap", HEATMAP_CMAP(np.linspace(0, 0.75, 256))
    )

    im = ax.imshow(plot_status, cmap=truncated_cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)

    # --- Y-axis Ticks and Labels ---
    ax.set_yticks(np.arange(num_sorted_features))
    ax.set_yticklabels(ordered_basis_names, fontsize=PLOT_HEATMAP_YLABEL_SIZE)
    ax.set_ylabel('Model Terms', fontsize=PLOT_LABEL_FONTSIZE)

    # --- X-axis Ticks and Labels ---
    if log_x:
        tick_indices = np.linspace(0, n_steps - 1, min(8, n_steps), dtype=int)
        ax.set_xticks(tick_indices)
        tick_labels = []
        for i in tick_indices:
            log_val = np.log10(plot_x_values[i] + EPS)
            if abs(log_val - round(log_val)) < 0.3:
                tick_labels.append(f'10$^{{{int(round(log_val))}}}$')
            else:
                tick_labels.append(f'10$^{{{log_val:.1f}}}$')
        ax.set_xticklabels(tick_labels, fontsize=PLOT_TICK_FONTSIZE)
    else:
        tick_indices = np.linspace(0, n_steps - 1, min(10, n_steps), dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{int(plot_x_values[i])}' for i in tick_indices], fontsize=PLOT_TICK_FONTSIZE)
    ax.set_xlabel(x_axis_label, fontsize=PLOT_LABEL_FONTSIZE)

    # --- Mark the chosen values using DIRECT INDICES ---
    legend_elements = [
        Patch(facecolor=HEATMAP_CMAP(0.75), label='Active'), 
        Patch(facecolor=HEATMAP_CMAP(0.0), edgecolor='darkgray', label='Inactive')
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
                label_val_str = f'10$^{{{np.log10(max(label_val_for_display, EPS)):.1f}}}$'
            else:
                label_val_str = f'{int(label_val_for_display)}' if is_omp else f'{label_val_for_display:.2f}'
            
            vline_label = f'{method}'
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=vline_label))
        else:
            print(f"WARN: Chosen index for {method} is invalid ({chosen_index}), cannot plot vertical line.")

    # --- Grid and Legend ---
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=PLOT_LEGEND_FONTSIZE)

    # --- Final Touches ---
    ax.set_title(title, fontsize=PLOT_TITLE_FONTSIZE)
    fig.subplots_adjust(bottom=0.25, left=0.3, right=0.95)

    if filename:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            print(f"Saved activation heatmap: {filename}")
        except Exception as e: print(f"Error saving heatmap: {e}")
    plt.close(fig)
    
def _annotate_iteration(ax, x_iter, color, text=None):
    """
    Place 'Iteration = X' rotated 90° beside the vertical line.
    """
    if text is None:
        try:
            x_show = int(x_iter)
        except Exception:
            x_show = x_iter
        text = f"Iteration = {x_show}"

    y0, y1 = ax.get_ylim()
    y_mid = 0.5 * (y0 + y1)

    ax.annotate(
        text,
        xy=(x_iter, y_mid), xytext=(4, 0),  # 4pt to the right of line
        textcoords='offset points',
        ha='left', va='center',
        fontsize=PLOT_TICK_FONTSIZE - 4,
        color=color,
        clip_on=False,
        rotation=90   # rotate vertical
    )
        
def run_lasso_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None):
    """
    Performs LassoCV (coordinate descent) and Lasso path AIC/BIC selection.
    Plots a single heatmap and criteria plots. Measures time.
    Returns a dictionary of results for 'LassoCV', 'LassoAIC', 'LassoBIC'.
    Enforces non-negative coefficients (positive=True) for physical consistency.
    """
    print(f"\n--- Running LASSO Analysis (CV, AIC, BIC) ({title_prefix}) ---")
    n_samples, n_features = X_scaled.shape
    base_results = { 'coeffs': np.zeros(n_features), 'optimal_k': np.nan, 'n_nonzero': 0, 'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan, 'full_basis_names': feature_names }
    results = { 'LassoCV': {**base_results, 'method': 'LassoCV'}, 'LassoAIC': {**base_results, 'method': 'LassoAIC'}, 'LassoBIC': {**base_results, 'method': 'LassoBIC'} }
    if n_features == 0: print("ERROR: No features."); return results

    # --- 1. Run LassoCV to find the optimal alpha (non-negative) ---
    print("  Running LassoCV to determine optimal alpha...")
    start_time_cv = datetime.datetime.now()
    best_alpha_from_cv = np.nan
    
    # Define a single source of truth for the alpha range
    alpha_min_log = -8
    alpha_max_log = 0
    alphas_cv_grid = np.logspace(alpha_min_log, alpha_max_log, 1000)  # dense for CV

    try:
        # Use the explicit grid for CV (bounded) and enforce β >= 0
        lasso_cv = LassoCV(
            cv=cv_folds,
            alphas=alphas_cv_grid,
            max_iter=5000,
            n_jobs=-1,
            random_state=random_seed,
            precompute='auto',
            positive=True
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            lasso_cv.fit(X_scaled, y)
        best_alpha_from_cv = lasso_cv.alpha_
        print(f"  LassoCV determined optimal alpha: {best_alpha_from_cv:.6f}")
        
        alphas_cv_plot = np.array([]); cv_nmse_mean_plot = np.array([]); cv_nmse_std_plot = np.array([])
        try:
            alphas_cv_full = np.asarray(lasso_cv.alphas_)           # shape (n_alphas,)
            mse_path_cv = np.asarray(lasso_cv.mse_path_)            # shape (n_alphas, n_folds[, n_targets])
            if mse_path_cv.ndim == 3:
                cv_mse_mean_full = np.nanmean(mse_path_cv, axis=(1, 2))
                cv_mse_std_full  = np.nanstd(mse_path_cv, axis=(1, 2))
            elif mse_path_cv.ndim == 2:
                cv_mse_mean_full = np.nanmean(mse_path_cv, axis=1)
                cv_mse_std_full  = np.nanstd(mse_path_cv, axis=1)
            else:
                cv_mse_mean_full = mse_path_cv.ravel()
                cv_mse_std_full  = np.zeros_like(cv_mse_mean_full)

            # Convert CV MSE to statistical NMSE, then robustly map to [0,1] for plotting
            y_centered = y - np.mean(y)
            y_var_cv = float(np.var(y_centered))
            if y_var_cv > EPS:
                nmse_mean_full = cv_mse_mean_full / y_var_cv
                nmse_std_full  = cv_mse_std_full  / y_var_cv
            else:
                nmse_mean_full = cv_mse_mean_full
                nmse_std_full  = cv_mse_std_full

            # Robust min–max via percentiles (protect from outliers at extreme alphas)
            p_lo = float(np.nanpercentile(nmse_mean_full, 5))
            p_hi = float(np.nanpercentile(nmse_mean_full, 95))
            den  = max(p_hi - p_lo, EPS)

            cv_nmse_mean_full = (nmse_mean_full - p_lo) / den
            cv_nmse_std_full  = nmse_std_full / den

            # Clip to [0,1] for display
            cv_nmse_mean_full = np.clip(cv_nmse_mean_full, 0.0, 1.0)
            cv_nmse_std_full  = np.clip(cv_nmse_std_full,  0.0, 1.0)

            # Subsample to 20 evenly spaced points to declutter
            n_plot = min(20, alphas_cv_full.size)
            if n_plot > 0:
                sample_idx = np.unique(np.linspace(0, alphas_cv_full.size - 1, n_plot, dtype=int))
                alphas_cv_plot     = alphas_cv_full[sample_idx]
                cv_nmse_mean_plot  = cv_nmse_mean_full[sample_idx]
                cv_nmse_std_plot   = cv_nmse_std_full[sample_idx]
        except Exception as e:
            print(f"WARN: Unable to extract/normalize LassoCV CV curve: {e}")
        
    except Exception as e:
        print(f"ERROR during LassoCV: {e}")
        results['LassoCV'].update({'error': str(e)})

    # --- 2. Create the UNIFIED PATH DataFrame (non-negative) ---
    print("\n  Generating a unified Lasso Path for all criteria...")
    start_time_path = datetime.datetime.now()
    df_path = pd.DataFrame()
    alphas_lasso, coefs_lasso = None, None
    try:
        y_centered = y - np.mean(y)
        
        # Generate a rich path, ensuring the CV alpha is included for exact matching
        alphas_base = np.logspace(alpha_min_log, alpha_max_log, 20)
        alphas_for_path = np.unique(np.concatenate([alphas_base, [best_alpha_from_cv] if pd.notna(best_alpha_from_cv) else []]))
        alphas_for_path = np.sort(alphas_for_path)[::-1] # Sort descending

        alphas_lasso, coefs_lasso, _ = lasso_path(
            X_scaled, y, alphas=alphas_for_path, max_iter=5000, precompute='auto', positive=True
        )
        # numerical guard
        coefs_lasso = np.maximum(coefs_lasso, 0.0)

        path_results_list = []
        for i in range(coefs_lasso.shape[1]):
            alpha_i = alphas_lasso[i]
            beta_i = np.maximum(coefs_lasso[:, i], 0.0)  # ensure non-negative
            n_params = np.sum(beta_i > EPS)
            
            if n_params == 0 and i < coefs_lasso.shape[1] - 1: continue

            y_pred_train = X_scaled @ beta_i
            mse = mean_squared_error(y_centered, y_pred_train)
            aic, bic = calculate_aic_bic(y, y_pred_train, int(n_params), n_samples)
            
            path_results_list.append({
                'alpha': alpha_i, 'k': int(n_params), 'coeffs': beta_i,
                'mse_train': mse, 'aic': aic, 'bic': bic
            })
        
        df_path = pd.DataFrame(path_results_list)
        print(f"  Unified path created with {len(df_path)} unique models.")

    except Exception as e:
        print(f"ERROR during unified path generation: {e}"); traceback.print_exc()
        results['LassoCV'].update({'error': str(e)}); results['LassoAIC'].update({'error': str(e)}); results['LassoBIC'].update({'error': str(e)})
        return results
    
    # --- 2b. Calculate Normalized Metrics for the Path ---
    y_var = np.var(y_centered)
    if y_var > EPS:
        df_path['norm_mse'] = df_path['mse_train'] / y_var
    else:
        df_path['norm_mse'] = df_path['mse_train'] # Avoid division by zero if y is constant

    df_path['l1_norm'] = df_path['coeffs'].apply(lambda x: np.sum(np.abs(x)))
    max_l1_norm = df_path['l1_norm'].max()
    if max_l1_norm > EPS:
        df_path['norm_l1'] = df_path['l1_norm'] / max_l1_norm
    else:
        df_path['norm_l1'] = 0.0

    # --- 3. Identify Optimal Models from the UNIFIED DataFrame ---
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    if not df_path.empty and pd.notna(best_alpha_from_cv):
        cv_model_row = df_path.iloc[(df_path['alpha'] - best_alpha_from_cv).abs().argsort()[:1]].iloc[0]
        results['LassoCV'].update({
            'coeffs': cv_model_row['coeffs'], 'alpha': cv_model_row['alpha'], 'n_nonzero': cv_model_row['k'], 
            'aic': cv_model_row['aic'], 'bic': cv_model_row['bic'], 
            'selected_features': [feature_names[i] for i, c in enumerate(cv_model_row['coeffs']) if c > EPS],  # positivity-aware
            'time': duration_cv
        })
        print(f"  LassoCV model (k={cv_model_row['k']}) identified on unified path in {duration_cv:.2f}s.")

    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()
    if not df_path.empty:
        aic_model_row = df_path.loc[df_path['aic'].idxmin()]
        results['LassoAIC'].update({
            'coeffs': aic_model_row['coeffs'], 'alpha': aic_model_row['alpha'], 'n_nonzero': aic_model_row['k'],
            'aic': aic_model_row['aic'], 'bic': aic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(aic_model_row['coeffs']) if c > EPS],  # positivity-aware
            'time': duration_path
        })
        print(f"  LassoAIC model (k={aic_model_row['k']}) identified on unified path.")

        bic_model_row = df_path.loc[df_path['bic'].idxmin()]
        results['LassoBIC'].update({
            'coeffs': bic_model_row['coeffs'], 'alpha': bic_model_row['alpha'], 'n_nonzero': bic_model_row['k'],
            'aic': bic_model_row['aic'], 'bic': bic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(bic_model_row['coeffs']) if c > EPS],  # positivity-aware
            'time': duration_path
        })
        print(f"  LassoBIC model (k={bic_model_row['k']}) identified on unified path.")

    # --- 4. Plotting from the UNIFIED DataFrame ---
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
                'color': plasma(0.25), 'linestyle': '--'
            },
            'AIC': {
                'index': find_alpha_index_in_path(aic_exact_model['alpha'], alphas_lasso) if aic_exact_model is not None else None,
                'label_val': aic_exact_model['k'] if aic_exact_model is not None else None, 
                'color': plasma(0.55), 'linestyle': '-'
            },
            'BIC': {
                'index': find_alpha_index_in_path(bic_exact_model['alpha'], alphas_lasso) if bic_exact_model is not None else None,
                'label_val': bic_exact_model['k'] if bic_exact_model is not None else None, 
                'color': plasma(0.85), 'linestyle': '-.'
            }
        }
        
        plot_activation_heatmap(alphas_lasso, np.maximum(coefs_lasso, 0.0), feature_names, chosen_values_heatmap,
                                title=f"Lasso Activation Path", x_axis_label=r'Regularization Strength',
                                filename=os.path.join(save_dir, f"{title_prefix}_Lasso_ActivationHeatmap.pdf"),
                                log_x=True, reverse_x=True, is_omp=False)
        
        df_plot = df_path.sort_values('alpha', ascending=False)
        
        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Use the EXACT alpha values from the same models
        alpha_for_cv = cv_exact_model['alpha'] if cv_exact_model is not None else None
        alpha_for_aic = aic_exact_model['alpha'] if aic_exact_model is not None else None
        alpha_for_bic = bic_exact_model['alpha'] if bic_exact_model is not None else None

        # Row 0: CV NMSE (mean ± std) and Norm. L1 on twin axis
        ax0_twin = axs[0].twinx()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0 and np.isfinite(cv_nmse_mean_plot).any():
            axs[0].plot(alphas_cv_plot, cv_nmse_mean_plot, 'o-', ms=3, color=plasma(0.25), label='CV NMSE')
            if isinstance(cv_nmse_std_plot, np.ndarray) and cv_nmse_std_plot.size == cv_nmse_mean_plot.size:
                try:
                    axs[0].fill_between(alphas_cv_plot,
                                        cv_nmse_mean_plot - cv_nmse_std_plot,
                                        cv_nmse_mean_plot + cv_nmse_std_plot,
                                        color=plasma(0.25), alpha=0.15, linewidth=0)
                except Exception:
                    pass
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes, ha='center', va='center')

        # Norm. L1 on twin axis
        ax0_twin.plot(df_plot['alpha'], df_plot['norm_l1'], 's-', ms=3, color=plasma(0.65), alpha=0.6, label='Norm. L1')
        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE, color=plasma(0.25), labelpad=-20)
        ax0_twin.set_ylabel('Norm. L1', fontsize=PLOT_LABEL_FONTSIZE, color=plasma(0.65), labelpad=-10)

        if pd.notna(alpha_for_cv):
            axs[0].axvline(alpha_for_cv, color=chosen_values_heatmap['CV']['color'], ls=chosen_values_heatmap['CV']['linestyle'], lw=1.5)
            axs[0].text(alpha_for_cv * 1.5, 0.5, f"$\\lambda_L={alpha_for_cv:.1e}$", rotation=90, verticalalignment='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=chosen_values_heatmap['CV']['color'])

        # Keep the same y-limits and ticks styling as before
        axs[0].set_ylim(-0.05, 1.05); ax0_twin.set_ylim(-0.05, 1.05)
        axs[0].set_yticks([0, 1.0]); ax0_twin.set_yticks([0, 1.0])
        ax0_twin.tick_params(axis='y', labelcolor=plasma(0.65))
        axs[0].tick_params(axis='y', labelcolor=plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_plot['alpha'], df_plot['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_aic):
            axs[1].axvline(alpha_for_aic, color=chosen_values_heatmap['AIC']['color'], ls=chosen_values_heatmap['AIC']['linestyle'], lw=1.5)
            axs[1].text(alpha_for_aic * 1.5, np.mean(axs[1].get_ylim()), f"$\\lambda_L={alpha_for_aic:.1e}$", rotation=90, verticalalignment='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=chosen_values_heatmap['AIC']['color'])
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['alpha'], df_plot['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_bic):
            axs[2].axvline(alpha_for_bic, color=chosen_values_heatmap['BIC']['color'], ls=chosen_values_heatmap['BIC']['linestyle'], lw=1.5)
            axs[2].text(alpha_for_bic * 1.5, np.mean(axs[2].get_ylim()), f"$\\lambda_L={alpha_for_bic:.1e}$", rotation=90, verticalalignment='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=chosen_values_heatmap['BIC']['color'])
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])

        # Formatting all axes
        axs[2].set_xlabel('Regularization Strength', fontsize=PLOT_LABEL_FONTSIZE)
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
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_Lasso_Criteria.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)

    return results

def run_lars_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None):
    """
    LARS with unified path for CV, AIC, and BIC.
    Enforces non-negative coefficients (β >= 0) via NNLS refits on each support.
    CV error is normalized (NMSE) to [0, 1] via min–max scaling across the path.
    """
    from scipy.optimize import nnls
    print(f"\n--- Running LARS Analysis (CV, AIC, BIC; unified path) ({title_prefix}) ---")

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
        print("ERROR: No features.")
        return results

    # ---------- helper: NNLS refit with unconstrained intercept ----------
    def _nnls_refit(Xmat, yvec, S):
        """
        Solve: min_{β_S>=0, b in R} || y - (b + X_S β_S) ||^2
        by centering to remove intercept from the NNLS and then recovering b.
        Returns (full_beta, intercept, y_hat_on_Xmat).
        """
        if len(S) == 0:
            b = float(np.mean(yvec))
            beta_full = np.zeros(Xmat.shape[1])
            yhat = np.full_like(yvec, b, dtype=float)
            return beta_full, b, yhat

        X_S = Xmat[:, S]
        y_mean = float(np.mean(yvec))
        X_means = np.mean(X_S, axis=0)

        y_c = yvec - y_mean
        X_c = X_S - X_means  # column-centering

        # NNLS on centered system
        beta_S, _ = nnls(X_c, y_c)

        # recover intercept
        b = y_mean - float(X_means @ beta_S)

        # assemble full β
        beta_full = np.zeros(Xmat.shape[1])
        beta_full[np.asarray(S, dtype=int)] = beta_S

        # predictions
        yhat = b + X_S @ beta_S
        return beta_full, b, yhat

    # 1) LARS path (support order only)
    start_time_path = datetime.datetime.now()
    y_centered = y - np.mean(y)
    try:
        from sklearn.linear_model import lars_path
        _, _, coefs_path_raw = lars_path(
            X_scaled, y_centered,
            method='lars',
            copy_X=True,
            eps=1e-10,
            max_iter=min(5 * n_features, max(n_features, 100))
        )
        if coefs_path_raw.shape[1] < 2:
            raise ValueError("LARS path contains fewer than 2 steps.")
        coefs_path_lars = coefs_path_raw[:, 1:]  # drop all-zero column
        n_steps = coefs_path_lars.shape[1]
        print(f"  LARS path computed with {n_steps} nonzero steps.")
    except Exception as e:
        print(f"ERROR: LARS path construction failed: {e}")
        traceback.print_exc()
        return results

    # Unique-k compression (based on support size; sign ignored for support detection)
    supports = []
    kept_indices = []
    seen_k = set()
    for t in range(n_steps):
        beta_t = coefs_path_lars[:, t]
        S_t = np.flatnonzero(np.abs(beta_t) > EPS)
        k_t = int(S_t.size)
        if k_t == 0 or k_t in seen_k:
            continue
        seen_k.add(k_t)
        supports.append((t, S_t))
        kept_indices.append(t)

    steps_for_plot = np.arange(1, len(supports) + 1)
    print(f"  Unique-k path length: {len(supports)} (k runs from 1 to {len(seen_k)}).")

    if len(supports) == 0:
        print("ERROR: No supports after unique-k compression.")
        return results

    # Build a β≥0 coefficient matrix for the heatmap using NNLS refits on the full data
    coefs_nnls_cols = []
    for _, S_t in supports:
        beta_full, b_full, _ = _nnls_refit(X_scaled, y, S_t)
        coefs_nnls_cols.append(beta_full)
    coefs_path_nonneg = np.column_stack(coefs_nnls_cols)  # shape (p, len(supports))

    # 2) AIC/BIC on full data (β≥0 via NNLS refit)
    df_path_rows = []
    for idx_in_path, (t, S_t) in enumerate(supports):
        k_t = len(S_t)
        if k_t == 0:
            continue
        try:
            beta_full, b_full, y_hat_full = _nnls_refit(X_scaled, y, S_t)
            # effective parameters: k coefficients + intercept
            aic, bic = calculate_aic_bic(
                y_true=y, y_pred=y_hat_full,
                n_params=k_t + 1, n_samples=n_samples
            )
            # training MSE (non-centered) for record
            mse_train = mean_squared_error(y, y_hat_full)
            df_path_rows.append({
                'step': idx_in_path + 1,
                'k': k_t,
                'coeffs': beta_full,
                'support': S_t,
                'mse_train': mse_train,
                'aic': aic,
                'bic': bic
            })
        except Exception as e_step:
            print(f"  WARN: Full-data NNLS refit failed at step {idx_in_path+1} (k={k_t}): {e_step}")

    if not df_path_rows:
        print("ERROR: No valid steps for AIC/BIC.")
        return results
    df_path = pd.DataFrame(df_path_rows)

    # 3) CV with the same supports (β≥0 via NNLS in each fold)
    start_time_cv = datetime.datetime.now()
    if isinstance(cv_folds, int):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        kf = cv_folds

    cv_mse_mean = []
    cv_mse_std  = []
    for _, S_t in supports:
        k_t = len(S_t)
        if k_t == 0:
            cv_mse_mean.append(np.nan)
            cv_mse_std.append(np.nan)
            continue

        fold_mse = []
        for tr_idx, va_idx in kf.split(X_scaled, y):
            X_tr = X_scaled[tr_idx]
            y_tr = y[tr_idx]
            X_va = X_scaled[va_idx]
            y_va = y[va_idx]

            try:
                beta_full_tr, b_tr, _ = _nnls_refit(X_tr, y_tr, S_t)
                # predict on validation
                y_va_hat = b_tr + X_va[:, S_t] @ beta_full_tr[S_t]
                fold_mse.append(mean_squared_error(y_va, y_va_hat))
            except Exception:
                fold_mse.append(np.inf)

        cv_mse_mean.append(np.mean(fold_mse))
        cv_mse_std.append(np.std(fold_mse, ddof=1) if len(fold_mse) > 1 else 0.0)

    cv_mse_mean = np.array(cv_mse_mean)
    cv_mse_std  = np.array(cv_mse_std)

    # Robust normalization to [0,1] across the path (display scaling)
    finite_mask = np.isfinite(cv_mse_mean)
    if np.any(finite_mask):
        min_mse = float(np.nanmin(cv_mse_mean))
        max_mse = float(np.nanmax(cv_mse_mean))
        if np.isclose(max_mse, min_mse):
            cv_nmse_mean = np.zeros_like(cv_mse_mean)
            cv_nmse_std  = np.zeros_like(cv_mse_std)
        else:
            rng = max(max_mse - min_mse, EPS)
            cv_nmse_mean = (cv_mse_mean - min_mse) / rng
            cv_nmse_std  = cv_mse_std / rng
    else:
        cv_nmse_mean = cv_mse_mean
        cv_nmse_std  = cv_mse_std

    best_idx_cv = int(np.nanargmin(cv_mse_mean))  # choose by true MSE; display uses normalized
    cv_row = df_path.iloc[best_idx_cv]
    best_k_from_cv = int(cv_row['k'])
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    print(f"  CV chose k={best_k_from_cv} on the unified LARS path (NNLS-refit).")

    # 4) Package results (β≥0)
    results['LarsCV'].update({
        'coeffs': cv_row['coeffs'],
        'optimal_k': cv_row['k'],
        'n_nonzero': int(np.sum(cv_row['coeffs'] > EPS)),
        'aic': cv_row['aic'],
        'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > EPS],
        'time': duration_cv
    })

    aic_row = df_path.loc[df_path['aic'].idxmin()]
    results['LarsAIC'].update({
        'coeffs': aic_row['coeffs'],
        'optimal_k': aic_row['k'],
        'n_nonzero': int(np.sum(aic_row['coeffs'] > EPS)),
        'aic': aic_row['aic'],
        'bic': aic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > EPS],
        'time': (datetime.datetime.now() - start_time_path).total_seconds()
    })

    bic_row = df_path.loc[df_path['bic'].idxmin()]
    results['LarsBIC'].update({
        'coeffs': bic_row['coeffs'],
        'optimal_k': bic_row['k'],
        'n_nonzero': int(np.sum(bic_row['coeffs'] > EPS)),
        'aic': bic_row['aic'],
        'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > EPS],
        'time': (datetime.datetime.now() - start_time_path).total_seconds()
    })

    # 5) Plotting (use NNLS-refit coefficients for heatmap)
    order_by_k = np.argsort(df_path['k'].values)
    k_sorted = df_path['k'].values[order_by_k].astype(int)
    cv_nmse_mean_sorted = cv_nmse_mean[order_by_k]
    cv_nmse_std_sorted  = cv_nmse_std[order_by_k]

    chosen_values_heatmap = {
        'CV':  {'index': best_idx_cv, 'label_val': int(cv_row['k']),  'color': plasma(0.25), 'linestyle': '--'},
        'AIC': {'index': int(np.where(df_path.index == df_path['aic'].idxmin())[0][0]),
                'label_val': int(aic_row['k']), 'color': plasma(0.55), 'linestyle': '-'},
        'BIC': {'index': int(np.where(df_path.index == df_path['bic'].idxmin())[0][0]),
                'label_val': int(bic_row['k']), 'color': plasma(0.85), 'linestyle': '-.'}
    }
    try:
        # Heatmap over iterations (columns are β≥0)
        plot_activation_heatmap(
            steps_for_plot, coefs_path_nonneg, feature_names, chosen_values_heatmap,
            title=f"LARS Activation Path", x_axis_label='LARS Iteration',
            filename=os.path.join(save_dir, f"{title_prefix}_Lars_ActivationHeatmap.pdf"),
            log_x=False, reverse_x=False, is_omp=False
        )

        df_plot = df_path.sort_values('k')
        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std), normalized to [0,1] for display; ticks forced
        if (k_sorted.size > 0 and np.any(np.isfinite(cv_nmse_mean_sorted))):
            lower = np.clip(cv_nmse_mean_sorted - cv_nmse_std_sorted, 0, 1)
            upper = np.clip(cv_nmse_mean_sorted + cv_nmse_std_sorted, 0, 1)
            axs[0].fill_between(k_sorted, lower, upper, alpha=0.2,
                                edgecolor='none', facecolor=plasma(0.25))
            axs[0].plot(k_sorted, cv_nmse_mean_sorted, 'o-', ms=3, color=plasma(0.25))
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes,
                        ha='center', va='center')

        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE,
                          color=plasma(0.25), labelpad=-10)
        axs[0].tick_params(axis='y', labelcolor=plasma(0.25))
        axs[0].axvline(int(cv_row['k']), color=plasma(0.25), ls='--', lw=1.5)
        _annotate_iteration(axs[0], int(cv_row['k']), color=plasma(0.25))
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([k_sorted.min(), k_sorted.max()])

        # Row 1: AIC
        axs[1].plot(df_plot['k'], df_plot['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['k']), color=plasma(0.55), ls='-', lw=1.5)
        _annotate_iteration(axs[1], int(aic_row['k']), color=plasma(0.55))
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])
        axs[1].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['k'], df_plot['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['k']), color=plasma(0.85), ls='-.', lw=1.5)
        _annotate_iteration(axs[2], int(bic_row['k']), color=plasma(0.85))
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])
        axs[2].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        axs[2].set_xlabel('LARS iteration', fontsize=PLOT_LABEL_FONTSIZE)
        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_Lars_Criteria.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)

    except Exception as e_plot:
        print(f"  WARN: LARS plotting failed: {e_plot}")

    return results

def run_omp_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None):
    """
    OMP with a SINGLE shared greedy path used for CV, AIC, and BIC.
    Enforces non-negative coefficients (β >= 0) throughout:
      - Path construction: NNLS on centered target (no intercept) and positive-correlation selection.
      - CV/AIC/BIC refits: NNLS with unconstrained intercept recovered analytically.

    Plots and tick behavior follow the same layout and ticks as elsewhere.
    """
    from scipy.optimize import nnls

    print(f"\n--- Running OMP Analysis (CV, AIC, BIC; unified path) ({title_prefix}) ---")
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

    # Helper: NNLS refit with unconstrained intercept
    def _nnls_refit_with_intercept(Xmat, yvec, S):
        """
        Solve: min_{β_S>=0, b in R} || y - (b + X_S β_S) ||^2
        by centering columns of X_S and y, NNLS on centered system, then recover b.
        Returns (full_beta, intercept, y_hat_on_full_X).
        """
        if len(S) == 0:
            b = float(np.mean(yvec))
            beta_full = np.zeros(Xmat.shape[1])
            yhat = np.full_like(yvec, b, dtype=float)
            return beta_full, b, yhat

        X_S = Xmat[:, S]
        y_mean = float(np.mean(yvec))
        X_means = np.mean(X_S, axis=0)

        y_c = yvec - y_mean
        X_c = X_S - X_means

        beta_S, _ = nnls(X_c, y_c)  # β_S >= 0

        b = y_mean - float(X_means @ beta_S)
        beta_full = np.zeros(Xmat.shape[1]); beta_full[np.asarray(S, dtype=int)] = beta_S
        yhat = b + X_S @ beta_S
        return beta_full, b, yhat

    # 1) Greedy OMP path on centered y (β ≥ 0 via NNLS; positive-correlation selection)
    y_centered = y - np.mean(y)
    X = X_scaled
    max_k = min(n_features, max(1, np.linalg.matrix_rank(X)))

    active = []
    cols = []
    r = y_centered.copy()

    for k in range(1, max_k + 1):
        corr = X.T @ r  # correlations with residual
        if active:
            corr[np.array(active, dtype=int)] = -np.inf  # exclude actives
        # positivity-consistent selection: require nonnegative correlation
        j = int(np.argmax(corr))
        if not np.isfinite(corr[j]) or corr[j] <= 0:
            print(f"  OMP stops at k={k}: no atom with positive correlation remains.")
            break
        if j in active:
            print(f"  OMP early stop at k={k} (duplicate)."); break
        active.append(j)

        # NNLS on current support (no intercept; target already centered)
        XA = X[:, active]
        try:
            beta_A, _ = nnls(XA, y_centered)  # β_A >= 0
        except Exception as e_nn:
            print(f"  WARN: NNLS failed at k={k}: {e_nn}; stopping.")
            break

        beta_full = np.zeros(n_features); beta_full[np.array(active, dtype=int)] = beta_A
        cols.append(beta_full)

        # Update residual
        r = y_centered - XA @ beta_A
        if np.linalg.norm(r) < 1e-14:
            print(f"  OMP residual ~0 at k={k}; stopping."); break

    if not cols:
        print("ERROR: empty OMP path."); return results

    coefs_path_nonneg = np.column_stack(cols)  # (p, K) nonnegative by construction
    steps = np.arange(1, coefs_path_nonneg.shape[1] + 1)
    supports = [np.flatnonzero(coefs_path_nonneg[:, i] > EPS) for i in range(coefs_path_nonneg.shape[1])]
    K = len(supports)
    print(f"  OMP path computed with K={K} steps (k from 1..{K}).")

    # 2) AIC/BIC on full data (NNLS with free intercept)
    df_path_rows = []
    for i, S in enumerate(supports, start=1):
        try:
            beta_full, b_full, y_hat_full = _nnls_refit_with_intercept(X, y, S)
            aic, bic = calculate_aic_bic(y_true=y, y_pred=y_hat_full,
                                         n_params=len(S) + 1, n_samples=n_samples)
            df_path_rows.append({
                'step': i,
                'k': len(S),
                'coeffs': beta_full,     # nonnegative β (full length)
                'support': S,
                'aic': aic,
                'bic': bic
            })
        except Exception as e_step:
            print(f"  WARN: Full-data NNLS refit failed at k={i}: {e_step}")

    if not df_path_rows:
        print("ERROR: No valid steps for AIC/BIC in OMP."); return results
    df_path = pd.DataFrame(df_path_rows)

    # 3) CV on the same supports (NNLS + free intercept in each fold)
    print("  Cross-validating on unified OMP supports…")
    start_time_cv = datetime.datetime.now()
    if isinstance(cv_folds, int):
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    else:
        kf = cv_folds

    cv_mse_mean = np.zeros(K)
    cv_mse_std  = np.zeros(K)
    for i, S in enumerate(supports):
        fold_mse = []
        for tr, va in kf.split(X, y):
            Xtr, ytr = X[tr], y[tr]
            Xva, yva = X[va], y[va]
            try:
                beta_tr, b_tr, _ = _nnls_refit_with_intercept(Xtr, ytr, S)
                yhat = b_tr + Xva[:, S] @ beta_tr[S]
                fold_mse.append(mean_squared_error(yva, yhat))
            except Exception:
                fold_mse.append(np.inf)
        cv_mse_mean[i] = np.mean(fold_mse)
        cv_mse_std[i]  = np.std(fold_mse, ddof=1) if len(fold_mse) > 1 else 0.0

    # Normalize to [0,1] for display (retain true MSE to choose the best)
    mm = float(np.nanmin(cv_mse_mean)); MM = float(np.nanmax(cv_mse_mean))
    if np.isclose(MM, mm):
        cv_nmse_mean = np.zeros_like(cv_mse_mean); cv_nmse_std = np.zeros_like(cv_mse_std)
    else:
        rng = max(MM - mm, EPS)
        cv_nmse_mean = (cv_mse_mean - mm) / rng
        cv_nmse_std  = cv_mse_std / rng

    best_idx = int(np.nanargmin(cv_mse_mean))
    cv_row  = df_path.iloc[best_idx]
    aic_row = df_path.iloc[df_path['aic'].idxmin()]
    bic_row = df_path.iloc[df_path['bic'].idxmin()]
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    print(f"  CV chose k={int(cv_row['k'])} on the unified OMP path (NNLS-refit).")

    # 4) Results (β ≥ 0)
    results['OMPCV'].update({
        'coeffs': cv_row['coeffs'],
        'optimal_k': int(cv_row['k']),
        'n_nonzero': int(np.sum(cv_row['coeffs'] > EPS)),
        'aic': cv_row['aic'],
        'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > EPS],
        'time': duration_cv
    })
    results['OMPAIC'].update({
        'coeffs': aic_row['coeffs'],
        'optimal_k': int(aic_row['k']),
        'n_nonzero': int(np.sum(aic_row['coeffs'] > EPS)),
        'aic': aic_row['aic'],
        'bic': aic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > EPS],
        'time': duration_cv
    })
    results['OMPBIC'].update({
        'coeffs': bic_row['coeffs'],
        'optimal_k': int(bic_row['k']),
        'n_nonzero': int(np.sum(bic_row['coeffs'] > EPS)),
        'aic': bic_row['aic'],
        'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > EPS],
        'time': duration_cv
    })

    # 5) Plots (heatmap uses β ≥ 0 from full-data NNLS)
    try:
        # For heatmap we want a (p × steps) matrix of nonnegative coefs at each step.
        # Use FULL-DATA NNLS refits so the heatmap matches AIC/BIC models.
        heat_cols = []
        for S in supports:
            beta_full, _, _ = _nnls_refit_with_intercept(X, y, S)
            heat_cols.append(beta_full)
        heat_coefs = np.column_stack(heat_cols)

        chosen_heat = {
            'CV':  {'index': best_idx,  'label_val': int(cv_row['k']),  'color': plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': int(np.where(df_path.index == df_path['aic'].idxmin())[0][0]),
                    'label_val': int(aic_row['k']), 'color': plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': int(np.where(df_path.index == df_path['bic'].idxmin())[0][0]),
                    'label_val': int(bic_row['k']), 'color': plasma(0.85), 'linestyle': '-.'},
        }
        plot_activation_heatmap(
            steps, heat_coefs, feature_names, chosen_heat,
            title="OMP Activation Path", x_axis_label='OMP Iteration',
            filename=os.path.join(save_dir, f"{title_prefix}_OMP_ActivationHeatmap.pdf"),
            log_x=False, reverse_x=False, is_omp=True
        )

        # 3×1 criteria
        df_plot = df_path.sort_values('k')
        order_by_k = np.argsort(df_path['k'].values)
        k_sorted = df_path['k'].values[order_by_k].astype(int)
        cv_nmse_mean_sorted = cv_nmse_mean[order_by_k]
        cv_nmse_std_sorted  = cv_nmse_std[order_by_k]

        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std), ticks y=[0,1], x=[min(k),max(k)]
        if (k_sorted.size > 0 and np.any(np.isfinite(cv_nmse_mean_sorted))):
            lower = np.clip(cv_nmse_mean_sorted - cv_nmse_std_sorted, 0, 1)
            upper = np.clip(cv_nmse_mean_sorted + cv_nmse_std_sorted, 0, 1)
            axs[0].fill_between(k_sorted, lower, upper, alpha=0.2,
                                edgecolor='none', facecolor=plasma(0.25))
            axs[0].plot(k_sorted, cv_nmse_mean_sorted, 'o-', ms=3, color=plasma(0.25))
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes,
                        ha='center', va='center')

        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE,
                          color=plasma(0.25), labelpad=-10)
        axs[0].tick_params(axis='y', labelcolor=plasma(0.25))
        axs[0].axvline(int(cv_row['k']), color=plasma(0.25), ls='--', lw=1.5)
        _annotate_iteration(axs[0], int(cv_row['k']), color=plasma(0.25))
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([k_sorted.min(), k_sorted.max()])

        # Row 1: AIC
        axs[1].plot(df_plot['k'], df_plot['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['k']), color=plasma(0.55), ls='-', lw=1.5)
        _annotate_iteration(axs[1], int(aic_row['k']), color=plasma(0.55))
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])
        axs[1].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['k'], df_plot['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['k']), color=plasma(0.85), ls='-.', lw=1.5)
        _annotate_iteration(axs[2], int(bic_row['k']), color=plasma(0.85))
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])
        axs[2].set_xticks([df_plot['k'].min(), df_plot['k'].max()])

        axs[2].set_xlabel('OMP iteration', fontsize=PLOT_LABEL_FONTSIZE)
        for ax in axs:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_OMP_Criteria.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)

    except Exception as e_plot:
        print(f"  WARN: OMP plotting failed: {e_plot}")

    return results

# ==============================================
# Entry Point (Specified SEFs, UT Fit, P11/P22 Eval via Framework Logic)
# ==============================================
if __name__ == "__main__":

    # Resolve output directories relative to this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Save outputs as a single-level hierarchy: outputs/<scenario>
    outputs_root = os.path.join(script_dir, "outputs")
    BASE_SAVE_DIR = outputs_root
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # Define the log file path inside BASE_SAVE_DIR
    log_file_name = os.path.join(
        BASE_SAVE_DIR,
        "log_clean.txt"
    )
    original_stdout = sys.stdout  # Save a reference to the original standard output
    log_file_handle = None # Initialize file handle

    try:
        # Redirect stdout to the log file
        log_file_handle = open(log_file_name, 'w')
        sys.stdout = log_file_handle
        
        # Configuration
        RANDOM_SEED = 42 # Master seed for all random operations
        np.random.seed(RANDOM_SEED) # Seed NumPy's random number generator
        print(f"INFO: Using global random seed: {RANDOM_SEED}")
        
        CV_FOLDS = 5
        # BASE_SAVE_DIR is already defined relative to the script directory above
        COEFF_THRESHOLD = 1e-6 
        RIDGE_ALPHA_REFIT = 1e-7 

        SYNTH_N_POINTS = 60
        SYNTH_LAMBDA_RANGE = (0.6, 5.0) 
        SYNTH_GAMMA_RANGE = (-1.0, 1.0) 
        
        SYNTH_RELATIVE_NOISE_LEVELS = [0.0, 0.05, 0.10] 

        # MR_MAX_ORDER = 3
        TARGET_OGDEN_EXPONENTS_INT = [-10, -5, 5, 1, -1] 
        STANDARD_OGDEN_EXPONENTS = [-9, -8, -7, -6, -4, -3, 3, 4, 5, 6, 7, 8, 9, 10]
        FULL_OGDEN_DISCOVERY_SET = sorted(list(set(STANDARD_OGDEN_EXPONENTS + TARGET_OGDEN_EXPONENTS_INT)))
    
        print(f"Potential Ogden Exponents for Discovery Libraries: {FULL_OGDEN_DISCOVERY_SET}")

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
           
        REFIT_REGRESSION_TYPE = f"Ridge(α={RIDGE_ALPHA_REFIT:.1e})" if RIDGE_ALPHA_REFIT > 0 else "OLS"

        overall_start_time = datetime.datetime.now()
        
        print(f"\nStarting Synthetic Data Rediscovery (UT+PS+EBT P11 Fit, Relative Noise) at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        os.makedirs(BASE_SAVE_DIR, exist_ok=True)

        all_initial_results = {}
        all_refit_summaries = {}

        for synth_model_name, term_coeff_map_Pa in synthetic_model_definitions.items(): 
            for rel_noise in SYNTH_RELATIVE_NOISE_LEVELS:
                scenario_name = f"{synth_model_name}_RelNoise{rel_noise*100:.0f}pct"
                print("\n" + "#"*80)
                print(f"### PROCESSING SCENARIO: {scenario_name} (Fit UT+PS+EBT P11) ###")

                print(f"--- Configuring Discovery Library for Ground Truth: {synth_model_name} ---")
                INCLUDE_NEOHOOKEAN=False; INCLUDE_MOONEY_RIVLIN=False; INCLUDE_YEOH=False;
                INCLUDE_ARRUDA_BOYCE=False; INCLUDE_OGDEN=False; INCLUDE_GENT=False;
                INCLUDE_DATA_DRIVEN=False; INCLUDE_FRACTIONAL_POWERS=False;
                INCLUDE_LOW_STRAIN_TERMS=False; INCLUDE_ULTRA_SMALL_STRAIN=False;
                OGDEN_EXPONENTS = [] 

                if synth_model_name in ["MR2", "MR3"]:
                    MODEL_CONFIG_NAME = "MR_Lib_Only"
                    INCLUDE_MOONEY_RIVLIN = True; #MR_MAX_ORDER = 4
                elif synth_model_name in ["O1", "O2"]:
                    MODEL_CONFIG_NAME = "Ogden_Lib_Only"
                    INCLUDE_OGDEN = True; OGDEN_EXPONENTS = FULL_OGDEN_DISCOVERY_SET
                elif synth_model_name in ["MR1O1", "MR2O1", "MR2O2"]:
                    MODEL_CONFIG_NAME = "MR_Ogden_Lib"
                    INCLUDE_MOONEY_RIVLIN = True; #MR_MAX_ORDER = 4
                    INCLUDE_OGDEN = True; OGDEN_EXPONENTS = FULL_OGDEN_DISCOVERY_SET
                else:
                    print(f"ERROR: Unknown synth_model_name '{synth_model_name}'. Skipping."); continue

                print(f"Discovery Library Configured: {MODEL_CONFIG_NAME}")
                CURRENT_BASIS_NAMES = generate_model_library()
                
                if not CURRENT_BASIS_NAMES: print(f"ERROR: Failed to generate library for {scenario_name}. Skipping."); continue
                print(f"Generated {len(CURRENT_BASIS_NAMES)} terms for discovery.")

                all_terms_found = True; print("Checking if synthetic model terms exist in the generated library...")
                for term in term_coeff_map_Pa.keys():
                    if term not in CURRENT_BASIS_NAMES:
                        print(f"ERROR: Term '{term}' needed for model '{synth_model_name}' is MISSING from library '{MODEL_CONFIG_NAME}'!"); all_terms_found = False
                    else: print(f"  OK: Term '{term}' found.")
                if not all_terms_found: print("\nCannot proceed."); continue

                # Collapse path: only outputs/<scenario>, no extra subfolder per library
                current_save_dir = os.path.join(BASE_SAVE_DIR, scenario_name)
                os.makedirs(current_save_dir, exist_ok=True)
                file_prefix = f"{scenario_name}"

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
                    if not df_raw_ut.empty: axes_raw[0].scatter(df_raw_ut['lambda1'], df_raw_ut['P11'], label=f'Noisy P11 (RelNoise {rel_noise*100:.0f}%)', s=15, alpha=0.7);
                    else: axes_raw[0].text(0.5, 0.5, "No UT Data", ha='center', va='center')
                    if 'P11_true' in df_raw_ut.columns: axes_raw[0].plot(df_raw_ut['lambda1'].sort_values(), df_raw_ut.sort_values('lambda1')['P11_true'], 'k--', label='True P11')
                    axes_raw[0].set_xlabel('$λ$'); axes_raw[0].set_ylabel('$P_{11}$ (MPa)'); axes_raw[0].set_title('Generated UT Data P11'); axes_raw[0].legend(frameon=False);
                    if not df_raw_ps.empty: axes_raw[1].scatter(df_raw_ps['lambda1'], df_raw_ps['P11'], label=f'Noisy P11 (RelNoise {rel_noise*100:.0f}%)', s=15, alpha=0.7, color='darkorange');
                    else: axes_raw[1].text(0.5, 0.5, "No PS Data (P11)", ha='center', va='center')
                    if 'P11_true' in df_raw_ps.columns: axes_raw[1].plot(df_raw_ps['lambda1'].sort_values(), df_raw_ps.sort_values('lambda1')['P11_true'], 'k:', label='True P11')
                    axes_raw[1].set_xlabel('$λ$'); axes_raw[1].set_ylabel('$P_{11}$ (MPa)'); axes_raw[1].set_title('Generated PS Data P11'); axes_raw[1].legend(frameon=False);
                    if not df_raw_ebt.empty: axes_raw[2].scatter(df_raw_ebt['lambda1'], df_raw_ebt['P11'], label=f'Noisy P11 (RelNoise {rel_noise*100:.0f}%)', s=15, alpha=0.7, color='green');
                    else: axes_raw[2].text(0.5, 0.5, "No EBT Data (P11)", ha='center', va='center')
                    if 'P11_true' in df_raw_ebt.columns: axes_raw[2].plot(df_raw_ebt['lambda1'].sort_values(), df_raw_ebt.sort_values('lambda1')['P11_true'], 'k:', label='True P11')
                    axes_raw[2].set_xlabel('$λ$'); axes_raw[2].set_ylabel('$P_{11}$ (MPa)'); axes_raw[2].set_title('Generated EBT Data P11'); axes_raw[2].legend(frameon=False);
                    fig_raw.suptitle(f"Raw Synthetic Data (Framework Logic): {scenario_name}")
                    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
                    raw_data_save_path = os.path.join(current_save_dir, f"{file_prefix}_RawSynthData_P11_MPa.pdf")
                    try: fig_raw.savefig(raw_data_save_path, format='pdf', bbox_inches='tight'); print(f"Saved raw synthetic data plot: {raw_data_save_path}")
                    except Exception as e: print(f"Error saving raw data plot: {e}")
                    plt.close(fig_raw)
                except Exception as e: print(f"ERROR during data generation/plotting: {e}"); traceback.print_exc(); continue

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
                    print(f"Model Library for Discovery: {MODEL_CONFIG_NAME}")
                    
                    design_matrix_unweighted, _, basis_names_from_matrix, _, _ = construct_design_matrix(
                        df_for_discovery
                    )
                    if design_matrix_unweighted.shape[0] == 0: raise ValueError("Unweighted matrix for scaling is empty.")
                    
                    scaler = StandardScaler()
                    scaler.fit(design_matrix_unweighted)

                    design_matrix_final, target_vector_weighted, _, _, _ = construct_design_matrix(
                        df_for_discovery
                    )
                    if design_matrix_final.shape[0] == 0: raise ValueError("Final design matrix is empty.")

                    X_weighted_scaled = scaler.transform(design_matrix_final)

                    if CURRENT_BASIS_NAMES != basis_names_from_matrix: print(f"WARN: Basis name mismatch; using list from matrix construction.")
                    basis_names_for_analysis = basis_names_from_matrix
                    if len(basis_names_for_analysis) != X_weighted_scaled.shape[1]: raise ValueError("Basis/matrix column mismatch.")
                    
                    if not np.all(np.isfinite(X_weighted_scaled)):
                        print("WARN: Non-finite values after scaling. Replacing with 0.")
                        X_weighted_scaled = np.nan_to_num(X_weighted_scaled)

                except Exception as e: print(f"ERROR in data prep: {e}"); traceback.print_exc(); continue

                scenario_initial_results = {}; scenario_refit_summary = {}
                
                analysis_functions = {'Lasso': run_lasso_analysis, 'LARS': run_lars_analysis, 'OMP': run_omp_analysis}
                for method_base_name, analysis_func in analysis_functions.items():
                    results_dict = None; print("\n" + "="*30 + f" Running {method_base_name} Variants for {scenario_name} ({MODEL_CONFIG_NAME}) " + "="*30)
                    try:
                        if method_base_name == 'Lasso':
                            results_dict = analysis_func(X_weighted_scaled, target_vector_weighted, basis_names_for_analysis, CV_FOLDS, file_prefix, current_save_dir, random_seed=RANDOM_SEED)
                        else:
                            results_dict = analysis_func(X_weighted_scaled, target_vector_weighted, basis_names_for_analysis, CV_FOLDS, file_prefix, current_save_dir)
                        scenario_initial_results.update(results_dict)
                    except Exception as outer_e: print(f"FATAL ERROR during {method_base_name} Analysis block: {outer_e}"); traceback.print_exc();


                print("\n" + "="*40 + f" Post-processing All Models for {scenario_name} ({MODEL_CONFIG_NAME}) " + "="*40)

                process_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']

                for method_key in process_order:
                    if method_key in scenario_initial_results:
                        result_data = scenario_initial_results[method_key]
                        print(f"\n--- Post-processing Result for: {method_key} ---")
                        scenario_refit_summary[method_key] = {'consistent': None, 'notes': ['Post-processing start']} 

                        if 'error' not in result_data and 'coeffs' in result_data:
                            initial_coeffs_MPa = result_data['coeffs'] 
                            n_nonzero_initial = result_data.get('n_nonzero', np.sum(np.abs(initial_coeffs_MPa) > EPS))

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
                                         coeff_threshold=COEFF_THRESHOLD, ridge_alpha=RIDGE_ALPHA_REFIT
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
                                    print(f"ERROR caught during post-processing call for {method_key}: UNPACKING/VALUE ERROR - {ve}")
                                    scenario_refit_summary[method_key]['error_postprocess'] = f"Unpacking/Value Error: {ve}"
                                    scenario_refit_summary[method_key]['notes'] = [f"Unpacking/Value Error: {ve}"]
                                    traceback.print_exc() 
                                except Exception as plot_e: 
                                    print(f"ERROR caught during post-processing call for {method_key}: {plot_e}"); traceback.print_exc()
                                    scenario_refit_summary[method_key]['error_postprocess'] = str(plot_e)
                                    scenario_refit_summary[method_key]['notes'] = [f"Plotting/PostProc Error: {plot_e}"]
                            else:
                                scenario_refit_summary[method_key]['notes'] = ['No features initially selected']
                                scenario_refit_summary[method_key]['consistent'] = None
                        else:
                            err_msg = result_data.get('error', 'Unknown initial fit error')
                            scenario_refit_summary[method_key]['notes'] = [f"Initial fit error: {err_msg}"]
                            scenario_refit_summary[method_key]['consistent'] = None
                    else:
                        scenario_refit_summary[method_key] = {'consistent': None, 'notes': [f'{method_key} not run or no results']}

                all_initial_results[scenario_name] = scenario_initial_results
                all_refit_summaries[scenario_name] = scenario_refit_summary
                print(f"\n### Finished All Analyses & Post-processing for Scenario: {scenario_name} ###\n")

        # Print overall summaries & verification table
        print("\n" + "="*80); print("### OVERALL INITIAL SPARSE SELECTION SUMMARY ###"); print("="*80)
        for scenario_name_sum, methods_results_sum in all_initial_results.items():
            print(f"\n--- Scenario: {scenario_name_sum} ---")
            print("| Method      | Optimal Param    | Non-Zero | AIC       | BIC       | Time (s) |")
            print("|-------------|------------------|----------|-----------|-----------|----------|")
            process_order = ['LassoCV', 'LassoAIC', 'LassoBIC', 'LarsCV', 'LarsAIC', 'LarsBIC', 'OMPCV', 'OMPAIC', 'OMPBIC']
            for method_key_sum in process_order:
                if method_key_sum in methods_results_sum:
                    res = methods_results_sum[method_key_sum] ; method_print_key = method_key_sum
                    if 'error' in res: time_str = f"{res.get('time', np.nan):<8.2f}" if pd.notna(res.get('time', np.nan)) else "N/A     " ; error_msg_short = str(res.get('error', 'Unknown Error'))[:15] + "..." ; print(f"| {method_print_key:<11} | ERROR             | ---      | ---       | ---       | {time_str} | {error_msg_short}")
                    else: n_nonzero = res.get('n_nonzero', 'N/A'); aic_str = f"{res.get('aic', np.nan):<9.2e}" if pd.notna(res.get('aic', np.nan)) else 'N/A        '; bic_str = f"{res.get('bic', np.nan):<9.2e}" if pd.notna(res.get('bic', np.nan)) else 'N/A        '; time_str = f"{res.get('time', np.nan):<8.4f}" if pd.notna(res.get('time', np.nan)) else 'N/A     '; opt_param_details = [];
                    if 'CV' in method_key_sum: opt_param_details.append("CV")
                    if 'AIC' in method_key_sum: opt_param_details.append("AIC")
                    if 'BIC' in method_key_sum: opt_param_details.append("BIC")
                    if 'alpha' in res and pd.notna(res['alpha']): opt_param_details.append(f"α≈{res['alpha']:.2g}")
                    if 'optimal_k' in res and pd.notna(res['optimal_k']): opt_param_details.append(f"k={int(res['optimal_k'])}")
                    elif 'n_features_opt' in res and pd.notna(res['n_features_opt']): opt_param_details.append(f"k={int(res['n_features_opt'])}")
                    opt_param_str = " ".join(opt_param_details) if opt_param_details else "N/A"; print(f"| {method_print_key:<11} | {opt_param_str:<16} | {n_nonzero:<8} | {aic_str} | {bic_str} | {time_str} |")

        header_width = 170 
        print("\n" + "="*header_width); print(f"### OVERALL FINAL REFIT MODEL SUMMARY (Refit: {REFIT_REGRESSION_TYPE}, Thresh={COEFF_THRESHOLD:.1e}) ###".center(header_width)); print("### Metrics: UT(P11), PS(P11), EBT(P11) on UNWEIGHTED Data | Consistency Check ###".center(header_width))
        print(f"| {'Method':<11} | {'N Terms':<7} | {'R²(UT)':<8} | {'RMSE(UT)[kPa]':<13} | {'NRMSE(UT)':<9} | {'R²(PS11)':<9} | {'RMSE(PS11)[kPa]':<15} | {'NRMSE(PS11)':<11} | {'R²(EBT11)':<10} | {'RMSE(EBT11)[kPa]':<16} | {'NRMSE(EBT11)':<12} | {'Consist':<7} | {'Notes':<30} |")
        print(f"|{'-'*13}|{'-'*9}|{'-'*10}|{'-'*15}|{'-'*11}|{'-'*11}|{'-'*17}|{'-'*13}|{'-'*12}|{'-'*18}|{'-'*14}|{'-'*9}|{'-'*32}|")

        for scenario_name_sum, methods_summary_sum in all_refit_summaries.items(): 
            print(f"\n--- Scenario: {scenario_name_sum} ---")
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
                    print(f"| {method_key_sum:<11} | {n_terms:<7} | {r2_ut:<8} | {rmse_ut:<13} | {nrmse_ut:<9} | {r2_ps11:<9} | {rmse_ps11:<15} | {nrmse_ps11:<11} | {r2_ebt11:<10} | {rmse_ebt11:<16} | {nrmse_ebt11:<12} | {consistent:<7} | {notes_str:<30} |")

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
            print(f"\nLog output saved to {log_file_name}")
        else: 
            sys.stdout = original_stdout # Still ensure stdout is reset
            print(f"\nLogging to file failed. Outputting to console.")

        current_time_str = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        print(f"\nCompleted all scenarios at: {current_time_str}")
        if 'overall_start_time' in locals() and overall_start_time is not None:
            total_duration = datetime.datetime.now() - overall_start_time
            print(f"Total analysis duration: {total_duration}")
        print("="*170)
