# ==============================================
# Necessary Imports
# ==============================================
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from matplotlib.patches import Patch # For custom legend in heatmap
import matplotlib.ticker # Added for tick formatting

from numpy.linalg import eigh, LinAlgError # Keep eigh if used elsewhere, check needed? No, remove.
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import (
    LassoCV, lasso_path,
    LarsCV, lars_path,
    OrthogonalMatchingPursuitCV, orthogonal_mp,
    # RidgeCV, # Not used
    Ridge,
    LinearRegression # Not used directly, but good to have if needed later
)
from sklearn.metrics import mean_squared_error, r2_score # Added r2_score
import re # For parsing term strings
import datetime # To use current time
import os # For file/directory handling
from scipy.special import erf # For erf function
import warnings
from sklearn.exceptions import ConvergenceWarning
import traceback # For detailed error printing
from sklearn.model_selection import KFold

# Filter convergence warnings from sklearn
warnings.filterwarnings("ignore", category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn') # Ignore n_features > n_samples warnings in Lars

# ==============================================
# Global Plotting Configuration 
# ==============================================
PLOT_TITLE_FONTSIZE = 14
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 14
PLOT_LEGEND_FONTSIZE = 10
PLOT_LINEWIDTH = 1.5
PLOT_PATH_ALPHA = 0.7 # Transparency for path lines (Not used for coef paths now)
PLOT_MARKER_SIZE = 30 # For scatter plots in predictions
PLOT_PRED_LINEWIDTH = 1.5 # For prediction lines
PLOT_HEATMAP_YLABEL_SIZE = 9 # Specific size for potentially long feature lists

SMALL_FIG_LASSO_LARS_OMP = (3.0, 4.5)

# ==============================================
# Global Colormap Configuration
# ==============================================
try:
    plasma = plt.get_cmap('plasma')
    blues_cmap = plt.get_cmap('Blues')
except Exception:
    plasma = plt.cm.plasma
    blues_cmap = plt.cm.Blues

# Define the truncated plasma colormap that's used in plotting functions
TRUNCATED_PLASMA_CMAP = plasma

# Colormap for Heatmaps (Binary: Inactive/Active) - Aligned with HearthCANN
HEATMAP_CMAP = blues_cmap

# ==============================================
# Model configuration - Consolidated MR
# (Set these globally for the functions that read them)
# ==============================================
# Standard model components (Will be overridden in main loop)
INCLUDE_NEOHOOKEAN = True
INCLUDE_MOONEY_RIVLIN = True
INCLUDE_YEOH = False
INCLUDE_ARRUDA_BOYCE = False
INCLUDE_OGDEN = True
INCLUDE_GENT = False
INCLUDE_DATA_DRIVEN = False # Not used here

# Enhanced components for S-shaped response (Will be overridden in main loop)
INCLUDE_FRACTIONAL_POWERS = False
INCLUDE_LOW_STRAIN_TERMS = False
INCLUDE_EXTENDED_OGDEN = False # Often used with Ogden
INCLUDE_ULTRA_SMALL_STRAIN = False

# Maximum powers for various expansions
MR_MAX_POWER_I1 = 3
MR_MAX_POWER_I2 = 3
max_total_power = 3
YEOH_MAX_POWER = 5

# --- Fractional powers config --- (Not used in default loop)
STANDARD_FRACTIONAL_POWERS = [0.25, 0.5, 0.75, 1.5, 2.5, 3.5]
SMALL_STRAIN_FRACTIONAL_POWERS = [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
FRACTIONAL_POWERS = STANDARD_FRACTIONAL_POWERS # Default

# --- Ogden config ---
# Note: Will be updated based on INCLUDE_EXTENDED_OGDEN in generate_model_library
OGDEN_EXPONENTS = [] # Placeholder, will be generated

# --- Numerical stability & Weighting ---
EPS = 1e-10
SMALL_STRAIN_THRESHOLD = 50.0 # Threshold for weighting (e.g., strain percent)
WEIGHT_FACTOR = 1.0 # Factor to multiply weights for small strains

# ==============================================
# Invariant/Derivative/Low-Strain Helpers
# ==============================================
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

def sigmoid_function(x, scale=1.0, offset=0.0):
    x_scaled = np.clip(-(x - offset) * scale, -700, 700)
    return 1.0 / (1.0 + np.exp(x_scaled))

def sigmoid_derivative(x, scale=1.0, offset=0.0):
    sig = sigmoid_function(x, scale, offset)
    return scale * sig * (1.0 - sig)

def tanh_function(x, scale=1.0, offset=0.0):
    return np.tanh((x - offset) * scale)

def tanh_derivative(x, scale=1.0, offset=0.0):
    return scale * (1.0 - np.tanh((x - offset) * scale)**2)

def erf_function(x, scale=1.0, offset=0.0):
    return erf((x - offset) * scale)

def erf_derivative(x, scale=1.0, offset=0.0):
    return scale * (2.0 / np.sqrt(np.pi)) * np.exp(-((x - offset) * scale)**2)

def safe_power(base, exponent):
    base = np.maximum(base, 0)
    if isinstance(base, (np.ndarray)):
        result = np.zeros_like(base, dtype=float)
        mask_safe = base > EPS
        # Use np.power for element-wise operation
        if np.isscalar(exponent) or len(exponent) == 1:
            result[mask_safe] = np.power(base[mask_safe], exponent)
        else: 
             result[mask_safe] = np.power(base[mask_safe], exponent[mask_safe])

        # Careful with exponent type check if it could be array
        is_zero_exp = np.all(exponent == 0) if isinstance(exponent, np.ndarray) else (exponent == 0)
        is_neg_exp = np.all(exponent < 0) if isinstance(exponent, np.ndarray) else (exponent < 0)

        if is_zero_exp: result[base <= EPS] = 1.0
        elif is_neg_exp: result[base <= EPS] = 0.0 
        return result
    else: # Scalar case
        if base > EPS: return base ** exponent
        elif exponent > 0: return 0.0
        elif exponent == 0: return 1.0
        else: return 0.0 


def safe_log(x):
    if isinstance(x, (np.ndarray)):
        res = np.full_like(x, -700.0, dtype=float)
        mask = x > EPS
        res[mask] = np.log(x[mask])
        return res
    else: # Scalar case
        return np.log(x) if x > EPS else -700.0

# ==============================================
# Consolidated Model Library Generation
# ==============================================
def generate_model_library():
    """Generates a list of basis function strings based on configuration flags."""
    global OGDEN_EXPONENTS # Allow modification based on INCLUDE_EXTENDED_OGDEN
    if INCLUDE_OGDEN:
        if INCLUDE_EXTENDED_OGDEN:
            STANDARD_OGDEN_EXPONENTS_local = [-4, -3, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 3, 4]
            SMALL_STRAIN_OGDEN_EXPONENTS_local = [-0.1, -0.05, -0.01, 0.01, 0.05, 0.1]
            OGDEN_EXPONENTS = [-4, -3, -1, 1, 3, 4] #STANDARD_OGDEN_EXPONENTS_local + SMALL_STRAIN_OGDEN_EXPONENTS_local if INCLUDE_ULTRA_SMALL_STRAIN else STANDARD_OGDEN_EXPONENTS_local
        else:
            OGDEN_EXPONENTS = [-4, -3, -1, 1, 3, 4] #[-4, -3, -1, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1, 3, 4]
    else:
        OGDEN_EXPONENTS = []

    basis_functions = []
    seen_terms = set()
    def add_term(term_string):
        if term_string not in seen_terms:
            basis_functions.append(term_string)
            seen_terms.add(term_string)
    if INCLUDE_NEOHOOKEAN and not INCLUDE_MOONEY_RIVLIN: add_term("NH: (I₁-3)")
    if INCLUDE_MOONEY_RIVLIN:
        # max_total_power = MR_MAX_POWER_I1 + MR_MAX_POWER_I2
        for total_power in range(1, max_total_power + 1):
            for i in range(min(total_power + 1, MR_MAX_POWER_I1 + 1)):
                j = total_power - i
                if j > MR_MAX_POWER_I2 or j < 0: continue  # Skip if power of I₂ exceeds maximum or is negative
                term_i1 = f"(I₁-3)^{i}" if i > 1 else ("(I₁-3)" if i == 1 else "")
                term_i2 = f"(I₂-3)^{j}" if j > 1 else ("(I₂-3)" if j == 1 else "")
                term_str = f"MR: {term_i1}{term_i2}" if term_i1 and term_i2 else (f"MR: {term_i1}" if term_i1 else f"MR: {term_i2}")
                add_term(term_str)
    if INCLUDE_YEOH:
        for n in range(1, YEOH_MAX_POWER + 1):
            term_str = f"Yeoh: (I₁-3)^{n}" if n > 1 else "Yeoh: (I₁-3)"
            mr_equiv = f"MR: (I₁-3)^{n}" if n > 1 else "MR: (I₁-3)"
            if not (INCLUDE_MOONEY_RIVLIN and n <= MR_MAX_POWER_I1 and mr_equiv in seen_terms): add_term(term_str)
    if INCLUDE_FRACTIONAL_POWERS:
        current_frac_powers = STANDARD_FRACTIONAL_POWERS + SMALL_STRAIN_FRACTIONAL_POWERS if INCLUDE_ULTRA_SMALL_STRAIN else STANDARD_FRACTIONAL_POWERS
        for p in current_frac_powers:
             if p != 0: add_term(f"Frac: (I₁-3)^{p:.3f}")
    if INCLUDE_LOW_STRAIN_TERMS or INCLUDE_ULTRA_SMALL_STRAIN:
        low_terms = ["Low: 1-exp(-(I₁-3))", "Low: tanh(I₁-3)", "Low: (I₁-3)/(1+(I₁-3))", "Low: ln(1+(I₁-3))", "Low: sigmoid((I₁-3)*10)", "Low: sigmoid((I₁-3)*5)", "Low: sigmoid((I₁-3)*2)", "Low: sigmoid((I₁-3)*1)", "Low: tanh((I₁-3)*10)", "Low: tanh((I₁-3)*5)", "Low: tanh((I₁-3)*2)", "Low: erf((I₁-3)*5)", "Low: erf((I₁-3)*2)", "Low: erf((I₁-3)*1)"]
        ultra_terms = ["Ultra: arctan((I₁-3)*50)", "Ultra: arctan((I₁-3)*20)", "Ultra: arctan((I₁-3)*10)", "Ultra: (I₁-3)/(0.001+(I₁-3))", "Ultra: (I₁-3)/(0.01+(I₁-3))", "Ultra: (I₁-3)/(0.1+(I₁-3))", "Ultra: (I₁-3)^2*(1-(I₁-3))", "Ultra: (I₁-3)^3*(1-(I₁-3)^2)"]
        if INCLUDE_LOW_STRAIN_TERMS:
            for term in low_terms: add_term(term)
        if INCLUDE_ULTRA_SMALL_STRAIN:
            for term in ultra_terms: add_term(term)
    if INCLUDE_OGDEN:
        for a in OGDEN_EXPONENTS:
             if a != 0: add_term(f"Ogden: λ^{a}")
    if INCLUDE_GENT:
        add_term("Gent: -ln(1-(I₁-3)/100)")
        if INCLUDE_ULTRA_SMALL_STRAIN: add_term("Gent: -ln(1-(I₁-3)/10)"); add_term("Gent: -ln(1-(I₁-3)/500)")
    if INCLUDE_ARRUDA_BOYCE:
        for n in range(1, 6):
            term_str = f"AB_Inspired: (I₁-3)^{n}" if n > 1 else "AB_Inspired: (I₁-3)"
            mr_equiv = f"MR: (I₁-3)^{n}" if n > 1 else "MR: (I₁-3)"
            yeoh_equiv = f"Yeoh: (I₁-3)^{n}" if n > 1 else "Yeoh: (I₁-3)"
            is_mr_dup = INCLUDE_MOONEY_RIVLIN and n <= MR_MAX_POWER_I1 and mr_equiv in seen_terms
            is_yeoh_dup = INCLUDE_YEOH and n <= YEOH_MAX_POWER and yeoh_equiv in seen_terms
            if not (is_mr_dup or is_yeoh_dup): add_term(term_str)
    if INCLUDE_DATA_DRIVEN: pass
    print(f"Generated {len(basis_functions)} unique basis terms for this scenario.")
    return basis_functions

# ==============================================
# Construct derivatives matrix (dW/dλᵢ)
# ==============================================
def construct_derivatives_matrix(df):
    num_samples = len(df)
    basis_names = generate_model_library()
    num_basis = len(basis_names)
    dW_dl1 = np.zeros((num_samples, num_basis)); dW_dl2 = np.zeros((num_samples, num_basis)); dW_dl3 = np.zeros((num_samples, num_basis))
    i1_power_pattern = re.compile(r'\(I₁-3\)(?:\^([\d\.-]+))?'); i2_power_pattern = re.compile(r'\(I₂-3\)(?:\^([\d\.-]+))?')
    ogden_power_pattern = re.compile(r'λ\^([\-\+\d\.]*\d)'); gent_jm_pattern = re.compile(r'/([\d\.]+)\)')
    frac_power_pattern = re.compile(r'\^(-?\d+\.?\d*)'); sigmoid_scale_pattern = re.compile(r'sigmoid\(\(I₁-3\)\*([\d\.]+)\)')
    tanh_scale_pattern = re.compile(r'tanh\(\(I₁-3\)\*([\d\.]+)\)'); erf_scale_pattern = re.compile(r'erf\(\(I₁-3\)\*([\d\.]+)\)')
    arctan_scale_pattern = re.compile(r'arctan\(\(I₁-3\)\*([\d\.]+)\)'); rational_denom_pattern = re.compile(r'\(I₁-3\)\/\(([\d\.]+)\+\(I₁-3\)\)')
    for idx in range(num_samples):
        row = df.iloc[idx]; l1, l2 = max(row['lambda1'], EPS), max(row['lambda2'], EPS)
        l3_inv_prod = l1 * l2; l3 = max(1.0 / l3_inv_prod if l3_inv_prod > EPS else 1.0 / EPS, EPS)
        I1, I2 = compute_invariants(l1, l2, l3)
        (dI1dl1, dI1dl2, dI1dl3), (dI2dl1, dI2dl2, dI2dl3) = compute_invariant_derivatives(l1, l2, l3)
        i1m3 = I1 - 3.0; i2m3 = I2 - 3.0; i1m3_safe = max(i1m3, 0.0); i2m3_safe = max(i2m3, 0.0)
        for basis_idx, term_name in enumerate(basis_names):
            dW_dI1_term = 0.0; dW_dI2_term = 0.0; dWterm_dl1 = 0.0; dWterm_dl2 = 0.0; dWterm_dl3 = 0.0
            if term_name.startswith("MR:"):
                math_part = term_name.split(":", 1)[1].strip(); i_match = i1_power_pattern.search(math_part); j_match = i2_power_pattern.search(math_part)
                i = float(i_match.group(1)) if i_match and i_match.group(1) else (1 if i_match else 0)
                j = float(j_match.group(1)) if j_match and j_match.group(1) else (1 if j_match else 0)
                if i > 0: dW_dI1_term = i * safe_power(i1m3_safe, i - 1.0) * safe_power(i2m3_safe, j)
                if j > 0: dW_dI2_term = safe_power(i1m3_safe, i) * j * safe_power(i2m3_safe, j - 1.0)
            elif term_name.startswith(("NH:", "Yeoh:", "AB_Inspired:")):
                match = i1_power_pattern.search(term_name); n = float(match.group(1)) if match and match.group(1) else (1 if match else 0)
                if n > 0: dW_dI1_term = n * safe_power(i1m3_safe, n - 1.0)
            elif term_name.startswith("Frac:"):
                match = frac_power_pattern.search(term_name); p = float(match.group(1)) if match else 0.0
                if p != 0: dW_dI1_term = p * safe_power(i1m3_safe, p - 1.0)
            elif term_name.startswith(("Low:", "Ultra:")):
                dTerm_di1m3 = 0.0
                if "1-exp(-(I₁-3))" in term_name: dTerm_di1m3 = np.exp(-max(i1m3, -700))
                elif "tanh(" in term_name: scale_match = tanh_scale_pattern.search(term_name); scale = float(scale_match.group(1)) if scale_match else 1.0; dTerm_di1m3 = tanh_derivative(i1m3, scale=scale)
                elif "(I₁-3)/(1+(I₁-3))" in term_name: denom = 1.0 + i1m3; dTerm_di1m3 = 1.0 / safe_power(max(denom, EPS), 2.0)
                elif "ln(1+(I₁-3))" in term_name: denom = 1.0 + i1m3; dTerm_di1m3 = 1.0 / max(denom, EPS)
                elif "sigmoid(" in term_name: scale_match = sigmoid_scale_pattern.search(term_name); scale = float(scale_match.group(1)) if scale_match else 1.0; dTerm_di1m3 = sigmoid_derivative(i1m3, scale=scale)
                elif "erf(" in term_name: scale_match = erf_scale_pattern.search(term_name); scale = float(scale_match.group(1)) if scale_match else 1.0; dTerm_di1m3 = erf_derivative(i1m3, scale=scale)
                elif "arctan(" in term_name: scale_match = arctan_scale_pattern.search(term_name); scale = float(scale_match.group(1)) if scale_match else 1.0; dTerm_di1m3 = scale / (1.0 + (scale * i1m3)**2)
                elif term_name.startswith("Ultra: (I₁-3)/("): match = rational_denom_pattern.search(term_name); d_val = float(match.group(1)) if match else 0.01; denom = d_val + i1m3; dTerm_di1m3 = d_val / safe_power(max(denom, EPS), 2.0)
                elif "Ultra: (I₁-3)^2*(1-(I₁-3))" in term_name: dTerm_di1m3 = 2.0*i1m3 - 3.0*safe_power(i1m3_safe, 2.0)
                elif "Ultra: (I₁-3)^3*(1-(I₁-3)^2)" in term_name: dTerm_di1m3 = 3.0*safe_power(i1m3_safe, 2.0) - 5.0*safe_power(i1m3_safe, 4.0)
                else: print(f"Warning: Derivative logic missing for Low/Ultra term '{term_name}'")
                dW_dI1_term = dTerm_di1m3
            elif term_name.startswith("Ogden:"):
                match = ogden_power_pattern.search(term_name); alpha = float(match.group(1)) if match else 0.0
                if alpha != 0.0:
                    dWterm_dl1 = alpha * safe_power(l1, alpha - 1.0); dWterm_dl2 = alpha * safe_power(l2, alpha - 1.0); dWterm_dl3 = alpha * safe_power(l3, alpha - 1.0)
                    dW_dI1_term = 0.0; dW_dI2_term = 0.0
            elif term_name.startswith("Gent:"):
                match = gent_jm_pattern.search(term_name); Jm = float(match.group(1)) if match else 100.0
                denom = Jm - i1m3; dW_dI1_term = min(1.0 / max(denom, EPS), 1e12)
            elif term_name.startswith("DD:"): pass
            if not term_name.startswith("Ogden:"):
                dWterm_dl1 = dW_dI1_term * dI1dl1 + dW_dI2_term * dI2dl1; dWterm_dl2 = dW_dI1_term * dI1dl2 + dW_dI2_term * dI2dl2; dWterm_dl3 = dW_dI1_term * dI1dl3 + dW_dI2_term * dI2dl3
            dW_dl1[idx, basis_idx] = dWterm_dl1; dW_dl2[idx, basis_idx] = dWterm_dl2; dW_dl3[idx, basis_idx] = dWterm_dl3
    dW_dl1 = np.nan_to_num(dW_dl1, nan=0.0, posinf=1e12, neginf=-1e12); dW_dl2 = np.nan_to_num(dW_dl2, nan=0.0, posinf=1e12, neginf=-1e12); dW_dl3 = np.nan_to_num(dW_dl3, nan=0.0, posinf=1e12, neginf=-1e12)
    return dW_dl1, dW_dl2, dW_dl3, basis_names

# ==============================================
# Weighting Function
# ==============================================
def weighting_function(strain_pct, emphasize_small_strain=True, weight_factor=WEIGHT_FACTOR, small_strain_threshold=SMALL_STRAIN_THRESHOLD):
    """Applies weights to data points, potentially emphasizing small strains."""
    weights = np.ones_like(strain_pct, dtype=float)
    if not emphasize_small_strain:
        print("INFO: Weighting disabled.")
        return weights
    small_strain_mask = strain_pct <= small_strain_threshold
    num_weighted = np.sum(small_strain_mask)
    if num_weighted > 0:
        weights[small_strain_mask] = weight_factor
        print(f"INFO: Applied weight factor {weight_factor} to {num_weighted} points <= {small_strain_threshold}% strain.")
    else:
        print("INFO: Weighting enabled, but no points met the small strain threshold.")
    return weights

# ==============================================
# Construct design matrix (Φ) and target vector (y)
# (Includes weighting) - PK1 STRESS Formulation
# ==============================================
def construct_design_matrix(df, use_small_strain_only=False, emphasize_small_strain=True, weight_factor=WEIGHT_FACTOR, small_strain_threshold=SMALL_STRAIN_THRESHOLD):
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
        print(f"Using small strain data subset: {len(df_subset)} points.")
        if len(df_subset) < 5:
            print(f"WARN: Not enough small strain points ({len(df_subset)}). Using full dataset instead.")
            df_subset = df.copy()
    else:
        df_subset = df.copy()

    df_subset = df_subset.reset_index(drop=True) # Ensure index is sequential for mapping later

    # Calculate weights based on the subset being used
    point_weights = weighting_function(df_subset['strain_pct'].values, emphasize_small_strain, weight_factor, small_strain_threshold)

    dW1_mat, dW2_mat, dW3_mat, basis_names = construct_derivatives_matrix(df_subset)

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
        l1, l2 = max(row_data['lambda1'], EPS), max(row_data['lambda2'], EPS)
        l3_inv_prod = l1 * l2
        l3 = max(1.0 / l3_inv_prod if l3_inv_prod > EPS else 1.0 / EPS, EPS)
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
                l3_over_l1 = l3 / l1 if l1 > EPS else l3 / EPS
                phi_row_p11 = dW1_vec - l3_over_l1 * dW3_vec
                # Apply weight to BOTH phi_row and target
                design_matrix[current_row_idx, :] = p_weight * phi_row_p11
                target_vector[current_row_idx] = p_weight * row_data['P11']
                row_weights_vector[current_row_idx] = p_weight # Store the weight used for this row
                current_row_idx += 1
                processed_indices.add(idx)
            else:
                print(f"WARN: Row index {current_row_idx} exceeds expected total {total_rows} adding P11 idx {idx}.")

        if has_p22:
            if current_row_idx < total_rows:
                l3_over_l2 = l3 / l2 if l2 > EPS else l3 / EPS
                phi_row_p22 = dW2_vec - l3_over_l2 * dW3_vec
                # Apply weight to BOTH phi_row and target
                design_matrix[current_row_idx, :] = p_weight * phi_row_p22
                target_vector[current_row_idx] = p_weight * row_data['P22']
                row_weights_vector[current_row_idx] = p_weight # Store the weight used for this row
                current_row_idx += 1
                processed_indices.add(idx)
            else:
                print(f"WARN: Row index {current_row_idx} exceeds expected total {total_rows} adding P22 idx {idx}.")

        end_row_for_idx = current_row_idx # Store ending row (exclusive)
        if end_row_for_idx > start_row_for_idx: # If rows were added for this index
             original_indices_map.append({
                 'original_df_index': df_subset.index[idx], # Get original index BEFORE reset_index
                 'matrix_row_start': start_row_for_idx,
                 'matrix_row_end': end_row_for_idx
             })


    if current_row_idx != total_rows:
        print(f"WARN: Final row index ({current_row_idx}) != Expected total ({total_rows}). Resizing arrays.")
        design_matrix = design_matrix[:current_row_idx, :]
        target_vector = target_vector[:current_row_idx]
        row_weights_vector = row_weights_vector[:current_row_idx]

    # Check for non-finite values (can happen with extreme derivatives/powers)
    finite_mask = np.all(np.isfinite(design_matrix), axis=1) & np.isfinite(target_vector)
    if not np.all(finite_mask):
        num_removed = np.sum(~finite_mask)
        print(f"WARN: Removing {num_removed} rows from design matrix/target due to non-finite values.")
        design_matrix = design_matrix[finite_mask, :]
        target_vector = target_vector[finite_mask]
        row_weights_vector = row_weights_vector[finite_mask]

    if design_matrix.shape[0] == 0:
        print("ERROR: Design matrix empty after filtering.")
        return np.zeros((0, n_basis)), np.zeros((0,)), basis_names, np.zeros((0,)), df_subset

    print(f"Constructed design matrix ({'Weighted' if emphasize_small_strain else 'Unweighted'}): {design_matrix.shape}")
    # Pass df_subset back, as it corresponds to the rows used (before filtering non-finite)
    # The mapping back to original df needs care in the plotting function.
    return design_matrix, target_vector, basis_names, row_weights_vector, df_subset

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


# ==============================================
# Plotting Functions for Sparse Regression
# ==============================================

# --- Heatmap Plotters ---
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
):
    """
    Generalized function to plot feature activation heatmaps with multiple selection criteria.
    Handles Lasso (vs log alpha) and LARS/OMP (vs step).
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

    # --- Mark the chosen values ---
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

            label_prefix = 'k=' if is_omp else 'α=' if log_x else ''
            vline_label = f'{method}' #f'Selected {method} ({label_prefix}{label_val_str})'
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=vline_label))
        else:
            print(f"WARN: Chosen value for {method} is None or NaN, cannot plot vertical line.")

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

# ==============================================
# Physical Consistency Check Helper 
# ==============================================
def check_physical_consistency(final_terms, final_coeffs):
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
        if c10_coeff <= EPS:
            notes.append("    WARN: C10 component is non-positive.")

    if mr_c01_term in term_coeff_map:
        notes.append(f"  MR C01 Term ('{mr_c01_term}') Coeff (C01) = {c01_coeff:.4f}")
        if c01_coeff < -EPS:
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

                if abs(alpha_i) < EPS:
                    mu_i = 0.0
                    notes.append(f"  WARN: Ogden term '{term}' has alpha near zero. mu_i set to 0.")
                else:
                    mu_i = coeff * alpha_i

                mu_alpha_product = mu_i * alpha_i
                ogden_contrib_sum_mu_alpha += mu_alpha_product # Sum (mu_i * alpha_i)

                ogden_coeffs_details.append({'coeff_Ci': coeff, 'mu_i': mu_i, 'alpha': alpha_i, 'mu_alpha': mu_alpha_product})
                notes.append(f"  Ogden Term: Coeff (C_i={coeff:.4f}), α={alpha_i:.4f} => Implied μ_i={mu_i:.4f}. (μ_i*α_i = {mu_alpha_product:.4f})")
                ogden_term_found = True
            except ValueError:
                notes.append(f"  WARN: Could not parse Ogden exponent in '{term}'.")
            except Exception as e:
                 notes.append(f"  ERROR processing Ogden term '{term}': {e}")

    # --- Calculate Final Shear Modulus using correct formula ---
    # mu_0 = 2 * (C10 + C01) + 0.5 * sum(mu_i * alpha_i)
    shear_modulus = 2.0 * mr_contrib_sum + 0.5 * ogden_contrib_sum_mu_alpha

    # --- Final Shear Modulus Check ---
    notes.insert(0, f"Calculated initial shear modulus μ₀ ≈ {shear_modulus:.4f} (using μ₀ = 2(C10+C01) + 0.5*Σ(μᵢαᵢ))")
    if ogden_term_found:
        notes.append(f"  MR contribution (2*(C10+C01)) = {2.0 * mr_contrib_sum:.4f}")
        notes.append(f"  Ogden contribution (0.5*Σ(μᵢαᵢ)) = {0.5 * ogden_contrib_sum_mu_alpha:.4f}")
        notes.append("  NOTE: Ogden consistency assumes discovered 'coeff' C_i relates to standard μ_i via μ_i = C_i * α_i.")


    is_consistent = True
    if shear_modulus <= EPS: # Use EPS for safety margin around zero
        notes.append("  CONSISTENCY FAIL: Initial shear modulus μ₀ is non-positive.")
        is_consistent = False
    else:
        notes.append("  CONSISTENCY PASS: Initial shear modulus μ₀ is positive.")


    return is_consistent, shear_modulus, notes

# Define coefficient threshold globally or pass as argument
COEFF_THRESHOLD = 1e-6 # Example threshold

def plot_model_predictions(
    df_fit_scope,        # Data used ONLY for the refit step
    df_plot_structure,   # Full data structure for prediction calculation & plotting
    sparse_coeffs,       # Coefficients from initial sparse fit (on df_fit_scope)
    basis_names,         # Full list of basis names from initial fit
    method_key,          # Name reflecting selection (e.g., "LassoCV")
    scenario_prefix,     # Includes fitting scenario info
    save_dir,
    weight_factor=WEIGHT_FACTOR,
    small_strain_threshold=SMALL_STRAIN_THRESHOLD,
    coeff_threshold=COEFF_THRESHOLD,
    ridge_alpha=1e-6
):
    """
    Performs Ridge refit *only* on df_fit_scope data.
    Calculates predictions and metrics across the full df_plot_structure.
    Plots performance across df_plot_structure (e.g., Treloar & Kawabata).
    Returns 8 values: final_terms, final_coeffs, R2/RMSE metrics (on full structure), consistency info.
    """
    print(f"\n--- Generating Final Model & Predictions ({method_key} / Ridge Refit on FIT SUBSET / Evaluate on FULL SET) ---")
    print(f"    Refit Scope: {len(df_fit_scope)} points. Evaluation Scope: {len(df_plot_structure)} points.")

    # --- Step 1: Identify Initial Selected Features (from sparse fit on subset) ---
    selected_indices = np.where(np.abs(sparse_coeffs) > EPS)[0]
    if len(selected_indices) == 0:
        print(f"WARN: No features initially selected by {method_key}. Skipping evaluation.")
        return [], [], np.nan, np.nan, np.nan, np.nan, False, ["No features selected in initial fit"]
    initial_selected_basis_names = [basis_names[i] for i in selected_indices]
    print(f"  Initial selection ({method_key}): {len(initial_selected_basis_names)} features.")

    # --- Step 2: Reconstruct WEIGHTED Design Matrix & Target for Refit (using FIT SUBSET) ---
    print(f"  Reconstructing WEIGHTED design matrix for refit using FIT SUBSET ({len(df_fit_scope)} points)...")
    phi_weighted_fit, target_weighted_fit = None, None
    try:
        phi_weighted_fit, target_weighted_fit, basis_names_refit, _, _ = construct_design_matrix(
            df_fit_scope,
            emphasize_small_strain=True,
            weight_factor=weight_factor,
            small_strain_threshold=small_strain_threshold
        )
        if phi_weighted_fit.shape[0] == 0: raise ValueError("Weighted design matrix for refit empty.")
        if phi_weighted_fit.shape[0] != len(target_weighted_fit): raise ValueError("Weighted refit matrix/target row mismatch.")
        if basis_names != basis_names_refit: print("WARN: Basis name mismatch during weighted refit matrix recon.")

        phi_selected_weighted_fit = phi_weighted_fit[:, selected_indices]

    except Exception as e:
        print(f"ERROR: Failed to reconstruct WEIGHTED matrix/target for refit using fit subset: {e}")
        traceback.print_exc()
        return [], [], np.nan, np.nan, np.nan, np.nan, False, [f"Refit Matrix Recon Error: {e}"]

    # --- Step 3: Refit using Ridge Regression (on FIT SUBSET data) ---
    print(f"  Refitting {len(initial_selected_basis_names)} features on WEIGHTED FIT SUBSET data (shape: {phi_selected_weighted_fit.shape}) using Ridge...")
    final_coeffs = np.zeros(len(initial_selected_basis_names))
    try:
        ridge_refit = Ridge(alpha=ridge_alpha, fit_intercept=False, solver='auto')
        # ridge_refit = Ridge(alpha=ridge_alpha, fit_intercept=False, solver='lbfgs', positive=True)
        ridge_refit.fit(phi_selected_weighted_fit, target_weighted_fit)
        refit_coeffs_weighted = ridge_refit.coef_
    except Exception as e:
        print(f"ERROR: Failed during Ridge Regression refit for {method_key} on fit subset: {e}"); traceback.print_exc()
        return [], [], np.nan, np.nan, np.nan, np.nan, False, [f"Ridge Refit Error: {e}"]

    # --- Step 4: Apply Thresholding (to coefficients from refit on subset) ---
    print(f"  Applying threshold ({coeff_threshold:.1e}) to Ridge coefficients...")
    significant_mask = np.abs(refit_coeffs_weighted) > coeff_threshold
    final_coeffs = refit_coeffs_weighted[significant_mask]
    final_global_indices = selected_indices[significant_mask]
    final_selected_basis_names = [basis_names[i] for i in final_global_indices]
    n_final = len(final_selected_basis_names)
    print(f"  Final selected features after thresholding: {n_final}")

    # --- Step 4.5: Check Physical Consistency ---
    is_consistent = False; shear_modulus = np.nan; consistency_notes = ["Thresholding removed all terms"]
    if n_final == 0:
        print(f"WARN: All features removed by thresholding for {method_key}.")
        is_consistent, shear_modulus, consistency_notes = check_physical_consistency([], [])
    else:
        print(f"  Final Model ({method_key} - based on fit subset refit):"); [print(f"    '{name}': {coeff:.6f}") for name, coeff in zip(final_selected_basis_names, final_coeffs)]
        print("  Checking physical consistency..."); is_consistent, shear_modulus, consistency_notes = check_physical_consistency(final_selected_basis_names, final_coeffs)
        print("  Consistency Check Results:"); [print(f"    {note}") for note in consistency_notes]
        if not is_consistent: print(f"  WARNING: {method_key} model potentially inconsistent (μ₀={shear_modulus:.4f}).")
        else: print(f"  INFO: {method_key} model passed basic consistency check (μ₀={shear_modulus:.4f}).")

    # --- Step 5: Reconstruct UNWEIGHTED Design Matrix for FULL EVALUATION SET ---
    print(f"\n  Reconstructing UNWEIGHTED design matrix for FULL EVALUATION set ({len(df_plot_structure)} points)...")
    phi_unweighted_full, target_unweighted_full = None, None
    row_to_plot_idx_map = None
    try:
        dW1_mat_uw_full, dW2_mat_uw_full, dW3_mat_uw_full, basis_names_unweighted = construct_derivatives_matrix(df_plot_structure)
        if basis_names != basis_names_unweighted: raise ValueError("Basis name mismatch!")

        n_samp_full = df_plot_structure.shape[0]; n_basis = len(basis_names)
        n_p11_rows_full = df_plot_structure[pd.notna(df_plot_structure['P11'])].shape[0]
        n_p22_rows_full = df_plot_structure[df_plot_structure['mode'].isin(['EBT', 'BIAX']) & pd.notna(df_plot_structure['P22'])].shape[0]
        total_rows_expected_full = n_p11_rows_full + n_p22_rows_full

        phi_unweighted_full = np.zeros((total_rows_expected_full, n_basis))
        target_unweighted_full = np.zeros(total_rows_expected_full)
        row_to_plot_idx_map = np.full(total_rows_expected_full, -1, dtype=int)

        print(f"    Building unweighted evaluation matrix with expected shape: ({total_rows_expected_full}, {n_basis})")
        current_row_idx = 0
        for idx_in_plot_df in range(n_samp_full):
            row_data = df_plot_structure.iloc[idx_in_plot_df]; l1, l2 = max(row_data['lambda1'], EPS), max(row_data['lambda2'], EPS)
            l3_inv_prod = l1 * l2; l3 = max(1.0 / l3_inv_prod if l3_inv_prod > EPS else 1.0 / EPS, EPS); mode = row_data['mode']
            dW1_vec_uw = dW1_mat_uw_full[idx_in_plot_df, :]; dW2_vec_uw = dW2_mat_uw_full[idx_in_plot_df, :]; dW3_vec_uw = dW3_mat_uw_full[idx_in_plot_df, :]
            has_p11 = pd.notna(row_data['P11']); has_p22 = pd.notna(row_data['P22']) and mode in ['EBT', 'BIAX']
            if has_p11:
                if current_row_idx < total_rows_expected_full: l3_over_l1 = l3 / l1 if l1 > EPS else l3 / EPS; phi_unweighted_full[current_row_idx, :] = dW1_vec_uw - l3_over_l1 * dW3_vec_uw; target_unweighted_full[current_row_idx] = row_data['P11']; row_to_plot_idx_map[current_row_idx] = idx_in_plot_df; current_row_idx += 1
            if has_p22:
                if current_row_idx < total_rows_expected_full: l3_over_l2 = l3 / l2 if l2 > EPS else l3 / EPS; phi_unweighted_full[current_row_idx, :] = dW2_vec_uw - l3_over_l2 * dW3_vec_uw; target_unweighted_full[current_row_idx] = row_data['P22']; row_to_plot_idx_map[current_row_idx] = idx_in_plot_df; current_row_idx += 1

        if current_row_idx != total_rows_expected_full:
            print(f"WARN (Unweighted Full Recon): Row count mismatch ({current_row_idx} vs {total_rows_expected_full}). Resizing.")
            phi_unweighted_full = phi_unweighted_full[:current_row_idx, :]; target_unweighted_full = target_unweighted_full[:current_row_idx]; row_to_plot_idx_map = row_to_plot_idx_map[:current_row_idx]

        print("    Filtering non-finite values from full unweighted matrix/target...")
        finite_mask_uw_full = np.all(np.isfinite(phi_unweighted_full), axis=1) & np.isfinite(target_unweighted_full)
        num_finite_uw_full = np.sum(finite_mask_uw_full)
        if num_finite_uw_full < phi_unweighted_full.shape[0]:
            num_removed = phi_unweighted_full.shape[0] - num_finite_uw_full; print(f"    WARN: Removing {num_removed} non-finite rows from unweighted full matrix/target.")
            phi_unweighted_full = phi_unweighted_full[finite_mask_uw_full, :]; target_unweighted_full = target_unweighted_full[finite_mask_uw_full]; row_to_plot_idx_map = row_to_plot_idx_map[finite_mask_uw_full]

        if phi_unweighted_full.shape[0] == 0:
            raise ValueError("Unweighted full matrix empty after filtering.")
        print(f"    Final unweighted evaluation matrix shape after filtering: {phi_unweighted_full.shape}")

    except Exception as e:
        print(f"ERROR: Failed to reconstruct UNWEIGHTED matrix/target for full evaluation set: {e}")
        traceback.print_exc()
        return final_selected_basis_names, final_coeffs, np.nan, np.nan, np.nan, np.nan, is_consistent, consistency_notes

    # --- Step 6: Calculate UNWEIGHTED Predictions across FULL Evaluation Set ---
    pred_unweighted_full = np.zeros(phi_unweighted_full.shape[0])
    if n_final > 0:
        phi_final_selected_unweighted_full = phi_unweighted_full[:, final_global_indices]
        print(f"  Calculating predictions on FULL evaluation set using matrix shape {phi_final_selected_unweighted_full.shape} and {len(final_coeffs)} coeffs (from subset refit)...")
        try:
            if phi_final_selected_unweighted_full.shape[1] != len(final_coeffs): raise ValueError(f"Matrix columns ({phi_final_selected_unweighted_full.shape[1]}) != Coeffs ({len(final_coeffs)})")
            pred_unweighted_full = phi_final_selected_unweighted_full @ final_coeffs
            if len(pred_unweighted_full) != len(target_unweighted_full): raise ValueError(f"Prediction length ({len(pred_unweighted_full)}) != Target length ({len(target_unweighted_full)})")
        except Exception as e: print(f"ERROR predicting on full set: {e}"); traceback.print_exc(); return final_selected_basis_names, final_coeffs, np.nan, np.nan, np.nan, np.nan, is_consistent, consistency_notes
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
            mapping_errors += 1; print(f"    ERROR mapping prediction {i} to df index {original_df_idx}: {map_e}")

    if successfully_mapped != len(pred_unweighted_full):
        print(f"WARN: Mapping count mismatch. Expected {len(pred_unweighted_full)}, mapped {successfully_mapped}. Errors: {mapping_errors}")
    else:
        print(f"  Successfully mapped {successfully_mapped} predictions onto the full dataset structure.")

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

    print(f"  Overall Metrics on Full Set: P11 R²={r2_P11_overall:.4f}, RMSE={rmse_P11_overall:.4f} | P22 R²={r2_P22_overall:.4f}, RMSE={rmse_P22_overall:.4f}")

    mode_specific_metrics = {}
    available_modes = df_plot['mode'].unique()
    print(f"  Calculating mode-specific metrics for modes: {available_modes}")
    
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
                print(f"    {mode} Mode P11: R²={r2_p11_mode:.4f}, RMSE={rmse_p11_mode:.4f} (n={n_points_p11})")
            if n_points_p22 > 0:
                print(f"    {mode} Mode P22: R²={r2_p22_mode:.4f}, RMSE={rmse_p22_mode:.4f} (n={n_points_p22})")
            if n_points_p11 == 0 and n_points_p22 == 0:
                print(f"    {mode} Mode: No valid data points for evaluation")

    # --- Step 9: Create Plots (using FULL df_plot structure) ---
    print("  Preparing plots using full dataset structure...")
    consistency_status_str = "(Phys OK)" if is_consistent else "(Phys WARN!)"
    try: global REFIT_REGRESSION_TYPE
    except NameError: REFIT_REGRESSION_TYPE = "Ridge Refit"
    title_suffix = f'Final Model ({method_key} / Refit on Fit Subset Thresh={coeff_threshold:.1e}) {consistency_status_str}'

    fig = None; plot_created = False
    datasets_in_plot_structure = df_plot['dataset'].unique()
    print(f"  DEBUG: Datasets found in plotting structure: {datasets_in_plot_structure}")
    treloar_present = 'Treloar' in datasets_in_plot_structure
    kawabata_present = 'Kawabata' in datasets_in_plot_structure
    print(f"  DEBUG: Plotting Treloar: {treloar_present}, Plotting Kawabata: {kawabata_present}")

    if treloar_present and kawabata_present:
        print(f"  DEBUG: Entering COMBINED plot block for {method_key}...")
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
            
            plot_treloar_prediction(treloar_data, axes=ax_treloar, r2_overall_precomputed=r2_treloar_p11)
        except Exception as e: print(f"  ERROR plotting Treloar: {e}"); traceback.print_exc()
        try: plot_Kawabata_prediction(kawabata_data, axes=ax_kawabata, r2_p11_precomputed=r2_kawabata_p11, r2_p22_precomputed=r2_kawabata_p22)
        except Exception as e: print(f"  ERROR plotting Kawabata: {e}"); traceback.print_exc()
        fig.suptitle(f'{title_suffix} vs. Exp. Data', fontsize=PLOT_TITLE_FONTSIZE+1);
        try: plt.tight_layout(rect=[0, 0, 1, 0.96])
        except Exception as e: print(f"  WARN: tight_layout failed: {e}")
        plot_created = True
    elif treloar_present:
        print(f"  DEBUG: Entering TRELOAR ONLY plot block for {method_key}..."); fig, axes = plt.subplots(1, 2, figsize=(8, 5));
        try: 
            treloar_data = df_plot[df_plot['dataset'] == 'Treloar']
            treloar_p11_valid = pd.notna(treloar_data['P11']) & pd.notna(treloar_data['P11_pred'])
            r2_treloar_p11 = r2_score(treloar_data['P11'][treloar_p11_valid], treloar_data['P11_pred'][treloar_p11_valid]) if treloar_p11_valid.sum() > 0 else np.nan
            plot_treloar_prediction(treloar_data, axes=axes, r2_overall_precomputed=r2_treloar_p11); fig.suptitle(f'{title_suffix} vs. Treloar Data', fontsize=PLOT_TITLE_FONTSIZE+1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plot_created = True;
        except Exception as e: print(f"  ERROR plotting Treloar: {e}"); traceback.print_exc(); plt.close(fig); fig=None
    elif kawabata_present:
        print(f"  DEBUG: Entering KAWABATA ONLY plot block for {method_key}..."); fig, axes = plt.subplots(1, 2, figsize=(8, 5));
        try: 
            kawabata_data = df_plot[df_plot['dataset'] == 'Kawabata']
            kawabata_p11_valid = pd.notna(kawabata_data['P11']) & pd.notna(kawabata_data['P11_pred'])
            kawabata_p22_valid = pd.notna(kawabata_data['P22']) & pd.notna(kawabata_data['P22_pred']) & kawabata_data['mode'].isin(['EBT', 'BIAX'])
            r2_kawabata_p11 = r2_score(kawabata_data['P11'][kawabata_p11_valid], kawabata_data['P11_pred'][kawabata_p11_valid]) if kawabata_p11_valid.sum() > 0 else np.nan
            r2_kawabata_p22 = r2_score(kawabata_data['P22'][kawabata_p22_valid], kawabata_data['P22_pred'][kawabata_p22_valid]) if kawabata_p22_valid.sum() > 0 else np.nan
            plot_Kawabata_prediction(kawabata_data, axes=axes, r2_p11_precomputed=r2_kawabata_p11, r2_p22_precomputed=r2_kawabata_p22); fig.suptitle(f'{title_suffix} vs. Kawabata Data', fontsize=PLOT_TITLE_FONTSIZE+1); plt.tight_layout(rect=[0, 0, 1, 0.95]); plot_created = True;
        except Exception as e: print(f"  ERROR plotting Kawabata: {e}"); traceback.print_exc(); plt.close(fig); fig=None
    else:
        print(f"No recognizable datasets ('Treloar', 'Kawabata') found in df_plot_structure for {method_key}. Skipping prediction plot.")

    # --- Step 10: Save Plot ---
    if save_dir and scenario_prefix and fig and plot_created:
        save_path = os.path.join(save_dir, f"{scenario_prefix}_{method_key}_RefitSubset_EvalFull_Predictions.pdf")
        try:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            fig.savefig(save_path, format='pdf', bbox_inches='tight')
            print(f"Saved final prediction plot: {save_path}")
        except Exception as e:
            print(f"Error saving final prediction plot {save_path}: {e}")
    elif fig:
        print(f"WARN: Plot not saved for {method_key} (save conditions not met or plot_created={plot_created}).")
    if fig:
        plt.close(fig)

        # --- Step 11: Return Final Model Details & EVALUATION METRICS ---
    return (final_selected_basis_names, final_coeffs, r2_P11_overall, rmse_P11_overall, 
            r2_P22_overall, rmse_P22_overall, is_consistent, consistency_notes, mode_specific_metrics)

def plot_treloar_prediction(df, axes, r2_overall_precomputed=np.nan):
    """Plot Treloar data with model predictions (Helper). Needs axes=[ax_full, ax_zoom]. Uses precomputed R2 for title."""
    ax_full, ax_zoom = axes
    colors = {'UT': TRUNCATED_PLASMA_CMAP(0.2), 'PS': TRUNCATED_PLASMA_CMAP(0.5), 'EBT': TRUNCATED_PLASMA_CMAP(0.8)}
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
                sc = ax.scatter(lambda1_sorted, exp_sorted, color=colors[mode], marker=markers[mode], s=PLOT_MARKER_SIZE, label=f"{mode} (Exp. data)" if i==0 else None, alpha=0.7, edgecolor='black', linewidth=0.5)
                pl, = ax.plot(lambda1_sorted, pred_sorted, color=colors[mode], linestyle='-', linewidth=PLOT_PRED_LINEWIDTH, label=f"{mode} (Model)" if i==0 else None)
                if i == 0: legend_handles.extend([sc, pl])
                if i == 1:
                    mask_zoom_mode = (lambda1_sorted >= 1.0) & (lambda1_sorted <= 4.5)
                    if mask_zoom_mode.sum() > 0:
                         max_y_zoom_overall = max(max_y_zoom_overall, np.nanmax(exp_sorted[mask_zoom_mode]), np.nanmax(pred_sorted[mask_zoom_mode]))
                         min_y_zoom_overall = min(min_y_zoom_overall, np.nanmin(exp_sorted[mask_zoom_mode]), np.nanmin(pred_sorted[mask_zoom_mode]))

        ax.set_xlabel('Stretch λ', fontsize=PLOT_LABEL_FONTSIZE)
        ax.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)

        if i == 0:
            title_r2_str = f"{r2_overall:.3f}" if pd.notna(r2_overall) else "N/A"
            ax.set_title(f'Treloar (Full Range, $R^2 \\approx {title_r2_str}$)', fontsize=PLOT_TITLE_FONTSIZE)
            ax.legend(handles=legend_handles, loc='best', fontsize=PLOT_LEGEND_FONTSIZE)
            ax.set_xlim(left=1.0, right=max(4.5, max_lambda_full * 1.05))
            ax.set_ylim(bottom=min(0, min_stress_overall - 0.2), top=max(0.2, max_stress_overall * 1.1))
        else:
            ax.set_title(f'Treloar (Zoomed)', fontsize=PLOT_TITLE_FONTSIZE)
            ax.set_xlim(1.0, 5.5)
            if not np.isfinite(min_y_zoom_overall) or not np.isfinite(max_y_zoom_overall): 
                ax.set_ylim(0, 1)
            else: 
                ax.set_ylim(bottom=min(0, min_y_zoom_overall - 0.1), top=max(0.2, max_y_zoom_overall * 1.1))
            
def plot_Kawabata_prediction(df, axes, r2_p11_precomputed=np.nan, r2_p22_precomputed=np.nan):
    """Plot Kawabata data with model predictions (Helper). Uses precomputed R2 values for titles."""
    ax_p11, ax_p22 = axes; df_plot = df[df['dataset']=='Kawabata'].copy()
    df_plot['P11'] = pd.to_numeric(df_plot['P11'], errors='coerce'); df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')
    df_plot['P22'] = pd.to_numeric(df_plot['P22'], errors='coerce'); df_plot['P22_pred'] = pd.to_numeric(df_plot['P22_pred'], errors='coerce')
    lambda1_values = sorted(df_plot['lambda1'].unique()); cmap = TRUNCATED_PLASMA_CMAP; colors = cmap(np.linspace(0.1, 0.9, len(lambda1_values)))

    r2_P11 = r2_p11_precomputed
    r2_P22 = r2_p22_precomputed

    valid_p11_plot = ~np.isnan(df_plot['P11']) & ~np.isnan(df_plot['P11_pred'])
    valid_p22_plot = ~np.isnan(df_plot['P22']) & ~np.isnan(df_plot['P22_pred'])
    
    handles_p11 = []; max_y_p11 = 0
    for i, lambda1 in enumerate(lambda1_values):
        subset = df_plot[df_plot['lambda1'] == lambda1].sort_values(by='lambda2')
        v_p11 = valid_p11_plot[subset.index]
        if v_p11.sum() == 0: continue
        sc = ax_p11.scatter(subset['lambda2'][v_p11], subset['P11'][v_p11], color=colors[i], s=PLOT_MARKER_SIZE, alpha=0.7, marker='o', edgecolor='k', linewidth=0.5)
        pl, = ax_p11.plot(subset['lambda2'][v_p11], subset['P11_pred'][v_p11], color=colors[i], linestyle='-', linewidth=PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}')
        if i == 0: 
            handles_p11.extend([Patch(color='gray', alpha=0.7, label='Exp'), Patch(color='black', label='Model')])
        handles_p11.append(pl)
        if v_p11.sum() > 0:
            max_y_p11 = max(max_y_p11, np.nanmax(subset['P11'][v_p11]), np.nanmax(subset['P11_pred'][v_p11]))

    ax_p11.set_xlabel('Stretch λ₂', fontsize=PLOT_LABEL_FONTSIZE); ax_p11.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=PLOT_LABEL_FONTSIZE)
    title_r2_p11_str = f"{r2_P11:.3f}" if pd.notna(r2_P11) else "N/A"
    ax_p11.set_title(f'Kawabata P₁₁ ($R^2 \\approx {title_r2_p11_str}$)', fontsize=PLOT_TITLE_FONTSIZE)
    ax_p11.legend(handles=handles_p11[:10], fontsize=PLOT_LEGEND_FONTSIZE*0.8, loc='best', ncol=2, frameon=False)
    ax_p11.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
    ax_p11.set_ylim(bottom= min(0, np.nanmin(df_plot['P11'][valid_p11_plot])-0.1) if valid_p11_plot.sum() > 0 else 0, top=max(0.2, max_y_p11*1.1))

    handles_p22 = []; max_y_p22 = 0
    for i, lambda1 in enumerate(lambda1_values):
        subset = df_plot[df_plot['lambda1'] == lambda1].sort_values(by='lambda2')
        v_p22 = valid_p22_plot[subset.index]
        if v_p22.sum() == 0: continue
        sc = ax_p22.scatter(subset['lambda2'][v_p22], subset['P22'][v_p22], color=colors[i], s=PLOT_MARKER_SIZE, alpha=0.7, marker='s', edgecolor='k', linewidth=0.5)
        pl, = ax_p22.plot(subset['lambda2'][v_p22], subset['P22_pred'][v_p22], color=colors[i], linestyle='--', linewidth=PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}')
        if i == 0: 
            handles_p22.extend([Patch(color='gray', alpha=0.7, label='Exp'), Patch(color='black', linestyle='--', label='Model')])
        handles_p22.append(pl)
        if v_p22.sum() > 0:
            max_y_p22 = max(max_y_p22, np.nanmax(subset['P22'][v_p22]), np.nanmax(subset['P22_pred'][v_p22]))

    ax_p22.set_xlabel('Stretch λ₂', fontsize=PLOT_LABEL_FONTSIZE); ax_p22.set_ylabel('Nominal Stress P₂₂ (MPa)', fontsize=PLOT_LABEL_FONTSIZE)
    title_r2_p22_str = f"{r2_P22:.3f}" if pd.notna(r2_P22) else "N/A"
    ax_p22.set_title(f'Kawabata P₂₂ ($R^2 \\approx {title_r2_p22_str}$)', fontsize=PLOT_TITLE_FONTSIZE)
    ax_p22.legend(handles=handles_p22[:10], fontsize=PLOT_LEGEND_FONTSIZE*0.8, loc='best', ncol=2, frameon=False)
    ax_p22.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
    ax_p22.set_ylim(bottom= min(0, np.nanmin(df_plot['P22'][valid_p22_plot])-0.1) if valid_p22_plot.sum() > 0 else 0, top=max(0.2, max_y_p22*1.1))
    
# ==============================================
# Data Plotting Helper
# ==============================================
def plot_combined_datasets(df, save_dir=None, filename_prefix=None):
    """Plot the raw data from all included datasets"""
    datasets = df['dataset'].unique()
    n_datasets = len(datasets)
    if n_datasets == 0: print("No data to plot."); return
    fig, axes = plt.subplots(1, n_datasets, figsize=(6 * n_datasets, 5), squeeze=False)
    for i, dataset in enumerate(datasets):
        ax = axes[0, i]; ds_data = df[df['dataset'] == dataset].copy()
        if dataset == 'Treloar': plot_treloar_data(ds_data, ax=ax)
        elif dataset == 'Kawabata': plot_Kawabata_data(ds_data, ax=ax)
        else: print(f"Warning: Unknown dataset type '{dataset}' for plotting."); ax.text(0.5, 0.5, f"Data for\n{dataset}", ha='center', va='center'); ax.set_title(f"{dataset} Data")
    plt.tight_layout()
    if save_dir and filename_prefix:
        save_path = os.path.join(save_dir, f"{filename_prefix}_RawData.pdf")
        try: os.makedirs(save_dir, exist_ok=True); fig.savefig(save_path, format='pdf', bbox_inches='tight'); print(f"Saved raw data plot: {save_path}")
        except Exception as e: print(f"Error saving raw data plot {save_path}: {e}")
    plt.close(fig)

def plot_treloar_data(df, ax=None):
    """Plot Treloar dataset - Helper for raw data plot"""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'UT': 'blue', 'PS': 'green', 'EBT': 'red'}; markers = {'UT': 'o', 'PS': 's', 'EBT': '^'}
    for mode in ['UT', 'PS', 'EBT']:
        mode_data = df[df['mode'] == mode]; p11_data = pd.to_numeric(mode_data['P11'], errors='coerce'); valid_mask = ~np.isnan(p11_data)
        if valid_mask.sum() > 0:
            strain = (mode_data['lambda1'][valid_mask] - 1) * 100; stress = p11_data[valid_mask]; sort_idx = np.argsort(strain)
            ax.scatter(strain.iloc[sort_idx], stress.iloc[sort_idx], color=colors[mode], marker=markers[mode], label=mode, alpha=0.8, s=PLOT_MARKER_SIZE, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Strain (%)', fontsize=PLOT_LABEL_FONTSIZE); ax.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_title('Treloar Data', fontsize=PLOT_TITLE_FONTSIZE); ax.legend(fontsize=PLOT_LEGEND_FONTSIZE); ax.grid(alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
    return ax

def plot_Kawabata_data(df, ax=None):
    """Plot Kawabata biaxial dataset - Helper for raw data plot"""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    lambda1_values = sorted(df['lambda1'].unique()); cmap = TRUNCATED_PLASMA_CMAP; colors = cmap(np.linspace(0.1, 0.9, len(lambda1_values)))
    handles = [] # For legend
    for i, lambda1 in enumerate(lambda1_values):
        subset = df[df['lambda1'] == lambda1].copy(); subset['P11'] = pd.to_numeric(subset['P11'], errors='coerce'); subset['P22'] = pd.to_numeric(subset['P22'], errors='coerce')
        subset = subset.sort_values(by='lambda2'); valid_p11 = ~np.isnan(subset['P11']); valid_p22 = ~np.isnan(subset['P22'])
        line_p11, = ax.plot(subset['lambda2'][valid_p11], subset['P11'][valid_p11], marker='o', markersize=4, color=colors[i], linestyle='-', linewidth=PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}') if valid_p11.sum() > 0 else (None,)
        line_p22, = ax.plot(subset['lambda2'][valid_p22], subset['P22'][valid_p22], marker='s', markersize=4, color=colors[i], linestyle='--', linewidth=PLOT_PRED_LINEWIDTH) if valid_p22.sum() > 0 else (None,)
        if i==0: handles.extend([Patch(color='gray', label='P₁₁'), Patch(color='gray', linestyle='--', label='P₂₂')]) # General labels
        if line_p11: handles.append(line_p11) # Add specific lambda line handle
    ax.set_xlabel('λ₂', fontsize=PLOT_LABEL_FONTSIZE); ax.set_ylabel('Nominal Stress (MPa)', fontsize=PLOT_LABEL_FONTSIZE)
    ax.set_title('Kawabata Biaxial Data', fontsize=PLOT_TITLE_FONTSIZE); ax.legend(handles=handles[:12], fontsize=PLOT_LEGEND_FONTSIZE*0.9); ax.grid(alpha=0.3); ax.tick_params(axis='both', which='major', labelsize=PLOT_TICK_FONTSIZE)
    return ax

def run_lasso_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None):
    """
    Performs LassoCV (coordinate descent) and Lasso path AIC/BIC selection with non-negative coefficients.
    - CV curve uses LassoCV(positive=True)
    - AIC/BIC path uses lasso_path(positive=True)
    - Coefficients are clipped to >= 0 for robustness
    Plots a single heatmap and criteria plots. Measures time.
    Returns a dictionary of results for 'LassoCV', 'LassoAIC', 'LassoBIC'.
    Updated to match HearthCANN/synthetic CV-NMSE plotting (robustly normalized to [0,1]).
    """
    print(f"\n--- Running LASSO Analysis (CV, AIC, BIC) ({title_prefix}) ---")
    n_samples, n_features = X_scaled.shape
    base_results = { 'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
                     'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan,
                     'full_basis_names': feature_names }
    results = { 'LassoCV': {**base_results, 'method': 'LassoCV'},
                'LassoAIC': {**base_results, 'method': 'LassoAIC'},
                'LassoBIC': {**base_results, 'method': 'LassoBIC'} }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # --- 1. LassoCV on a controlled alpha range (non-negative) ---
    print("  Running LassoCV to determine optimal alpha...")
    start_time_cv = datetime.datetime.now()
    best_alpha_from_cv = np.nan

    alpha_min_log = -6
    alpha_max_log = 0
    alphas_cv_grid = np.logspace(alpha_min_log, alpha_max_log, 1000)  # dense for CV

    # For Row-0 CV curve
    alphas_cv_plot = np.array([])
    cv_nmse_mean_plot = np.array([])
    cv_nmse_std_plot  = np.array([])

    try:
        lasso_cv = LassoCV(
            cv=cv_folds,
            alphas=alphas_cv_grid,
            max_iter=5000,
            n_jobs=-1,
            random_state=random_seed,
            precompute='auto',     # keep your environment setting
            positive=True          # <<< enforce β >= 0 in CV
        )
        lasso_cv.fit(X_scaled, y)
        best_alpha_from_cv = lasso_cv.alpha_
        print(f"  LassoCV determined optimal alpha: {best_alpha_from_cv:.6f}")

        # ---- Extract CV MSE and robust-normalize NMSE to [0,1] for plotting ----
        try:
            alphas_cv_full = np.asarray(lasso_cv.alphas_)      # (n_alphas,)
            mse_path_cv = np.asarray(lasso_cv.mse_path_)       # (n_alphas, n_folds[, n_targets])

            if mse_path_cv.ndim == 3:
                cv_mse_mean_full = np.nanmean(mse_path_cv, axis=(1, 2))
                cv_mse_std_full  = np.nanstd( mse_path_cv, axis=(1, 2))
            elif mse_path_cv.ndim == 2:
                cv_mse_mean_full = np.nanmean(mse_path_cv, axis=1)
                cv_mse_std_full  = np.nanstd( mse_path_cv, axis=1)
            else:
                cv_mse_mean_full = mse_path_cv.ravel()
                cv_mse_std_full  = np.zeros_like(cv_mse_mean_full)

            # Convert to statistical NMSE (for semantics), then robustly map to [0,1]
            y_centered = y - np.mean(y)
            y_var_cv = float(np.var(y_centered))
            if y_var_cv > EPS:
                nmse_mean_full = cv_mse_mean_full / y_var_cv
                nmse_std_full  = cv_mse_std_full  / y_var_cv
            else:
                nmse_mean_full = cv_mse_mean_full
                nmse_std_full  = cv_mse_std_full

            # Robust min–max via percentiles (protect from outliers)
            p_lo = float(np.nanpercentile(nmse_mean_full, 5))
            p_hi = float(np.nanpercentile(nmse_mean_full, 95))
            den  = max(p_hi - p_lo, EPS)

            nmse_mean_scaled = (nmse_mean_full - p_lo) / den
            nmse_std_scaled  = nmse_std_full / den  # scale only (std is not centered)

            # Clip to strict [0,1] display range
            nmse_mean_scaled = np.clip(nmse_mean_scaled, 0.0, 1.0)
            nmse_std_scaled  = np.clip(nmse_std_scaled,  0.0, 1.0)

            # Subsample to 20 evenly spaced points to declutter
            n_plot = min(20, alphas_cv_full.size)
            if n_plot > 0:
                sample_idx = np.unique(np.linspace(0, alphas_cv_full.size - 1, n_plot, dtype=int))
                alphas_cv_plot     = alphas_cv_full[sample_idx]
                cv_nmse_mean_plot  = nmse_mean_scaled[sample_idx]
                cv_nmse_std_plot   = nmse_std_scaled[sample_idx]

        except Exception as e:
            print(f"WARN: Unable to extract/normalize LassoCV CV curve: {e}")

    except Exception as e:
        print(f"ERROR during LassoCV: {e}")
        results['LassoCV'].update({'error': str(e)})

    # --- 2. Unified Lasso Path (AIC/BIC on same domain; non-negative) ---
    print("\n  Generating a unified Lasso Path for all criteria...")
    start_time_path = datetime.datetime.now()
    df_path = pd.DataFrame()
    alphas_lasso, coefs_lasso = None, None
    try:
        alphas_base = np.logspace(alpha_min_log, alpha_max_log, 20)
        alphas_for_path = np.unique(np.concatenate(
            [alphas_base, [best_alpha_from_cv] if pd.notna(best_alpha_from_cv) else []]
        ))
        alphas_for_path = np.sort(alphas_for_path)[::-1]  # descending

        alphas_lasso, coefs_lasso, _ = lasso_path(
            X_scaled, y, alphas=alphas_for_path, max_iter=5000, precompute='auto', positive=True  # <<< enforce β >= 0
        )
        # Numerical guard: clip tiny negatives that can appear due to tolerances
        coefs_lasso = np.maximum(coefs_lasso, 0.0)

        path_results_list = []
        for i in range(coefs_lasso.shape[1]):
            alpha_i = alphas_lasso[i]
            beta_i  = np.maximum(coefs_lasso[:, i], 0.0)  # ensure non-negative
            n_params = int(np.sum(beta_i > EPS))
            if n_params == 0 and i < coefs_lasso.shape[1] - 1:
                continue

            y_pred_train = X_scaled @ beta_i
            mse = mean_squared_error(y, y_pred_train)
            aic, bic = calculate_aic_bic(y, y_pred_train, n_params, n_samples)
            path_results_list.append({
                'alpha': alpha_i, 'k': n_params, 'coeffs': beta_i,
                'mse_train': mse, 'aic': aic, 'bic': bic
            })

        df_path = pd.DataFrame(path_results_list)
        print(f"  Unified path created with {len(df_path)} unique models.")
    except Exception as e:
        print(f"ERROR during unified path generation: {e}")
        traceback.print_exc()
        results['LassoCV'].update({'error': str(e)})
        results['LassoAIC'].update({'error': str(e)})
        results['LassoBIC'].update({'error': str(e)})
        return results

    # --- 2b. Normalized metrics on path (unchanged semantics) ---
    y_var = np.var(y)
    if y_var > EPS:
        df_path['norm_mse'] = df_path['mse_train'] / y_var
    else:
        df_path['norm_mse'] = df_path['mse_train']

    print(f"  DEBUG: NMSE range: {df_path['norm_mse'].min():.3f} to {df_path['norm_mse'].max():.3f}")
    print(f"  DEBUG: y variance: {y_var:.3f}")

    df_path['l1_norm'] = df_path['coeffs'].apply(lambda x: np.sum(np.abs(x)))
    max_l1_norm = df_path['l1_norm'].max()
    df_path['norm_l1'] = (df_path['l1_norm'] / max_l1_norm) if max_l1_norm > EPS else 0.0

    # --- 3. Pick models from unified path (unchanged; positivity respected in coeffs) ---
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    if not df_path.empty and pd.notna(best_alpha_from_cv):
        cv_model_row = df_path.iloc[(df_path['alpha'] - best_alpha_from_cv).abs().argsort()[:1]].iloc[0]
        results['LassoCV'].update({
            'coeffs': cv_model_row['coeffs'], 'alpha': cv_model_row['alpha'], 'n_nonzero': cv_model_row['k'],
            'aic': cv_model_row['aic'], 'bic': cv_model_row['bic'],
            'selected_features': [feature_names[i] for i, c in enumerate(cv_model_row['coeffs']) if c > EPS],
            'time': duration_cv
        })
        print(f"  LassoCV model (k={cv_model_row['k']}) identified on unified path in {duration_cv:.2f}s.")

    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()
    if not df_path.empty:
        aic_model_row = df_path.loc[df_path['aic'].idxmin()]
        results['LassoAIC'].update({
            'coeffs': aic_model_row['coeffs'], 'alpha': aic_model_row['alpha'], 'n_nonzero': aic_model_row['k'],
            'aic': aic_model_row['aic'], 'bic': aic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(aic_model_row['coeffs']) if c > EPS],
            'time': duration_path
        })
        print(f"  LassoAIC model (k={aic_model_row['k']}) identified on unified path.")

        bic_model_row = df_path.loc[df_path['bic'].idxmin()]
        results['LassoBIC'].update({
            'coeffs': bic_model_row['coeffs'], 'alpha': bic_model_row['alpha'], 'n_nonzero': bic_model_row['k'],
            'aic': bic_model_row['aic'], 'bic': bic_model_row['bic'],
            'selected_features': [feature_names[j] for j, c in enumerate(bic_model_row['coeffs']) if c > EPS],
            'time': duration_path
        })
        print(f"  LassoBIC model (k={bic_model_row['k']}) identified on unified path.")

    # --- 4. Plots (unchanged layout; Row 0 uses robustly normalized CV NMSE) ---
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
                    'color': plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': find_alpha_index_in_path(aic_exact_model['alpha'], alphas_lasso) if aic_exact_model is not None else None,
                    'label_val': aic_exact_model['k'] if aic_exact_model is not None else None,
                    'color': plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': find_alpha_index_in_path(bic_exact_model['alpha'], alphas_lasso) if bic_exact_model is not None else None,
                    'label_val': bic_exact_model['k'] if bic_exact_model is not None else None,
                    'color': plasma(0.85), 'linestyle': '-.'}
        }

        plot_activation_heatmap(
            alphas_lasso, np.maximum(coefs_lasso, 0.0), feature_names, chosen_values_heatmap,  # pass non-neg
            title="Lasso Activation Path", x_axis_label=r'Regularization Strength',
            filename=os.path.join(save_dir, f"{title_prefix}_Lasso_ActivationHeatmap.pdf"),
            log_x=True, reverse_x=True, is_omp=False
        )

        df_plot = df_path.sort_values('alpha', ascending=False)
        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        alpha_for_cv  = cv_exact_model['alpha']  if cv_exact_model  is not None else None
        alpha_for_aic = aic_exact_model['alpha'] if aic_exact_model is not None else None
        alpha_for_bic = bic_exact_model['alpha'] if aic_exact_model is not None else None

        # Row 0: CV NMSE (mean ± std) vs alpha (robustly normalized to [0,1])
        ax0_twin = axs[0].twinx()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0 and np.isfinite(cv_nmse_mean_plot).any():
            axs[0].plot(alphas_cv_plot, cv_nmse_mean_plot, 'o-', ms=3, color=plasma(0.25))
            if isinstance(cv_nmse_std_plot, np.ndarray) and cv_nmse_std_plot.size == cv_nmse_mean_plot.size:
                try:
                    axs[0].fill_between(alphas_cv_plot,
                                        np.clip(cv_nmse_mean_plot - cv_nmse_std_plot, 0.0, 1.0),
                                        np.clip(cv_nmse_mean_plot + cv_nmse_std_plot, 0.0, 1.0),
                                        color=plasma(0.25), alpha=0.15, linewidth=0)
                except Exception:
                    pass
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes, ha='center', va='center')

        ax0_twin.plot(df_plot['alpha'], df_plot['norm_l1'], 's-', ms=3, color=plasma(0.65), alpha=0.6)
        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE, color=plasma(0.25), labelpad=-20)
        ax0_twin.set_ylabel('Norm. L1', fontsize=PLOT_LABEL_FONTSIZE, color=plasma(0.65), labelpad=-10)

        if pd.notna(alpha_for_cv):
            axs[0].axvline(alpha_for_cv, color=plasma(0.25), ls='--', lw=1.5)
            axs[0].text(alpha_for_cv * 1.5, 0.5, f"$\\lambda_L={alpha_for_cv:.1e}$",
                        rotation=90, va='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.25))

        axs[0].set_ylim(-0.05, 1.05); ax0_twin.set_ylim(-0.05, 1.05)
        axs[0].set_yticks([0.0, 1.0]); ax0_twin.set_yticks([0.0, 1.0])
        ax0_twin.tick_params(axis='y', labelcolor=plasma(0.65))
        axs[0].tick_params(axis='y', labelcolor=plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_plot['alpha'], df_plot['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_aic):
            axs[1].axvline(alpha_for_aic, color=plasma(0.55), ls='-', lw=1.5)
            axs[1].text(alpha_for_aic * 1.5, np.mean(axs[1].get_ylim()), f"$\\lambda_L={alpha_for_aic:.1e}$",
                        rotation=90, va='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.55))
        axs[1].set_yticks([df_plot['aic'].min(), df_plot['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_plot['alpha'], df_plot['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_bic):
            axs[2].axvline(alpha_for_bic, color=plasma(0.85), ls='-.', lw=1.5)
            axs[2].text(alpha_for_bic * 1.5, np.mean(axs[2].get_ylim()), f"$\\lambda_L={alpha_for_bic:.1e}$",
                        rotation=90, va='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.85))
        axs[2].set_yticks([df_plot['bic'].min(), df_plot['bic'].max()])

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

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_Lasso_Criteria.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)

    return results

def run_lars_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None):
    """
    LARS with a SINGLE shared path (supports) used for CV, AIC, and BIC.
    Enforces non-negative coefficients (physical consistency) both on the path and on refits.
    """
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
        print("ERROR: No features."); return results

    # ---------- helpers: NNLS with intercept via centering ----------
    def _pgd_nnls(A, b, max_iter=2000, tol=1e-8):
        # Simple projected-gradient NNLS fallback: minimize 0.5||Aβ-b||^2 s.t. β>=0
        A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float)
        if A.size == 0:
            return np.zeros((A.shape[1] if A.ndim==2 else 0,), dtype=float)
        # Lipschitz constant (spectral norm squared)
        try:
            L = np.linalg.norm(A, 2) ** 2
        except Exception:
            L = np.linalg.norm(A, ord=None) ** 2
        L = max(L, EPS)
        beta = np.zeros(A.shape[1], dtype=float)
        for _ in range(max_iter):
            grad = A.T @ (A @ beta - b)
            beta_new = beta - grad / L
            # projection
            beta_new = np.maximum(beta_new, 0.0)
            if np.linalg.norm(beta_new - beta) <= tol * (np.linalg.norm(beta) + tol):
                beta = beta_new
                break
            beta = beta_new
        return beta

    def _nnls_refit_with_intercept(Xmat, yvec):
        """
        Solve: min_{μ, β>=0} || y - μ - X β ||^2
        by centering X, y on the fitting set, solve NNLS for β on centered data,
        then μ = mean(y) - mean(X,0)·β. Returns (beta, intercept).
        """
        Xmat = np.asarray(Xmat, dtype=float)
        yvec = np.asarray(yvec, dtype=float).ravel()
        if Xmat.size == 0 or Xmat.shape[1] == 0:
            return np.zeros(0, dtype=float), float(np.mean(yvec))
        xmean = Xmat.mean(axis=0)
        ymean = float(np.mean(yvec))
        Xc = Xmat - xmean
        yc = yvec - ymean

        # Try scipy.nnls, fallback to PGD NNLS
        beta = None
        try:
            from scipy.optimize import nnls as _scipy_nnls
            beta, _ = _scipy_nnls(Xc, yc)
        except Exception:
            beta = _pgd_nnls(Xc, yc)

        # Numerical guard: clip tiny negatives
        beta = np.maximum(beta, 0.0)
        mu = ymean - float(xmean @ beta)
        return beta, mu

    # ---------- 1) LARS path on centered y, with non-negative coefficients ----------
    y_centered = y - np.mean(y)
    # positive=True enforces non-negative coefficients along the LARS path
    _, _, coefs_raw = lars_path(X_scaled, y_centered, method='lars', positive=True)
    if coefs_raw.shape[1] < 2:
        print("ERROR: LARS path too short."); return results
    coefs_all = coefs_raw[:, 1:]  # remove all-zero start
    T = coefs_all.shape[1]

    # Unique-k compression (keep first occurrence of each support size)
    supports, kept_idx = [], []
    seen_k = set()
    for t in range(T):
        b_t = coefs_all[:, t]
        # numerical guard to keep path strictly non-negative
        b_t = np.maximum(b_t, 0.0)
        S_t = np.flatnonzero(b_t > EPS)
        k_t = int(S_t.size)
        if k_t == 0 or k_t in seen_k:
            continue
        seen_k.add(k_t)
        supports.append(S_t)
        kept_idx.append(t)
    coefs_path = np.maximum(coefs_all[:, kept_idx], 0.0)  # (p, n_steps), clipped to >=0
    steps = np.arange(1, len(supports) + 1)               # 1..K

    # ---------- 2) CV on the SAME supports (NNLS refit with intercept) ----------
    print("  Cross-validating on unified LARS supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_mse = np.zeros(len(supports))
    cv_mse_sq = np.zeros(len(supports))

    for tr, va in kf.split(X_scaled):
        Xtr, Xva = X_scaled[tr], X_scaled[va]
        ytr, yva = y[tr], y[va]
        ytr_mean = float(np.mean(ytr))  # used for S=∅ case

        for i, S in enumerate(supports):
            if len(S) == 0:
                # intercept-only model: μ = mean(y_tr)
                yhat = np.full_like(yva, fill_value=ytr_mean)
                mse = float(np.mean((yva - yhat) ** 2))
            else:
                beta_i, mu_i = _nnls_refit_with_intercept(Xtr[:, S], ytr)
                yhat = mu_i + Xva[:, S] @ beta_i
                mse = float(np.mean((yva - yhat) ** 2))
            cv_mse[i] += mse
            cv_mse_sq[i] += mse ** 2

    cv_mse_mean = cv_mse / cv_folds
    cv_mse_std  = np.sqrt(np.maximum(cv_mse_sq / cv_folds - cv_mse_mean ** 2, 0.0))

    # Normalize to [0,1] for plotting (min–max on mean; std scaled by same range)
    mm, MM = float(np.min(cv_mse_mean)), float(np.max(cv_mse_mean))
    rng = max(MM - mm, EPS)
    cv_nmse_mean = (cv_mse_mean - mm) / rng
    cv_nmse_std  = cv_mse_std / rng
    best_idx = int(np.argmin(cv_mse_mean))
    best_step = int(steps[best_idx])

    # ---------- 3) AIC/BIC on FULL data (NNLS refit with intercept) ----------
    rows = []
    for i, S in enumerate(supports):
        if len(S) == 0:
            mu_full = float(np.mean(y))
            y_pred = np.full_like(y, fill_value=mu_full)
            k_eff = 1  # intercept only
            coeffs_full = np.zeros(n_features)
        else:
            beta_i, mu_full = _nnls_refit_with_intercept(X_scaled[:, S], y)
            y_pred = mu_full + X_scaled[:, S] @ beta_i
            k_eff = len(S) + 1  # β (nonneg) + intercept
            coeffs_full = np.zeros(n_features); coeffs_full[S] = beta_i
        aic, bic = calculate_aic_bic(y, y_pred, n_params=k_eff, n_samples=n_samples)
        rows.append({
            'step': int(steps[i]),
            'k': len(S),
            'coeffs': coeffs_full,
            'aic': aic,
            'bic': bic
        })
    df_path = pd.DataFrame(rows)

    # ---------- 4) Best rows & results ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    cv_row  = df_path.iloc[best_idx]
    aic_row = df_path.iloc[df_path['aic'].idxmin()]
    bic_row = df_path.iloc[df_path['bic'].idxmin()]

    results['LarsCV'].update({
        'coeffs': cv_row['coeffs'],
        'optimal_k': cv_row['step'],
        'n_nonzero': int(np.sum(cv_row['coeffs'] > EPS)),
        'aic': cv_row['aic'],
        'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > EPS],
        'time': duration_cv
    })
    results['LarsAIC'].update({
    'coeffs': aic_row['coeffs'], 'optimal_k': aic_row['step'],
    'n_nonzero': int(np.sum(aic_row['coeffs'] > EPS)),
    'aic': aic_row['aic'], 'bic': aic_row['bic'],
    'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > EPS],
    'time': duration_cv
    })
    results['LarsBIC'].update({
        'coeffs': bic_row['coeffs'], 'optimal_k': bic_row['step'],
        'n_nonzero': int(np.sum(bic_row['coeffs'] > EPS)),
        'aic': bic_row['aic'], 'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > EPS],
        'time': duration_cv
    })

    # ---------- 5) Plots (unchanged; now path is non-negative and refits are NNLS) ----------
    try:
        chosen_heat = {
            'CV':  {'index': int(cv_row['step']) - 1,  'label_val': int(cv_row['step']),  'color': plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': int(aic_row['step']) - 1, 'label_val': int(aic_row['step']), 'color': plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': int(bic_row['step']) - 1, 'label_val': int(bic_row['step']), 'color': plasma(0.85), 'linestyle': '-.'},
        }
        plot_activation_heatmap(
            steps, coefs_path, feature_names, chosen_heat,
            title="LARS Activation Path", x_axis_label='LARS Iteration',
            filename=os.path.join(save_dir, f"{title_prefix}_Lars_ActivationHeatmap.pdf"),
            log_x=False, reverse_x=False, is_omp=False
        )

        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std)
        axs[0].plot(steps, cv_nmse_mean, 'o-', ms=3, color=plasma(0.25))
        axs[0].fill_between(steps,
                            np.maximum(0, cv_nmse_mean - cv_nmse_std),
                            np.minimum(1, cv_nmse_mean + cv_nmse_std),
                            alpha=0.2, color=plasma(0.25))
        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20, color=plasma(0.25))
        axs[0].axvline(int(cv_row['step']), color=plasma(0.25), ls='--', lw=1.5)
        axs[0].annotate(f"Iteration = {int(cv_row['step'])}",
                        xy=(int(cv_row['step']), 0.5), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.25))
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([int(steps.min()), int(steps.max())])

        # Row 1: AIC
        axs[1].plot(df_path['step'], df_path['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['step']), color=plasma(0.55), ls='-', lw=1.5)
        axs[1].annotate(f"Iteration = {int(aic_row['step'])}",
                        xy=(int(aic_row['step']), np.mean(axs[1].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])
        axs[1].set_xticks([int(steps.min()), int(steps.max())])

        # Row 2: BIC
        axs[2].plot(df_path['step'], df_path['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['step']), color=plasma(0.85), ls='-.', lw=1.5)
        axs[2].annotate(f"Iteration = {int(bic_row['step'])}",
                        xy=(int(bic_row['step']), np.mean(axs[2].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])
        axs[2].set_xticks([int(steps.min()), int(steps.max())])

        axs[2].set_xlabel('LARS Iteration', fontsize=PLOT_LABEL_FONTSIZE)
        for ax in axs:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_Lars_Criteria.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e_plot:
        print(f"  WARN: LARS plotting failed: {e_plot}")

    return results

def run_omp_analysis(X_scaled, y, feature_names, cv_folds, title_prefix, save_dir, random_seed=None):
    """
    OMP with a SINGLE shared greedy path used for CV, AIC, and BIC.
    Enforces non-negative coefficients throughout:
      - Greedy selection uses only positively-correlated atoms.
      - Coefficients at each step are refit via NNLS (β >= 0).
      - CV and AIC/BIC refits use NNLS-with-intercept via centering.
    """

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

    # ---------- helpers: NNLS (fallback PGD) and NNLS-with-intercept ----------
    def _pgd_nnls(A, b, max_iter=2000, tol=1e-8):
        """Projected-gradient NNLS fallback: min 0.5||Aβ-b||^2 s.t. β>=0"""
        A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float).ravel()
        if A.size == 0 or A.shape[1] == 0:
            return np.zeros((0,), dtype=float)
        try:
            L = np.linalg.norm(A, 2)**2
        except Exception:
            L = np.linalg.norm(A)**2
        L = max(L, EPS)
        beta = np.zeros(A.shape[1], dtype=float)
        for _ in range(max_iter):
            grad = A.T @ (A @ beta - b)
            beta_new = beta - grad / L
            beta_new = np.maximum(beta_new, 0.0)
            if np.linalg.norm(beta_new - beta) <= tol * (np.linalg.norm(beta) + tol):
                beta = beta_new
                break
            beta = beta_new
        return beta

    def _nnls(A, b):
        """Prefer scipy.optimize.nnls, else PGD fallback."""
        A = np.asarray(A, dtype=float); b = np.asarray(b, dtype=float).ravel()
        if A.size == 0 or A.shape[1] == 0:
            return np.zeros((0,), dtype=float)
        try:
            from scipy.optimize import nnls as _scipy_nnls
            beta, _ = _scipy_nnls(A, b)
        except Exception:
            beta = _pgd_nnls(A, b)
        return np.maximum(beta, 0.0)

    def _nnls_refit_with_intercept(Xmat, yvec):
        """
        Solve:  min_{μ, β>=0} || y - μ - X β ||^2
        by centering X, y on the fitting set, solve NNLS for β on centered data,
        then μ = mean(y) - mean(X,0)·β. Returns (beta, intercept).
        """
        Xmat = np.asarray(Xmat, dtype=float)
        yvec = np.asarray(yvec, dtype=float).ravel()
        if Xmat.size == 0 or Xmat.shape[1] == 0:
            return np.zeros(0, dtype=float), float(np.mean(yvec))
        xmean = Xmat.mean(axis=0)
        ymean = float(np.mean(yvec))
        Xc = Xmat - xmean
        yc = yvec - ymean
        beta = _nnls(Xc, yc)
        mu = ymean - float(xmean @ beta)
        return beta, mu

    # ---------- 1) Greedy OMP path on centered y (NNLS refit, positive-only) ----------
    X = X_scaled
    y_centered = y - np.mean(y)
    max_k = min(n_features, max(1, np.linalg.matrix_rank(X)))
    active, cols = [], []
    r = y_centered.copy()

    for k in range(1, max_k + 1):
        corr = X.T @ r  # choose only positively-correlated atoms
        if active:
            corr[np.array(active, int)] = -np.inf
        corr[corr <= 0] = -np.inf
        j = int(np.argmax(corr))
        if not np.isfinite(corr[j]):
            print(f"  OMP stopped at k={k-1}: no positively-correlated atoms remain."); break
        if j in active:
            print(f"  OMP early stop at k={k} (duplicate)."); break
        active.append(j)

        # Refit β>=0 at current support on centered target (no intercept in residual update)
        beta_S = _nnls(X[:, active], y_centered)
        beta = np.zeros(n_features); beta[active] = beta_S
        cols.append(np.maximum(beta.copy(), 0.0))

        # Update residual
        r = y_centered - X @ beta
        if np.linalg.norm(r) < 1e-14:
            print(f"  OMP residual ~0 at k={k}; stopping."); break

    if not cols:
        print("ERROR: empty OMP path."); return results

    coefs_path = np.maximum(np.column_stack(cols), 0.0)  # (p, n_steps), ensure non-negative
    steps = np.arange(1, coefs_path.shape[1] + 1)
    supports = [np.flatnonzero(coefs_path[:, i] > EPS) for i in range(coefs_path.shape[1])]

    # ---------- 2) CV on SAME supports (NNLS-with-intercept refit) ----------
    print("  Cross-validating on unified OMP supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_seed)
    cv_mse = np.zeros(len(steps))
    cv_mse_sq = np.zeros(len(steps))

    for tr, va in kf.split(X):
        Xtr, Xva = X[tr], X[va]
        ytr, yva = y[tr], y[va]
        ytr_mean = float(np.mean(ytr))
        for i, S in enumerate(supports):
            if len(S) == 0:
                yhat = np.full_like(yva, fill_value=ytr_mean)
                mse = float(np.mean((yva - yhat) ** 2))
            else:
                beta_i, mu_i = _nnls_refit_with_intercept(Xtr[:, S], ytr)
                yhat = mu_i + Xva[:, S] @ beta_i
                mse = float(np.mean((yva - yhat) ** 2))
            cv_mse[i] += mse
            cv_mse_sq[i] += mse**2

    cv_mse_mean = cv_mse / cv_folds
    cv_mse_std  = np.sqrt(np.maximum(cv_mse_sq / cv_folds - cv_mse_mean**2, 0.0))

    # Normalize to [0,1] for plotting (min–max on mean; std scaled by same range)
    mm, MM = float(np.min(cv_mse_mean)), float(np.max(cv_mse_mean))
    rng = max(MM - mm, EPS)
    cv_nmse_mean = (cv_mse_mean - mm) / rng
    cv_nmse_std  = cv_mse_std / rng

    best_idx = int(np.argmin(cv_mse_mean))
    best_step = int(steps[best_idx])

    # ---------- 3) AIC/BIC on FULL data (NNLS-with-intercept refit) ----------
    rows = []
    for i, S in enumerate(supports):
        if len(S) == 0:
            mu_full = float(np.mean(y))
            y_pred = np.full_like(y, fill_value=mu_full)
            k_eff = 1
            coeffs_full = np.zeros(n_features)
        else:
            beta_i, mu_full = _nnls_refit_with_intercept(X[:, S], y)
            y_pred = mu_full + X[:, S] @ beta_i
            k_eff = len(S) + 1
            coeffs_full = np.zeros(n_features); coeffs_full[S] = beta_i
        aic, bic = calculate_aic_bic(y, y_pred, n_params=k_eff, n_samples=n_samples)
        rows.append({'step': int(steps[i]), 'k': len(S), 'coeffs': coeffs_full, 'aic': aic, 'bic': bic})
    df_path = pd.DataFrame(rows)

    # ---------- 4) Results ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    cv_row  = df_path.iloc[best_idx]
    aic_row = df_path.iloc[df_path['aic'].idxmin()]
    bic_row = df_path.iloc[df_path['bic'].idxmin()]

    results['OMPCV'].update({
        'coeffs': cv_row['coeffs'], 'optimal_k': cv_row['step'],
        'n_nonzero': int(np.sum(cv_row['coeffs'] > EPS)),
        'aic': cv_row['aic'], 'bic': cv_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(cv_row['coeffs']) if c > EPS],
        'time': duration_cv
    })
    results['OMPAIC'].update({
    'coeffs': aic_row['coeffs'], 'optimal_k': aic_row['step'],
    'n_nonzero': int(np.sum(aic_row['coeffs'] > EPS)),
    'aic': aic_row['aic'], 'bic': aic_row['bic'],
    'selected_features': [feature_names[j] for j, c in enumerate(aic_row['coeffs']) if c > EPS],
    'time': duration_cv  
    })
    results['OMPBIC'].update({
        'coeffs': bic_row['coeffs'], 'optimal_k': bic_row['step'],
        'n_nonzero': int(np.sum(bic_row['coeffs'] > EPS)),
        'aic': bic_row['aic'], 'bic': bic_row['bic'],
        'selected_features': [feature_names[j] for j, c in enumerate(bic_row['coeffs']) if c > EPS],
        'time': duration_cv  
    })

    # ---------- 5) Plots (unchanged layout; positive path) ----------
    try:
        chosen_heat = {
            'CV':  {'index': int(cv_row['step'])  - 1, 'label_val': int(cv_row['step']),  'color': plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': int(aic_row['step']) - 1, 'label_val': int(aic_row['step']), 'color': plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': int(bic_row['step']) - 1, 'label_val': int(bic_row['step']), 'color': plasma(0.85), 'linestyle': '-.'},
        }
        plot_activation_heatmap(
            steps, coefs_path, feature_names, chosen_heat,
            title="OMP Activation Path", x_axis_label='OMP Iteration',
            filename=os.path.join(save_dir, f"{title_prefix}_OMP_ActivationHeatmap.pdf"),
            log_x=False, reverse_x=False, is_omp=True
        )

        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        # Row 0: CV NMSE (±std)
        axs[0].plot(steps, cv_nmse_mean, 'o-', ms=3, color=plasma(0.25))
        axs[0].fill_between(steps,
                            np.maximum(0, cv_nmse_mean - cv_nmse_std),
                            np.minimum(1, cv_nmse_mean + cv_nmse_std),
                            alpha=0.2, color=plasma(0.25))
        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20, color=plasma(0.25))
        axs[0].axvline(int(cv_row['step']), color=plasma(0.25), ls='--', lw=1.5)
        axs[0].annotate(f"Iteration = {int(cv_row['step'])}",
                        xy=(int(cv_row['step']), 0.5), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.25))
        axs[0].set_yticks([0.0, 1.0])
        axs[0].set_xticks([int(steps.min()), int(steps.max())])

        # Row 1: AIC
        axs[1].plot(df_path['step'], df_path['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[1].axvline(int(aic_row['step']), color=plasma(0.55), ls='-', lw=1.5)
        axs[1].annotate(f"Iteration = {int(aic_row['step'])}",
                        xy=(int(aic_row['step']), np.mean(axs[1].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])
        axs[1].set_xticks([int(steps.min()), int(steps.max())])

        # Row 2: BIC
        axs[2].plot(df_path['step'], df_path['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        axs[2].axvline(int(bic_row['step']), color=plasma(0.85), ls='-.', lw=1.5)
        axs[2].annotate(f"Iteration = {int(bic_row['step'])}",
                        xy=(int(bic_row['step']), np.mean(axs[2].get_ylim())), xytext=(4, 0), textcoords='offset points',
                        rotation=90, va='center', ha='left', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])
        axs[2].set_xticks([int(steps.min()), int(steps.max())])

        axs[2].set_xlabel('OMP Iteration', fontsize=PLOT_LABEL_FONTSIZE)
        for ax in axs:
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_OMP_Criteria.pdf"), format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e_plot:
        print(f"  WARN: OMP plotting failed: {e_plot}")

    return results

# ==============================================
# Load Data Functions
# ==============================================
def load_treloar_data(include_data=True):
    if not include_data: print("Skipping Treloar data."); return pd.DataFrame()
    base_path = "."; ut_file_path = os.path.join(base_path, "input/TreloarDataUT.csv"); ps_file_path = os.path.join(base_path, "input/TreloarDataPS.csv"); ebt_file_path = os.path.join(base_path, "input/TreloarDataEBT.csv")
    all_data = []; loaded_files = 0
    if os.path.exists(ut_file_path):
        try: ut_df=pd.read_csv(ut_file_path,sep=';',decimal='.',skiprows=1,header=None,names=['lambda','P11']); [all_data.append({'lambda1':r['lambda'],'lambda2':1./np.sqrt(max(r['lambda'],EPS)),'lambda3':1./np.sqrt(max(r['lambda'],EPS)),'P11':r['P11'],'P22':0.0,'mode':'UT','dataset':'Treloar'}) for _,r in ut_df.iterrows()]; print(f"Loaded Treloar UT: {ut_file_path}"); loaded_files+=1
        except Exception as e: print(f"Error UT: {e}")
    else: print(f"UT file not found: {ut_file_path}")
    if os.path.exists(ps_file_path):
        try: ps_df=pd.read_csv(ps_file_path,sep=';',decimal='.',skiprows=1,header=None,names=['lambda','P11']); [all_data.append({'lambda1':r['lambda'],'lambda2':1.0,'lambda3':1./max(r['lambda'],EPS),'P11':r['P11'],'P22':np.nan,'mode':'PS','dataset':'Treloar'}) for _,r in ps_df.iterrows()]; print(f"Loaded Treloar PS: {ps_file_path}"); loaded_files+=1
        except Exception as e: print(f"Error PS: {e}")
    else: print(f"PS file not found: {ps_file_path}")
    if os.path.exists(ebt_file_path):
        try: ebt_df=pd.read_csv(ebt_file_path,sep=';',decimal='.',skiprows=1,header=None,names=['lambda','P11']); [all_data.append({'lambda1':r['lambda'],'lambda2':r['lambda'],'lambda3':1./max(r['lambda']**2,EPS),'P11':r['P11'],'P22':r['P11'],'mode':'EBT','dataset':'Treloar'}) for _,r in ebt_df.iterrows()]; print(f"Loaded Treloar EBT: {ebt_file_path}"); loaded_files+=1
        except Exception as e: print(f"Error EBT: {e}")
    else: print(f"EBT file not found: {ebt_file_path}")
    if not all_data: print("WARN: No Treloar data loaded."); return pd.DataFrame()
    treloar_df = pd.DataFrame(all_data); treloar_df['strain_pct'] = (treloar_df['lambda1'] - 1.0) * 100.0; return treloar_df

def load_Kawabata_data(csv_path="input/Kawabata.csv", include_data=True):
    if not include_data: print("Skipping Kawabata data."); return pd.DataFrame()
    if not os.path.exists(csv_path): print(f"Kawabata file not found: {csv_path}"); return pd.DataFrame()
    df = None
    try: df=pd.read_csv(csv_path,sep=';',decimal='.',skiprows=1); df.columns=['lambda1','lambda2','P11','P22'][:len(df.columns)]; print(f"Loaded Kawabata (';')")
    except Exception: print(f"Info: Kawabata with ';' failed. Trying ','."); df=None
    if df is None:
        try: df=pd.read_csv(csv_path,sep=',',decimal='.',skiprows=1); df.columns=['lambda1','lambda2','P11','P22'][:len(df.columns)]; print(f"Loaded Kawabata (',')")
        except Exception as e2: print(f"Error: Failed Kawabata {csv_path}: {e2}"); return pd.DataFrame()
    if len(df.columns)<4: print(f"Error: Kawabata needs at least 4 cols."); return pd.DataFrame()
    df['lambda3']=1./(df['lambda1'].apply(lambda x:max(x,EPS))*df['lambda2'].apply(lambda x:max(x,EPS))); df['lambda3']=df['lambda3'].apply(lambda x:max(x,EPS))
    df['strain_pct']=(df['lambda1']-1.)*100.; df['mode']='BIAX'; df['dataset']='Kawabata'; print(f"Processed {len(df)} Kawabata points.")
    for col in ['P11','P22']: df[col]=pd.to_numeric(df[col], errors='coerce') # Ensure numeric
    return df[['lambda1','lambda2','lambda3','P11','P22','mode','dataset','strain_pct']]

def combine_datasets(include_treloar=True, include_Kawabata=True, Kawabata_path="input/Kawabata.csv"):
    print("\n--- Loading Data ---"); df_treloar=load_treloar_data(include_data=include_treloar); df_kawabata=load_Kawabata_data(csv_path=Kawabata_path,include_data=include_Kawabata)
    combined_df=pd.concat([df_treloar,df_kawabata],ignore_index=True)
    if len(combined_df)==0: raise ValueError("No data loaded. Check file paths and include flags.")
    print(f"\n--- Dataset Summary ---"); print(f"Total data points combined: {len(combined_df)}"); print(f"Sources: {combined_df['dataset'].unique()}"); print(f"Modes: {combined_df['mode'].unique()}")
    return combined_df.reset_index(drop=True)

# ==============================================
# Entry Point for Sparse Regression Analysis (Fit on Subset, Evaluate Full)
# ==============================================
if __name__ == "__main__":

    # --- Configuration ---
    KAWABATA_FILE_PATH = 'input/Kawabata.csv'

    EMPHASIZE_SMALL_STRAIN = False
    CV_FOLDS = 5
    BASE_SAVE_DIR = "outputs" # New version name
    COEFF_THRESHOLD = 1e-6 # Threshold for final coefficients after refit
    RIDGE_ALPHA_REFIT = 1e-6 # Regularization for Ridge refit step
    WEIGHT_FACTOR = 2.0 # Factor for emphasizing small strains (if enabled)
    SMALL_STRAIN_THRESHOLD = 50.0 # Strain % threshold for weighting

    # Define Fitting Scenarios
    # Each tuple: (Scenario Name, Use Treloar for Fit, Use Kawabata for Fit)
    # fitting_scenarios = [
    #     ("Fit_Treloar_Only", True, False),
    #     ("Fit_Kawabata_Only", False, True),
    #     ("Fit_Both", True, True)
    # ]
    
    fitting_scenarios = [
        ("Fit_Treloar_Only", True, False)
    ]

    # Define Model Library Configuration (Ensure only one set of parameters is active)
    MODEL_CONFIG_NAME = "MR3_OgdenExt"
    INCLUDE_NEOHOOKEAN = False
    INCLUDE_MOONEY_RIVLIN = True   # Use Mooney-Rivlin based terms
    INCLUDE_YEOH = False
    INCLUDE_ARRUDA_BOYCE = False
    INCLUDE_OGDEN = True           # Use Ogden terms
    INCLUDE_EXTENDED_OGDEN = True  # Use the extended set of Ogden exponents
    OGDEN_EXPONENTS = []           # Will be generated based on flags
    INCLUDE_GENT = False
    INCLUDE_DATA_DRIVEN = False    # Not used
    INCLUDE_FRACTIONAL_POWERS = False
    INCLUDE_LOW_STRAIN_TERMS = False
    INCLUDE_ULTRA_SMALL_STRAIN = False # Controls specific Ogden/Frac/Low terms

    # Determine Refit Regressor Type based on alpha
    REFIT_REGRESSION_TYPE = f"Ridge(α={RIDGE_ALPHA_REFIT:.1e})" if RIDGE_ALPHA_REFIT > 0 else "OLS"

    overall_start_time = datetime.datetime.now()
    print(f"Starting Sparse Regression Analysis (Fit Subset / Evaluate Full) at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Using Model Library: {MODEL_CONFIG_NAME}")
    print(f"Base Save Directory: {BASE_SAVE_DIR}")
    os.makedirs(BASE_SAVE_DIR, exist_ok=True)

    # --- Load the FULL combined dataset ONCE for evaluation/plotting purposes ---
    print("\n--- Loading Full Combined Dataset (for evaluation structure) ---")
    # Check for existence of necessary files first
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
        print(f"FATAL ERROR: Could not load the full combined dataset: {e}")
        print("Cannot proceed without the full dataset structure for evaluation.")
        traceback.print_exc()
        exit()

    # --- Loop Through Fitting Scenarios ---
    all_initial_results = {} # Store initial sparse selection results per scenario
    all_refit_summaries = {} # Store final post-processed results per scenario

    for fitting_name, use_treloar_fit, use_kawabata_fit in fitting_scenarios:

        print("\n" + "#"*80)
        print(f"### PROCESSING FITTING SCENARIO: {fitting_name} ###")
        print(f"### Model Library: {MODEL_CONFIG_NAME} ###")
        print(f"### Refit Method (on subset): {REFIT_REGRESSION_TYPE}, Threshold={COEFF_THRESHOLD:.1e} ###")
        print(f"### Weight Factor: {WEIGHT_FACTOR}, Small Strain Thresh: {SMALL_STRAIN_THRESHOLD}% ###")
        print("#"*80 + "\n")

        current_save_dir = os.path.join(BASE_SAVE_DIR, fitting_name)
        os.makedirs(current_save_dir, exist_ok=True)
        # Prefix includes both fitting scenario and model config
        scenario_prefix = f"{fitting_name}_{MODEL_CONFIG_NAME}"

        # --- Create Fitting Subset Data ---
        print(f"--- Creating data subset for fitting: Treloar={use_treloar_fit}, Kawabata={use_kawabata_fit} ---")
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

        print(f"  Total data points in fitting subset: {len(df_fitting_subset)}")

        # --- Construct Matrix, Scale (using ONLY fitting subset) ---
        basis_names = [] # Ensure scope
        try:
            print(f"\nGenerating Model Library & Constructing Design Matrix from Fitting Subset...")
            # Generate basis functions based on global config (done within construct_design_matrix)
            # The global flags determine the library structure used here.
            design_matrix_weighted_fit, target_vector_weighted_fit, basis_names, _, df_subset_info = construct_design_matrix(
                 df_fitting_subset, # Use the fitting subset here
                 emphasize_small_strain=EMPHASIZE_SMALL_STRAIN,
                 weight_factor=WEIGHT_FACTOR,
                 small_strain_threshold=SMALL_STRAIN_THRESHOLD
            )

            if design_matrix_weighted_fit.shape[0] == 0:
                raise ValueError("Weighted design matrix for fitting is empty.")
            if not basis_names:
                 raise ValueError("Basis names list is empty after matrix construction.")
            if len(basis_names) != design_matrix_weighted_fit.shape[1]:
                raise ValueError(f"Basis ({len(basis_names)}) / Matrix column ({design_matrix_weighted_fit.shape[1]}) mismatch for fitting matrix.")

            print(f"Weighted design matrix for FITTING created with shape: {design_matrix_weighted_fit.shape}")
            
            # Scale the fitting data - SCALE BOTH X AND Y
            print("Scaling fitting data (both X and y)...")
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()
            X_weighted_scaled_fit = scaler_X.fit_transform(design_matrix_weighted_fit)
            target_vector_scaled_fit = scaler_y.fit_transform(target_vector_weighted_fit.reshape(-1, 1)).flatten()
            
            if not np.all(np.isfinite(X_weighted_scaled_fit)):
                print("WARN: Non-finite values after scaling fitting data. Applying nan_to_num.")
                X_weighted_scaled_fit = np.nan_to_num(X_weighted_scaled_fit)

            print(f"Fitting data prepared and scaled successfully. X shape: {X_weighted_scaled_fit.shape}, y scaled variance: {np.var(target_vector_scaled_fit):.4f}")


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
                    if key not in scenario_refit_summary: scenario_refit_summary[key] = {'consistent': None, 'notes': [f"Fatal Error in initial fit: {outer_e}"]}

        # --- Post-process ALL results for this fitting scenario ---
        # Refit on subset, Evaluate/Plot on full combined set
        print("\n" + "="*40 + f" Post-processing All Models for Fitting Scenario: {fitting_name} " + "="*40)
        print(f"--- Refitting on FIT SUBSET, Evaluating on FULL Combined Dataset ---")

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
                    n_nonzero_initial = result_data.get('n_nonzero', np.sum(np.abs(result_data['coeffs']) > EPS))
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
                                 weight_factor=WEIGHT_FACTOR,
                                 small_strain_threshold=SMALL_STRAIN_THRESHOLD,
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
                            scenario_refit_summary[method_key]['notes'] = [f"Plotting/Evaluation Error: {plot_e}"]
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
                    scenario_refit_summary[method_key]['notes'] = [f"Initial fit error: {err_msg}"]


        # --- Store results for this fitting scenario ---
        all_initial_results[fitting_name] = scenario_initial_results
        all_refit_summaries[fitting_name] = scenario_refit_summary
        print(f"\n### Finished All Analyses & Post-processing for Fitting Scenario: {fitting_name} ###\n")


    # --- Print Overall Summaries ---
    print("\n" + "="*80)
    print("### INITIAL SPARSE SELECTION SUMMARY (Fit on Scenario Subset, Weighted) ###")
    print("="*80)
    for fitting_scenario_name, methods_results in all_initial_results.items():
        print(f"\n--- Fitting Scenario Used: {fitting_scenario_name} ---")
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
                    print(f"| {method_print_key:<11} | ERROR ({error_msg_short}...) | ---      | ---       | ---       | {time_str:<8} |")
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
                    print(f"| {method_print_key:<11} | {opt_param_str:<16} | {n_nonzero:<8} | {aic_str:<9} | {bic_str:<9} | {time_str:<8} |")

    # --- Print Final Refit/Evaluation Summary Table ---
    print("\n" + "="*80)
    print(f"### FINAL MODEL EVALUATION SUMMARY (Refit: {REFIT_REGRESSION_TYPE} on Fit Subset, Threshold={COEFF_THRESHOLD:.1e}) ###")
    print("### Metrics Calculated on FULL UNWEIGHTED Combined Data | Consistency Check (μ₀ > 0) ###")
    print("="*80)
    for fitting_scenario_name, methods_summary in all_refit_summaries.items():
        print(f"\n--- Fitting Scenario Used: {fitting_scenario_name} ---")
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
                     print(f"| {method_print_key:<11} | {n_terms:<7} | {r2_p11_str:<8} | {rmse_p11_str:<10} | {r2_p22_str:<8} | {rmse_p22_str:<10} | {consistent_str:<10} |")
                     print(f"|           -> Note: {note_summary}")
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
                     print(f"| {method_print_key:<11} | {n_terms:<7} | {r2_p11_str:<8} | {rmse_p11_str:<10} | {r2_p22_str:<8} | {rmse_p22_str:<10} | {consistent_str:<10} |")
                     if summary.get('terms'):
                         print(f"|  -> Final Model Terms & Coefficients (from subset refit):")
                         max_terms_to_print = 10 # Show fewer terms in summary table
                         for i, (term, coeff) in enumerate(zip(summary['terms'], summary['coeffs'])):
                             if i < max_terms_to_print:
                                 # Use format_term_for_latex for better display if needed, or keep raw
                                 term_display = term # format_term_for_latex(term).strip('$') # Optional LaTeX formatting
                                 print(f"|      - {term_display:<40}: {coeff:.6f}")
                             elif i == max_terms_to_print:
                                 print(f"|      ... (remaining {len(summary['terms'])-max_terms_to_print} terms not shown)")
                         notes = summary.get('notes', ['N/A'])
                         mu0_note = next((note for note in notes if 'μ₀ ≈' in note), 'μ₀ calculation note N/A')
                         consistency_note = next((note for note in notes if 'CONSISTENCY' in note), 'Consistency note N/A')
                         print(f"|  -> {mu0_note.strip()}")
                         print(f"|  -> {consistency_note.strip()}")
                     else:
                         print("|  -> (No terms remaining after thresholding/refit)")
                     print("|" + "-"*78 + "|")
                     
        # --- Print Mode-Specific Evaluation Summary Table ---
    print("\n" + "="*80)
    print("### MODE-SPECIFIC R² EVALUATION SUMMARY ###")
    print("="*80)
    for fitting_scenario_name, methods_summary in all_refit_summaries.items():
        print(f"\n--- Fitting Scenario Used: {fitting_scenario_name} ---")
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
                
                print(f"| {method_print_key:<11} | {ut_r2:<10} | {ps_r2:<10} | {ebt_r2_p11:<11} | {ebt_r2_p22:<11} | {biax_r2_p11:<12} | {biax_r2_p22:<12} |")
        
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
                            p11_str = f"P11: n={n_p11}" if n_p11 > 0 else ""
                            p22_str = f"P22: n={n_p22}" if n_p22 > 0 else ""
                            both_str = f"{p11_str}, {p22_str}" if p11_str and p22_str else (p11_str or p22_str)
                            print(f"|   {mode}: {both_str}")
                break  # Only need to print sample sizes once

    print(f"\nCompleted all scenarios at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total analysis duration: {datetime.datetime.now() - overall_start_time}")
    print("="*80)