# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors 
import matplotlib.cm as cm
import matplotlib.ticker 
from matplotlib.patches import Patch
from sklearn.linear_model import LassoCV, Lasso, lasso_path, LarsCV, lars_path, OrthogonalMatchingPursuitCV, orthogonal_mp, OrthogonalMatchingPursuit, LinearRegression 
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import minimize
from scipy.optimize import nnls
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
import time 
import datetime
import os
import warnings
import traceback
from sklearn.exceptions import ConvergenceWarning
import re

# Suppress convergence/runtime warnings from third-party libraries
warnings.filterwarnings('ignore', category=ConvergenceWarning)
warnings.filterwarnings("ignore", category=UserWarning, module='sklearn')
warnings.filterwarnings('ignore', category=RuntimeWarning)

# --- Set a global random seed for reproducibility ---
np.random.seed(42)

EVAL_R2_AGAINST_CLEAN = True

# --- Configuration ---
EXCEL_FILE_NAME = './input/CANNsHEARTdata_shear05.xlsx' 
SHEET_NAME = 'Sheet1' 
DATA_START_ROW = 4  
WEIGHTING_EPSILON = 1e-6 
NON_LINEAR_OPTIMIZER_METHOD = 'L-BFGS-B' 
NON_LINEAR_OPTIMIZER_TOL = 1e-6
CV_FOLDS = 5 # Folds for CV methods
BASE_SAVE_DIR = "outputs" 
RELATIVE_NOISE_LEVELS = [0.0, 0.05, 0.10] # Noise levels to test
EPS = 1e-10 # Small number for safe division/log

# --- Plotting Configuration ---
PLOT_TITLE_FONTSIZE = 14
PLOT_LABEL_FONTSIZE = 14
PLOT_TICK_FONTSIZE = 14
PLOT_LEGEND_FONTSIZE = 10
PLOT_MARKER_SIZE = 5
PLOT_LINEWIDTH = 2
PLOT_HEATMAP_YLABEL_SIZE = 9

SMALL_FIG_LASSO_LARS_OMP = (3.0, 4.5) 

# Define colors from Plasma palette
try:
    plasma = plt.get_cmap('plasma')
    blues_cmap = plt.get_cmap('Blues')
except ValueError:
    plasma = plt.cm.plasma
    blues_cmap = plt.cm.Blues

# Define the truncated plasma colormap that's used in plotting functions
TRUNCATED_PLASMA_CMAP = plasma

# Colormap for Heatmaps (Binary: Inactive/Active) - Aligned with HearthCANN
HEATMAP_CMAP = blues_cmap

color_refined = plasma(0.15)
color_pdf = plasma(0.55)
color_original = 'black'
color_noisy_data = plasma(0.7)

# --- 1. Load Dataset (Robust Loading) ---
df_experimental = None 
try:
    df_temp = pd.read_excel(EXCEL_FILE_NAME, sheet_name=SHEET_NAME, header=None)
    if isinstance(df_temp, dict): 
        df_experimental = df_temp.get(SHEET_NAME)
        if df_experimental is None and df_temp: 
            first_sheet_key = list(df_temp.keys())[0]
            df_experimental = df_temp[first_sheet_key]
            df_experimental.attrs['sheet_name'] = first_sheet_key
            print(f"Warning: Sheet '{SHEET_NAME}' not found. Loaded sheet '{first_sheet_key}' instead.")
        elif df_experimental is not None:
             df_experimental.attrs['sheet_name'] = SHEET_NAME
    else: 
        df_experimental = df_temp
        df_experimental.attrs['sheet_name'] = SHEET_NAME
    if df_experimental is not None:
        print(f"Successfully loaded Excel: '{EXCEL_FILE_NAME}', Sheet: '{df_experimental.attrs['sheet_name']}'")
except FileNotFoundError: exit(f"Error: '{EXCEL_FILE_NAME}' not found.")
except ValueError as e: 
    if "Worksheet named" in str(e) and f"'{SHEET_NAME}'" in str(e):
        print(f"Error: Worksheet named '{SHEET_NAME}' not found in '{EXCEL_FILE_NAME}'.")
        try:
            print(f"Attempting to load the first sheet...")
            xls = pd.ExcelFile(EXCEL_FILE_NAME); first_sheet_name_actual = xls.sheet_names[0]
            df_experimental = pd.read_excel(xls, sheet_name=first_sheet_name_actual, header=None)
            df_experimental.attrs['sheet_name'] = first_sheet_name_actual
            print(f"Successfully loaded the first sheet ('{first_sheet_name_actual}').")
        except Exception as e_fallback: exit(f"Could not load the first sheet either: {e_fallback}")
    else: exit(f"An error occurred while reading the Excel file: {e}")
except Exception as e_generic: exit(f"A generic error occurred during file loading: {e_generic}")
if df_experimental is None: exit("CRITICAL ERROR: Data loading failed.")

# --- Data Extraction Logic ---
shear_test_definitions = {
    'fs': {'P_col': 1, 'gamma_col': 0, 'plot_title': 'Shear $\sigma_{fs}$', 'stress_label': r'$\sigma_{fs}$', 'F_func': lambda g: np.array([[1,0,0],[g,1,0],[0,0,1]]), 'stress_to_calc': 'fs'},
    'fn': {'P_col': 3, 'gamma_col': 2, 'plot_title': 'Shear $\sigma_{fn}$', 'stress_label': r'$\sigma_{fn}$', 'F_func': lambda g: np.array([[1,0,0],[0,1,0],[g,0,1]]), 'stress_to_calc': 'fn'},
    'sf': {'P_col': 6, 'gamma_col': 5, 'plot_title': 'Shear $\sigma_{sf}$', 'stress_label': r'$\sigma_{sf}$', 'F_func': lambda g: np.array([[1,g,0],[0,1,0],[0,0,1]]), 'stress_to_calc': 'sf'},
    'sn': {'P_col': 8, 'gamma_col': 7, 'plot_title': 'Shear $\sigma_{sn}$', 'stress_label': r'$\sigma_{sn}$', 'F_func': lambda g: np.array([[1,0,0],[0,1,0],[0,g,1]]), 'stress_to_calc': 'sn'},
    'nf': {'P_col': 11, 'gamma_col': 10, 'plot_title': 'Shear $\sigma_{nf}$', 'stress_label': r'$\sigma_{nf}$', 'F_func': lambda g: np.array([[1,0,g],[0,1,0],[0,0,1]]), 'stress_to_calc': 'nf'},
    'ns': {'P_col': 13, 'gamma_col': 12, 'plot_title': 'Shear $\sigma_{ns}$', 'stress_label': r'$\sigma_{ns}$', 'F_func': lambda g: np.array([[1,0,0],[0,1,g],[0,0,1]]), 'stress_to_calc': 'ns'}
}
experimental_data_list = [] 
experimental_plot_series = {} # Store x, y_exp for plotting
for mode, D in shear_test_definitions.items():
    gamma_vals_series = pd.to_numeric(df_experimental.iloc[DATA_START_ROW:, D['gamma_col']], errors='coerce')
    P_vals_series = pd.to_numeric(df_experimental.iloc[DATA_START_ROW:, D['P_col']], errors='coerce')
    valid_indices = gamma_vals_series.notna() & P_vals_series.notna()
    gamma_vals = gamma_vals_series[valid_indices]; P_vals = P_vals_series[valid_indices]
    if not gamma_vals.empty:
        experimental_plot_series[mode] = {'x': gamma_vals.values, 'y_exp': P_vals.values, 'type': 'shear','xlabel': 'Shear Strain $\gamma$ [-]', 'ylabel': f"Stress {D['stress_label']} [kPa]", 'title': D['plot_title']}
        for i in range(len(gamma_vals)):
            g = gamma_vals.iloc[i]; F = D['F_func'](g)
            experimental_data_list.append({'F': F, 'type': 'shear', 'target_stress': P_vals.iloc[i], 'test_name': mode, 'gamma_val': g, 'stress_to_calc': D['stress_to_calc'], 'original_x': g})
biaxial_test_definitions = {
    '1:1':    {'MFs':15,'MFp':16,'CFs':17,'CFp':18, 'plot_title_f':'Biax. $\sigma_{ff}$ (1:1)','plot_title_n':'Biax. $\sigma_{nn}$ (1:1)'},
    '1:0.75': {'MFs':20,'MFp':21,'CFs':22,'CFp':23, 'plot_title_f':'Biax. $\sigma_{ff}$ (1:0.75)','plot_title_n':'Biax. $\sigma_{nn}$ (1:0.75)'},
    '0.75:1': {'MFs':25,'MFp':26,'CFs':27,'CFp':28, 'plot_title_f':'Biax. $\sigma_{ff}$ (0.75:1)','plot_title_n':'Biax. $\sigma_{nn}$ (0.75:1)'},
    '1:0.5':  {'MFs':30,'MFp':31,'CFs':32,'CFp':33, 'plot_title_f':'Biax. $\sigma_{ff}$ (1:0.5)','plot_title_n':'Biax. $\sigma_{nn}$ (1:0.5)'},
    '0.5:1':  {'MFs':35,'MFp':36,'CFs':37,'CFp':38, 'plot_title_f':'Biax. $\sigma_{ff}$ (0.5:1)','plot_title_n':'Biax. $\sigma_{nn}$ (0.5:1)'}
}
for ratio_name, D in biaxial_test_definitions.items():
    lambda_f_col_data = pd.to_numeric(df_experimental.iloc[DATA_START_ROW:, D['MFs']], errors='coerce')
    P_ff_col_data = pd.to_numeric(df_experimental.iloc[DATA_START_ROW:, D['MFp']], errors='coerce')
    lambda_n_col_data = pd.to_numeric(df_experimental.iloc[DATA_START_ROW:, D['CFs']], errors='coerce')
    P_nn_col_data = pd.to_numeric(df_experimental.iloc[DATA_START_ROW:, D['CFp']], errors='coerce')
    valid_ff = lambda_f_col_data.notna() & P_ff_col_data.notna() & lambda_n_col_data.notna()
    lf_vals_ff, ln_vals_ff, p_ff_vals = lambda_f_col_data[valid_ff], lambda_n_col_data[valid_ff], P_ff_col_data[valid_ff]
    plot_key_f = ratio_name + '_f'
    if not lf_vals_ff.empty:
        experimental_plot_series[plot_key_f] = {'x': lf_vals_ff.values, 'y_exp': p_ff_vals.values, 'type': 'biaxial_f','xlabel': 'Stretch $\lambda_f$ [-]', 'ylabel': r'Stress $\sigma_{ff}$ [kPa]', 'title': D['plot_title_f']}
        for i in range(len(lf_vals_ff)):
            lf, ln = lf_vals_ff.iloc[i], ln_vals_ff.iloc[i]
            if lf * ln == 0: continue
            ls = 1.0 / (lf * ln); F = np.diag([lf, ls, ln])
            experimental_data_list.append({'F': F, 'type': 'biaxial', 'target_stress': p_ff_vals.iloc[i], 'test_name': plot_key_f, 'lambda_f_val': lf, 'lambda_n_val': ln, 'stress_to_calc': 'ff', 'original_x': lf})
    valid_nn = lambda_n_col_data.notna() & P_nn_col_data.notna() & lambda_f_col_data.notna()
    ln_vals_nn, lf_vals_nn, p_nn_vals = lambda_n_col_data[valid_nn], lambda_f_col_data[valid_nn], P_nn_col_data[valid_nn]
    plot_key_n = ratio_name + '_n'
    if not ln_vals_nn.empty:
        experimental_plot_series[plot_key_n] = {'x': ln_vals_nn.values, 'y_exp': p_nn_vals.values, 'type': 'biaxial_n','xlabel': 'Stretch $\lambda_n$ [-]', 'ylabel': r'Stress $\sigma_{nn}$ [kPa]', 'title': D['plot_title_n']}
        for i in range(len(ln_vals_nn)):
            ln, lf = ln_vals_nn.iloc[i], lf_vals_nn.iloc[i]
            if lf * ln == 0: continue
            ls = 1.0 / (lf * ln); F = np.diag([lf, ls, ln])
            experimental_data_list.append({'F': F, 'type': 'biaxial', 'target_stress': p_nn_vals.iloc[i], 'test_name': plot_key_n, 'lambda_f_val': lf, 'lambda_n_val': ln, 'stress_to_calc': 'nn', 'original_x': ln})

# Store original target stresses before adding noise
y_target_original = np.array([dp['target_stress'] for dp in experimental_data_list])

# --- 2. Invariant, Psi Terms, and Stress Definitions ---
def get_invariants(F):
    C = F.T @ F; I1 = np.trace(C); I2 = 0.5 * (I1**2 - np.trace(C @ C))
    I4f, I4s, I4n = C[0,0], C[1,1], C[2,2]; I8fs, I8fn, I8sn = C[0,1], C[0,2], C[1,2]
    return {'I1':I1,'I2':I2,'I4f':I4f,'I4s':I4s,'I4n':I4n,'I4f_bar':max(I4f,1.0),'I4s_bar':max(I4s,1.0),
            'I4n_bar':max(I4n,1.0),'I8fs':I8fs,'I8fn':I8fn,'I8sn':I8sn}

psi_terms_definitions = [
    {'idx': 0, 'inv_key': 'I1', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'term_label': 'c1(I1-3)', 'raw_inv_transform': lambda I1: I1 - 3.0},
    {'idx': 1, 'inv_key': 'I1', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 0, 'term_label': 'c2exp(w1(I1-3))', 'raw_inv_transform': lambda I1: I1 - 3.0},
    {'idx': 2, 'inv_key': 'I1', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'term_label': 'c3(I1-3)^2', 'raw_inv_transform': lambda I1: I1 - 3.0},
    {'idx': 3, 'inv_key': 'I1', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 1, 'term_label': 'c4exp(w2(I1-3)^2)', 'raw_inv_transform': lambda I1: I1 - 3.0},
    {'idx': 4, 'inv_key': 'I2', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'term_label': 'c5(I2-3)', 'raw_inv_transform': lambda I2: I2 - 3.0},
    {'idx': 5, 'inv_key': 'I2', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 2, 'term_label': 'c6exp(w3(I2-3))', 'raw_inv_transform': lambda I2: I2 - 3.0},
    {'idx': 6, 'inv_key': 'I2', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'term_label': 'c7(I2-3)^2', 'raw_inv_transform': lambda I2: I2 - 3.0},
    {'idx': 7, 'inv_key': 'I2', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 3, 'term_label': 'c8exp(w4(I2-3)^2)', 'raw_inv_transform': lambda I2: I2 - 3.0},
    {'idx': 8, 'inv_key': 'I4f_bar', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'original_inv_key': 'I4f', 'term_label': 'c9(I4f-1)', 'raw_inv_transform': lambda I4fb: I4fb - 1.0},
    {'idx': 9, 'inv_key': 'I4f_bar', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 4, 'original_inv_key': 'I4f', 'term_label': 'c10exp(w5(I4f-1))', 'raw_inv_transform': lambda I4fb: I4fb - 1.0},
    {'idx': 10, 'inv_key': 'I4f_bar', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'original_inv_key': 'I4f', 'term_label': 'c11(I4f-1)^2', 'raw_inv_transform': lambda I4fb: I4fb - 1.0},
    {'idx': 11, 'inv_key': 'I4f_bar', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 5, 'original_inv_key': 'I4f', 'term_label': 'c12exp(w6(I4f-1)^2)', 'raw_inv_transform': lambda I4fb: I4fb - 1.0},
    {'idx': 12, 'inv_key': 'I4s_bar', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'original_inv_key': 'I4s', 'term_label': 'c13(I4s-1)', 'raw_inv_transform': lambda I4sb: I4sb - 1.0},
    {'idx': 13, 'inv_key': 'I4s_bar', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 6, 'original_inv_key': 'I4s', 'term_label': 'c14exp(w7(I4s-1))', 'raw_inv_transform': lambda I4sb: I4sb - 1.0},
    {'idx': 14, 'inv_key': 'I4s_bar', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'original_inv_key': 'I4s', 'term_label': 'c15(I4s-1)^2', 'raw_inv_transform': lambda I4sb: I4sb - 1.0},
    {'idx': 15, 'inv_key': 'I4s_bar', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 7, 'original_inv_key': 'I4s', 'term_label': 'c16exp(w8(I4s-1)^2)', 'raw_inv_transform': lambda I4sb: I4sb - 1.0},
    {'idx': 16, 'inv_key': 'I4n_bar', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'original_inv_key': 'I4n', 'term_label': 'c17(I4n-1)', 'raw_inv_transform': lambda I4nb: I4nb - 1.0},
    {'idx': 17, 'inv_key': 'I4n_bar', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 8, 'original_inv_key': 'I4n', 'term_label': 'c18exp(w9(I4n-1))', 'raw_inv_transform': lambda I4nb: I4nb - 1.0},
    {'idx': 18, 'inv_key': 'I4n_bar', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'original_inv_key': 'I4n', 'term_label': 'c19(I4n-1)^2', 'raw_inv_transform': lambda I4nb: I4nb - 1.0},
    {'idx': 19, 'inv_key': 'I4n_bar', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 9, 'original_inv_key': 'I4n', 'term_label': 'c20exp(w10(I4n-1)^2)', 'raw_inv_transform': lambda I4nb: I4nb - 1.0},
    {'idx': 20, 'inv_key': 'I8fs', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'term_label': 'c21(I8fs)', 'raw_inv_transform': lambda I8fs: I8fs},
    {'idx': 21, 'inv_key': 'I8fs', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 10, 'term_label': 'c22exp(w11I8fs)', 'raw_inv_transform': lambda I8fs: I8fs},
    {'idx': 22, 'inv_key': 'I8fs', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'term_label': 'c23(I8fs)^2', 'raw_inv_transform': lambda I8fs: I8fs},
    {'idx': 23, 'inv_key': 'I8fs', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 11, 'term_label': 'c24exp(w12(I8fs)^2)', 'raw_inv_transform': lambda I8fs: I8fs},
    {'idx': 24, 'inv_key': 'I8fn', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'term_label': 'c25(I8fn)', 'raw_inv_transform': lambda I8fn: I8fn},
    {'idx': 25, 'inv_key': 'I8fn', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 12, 'term_label': 'c26exp(w13I8fn)', 'raw_inv_transform': lambda I8fn: I8fn},
    {'idx': 26, 'inv_key': 'I8fn', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'term_label': 'c27(I8fn)^2', 'raw_inv_transform': lambda I8fn: I8fn},
    {'idx': 27, 'inv_key': 'I8fn', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 13, 'term_label': 'c28exp(w14(I8fn)^2)', 'raw_inv_transform': lambda I8fn: I8fn},
    {'idx': 28, 'inv_key': 'I8sn', 'type': 'lin', 'is_exp': False, 'w1_idx': None, 'term_label': 'c29(I8sn)', 'raw_inv_transform': lambda I8sn: I8sn},
    {'idx': 29, 'inv_key': 'I8sn', 'type': 'exp_lin', 'is_exp': True, 'w1_idx': 14, 'term_label': 'c30exp(w15I8sn)', 'raw_inv_transform': lambda I8sn: I8sn},
    {'idx': 30, 'inv_key': 'I8sn', 'type': 'sq', 'is_exp': False, 'w1_idx': None, 'term_label': 'c31(I8sn)^2', 'raw_inv_transform': lambda I8sn: I8sn},
    {'idx': 31, 'inv_key': 'I8sn', 'type': 'exp_sq', 'is_exp': True, 'w1_idx': 15, 'term_label': 'c32exp(w16(I8sn)^2)', 'raw_inv_transform': lambda I8sn: I8sn},
]
num_psi_terms = len(psi_terms_definitions) 
num_exp_terms = sum(1 for t in psi_terms_definitions if t['is_exp'])

def get_psi_deriv_for_term(term_def, invariants, w1=1.0):
    derivs = {key: 0.0 for key in ['I1', 'I2', 'I4f_bar', 'I4s_bar', 'I4n_bar', 'I8fs', 'I8fn', 'I8sn']}
    inv_name = term_def['inv_key']
    if inv_name not in invariants: return derivs 
    raw_inv = invariants[inv_name]
    if 'raw_inv_transform' not in term_def: return derivs
    trans_inv = term_def['raw_inv_transform'](raw_inv)
    d_term_d_trans = 0.0
    with warnings.catch_warnings(): 
        warnings.filterwarnings('ignore', category=RuntimeWarning)
        try: 
            if term_def['is_exp']:
                exp_arg = w1 * trans_inv
                exp_arg_sq = w1 * trans_inv**2
                exp_arg = min(exp_arg, 700); exp_arg_sq = min(exp_arg_sq, 700)
                if term_def['type'] == 'exp_lin': d_term_d_trans = w1 * np.exp(exp_arg) 
                elif term_def['type'] == 'exp_sq': d_term_d_trans = 2 * w1 * trans_inv * np.exp(exp_arg_sq)
            else: 
                if term_def['type'] == 'lin': d_term_d_trans = 1.0
                elif term_def['type'] == 'sq': d_term_d_trans = 2 * trans_inv
        except (OverflowError, ValueError): d_term_d_trans = np.nan 
    derivs[inv_name] = d_term_d_trans
    return derivs

# def calculate_stress_contribution_linearized(data_point, psi_derivs_for_one_term):
def calculate_stress_contribution_linearized(data_point, psi_derivs_for_one_term, term_def):
    """
    Correctly calculates the linearized contribution of a single psi term to the
    final stress value for use in the Stage 1 feature matrix (X).
    For biaxial tests, this is the contribution to the stress *difference* (e.g., s_ff - s_ss).
    For shear tests, this is the contribution to the shear stress component.
    """
    inv_key = term_def['inv_key']
    term_type = term_def['type']
    if ('I4' in inv_key or 'I8' in inv_key) and term_type in ['lin', 'exp_lin']:
        return 0.0    
    d_psi_d_I1 = psi_derivs_for_one_term.get('I1', 0.0)
    d_psi_d_I2 = psi_derivs_for_one_term.get('I2', 0.0)
    d_psi_d_I4f_bar = psi_derivs_for_one_term.get('I4f_bar', 0.0)
    d_psi_d_I4s_bar = psi_derivs_for_one_term.get('I4s_bar', 0.0)
    d_psi_d_I4n_bar = psi_derivs_for_one_term.get('I4n_bar', 0.0)
    d_psi_d_I8fs = psi_derivs_for_one_term.get('I8fs', 0.0)
    d_psi_d_I8fn = psi_derivs_for_one_term.get('I8fn', 0.0)
    d_psi_d_I8sn = psi_derivs_for_one_term.get('I8sn', 0.0)

    if any(np.isnan(v) or np.isinf(v) for v in psi_derivs_for_one_term.values()):
        return np.nan

    stress_contribution = 0.0
    try:
        if data_point['type'] == 'biaxial':
            lf, ln = data_point['lambda_f_val'], data_point['lambda_n_val']
            if lf * ln == 0: return np.nan
            ls = 1.0 / (lf * ln)
            
            # Isotropic contribution to stress difference (from textbook formulas)
            iso_contrib_ff = 2 * (lf**2 - ls**2) * (d_psi_d_I1 + ln**2 * d_psi_d_I2)
            iso_contrib_nn = 2 * (ln**2 - ls**2) * (d_psi_d_I1 + lf**2 * d_psi_d_I2)
            
            # Anisotropic contribution to stress difference (derived from full tensor)
            aniso_contrib_ff = 2 * d_psi_d_I4f_bar - 2 * d_psi_d_I4s_bar
            aniso_contrib_nn = 2 * d_psi_d_I4n_bar - 2 * d_psi_d_I4s_bar

            if data_point['stress_to_calc'] == 'ff':
                stress_contribution = iso_contrib_ff + aniso_contrib_ff
            elif data_point['stress_to_calc'] == 'nn':
                stress_contribution = iso_contrib_nn + aniso_contrib_nn

        elif data_point['type'] == 'shear':
            gamma = data_point['gamma_val']
            s_calc = data_point['stress_to_calc']
            
            # Isotropic contribution to shear stress
            stress_contribution = 2 * gamma * (d_psi_d_I1 + d_psi_d_I2)
            
            # Anisotropic contribution to shear stress
            if s_calc in ['fs', 'sf']:
                stress_contribution += d_psi_d_I8fs
            elif s_calc in ['fn', 'nf']:
                stress_contribution += d_psi_d_I8fn
            elif s_calc in ['sn', 'ns']:
                stress_contribution += d_psi_d_I8sn
        
        if np.isnan(stress_contribution) or np.isinf(stress_contribution):
            return np.nan

    except (OverflowError, ValueError):
        return np.nan
        
    return stress_contribution

def calculate_single_prediction_full_tensor(data_point, invariants, coeffs_w2, w1_map):
    """
    Calculates a single stress component for one data point using the full, 
    physically correct stress tensor formulation. Used for Stage 2 refinement and final predictions.
    """
    F = data_point['F']
    f0, s0, n0 = np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])

    try:
        B = F @ F.T
        I1_val = invariants.get('I1', np.trace(B))
    except np.linalg.LinAlgError:
        return np.nan

    sigma_hyper_tensor = np.zeros((3, 3))

    for term_idx, term_def in enumerate(psi_terms_definitions):
        w2 = coeffs_w2[term_idx]
        if abs(w2) < EPS: continue
        
        inv_key = term_def['inv_key']
        term_type = term_def['type']
        if ('I4' in inv_key or 'I8' in inv_key) and term_type in ['lin', 'exp_lin']:
            continue # Skip this term's contribution entirely
                
        w1 = w1_map.get(term_idx, 1.0)
        psi_derivs = get_psi_deriv_for_term(term_def, invariants, w1)
        
        if 'I1' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I1'] * 2 * B
        if 'I2' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I2'] * 2 * (I1_val * B - B @ B)
        if 'I4f_bar' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I4f_bar'] * 2 * np.outer(f0, f0)
        if 'I4s_bar' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I4s_bar'] * 2 * np.outer(s0, s0)
        if 'I4n_bar' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I4n_bar'] * 2 * np.outer(n0, n0)
        if 'I8fs' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I8fs'] * (np.outer(f0, s0) + np.outer(s0, f0))
        if 'I8fn' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I8fn'] * (np.outer(f0, n0) + np.outer(n0, f0))
        if 'I8sn' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I8sn'] * (np.outer(s0, n0) + np.outer(n0, s0))

    # Enforce sigma_ss = 0 by defining pressure p = sigma_hyper_ss
    pressure_p = sigma_hyper_tensor[1, 1]
    sigma_total_tensor = sigma_hyper_tensor - pressure_p * np.eye(3)

    component_map = {'ff': (0, 0), 'ss': (1, 1), 'nn': (2, 2), 'fs': (0, 1), 'sf': (1, 0), 'fn': (0, 2), 'nf': (2, 0), 'sn': (1, 2), 'ns': (2, 1)}
    stress_comp = data_point['stress_to_calc']
    return sigma_total_tensor[component_map[stress_comp]] if stress_comp in component_map else np.nan

def objective_function(params, active_term_indices, num_active_terms_in_subset, data_list_indices, weights, y_target_for_obj, all_invariants_list):
    """ Objective function for Stage 2 non-linear refinement. Uses the full tensor calculation. """
    w2_params = params[:num_active_terms_in_subset]
    w1_params = params[num_active_terms_in_subset:]
    
    # Build the full coefficient vectors for the prediction function
    coeffs_w2_full = np.zeros(num_psi_terms)
    coeffs_w2_full[active_term_indices] = w2_params
    w1_map_full = {}
    w1_counter_local = 0
    for term_idx_global in active_term_indices:
        if psi_terms_definitions[term_idx_global]['is_exp']:
            if w1_counter_local < len(w1_params):
                w1_map_full[term_idx_global] = w1_params[w1_counter_local]
                w1_counter_local += 1

    predictions = np.zeros(len(data_list_indices))
    for i_dp_idx, data_idx in enumerate(data_list_indices):
        data_point = experimental_data_list_filtered[data_idx]
        invariants = all_invariants_list[data_idx]
        pred = calculate_single_prediction_full_tensor(data_point, invariants, coeffs_w2_full, w1_map_full)
        predictions[i_dp_idx] = pred if not np.isnan(pred) else np.inf

    y_target_subset = y_target_for_obj[data_list_indices]
    weights_subset = weights[data_list_indices]
    if np.any(np.isinf(predictions)) or np.any(np.isnan(predictions)): return float('inf')
    total_weighted_ssr = np.sum(weights_subset * (y_target_subset - predictions)**2)
    return total_weighted_ssr

def calculate_predictions(coeffs_w2_full, w1_map, invar_list, data_list):
    """ Calculates final predictions for plotting/evaluation. Uses the full tensor calculation. """
    y_pred = np.zeros(len(data_list))
    for i, dp in enumerate(data_list):
        invariants = invar_list[i]
        y_pred[i] = calculate_single_prediction_full_tensor(dp, invariants, coeffs_w2_full, w1_map)
    return y_pred

# --- Helper: AIC/BIC Calculation ---
def calculate_aic_bic(y_true, y_pred, n_params, n_samples, weights=None):
    if n_samples <= 0 or n_params >= n_samples or n_params == 0: return np.nan, np.nan 
    if weights is None: weights = np.ones_like(y_true)
    if len(weights) != len(y_true): weights = np.ones_like(y_true)
    # Ensure y_pred has same length as y_true and weights
    if len(y_pred) != len(y_true): 
        print(f"WARN: y_pred length ({len(y_pred)}) != y_true length ({len(y_true)}) in AIC/BIC calc. Returning NaN.")
        return np.nan, np.nan
    
    # Filter out NaN predictions before calculating SSR
    valid_mask = ~np.isnan(y_pred)
    if np.sum(valid_mask) < n_params + 1: # Need more points than params
        return np.nan, np.nan
        
    y_true_f = y_true[valid_mask]
    y_pred_f = y_pred[valid_mask]
    weights_f = weights[valid_mask]
    n_samples_f = len(y_true_f)
    if n_samples_f <= n_params: return np.nan, np.nan # Check again after filtering

    weighted_ssr = np.sum(weights_f * (y_true_f - y_pred_f)**2)
    if weighted_ssr < EPS: return -np.inf, -np.inf 
    log_likelihood_proxy = -n_samples_f / 2.0 * np.log(max(weighted_ssr / n_samples_f, EPS))
    aic = -2 * log_likelihood_proxy + 2 * n_params
    bic = -2 * log_likelihood_proxy + n_params * np.log(max(n_samples_f, 1))
    return aic, bic

def build_stress_magnitude_weights(data_list):
    """
    Calculates weights for each data point based on the inverse RMS stress of its mode.
    This is analogous to the stress normalization in the synthetic data script.
    """
    print("INFO: Building weights based on inverse stress magnitude per mode...")
    
    # 1. Group stresses by mode ('test_name') from the data list
    stresses_by_mode = {}
    for dp in data_list:
        mode = dp.get('test_name')
        # Use the target stress for weighting
        stress = dp.get('target_stress') 
        if mode and stress is not None:
            if mode not in stresses_by_mode:
                stresses_by_mode[mode] = []
            stresses_by_mode[mode].append(stress)
            
    # 2. Calculate RMS stress for each mode
    stress_stats = {}
    for mode, stresses in stresses_by_mode.items():
        if stresses:
            stresses_arr = np.array(stresses)
            stress_stats[mode] = {
                'rms': np.sqrt(np.mean(stresses_arr**2))
            }
            print(f"    Mode '{mode}': RMS Stress = {stress_stats[mode]['rms']:.3f} kPa")

    # 3. Calculate the global RMS across all modes
    all_rms_values = [stats['rms'] for stats in stress_stats.values() if stats['rms'] > EPS]
    if not all_rms_values:
        print("WARN: No valid stress data for weighting. Returning equal weights.")
        return np.ones(len(data_list))
        
    global_rms = np.sqrt(np.mean(np.array(all_rms_values)**2))
    print(f"    Global RMS Stress (for normalization): {global_rms:.3f} kPa")

    # 4. Calculate the final weight factor for each mode
    mode_weights = {}
    for mode, stats in stress_stats.items():
        mode_weights[mode] = global_rms / stats['rms'] if stats['rms'] > EPS else 1.0
        print(f"    Mode '{mode}': Final Weight Factor = {mode_weights[mode]:.3f}")

    # 5. Create the final weight array, mapping the correct weight to each data point
    final_weights = np.array([mode_weights.get(dp.get('test_name'), 1.0) for dp in data_list])
    
    return final_weights

# --- Define PDF Model Parameters ---
pdf_model_active_indices = [6, 11, 19, 23] # 0-based indices: Terms 7, 12, 20, 24
pdf_model_coeffs_w2 = np.zeros(num_psi_terms)
pdf_model_w1_map = {}
pdf_model_coeffs_w2[6] = 5.162 
pdf_model_coeffs_w2[11] = 0.081; pdf_model_w1_map[11] = 21.151
pdf_model_coeffs_w2[19] = 0.315; pdf_model_w1_map[19] = 4.371
pdf_model_coeffs_w2[23] = 0.486; pdf_model_w1_map[23] = 0.508

def refine_model(stage1_model_dict, model_name, y_target_current, sample_weights_current, all_invariants_current):
    print(f"\n--- Stage 2: Refining {model_name} Model ---")
    
    # Handle potential negative coefficients from LARS/OMP by applying NNLS pre-processing
    active_coeffs_s1 = stage1_model_dict['coeffs'].copy()
    
    # Check if this is a LARS or OMP method that might have negative coefficients
    is_lars_or_omp = any(keyword in model_name for keyword in ['Lars', 'OMP'])
    
    if is_lars_or_omp:
        print(f"  INFO: Applying NNLS pre-refinement for {model_name} coefficients.")
        # Identify features selected by LARS/OMP in Stage 1
        selected_feature_indices_stage1 = np.where(np.abs(active_coeffs_s1) > 1e-9)[0]
        
        if len(selected_feature_indices_stage1) > 0:
            # Build design matrix for selected features
            X_selected_for_nnls = X_features_stage1[:, selected_feature_indices_stage1]
            
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
                print(f"    NNLS for {model_name}: {len(selected_feature_indices_stage1)} initially selected -> {n_positive_after_nnls} positive coefficients.")
            except Exception as e:
                print(f"    WARNING: NNLS pre-refinement failed for {model_name}: {e}. Falling back to clamping.")
                active_coeffs_s1 = np.maximum(active_coeffs_s1, 0)
        else:
            print(f"    No features originally selected by {model_name} in Stage 1.")
            active_coeffs_s1 = np.zeros_like(active_coeffs_s1)

    # Select terms based on coefficients being meaningfully positive
    active_term_indices = np.where(active_coeffs_s1 > 1e-6)[0]
    
    if len(active_term_indices) == 0:
        print(f"  INFO: {model_name} (after Stage 1 pre-processing) has 0 active terms with coeff > 1e-6. Skipping refinement.")
        return {'coeffs': np.zeros(num_psi_terms), 'w1_map': {}, 'success': False, 'message': 'No positive terms selected/retained from Stage 1 for refinement', 'time':0.0}
    
    active_terms_defs = [psi_terms_definitions[i] for i in active_term_indices]
    active_w2_coeffs_initial = active_coeffs_s1[active_term_indices] # Initial guess for w2 parameters
    
    active_exp_term_indices_in_subset = [i for i, term_idx in enumerate(active_term_indices) if psi_terms_definitions[term_idx]['is_exp']]
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
    
    print(f"  Optimizing {num_params_to_optimize} parameters ({num_active_terms_subset} w2, {num_active_exp_terms} w1) for {len(active_term_indices)} active terms...")
    opt_start_time = time.time()
    
    n_samples_for_obj = len(y_target_current) 

    optimization_result = minimize(
        objective_function, initial_guess, 
        args=(active_term_indices, num_active_terms_subset, np.arange(n_samples_for_obj), sample_weights_current, y_target_current, all_invariants_current),
        method=NON_LINEAR_OPTIMIZER_METHOD, bounds=bounds,
        options={'maxiter': 1000, 'ftol': NON_LINEAR_OPTIMIZER_TOL}
    )
    opt_end_time = time.time()
    print(f"  Optimization finished in {opt_end_time - opt_start_time:.2f} seconds. Success: {optimization_result.success}")
    
    final_coeffs_full = np.zeros(num_psi_terms)
    optimized_w1_map = {}
    refined_model_summary = {'success': optimization_result.success, 'message': optimization_result.message, 'time': opt_end_time - opt_start_time}

    if optimization_result.success:
        optimized_params = optimization_result.x
        final_w2_coeffs = optimized_params[:num_active_terms_subset]
        final_w1_coeffs = optimized_params[num_active_terms_subset:]
        
        final_coeffs_full[active_term_indices] = final_w2_coeffs # Assign refined w2
        
        w1_counter = 0
        print(f"\n  --- Refined {model_name} Model Summary ---")
        print(f"  Number of active terms: {len(active_term_indices)}")
        refined_model_summary['coeffs'] = final_coeffs_full
        refined_model_summary['k_w2'] = np.sum(final_coeffs_full > 1e-6) # Count actual non-zero after refinement
        refined_model_summary['k_w1'] = num_active_exp_terms # Number of w1 params optimized
        refined_model_summary['w1_map'] = {}

        for i_term_subset, term_idx_global in enumerate(active_term_indices):
            term_def = psi_terms_definitions[term_idx_global]
            w2_val = final_w2_coeffs[i_term_subset]
            print(f"    Term {term_idx_global+1} ({term_def['term_label']}): w2 = {w2_val:.4f}", end="")
            if term_def['is_exp']:
                if w1_counter < len(final_w1_coeffs):
                    w1_val = final_w1_coeffs[w1_counter]
                    optimized_w1_map[term_idx_global] = w1_val
                    refined_model_summary['w1_map'][term_idx_global] = w1_val
                    print(f", w1 = {w1_val:.4f}")
                    w1_counter += 1
                else: # Should not happen if logic is correct
                    print(", w1 = N/A (error in indexing)")
            else:
                print("") 
        # Recalculate k_w2 based on refined coefficients being > threshold
        refined_model_summary['k_w2'] = np.sum(final_coeffs_full[active_term_indices] > 1e-6)

    else:
        print(f"  Non-linear optimization failed for {model_name}. Using Stage 1 (pre-processed) coefficients as initial guess for reporting.")
        # Report based on the initial guess that went into the failed optimization
        final_coeffs_full[active_term_indices] = active_w2_coeffs_initial
        refined_model_summary['coeffs'] = final_coeffs_full
        refined_model_summary['w1_map'] = {} # No w1 values optimized
        # For w1_map in failure, could fill with initial 1.0s for active exp terms for consistency if needed by downstream
        w1_counter_fail = 0
        for i_term_subset, term_idx_global in enumerate(active_term_indices):
            term_def = psi_terms_definitions[term_idx_global]
            if term_def['is_exp']:
                refined_model_summary['w1_map'][term_idx_global] = initial_guess_w1[w1_counter_fail] # Report initial guess
                w1_counter_fail +=1
        
        refined_model_summary['k_w2'] = np.sum(active_w2_coeffs_initial > 1e-6)
        refined_model_summary['k_w1'] = num_active_exp_terms if refined_model_summary['k_w2'] > 0 else 0

    return refined_model_summary

def generate_stress_strain_plot_refined(coeffs_w2_full, w1_map, pdf_preds, y_target_noisy,
                                        plot_title_suffix, save_dir, file_prefix,
                                        all_invariants_current):
    """
    Plot refined model vs. data across modes with consistent layout and labels.
    """
    # Predictions
    y_pred_plot_specific = calculate_predictions(
        coeffs_w2_full, w1_map, all_invariants_current, experimental_data_list_filtered
    )

    plot_metrics = {}
    temp_plot_series = {k: v.copy() for k, v in experimental_plot_series.items()}
    for k in temp_plot_series:
        base_len = len(temp_plot_series[k].get('x', []))
        for arr_key in ['y_exp', 'y_noisy', 'y_pred_refined', 'y_pred_pdf']:
            if arr_key not in temp_plot_series[k]:
                temp_plot_series[k][arr_key] = np.full(base_len, np.nan, dtype=float)

    # Map predictions / noisy
    pred_map_refined, pred_map_pdf, noisy_target_map = {}, {}, {}
    for i, dp in enumerate(experimental_data_list_filtered):
        key = (dp.get('test_name'), dp.get('original_x'))
        if None in key: continue
        if i < len(y_pred_plot_specific): pred_map_refined[key] = y_pred_plot_specific[i]
        if i < len(pdf_preds):            pred_map_pdf[key]     = pdf_preds[i]
        if i < len(y_target_noisy):       noisy_target_map[key] = y_target_noisy[i]

    for test_key, series in temp_plot_series.items():
        x_arr = np.array(series.get('x', []))
        series['y_pred_refined'] = np.full_like(x_arr, np.nan, dtype=float)
        series['y_pred_pdf']     = np.full_like(x_arr, np.nan, dtype=float)
        series['y_noisy']        = np.full_like(x_arr, np.nan, dtype=float)
        for j, xv in enumerate(x_arr):
            mk = (test_key, xv)
            if mk in pred_map_refined: series['y_pred_refined'][j] = pred_map_refined[mk]
            if mk in pred_map_pdf:     series['y_pred_pdf'][j]     = pred_map_pdf[mk]
            if mk in noisy_target_map: series['y_noisy'][j]        = noisy_target_map[mk]

    # Figure (4:3 ratio)
    fig_w, fig_h = 9.5, 12.5  # 4:3
    fig, axs = plt.subplots(6, 3, figsize=(fig_w, fig_h))

    # Legend prototypes
    ax_proto = axs.flat[0]
    line_data,   = ax_proto.plot([], [], 'o', markersize=5, markerfacecolor='none',
                                 color=color_original, label='Original Data')
    line_noise,  = ax_proto.plot([], [], 'o', markersize=5, color=color_noisy_data,
                                 alpha=0.7, label='Noisy Data')
    line_refined,= ax_proto.plot([], [], '-',  lw=2, color=color_refined,
                                 label=f'Our Disc. Model ({plot_title_suffix})')
    line_pdf,    = ax_proto.plot([], [], '--', lw=2, color=color_pdf,
                                 label='Martonová et al. (2024)')

    shear_modes_sorted    = ['fs', 'fn', 'sf', 'sn', 'nf', 'ns']
    biaxial_ratios_sorted = ['1:1', '1:0.75', '0.75:1', '1:0.5', '0.5:1']
    lambda_f_keys         = [f"{r}_f" for r in biaxial_ratios_sorted]
    lambda_n_keys         = [f"{r}_n" for r in biaxial_ratios_sorted]

    def plot_single(ax, mode_key, data_dict, noise_prefix, active=True):
        if not active:
            ax.axis('off'); return
        d = data_dict.get(mode_key)
        if d is None or d.get('x') is None or d.get('y_exp') is None or len(d['x']) == 0:
            ax.set_title("No Data", fontsize=PLOT_LABEL_FONTSIZE)
            return
        x_raw = np.array(d['x']); y_exp = np.array(d['y_exp'])
        idx = np.argsort(x_raw)
        x = x_raw[idx]; y_exp = y_exp[idx]
        y_noisy = np.array(d.get('y_noisy', np.full_like(x, np.nan)))[idx]
        y_ref   = np.array(d.get('y_pred_refined', np.full_like(x, np.nan)))[idx]
        y_pdf   = np.array(d.get('y_pred_pdf', np.full_like(x, np.nan)))[idx]

        ax.plot(x, y_exp, 'o', markersize=5, markerfacecolor='none', color=color_original)
        if noise_prefix.split('_')[0] != "RelNoise0pct" and not np.all(np.isnan(y_noisy)):
            ax.plot(x, y_noisy, 'o', markersize=5, color=color_noisy_data, alpha=0.7)
        if not np.all(np.isnan(y_ref)): ax.plot(x, y_ref, '-', lw=2, color=color_refined)
        if not np.all(np.isnan(y_pdf)): ax.plot(x, y_pdf, '--', lw=2, color=color_pdf)

        if x.size:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            if xmin != xmax:
                ax.set_xticks([xmin, xmax])
                ax.xaxis.set_major_formatter(
                    plt.FormatStrFormatter('%.2f' if (xmax - xmin) < 0.2 else '%.1f')
                )
        y_all = np.concatenate([
            y_exp[~np.isnan(y_exp)],
            y_noisy[~np.isnan(y_noisy)],
            y_ref[~np.isnan(y_ref)],
            y_pdf[~np.isnan(y_pdf)]
        ]) if any([~np.all(np.isnan(a)) for a in [y_exp, y_noisy, y_ref, y_pdf]]) else np.array([])
        if y_all.size:
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
            if ymin != ymax:
                ax.set_yticks([ymin, ymax])
                ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter('%.2f' if (ymax - ymin) < 0.2 else '%.1f')
                )

        r2_val = rmse_val = np.nan
        mask = ~np.isnan(y_ref) & ~np.isnan(y_exp)
        if np.sum(mask) > 1:
            try:
                r2_val = r2_score(y_exp[mask], y_ref[mask])
                rmse_val = np.sqrt(mean_squared_error(y_exp[mask], y_ref[mask]))
            except ValueError:
                pass
        base_title = d.get('title', mode_key)
        if not np.isnan(r2_val):
            title_full = f"{base_title}\n$R^2$: {r2_val:.3f}, RMSE: {rmse_val:.3f}"
        else:
            title_full = f"{base_title}\n$R^2$/RMSE: N/A"
        ax.set_title(title_full, fontsize=PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=PLOT_TICK_FONTSIZE)
        plot_metrics[mode_key] = {'R2': r2_val, 'RMSE': rmse_val}

    # Populate
    for r in range(6):
        plot_single(axs[r, 0], shear_modes_sorted[r], temp_plot_series, file_prefix,
                    active=(r < len(shear_modes_sorted)))
    for r in range(6):
        plot_single(axs[r, 1], lambda_f_keys[r] if r < len(lambda_f_keys) else '',
                    temp_plot_series, file_prefix, active=(r < len(lambda_f_keys)))
    for r in range(6):
        plot_single(axs[r, 2], lambda_n_keys[r] if r < len(lambda_n_keys) else '',
                    temp_plot_series, file_prefix, active=(r < len(lambda_n_keys)))

    # Clear labels (custom placement)
    for ax in axs.ravel():
        ax.set_xlabel(""); ax.set_ylabel("")

    # Assign labels to ALL subplots now
    for r in range(6):
        axs[r, 0].set_ylabel("Stress [kPa]", fontsize=PLOT_LABEL_FONTSIZE)#, rotation=0,
                            #  ha='right', va='center')

    for r in range(6):
        axs[r, 0].set_xlabel("Shear Strain $\\gamma$ [-]", fontsize=PLOT_LABEL_FONTSIZE)
        axs[r, 1].set_xlabel("Stretch $\\lambda_f$ [-]", fontsize=PLOT_LABEL_FONTSIZE)
        axs[r, 2].set_xlabel("Stretch $\\lambda_n$ [-]", fontsize=PLOT_LABEL_FONTSIZE)

    # Draw for positioning
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Incrust Y labels
    for r in range(6):
        ax = axs[r, 0]
        tick_labels = ax.get_yticklabels()
        widths = []
        for tl in tick_labels:
            if tl.get_text():
                widths.append(tl.get_window_extent(renderer=renderer).width)
        if not widths: continue
        max_w = max(widths)
        fig_w_pixels = fig_w * fig.dpi
        gap = 150
        offset_fig_frac = (max_w + gap) / fig_w_pixels
        ax.yaxis.set_label_coords(-offset_fig_frac, 0.5)

    # Incrust X labels (all subplots)
    for ax in axs.ravel():
        tick_labels_x = ax.get_xticklabels()
        heights = []
        for tl in tick_labels_x:
            if tl.get_text():
                heights.append(tl.get_window_extent(renderer=renderer).height)
        if not heights: continue
        max_h = max(heights)
        fig_h_pixels = fig_h * fig.dpi
        gap_pix = 150
        offset = (max_h + gap_pix) / fig_h_pixels
        ax.xaxis.set_label_coords(0.5, -offset)

    # Spacing
    title_lines = [ax.get_title().count('\n') + 1 for ax in axs.ravel() if ax.get_title()]
    max_lines = max(title_lines) if title_lines else 1
    line_h_in = (PLOT_LABEL_FONTSIZE / 72.0) * 0.9
    needed_in = max_lines * line_h_in
    hspace = min(0.9, max(0.15, 1.05 * needed_in / fig_h))
    wspace = 0.35
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.055,
                        hspace=hspace, wspace=wspace)

    # Legend (top center)
    handles = [line_data]
    if file_prefix.split('_')[0] != "RelNoise0pct":
        handles.append(line_noise)
    handles += [line_refined, line_pdf]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=PLOT_LEGEND_FONTSIZE + 4)

    # Tighten (no suptitle region needed now)
    fig.tight_layout(rect=[0, 0.035, 1, 0.93], pad=0.25)

    # Save
    filename = f"{file_prefix}_{plot_title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}_Comparison.pdf"
    save_path = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f"Saved plot: {save_path}")
    except Exception as e:
        print(f"Error saving plot {save_path}: {e}")
    plt.close(fig)
    return plot_metrics

def format_term_for_latex(term_name):
    """Formats a term string like 'c2exp(w1(I1-3))' into a LaTeX string."""
    term = term_name
    
    # Handle special cases first
    term = term.replace('_bar', '')  # remove _bar from internal key name
    
    # Add subscripts to c, w, I using regex (be more careful with boundaries)
    term = re.sub(r'\bc(\d+)', r'c_{\1}', term)
    term = re.sub(r'\bw(\d+)', r'w_{\1}', term)
    term = re.sub(r'\bI(\d+[a-z]*)', r'I_{\1}', term)
    
    # Handle bar for I4 invariants after subscripting
    term = re.sub(r'(I_{4[fsn]})', r'\\bar{\1}', term)
    
    # Format exp - be more careful to not break it
    term = re.sub(r'\bexp\b', r'\\exp', term)
    
    # Format powers
    term = re.sub(r'\)\^(\d+)', r')^{\1}', term)
    
    return f"${term}$"

# --- Heatmap Plotting Function ---
def plot_activation_heatmap(path_variable, coef_path, feature_names_defs, chosen_values_dict,
                            title, x_axis_label, filename=None, log_x=False,
                            reverse_x=True, activation_thresh=1e-5, is_omp=False):
    """
    Generates a single activation heatmap with multiple vertical lines for different selection criteria.
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
         ordered_basis_names = [format_term_for_latex(feature_names_defs[i]['term_label']) for i in sorted_active_features]
    except Exception as e:
         print(f"WARN: LaTeX formatting failed for heatmap labels: {e}. Using raw names.")
         ordered_basis_names = [feature_names_defs[i]['term_label'] for i in sorted_active_features]

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
    
    # Use the same colormap as synthetic data
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

def compute_weighted_stats(y, sample_weight):
    if sample_weight is None:
        w = np.ones_like(y, dtype=float)
    else:
        w = np.asarray(sample_weight, dtype=float)
        if w.shape != y.shape:  # fallback
            w = np.ones_like(y, dtype=float)
    w_sum = np.sum(w)
    if w_sum <= 0:
        w = np.ones_like(y); w_sum = len(y)
    y_mean_w = np.sum(w * y) / w_sum
    var_w_num = np.sum(w * (y - y_mean_w)**2)
    var_w = var_w_num / w_sum if w_sum > 0 else np.nan
    return w, w_sum, y_mean_w, var_w

def run_lasso_analysis(X_unscaled, y, feature_names, cv_folds, title_prefix, save_dir, sample_weight=None):
    """
    LASSO with positive coefficients:
      - LassoCV(positive=True, fit_intercept=False)
      - lasso_path(positive=True)
      - Coefficients clipped to >=0 before metrics/AIC/BIC/plots/results.
    """
    print(f"\n--- Running LASSO Analysis (CV, AIC, BIC) ({title_prefix}) ---")
    n_samples, n_features = X_unscaled.shape
    base_results = { 'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
                     'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan, 'k': 0 }
    results = { 'LassoCV': {**base_results, 'method': 'LassoCV'},
                'LassoAIC': {**base_results, 'method': 'LassoAIC'},
                'LassoBIC': {**base_results, 'method': 'LassoBIC'} }
    if n_features == 0:
        print("ERROR: No features."); return results

    # Small helper to ensure strict nonnegativity (guards numerical fuzz)
    def _clip_nonneg(beta, tol=1e-12):
        b = np.array(beta, dtype=float, copy=True)
        b[b < tol] = 0.0
        return b

    # --- 1) LassoCV (positive) on controlled alpha range ---
    print("  Running LassoCV to determine optimal alpha...")
    start_time_cv = datetime.datetime.now()
    best_alpha_from_cv = np.nan

    alpha_min_log = -8
    alpha_max_log = 0
    alphas_cv_grid = np.logspace(alpha_min_log, alpha_max_log, 1000)

    alphas_cv_plot = np.array([])
    cv_nmse_mean_plot = np.array([])
    cv_nmse_std_plot  = np.array([])

    try:
        lasso_cv = LassoCV(
            cv=cv_folds,
            alphas=alphas_cv_grid,
            max_iter=5000,
            n_jobs=-1,
            random_state=42,
            positive=True,
            fit_intercept=False
        )
        lasso_cv.fit(X_unscaled, y)   # sample_weight=None by design (already baked into X,y if any)
        best_alpha_from_cv = lasso_cv.alpha_
        print(f"  LassoCV determined optimal alpha: {best_alpha_from_cv:.6f}")

        # Extract CV MSE and robust-normalize NMSE to [0,1] for plotting
        try:
            alphas_cv_full = np.asarray(lasso_cv.alphas_)
            mse_path_cv    = np.asarray(lasso_cv.mse_path_)
            if mse_path_cv.ndim == 3:
                cv_mse_mean_full = np.nanmean(mse_path_cv, axis=(1, 2))
                cv_mse_std_full  = np.nanstd( mse_path_cv, axis=(1, 2))
            elif mse_path_cv.ndim == 2:
                cv_mse_mean_full = np.nanmean(mse_path_cv, axis=1)
                cv_mse_std_full  = np.nanstd( mse_path_cv, axis=1)
            else:
                cv_mse_mean_full = mse_path_cv.ravel()
                cv_mse_std_full  = np.zeros_like(cv_mse_mean_full)

            y_centered = y - np.mean(y)
            y_var_cv = float(np.var(y_centered))
            if y_var_cv > EPS:
                nmse_mean_full = cv_mse_mean_full / y_var_cv
                nmse_std_full  = cv_mse_std_full  / y_var_cv
            else:
                nmse_mean_full = cv_mse_mean_full
                nmse_std_full  = cv_mse_std_full

            p_lo = float(np.nanpercentile(nmse_mean_full, 5))
            p_hi = float(np.nanpercentile(nmse_mean_full, 95))
            den  = max(p_hi - p_lo, EPS)
            cv_nmse_mean_full = np.clip((nmse_mean_full - p_lo) / den, 0.0, 1.0)
            cv_nmse_std_full  = np.clip(nmse_std_full / den, 0.0, 1.0)

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

    # --- 2) Unified positive Lasso path (AIC/BIC use the same alpha domain) ---
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
            X_unscaled, y, alphas=alphas_for_path,
            positive=True,
            max_iter=5000, tol=1e-4
        )

        # Weighted stats (as in your code)
        w_mse, w_sum, y_mean_w, y_var_w = compute_weighted_stats(y, None)

        path_results_list = []
        for i in range(coefs_lasso.shape[1]):
            alpha_i = alphas_lasso[i]
            beta_i  = _clip_nonneg(coefs_lasso[:, i])
            n_params = int(np.sum(beta_i > 1e-6))
            if n_params == 0 and i < coefs_lasso.shape[1] - 1:
                continue

            y_pred_train = X_unscaled @ beta_i
            residual = y - y_pred_train
            mse_w = np.sum(w_mse * residual**2) / w_sum
            aic, bic = calculate_aic_bic(y, y_pred_train, n_params, X_unscaled.shape[0], weights=None)
            l1_norm = float(np.sum(beta_i))

            path_results_list.append({
                'alpha': alpha_i, 'k': n_params, 'coeffs': beta_i,
                'mse_train': mse_w, 'aic': aic, 'bic': bic, 'l1_norm': l1_norm
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

    # --- 2b) Normalized metrics over the path ---
    if 'df_path' in locals() and not df_path.empty:
        if y_var_w > 0:
            df_path['norm_mse'] = df_path['mse_train'] / y_var_w
        else:
            df_path['norm_mse'] = np.nan
        print(f"  DEBUG (Lasso+pos): weighted var={y_var_w:.6e}, NMSE min={df_path['norm_mse'].min():.3f}, max={df_path['norm_mse'].max():.3f}")

    max_l1_norm = df_path['l1_norm'].max()
    df_path['norm_l1'] = (df_path['l1_norm'] / max_l1_norm) if max_l1_norm > EPS else 0.0

    # --- 3) Pick models from unified path (use clipped nonnegative coeffs) ---
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    if not df_path.empty and pd.notna(best_alpha_from_cv):
        cv_model_row = df_path.iloc[(np.abs(df_path['alpha'] - best_alpha_from_cv)).argsort()[:1]].iloc[0]
        results['LassoCV'].update({
            'coeffs': cv_model_row['coeffs'], 'alpha': cv_model_row['alpha'],
            'n_nonzero': cv_model_row['k'], 'aic': cv_model_row['aic'], 'bic': cv_model_row['bic'],
            'selected_features': [feature_names[i]['term_label'] for i, c in enumerate(cv_model_row['coeffs']) if c > 1e-6],
            'time': duration_cv, 'k': cv_model_row['k']
        })
        print(f"  LassoCV model (k={cv_model_row['k']}) identified on unified path in {duration_cv:.2f}s.")

    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()
    if not df_path.empty:
        aic_model_row = df_path.loc[df_path['aic'].idxmin()]
        results['LassoAIC'].update({
            'coeffs': aic_model_row['coeffs'], 'alpha': aic_model_row['alpha'], 'n_nonzero': aic_model_row['k'],
            'aic': aic_model_row['aic'], 'bic': aic_model_row['bic'],
            'selected_features': [feature_names[j]['term_label'] for j, c in enumerate(aic_model_row['coeffs']) if c > 1e-6],
            'time': duration_path, 'k': aic_model_row['k']
        })
        print(f"  LassoAIC model (k={aic_model_row['k']}) identified on unified path.")

        bic_model_row = df_path.loc[df_path['bic'].idxmin()]
        results['LassoBIC'].update({
            'coeffs': bic_model_row['coeffs'], 'alpha': bic_model_row['alpha'], 'n_nonzero': bic_model_row['k'],
            'aic': bic_model_row['aic'], 'bic': bic_model_row['bic'],
            'selected_features': [feature_names[j]['term_label'] for j, c in enumerate(bic_model_row['coeffs']) if c > 1e-6],
            'time': duration_path, 'k': bic_model_row['k']
        })
        print(f"  LassoBIC model (k={bic_model_row['k']}) identified on unified path.")

    # --- 4) Plots ---
    if not df_path.empty and alphas_lasso is not None and coefs_lasso is not None:
        print("\n  Generating plots from unified path data...")

        # Exact models for markers
        cv_exact_model = None; aic_exact_model = None; bic_exact_model = None
        if pd.notna(best_alpha_from_cv):
            cv_idx = (np.abs(df_path['alpha'] - best_alpha_from_cv)).argsort()[0]
            cv_exact_model = df_path.iloc[cv_idx]
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
            alphas_lasso,
            # pass the already clipped nonnegative coefs (recompute just in case)
            np.column_stack([_clip_nonneg(coefs_lasso[:, i]) for i in range(coefs_lasso.shape[1])]),
            feature_names, chosen_values_heatmap,
            title=f"Lasso Activation Path", x_axis_label=r'Regularization Strength',
            filename=os.path.join(save_dir, f"{title_prefix}_Lasso_ActivationHeatmap.pdf"),
            log_x=True, reverse_x=True, is_omp=False
        )

        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        alpha_for_cv  = cv_exact_model['alpha']  if cv_exact_model  is not None else None
        alpha_for_aic = aic_exact_model['alpha'] if aic_exact_model is not None else None
        alpha_for_bic = bic_exact_model['alpha'] if bic_exact_model is not None else None

        # Row 0: CV NMSE (mean ± std) and Norm. L1
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

        axs[0].set_ylabel('CV\nNMSE', fontsize=PLOT_LABEL_FONTSIZE, color=plasma(0.25), labelpad=-20)
        ax0_twin.set_ylabel('Norm. L1', fontsize=PLOT_LABEL_FONTSIZE, color=plasma(0.65), labelpad=-10)

        ax0_twin.plot(df_path['alpha'], df_path['norm_l1'], 's-', ms=3, color=plasma(0.65), alpha=0.6)
        if pd.notna(alpha_for_cv):
            axs[0].axvline(alpha_for_cv, color=plasma(0.25), ls='--', lw=1.5)
            axs[0].text(alpha_for_cv * 1.5, 0.5, f"$\\lambda_L={alpha_for_cv:.1e}$",
                        rotation=90, va='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.25))

        axs[0].set_ylim(-0.05, 1.05); ax0_twin.set_ylim(-0.05, 1.05)
        axs[0].set_yticks([0.0, 1.0]); ax0_twin.set_yticks([0.0, 1.0])
        ax0_twin.tick_params(axis='y', labelcolor=plasma(0.65))
        axs[0].tick_params(axis='y', labelcolor=plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_path['alpha'], df_path['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_aic):
            axs[1].axvline(alpha_for_aic, color=plasma(0.55), ls='-', lw=1.5)
            axs[1].text(alpha_for_aic * 1.5, np.mean(axs[1].get_ylim()), f"$\\lambda_L={alpha_for_aic:.1e}$",
                        rotation=90, va='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_path['alpha'], df_path['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_bic):
            axs[2].axvline(alpha_for_bic, color=plasma(0.85), ls='-.', lw=1.5)
            axs[2].text(alpha_for_bic * 1.5, np.mean(axs[2].get_ylim()), f"$\\lambda_L={alpha_for_bic:.1e}$",
                        rotation=90, va='center', fontsize=PLOT_LEGEND_FONTSIZE-1, color=plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])

        axs[2].set_xlabel('Regularization Strength', fontsize=PLOT_LABEL_FONTSIZE)
        x_min = df_path['alpha'].min(); x_max = df_path['alpha'].max()
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

    path_data_for_criteria = []
    if not df_path.empty:
        for _, row in df_path.iterrows():
            beta_pos = _clip_nonneg(row['coeffs'])
            y_pred_unweighted = X_unscaled @ beta_pos
            mse_unweighted = mean_squared_error(y, y_pred_unweighted)
            path_data_for_criteria.append({
                'alpha': row['alpha'],
                'k': row['k'],
                'mse_train': row['mse_train'],
                'mse_train_unweighted': mse_unweighted,
                'aic': row['aic'],
                'bic': row['bic'],
                'coeffs': beta_pos
            })

    return results, path_data_for_criteria


def run_lars_analysis(X_unscaled, y, feature_names, cv_folds, title_prefix, save_dir, sample_weight=None):
    """
    LARS (positive=True) with a unified path used consistently for CV, AIC, and BIC.
    Enforces nonnegative coefficients end-to-end (path, CV refits, full refits).
    Returns ONLY a dict (results) to match the caller's expectation.
    """
    print(f"\n--- Running LARS Analysis (CV, AIC, BIC; unified path, POSITIVE) ({title_prefix}) ---")
    n_samples, n_features = X_unscaled.shape
    base = {'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
            'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan,
            'optimal_k': np.nan, 'k': 0, 'method': None}
    results = {
        'LarsCV':  {**base, 'method': 'LarsCV'},
        'LarsAIC': {**base, 'method': 'LarsAIC'},
        'LarsBIC': {**base, 'method': 'LarsBIC'}
    }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # ---------- helpers ----------
    def _clip_nonneg(beta, tol=1e-12):
        b = np.array(beta, float, copy=True)
        b[b < tol] = 0.0
        return b

    def _safe_weighted_mse(y_true, y_pred, w=None):
        if w is None:
            return mean_squared_error(y_true, y_pred)
        resid2 = (y_true - y_pred) ** 2
        w = np.asarray(w).reshape(-1)
        w_sum = np.sum(w)
        if w_sum <= EPS:
            return np.mean(resid2)
        return float(np.sum(w * resid2) / w_sum)

    def _fit_nnls_with_intercept(X, y, w=None):
        """
        Solve: minimize || W^(1/2) * (y - (b0 + X @ b)) ||_2  s.t. b >= 0
        by centering with (weighted) means and calling NNLS on centered/scaled data.
        Returns (b0, b>=0).
        """
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        n, p = X.shape
        if w is not None:
            sw = np.sqrt(np.clip(np.asarray(w).ravel(), EPS, None))
            Xw = X * sw[:, None]
            yw = y * sw
            # weighted means (with weights w) == ordinary means after multiplying by sqrt(w)
            mu_x = (Xw.sum(axis=0) / np.clip(sw.sum(), EPS, None))
            mu_y = (yw.sum()      / np.clip(sw.sum(), EPS, None))
            Xc = Xw - mu_x[None, :]
            yc = yw - mu_y
        else:
            mu_x = X.mean(axis=0)
            mu_y = y.mean()
            Xc = X - mu_x[None, :]
            yc = y - mu_y

        # Try SciPy NNLS first
        beta = None
        try:
            from scipy.optimize import nnls
            beta, _ = nnls(Xc, yc)
        except Exception:
            pass

        # Fallback: simple projected gradient steps to enforce nonnegativity
        if beta is None:
            beta = np.zeros(Xc.shape[1], float)
            lr = 1.0 / max(np.linalg.norm(Xc, ord=2)**2, 1e-6)
            for _ in range(1000):
                grad = -(Xc.T @ (yc - Xc @ beta))
                beta -= lr * grad
                beta = np.maximum(beta, 0.0)

        # Recover intercept consistent with the centering we used
        b0 = float(mu_y - mu_x @ beta)
        return b0, beta

    def _annotate_iter(ax, k_val, color):
        y0, y1 = ax.get_ylim()
        y_mid = 0.5 * (y0 + y1)
        x0, x1 = ax.get_xlim()
        dx = max(0.02 * (x1 - x0), 0.1)
        ax.text(k_val + dx, y_mid, f"Iteration = {int(k_val)}",
                rotation=90, va='center', ha='left',
                fontsize=PLOT_LEGEND_FONTSIZE-1, color=color)

    # ---------- 1) unified positive LARS path on full data ----------
    start_time_path = datetime.datetime.now()
    try:
        from sklearn.linear_model import lars_path
        _, _, coefs_raw = lars_path(X_unscaled, y, method='lars', positive=True) 
        clean_cols, steps_k = [], []
        last_k = 0
        for j in range(1, coefs_raw.shape[1]):  # skip the initial all-zero column
            b = _clip_nonneg(coefs_raw[:, j])   # <— ensure pure nonnegativity on the path
            k_now = int(np.sum(b > 1e-6))
            if k_now > last_k:
                clean_cols.append(b)
                steps_k.append(k_now)
                last_k = k_now
        if len(clean_cols) == 0:
            print("WARN: LARS path is empty after cleaning.")
            return results
        coefs_path = np.column_stack(clean_cols)  # (p, n_steps), all >= 0
        steps_k = np.array(steps_k, dtype=int)
    except Exception as e:
        print(f"FATAL ERROR building LARS path: {e}")
        traceback.print_exc()
        return results

    supports = [np.flatnonzero(coefs_path[:, i] > 1e-6) for i in range(coefs_path.shape[1])]

    # ---------- 2) AIC/BIC and train MSE on full data (NNLS refit with intercept) ----------
    path_rows = []
    for i, S in enumerate(supports):
        beta_i = np.zeros(n_features)
        if len(S) == 0:
            # constant-only model (intercept = mean(y) under weights)
            if sample_weight is None:
                b0 = float(np.mean(y))
            else:
                w = np.asarray(sample_weight).ravel()
                b0 = float(np.sum(w * y) / np.clip(np.sum(w), EPS, None))
            y_hat = np.full_like(y, b0)
        else:
            b0, beta_S = _fit_nnls_with_intercept(X_unscaled[:, S], y, sample_weight)
            beta_i[S] = beta_S
            y_hat = b0 + X_unscaled[:, S] @ beta_S
        mse_tr = _safe_weighted_mse(y, y_hat, sample_weight)
        # n_params: k + 1 intercept if k>0; if k==0, count 1 parameter (intercept)
        k_eff = len(S) + 1
        aic_i, bic_i = calculate_aic_bic(y, y_hat, k_eff, n_samples)
        path_rows.append({'k': int(steps_k[i]), 'coeffs': beta_i,
                          'mse_train': mse_tr, 'aic': aic_i, 'bic': bic_i})
    df_path = pd.DataFrame(path_rows).drop_duplicates(subset='k').sort_values('k').reset_index(drop=True)

    # ---------- 3) CV on the SAME supports (NNLS refit with intercept in each fold) ----------
    print("  Cross-validating on unified LARS supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_mse_mean, cv_mse_std = [], []
    for i, S in enumerate(supports):
        fold_mse = []
        for tr_idx, va_idx in kf.split(X_unscaled):
            Xtr, Xva = X_unscaled[tr_idx][:, S], X_unscaled[va_idx][:, S]
            ytr, yva = y[tr_idx], y[va_idx]
            wtr = sample_weight[tr_idx] if sample_weight is not None else None
            wva = sample_weight[va_idx] if sample_weight is not None else None
            try:
                if len(S) == 0:
                    # intercept-only in fold
                    if wtr is None:
                        b0 = float(np.mean(ytr))
                    else:
                        sw = np.asarray(wtr).ravel()
                        b0 = float(np.sum(sw * ytr) / np.clip(np.sum(sw), EPS, None))
                    yhat_va = np.full_like(yva, b0)
                else:
                    b0, beta_S = _fit_nnls_with_intercept(Xtr, ytr, wtr)
                    yhat_va = b0 + Xva @ beta_S
                mse_va = _safe_weighted_mse(yva, yhat_va, wva)
            except Exception:
                mse_va = np.inf
            fold_mse.append(mse_va)
        fold_mse = np.array(fold_mse, dtype=float)
        cv_mse_mean.append(np.nanmean(fold_mse))
        cv_mse_std.append(np.nanstd(fold_mse))
    cv_mse_mean = np.array(cv_mse_mean); cv_mse_std = np.array(cv_mse_std)

    # robust min–max normalization of CV MSE to [0,1]
    mm_min = np.nanmin(cv_mse_mean); mm_max = np.nanmax(cv_mse_mean)
    scale = (mm_max - mm_min)
    if not np.isfinite(scale) or scale < EPS:
        cv_nmse_mean = np.zeros_like(cv_mse_mean)
        cv_nmse_std = np.zeros_like(cv_mse_std)
    else:
        cv_nmse_mean = (cv_mse_mean - mm_min) / scale
        cv_nmse_std = cv_mse_std / scale

    best_idx = int(np.nanargmin(cv_mse_mean))
    best_k = int(steps_k[best_idx])

    # ---------- 4) results (coeffs already nonnegative) ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()

    row_cv = df_path.iloc[best_idx]
    results['LarsCV'].update({
        'coeffs': row_cv['coeffs'], 'n_nonzero': int(row_cv['k']),
        'aic': row_cv['aic'], 'bic': row_cv['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_cv['coeffs'][j] > 1e-6 ],
        'time': duration_cv, 'optimal_k': best_k, 'k': best_k
    })
    idx_aic = int(df_path['aic'].values.argmin())
    idx_bic = int(df_path['bic'].values.argmin())
    row_aic = df_path.iloc[idx_aic]; row_bic = df_path.iloc[idx_bic]
    results['LarsAIC'].update({
        'coeffs': row_aic['coeffs'], 'n_nonzero': int(row_aic['k']),
        'aic': row_aic['aic'], 'bic': row_aic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_aic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_aic['k']), 'k': int(row_aic['k'])
    })
    results['LarsBIC'].update({
        'coeffs': row_bic['coeffs'], 'n_nonzero': int(row_bic['k']),
        'aic': row_bic['aic'], 'bic': row_bic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_bic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_bic['k']), 'k': int(row_bic['k'])
    })

    # ---------- 5) plots ----------
    try:
        chosen = {
            'CV':  {'index': best_idx, 'label_val': best_k,               'color': plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': idx_aic,  'label_val': int(row_aic['k']),    'color': plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': idx_bic,  'label_val': int(row_bic['k']),    'color': plasma(0.85), 'linestyle': '-.'}
        }
        # coefs_path already clipped to >=0
        plot_activation_heatmap(
            steps_k, coefs_path, feature_names, chosen,
            title="LARS Activation Path (positive)", x_axis_label="Iteration (k)",
            filename=os.path.join(save_dir, f"{title_prefix}_LARS_ActivationHeatmap.pdf"),
            log_x=False, reverse_x=False, is_omp=False
        )
    except Exception as e:
        print(f"WARN: LARS heatmap plotting failed: {e}")

    try:
        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)
        # Row 0: CV NMSE mean ± std
        axs[0].plot(steps_k, cv_nmse_mean, 'o-', ms=3, lw=1.5, color=plasma(0.25))
        try:
            axs[0].fill_between(steps_k, cv_nmse_mean - cv_nmse_std, cv_nmse_mean + cv_nmse_std,
                                alpha=0.15, linewidth=0, color=plasma(0.25))
        except Exception:
            pass
        axs[0].set_ylabel('CV NMSE', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-10, color=plasma(0.25))
        axs[0].set_ylim(-0.05, 1.05); axs[0].set_yticks([0.0, 1.0])
        axs[0].axvline(best_k, color=plasma(0.25), ls='--', lw=1.5)
        _annotate_iter(axs[0], best_k, plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_path['k'], df_path['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[1].axvline(int(row_aic['k']), color=plasma(0.55), ls='-', lw=1.5)
        _annotate_iter(axs[1], int(row_aic['k']), plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_path['k'], df_path['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[2].axvline(int(row_bic['k']), color=plasma(0.85), ls='-.', lw=1.5)
        _annotate_iter(axs[2], int(row_bic['k']), plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])

        # X ticks min–max
        k_min, k_max = int(df_path['k'].min()), int(df_path['k'].max())
        for ax in axs:
            ax.set_xlim(k_min - 0.5, k_max + 0.5)
            ax.set_xticks([k_min, k_max])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        axs[2].set_xlabel('LARS Iteration', fontsize=PLOT_LABEL_FONTSIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_LARS_Criteria.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"WARN: LARS plotting failed: {e}")

    return results

def run_omp_analysis(X_unscaled, y, feature_names, cv_folds, title_prefix, save_dir, sample_weight=None):
    """
    OMP with a unified path used consistently for CV, AIC, and BIC.
    NOW ENFORCES NONNEGATIVE COEFFICIENTS end-to-end (path, CV refits, full refits).
    Returns ONLY a dict (results) to match the caller's expectation.
    """
    print(f"\n--- Running OMP Analysis (CV, AIC, BIC; unified path, POSITIVE) ({title_prefix}) ---")
    n_samples, n_features = X_unscaled.shape
    base = {'coeffs': np.zeros(n_features), 'alpha': np.nan, 'n_nonzero': 0,
            'aic': np.nan, 'bic': np.nan, 'selected_features': [], 'time': np.nan,
            'optimal_k': np.nan, 'k': 0, 'method': None}
    results = {
        'OMPCV':  {**base, 'method': 'OMPCV'},
        'OMPAIC': {**base, 'method': 'OMPAIC'},
        'OMPBIC': {**base, 'method': 'OMPBIC'}
    }
    if n_features == 0:
        print("ERROR: No features.")
        return results

    # ---------- helpers ----------
    def _clip_nonneg(beta, tol=1e-12):
        b = np.array(beta, float, copy=True)
        b[b < tol] = 0.0
        return b

    def _safe_weighted_mse(y_true, y_pred, w=None):
        if w is None:
            return mean_squared_error(y_true, y_pred)
        resid2 = (y_true - y_pred) ** 2
        w = np.asarray(w).reshape(-1)
        w_sum = np.sum(w)
        if w_sum <= EPS:
            return np.mean(resid2)
        return float(np.sum(w * resid2) / w_sum)

    def _fit_nnls_with_intercept(X, y, w=None):
        """
        Solve min ||W^(1/2) (y - (b0 + X b))||_2 subject to b >= 0.
        Weighted by w if provided. Uses SciPy nnls if available,
        otherwise falls back to projected gradient (nonnegative) on centered data.
        Returns (b0, b>=0).
        """
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        if X.size == 0:  # no features -> intercept-only model
            if w is None:
                return float(np.mean(y)), np.zeros(0)
            wv = np.asarray(w).ravel()
            return float(np.sum(wv * y) / np.clip(np.sum(wv), EPS, None)), np.zeros(0)

        if w is not None:
            sw = np.sqrt(np.clip(np.asarray(w).ravel(), EPS, None))
            Xw = X * sw[:, None]
            yw = y * sw
            mu_x = (Xw.sum(axis=0) / np.clip(sw.sum(), EPS, None))
            mu_y = (yw.sum()      / np.clip(sw.sum(), EPS, None))
            Xc = Xw - mu_x[None, :]
            yc = yw - mu_y
        else:
            mu_x = X.mean(axis=0)
            mu_y = y.mean()
            Xc = X - mu_x[None, :]
            yc = y - mu_y

        beta = None
        try:
            from scipy.optimize import nnls
            beta, _ = nnls(Xc, yc)
        except Exception:
            pass

        if beta is None:  # projected gradient fallback
            beta = np.zeros(Xc.shape[1], float)
            L = max(np.linalg.norm(Xc, ord=2)**2, 1e-6)
            lr = 1.0 / L
            for _ in range(1000):
                grad = -(Xc.T @ (yc - Xc @ beta))
                beta -= lr * grad
                beta = np.maximum(beta, 0.0)

        b0 = float(mu_y - mu_x @ beta)
        return b0, beta

    def _annotate_iter(ax, k_val, color):
        y0, y1 = ax.get_ylim(); y_mid = 0.5*(y0+y1)
        x0, x1 = ax.get_xlim(); dx = max(0.02*(x1-x0), 0.1)
        ax.text(k_val + dx, y_mid, f"Iteration = {int(k_val)}",
                rotation=90, va='center', ha='left',
                fontsize=PLOT_LEGEND_FONTSIZE-1, color=color)

    # ---------- 1) unified OMP path on full data (then clipped to >=0) ----------
    start_time_path = datetime.datetime.now()
    try:
        Kmax = min(n_features, n_samples - 1) if n_samples > 1 else n_features
        coef_cols, steps_k = [], []
        last_k = 0
        for k in range(1, Kmax + 1):
            b_raw = orthogonal_mp(X_unscaled, y, n_nonzero_coefs=k, precompute=True)
            b_pos = _clip_nonneg(b_raw)  # enforce positivity on path
            k_now = int(np.sum(b_pos > 1e-6))
            if k_now > last_k:
                coef_cols.append(b_pos)
                steps_k.append(k_now)
                last_k = k_now
        if len(coef_cols) == 0:
            print("WARN: OMP path is empty.")
            return results
        coefs_path = np.column_stack(coef_cols)  # (p, n_steps), all >= 0
        steps_k = np.array(steps_k, dtype=int)
    except Exception as e:
        print(f"FATAL ERROR building OMP path: {e}")
        traceback.print_exc()
        return results

    supports = [np.flatnonzero(coefs_path[:, i] > 1e-6) for i in range(coefs_path.shape[1])]

    # ---------- 2) AIC/BIC and train MSE on full data (NNLS refit with intercept) ----------
    path_rows = []
    for i, S in enumerate(supports):
        beta_i = np.zeros(n_features)
        if len(S) == 0:
            # intercept-only
            if sample_weight is None:
                b0 = float(np.mean(y))
            else:
                w = np.asarray(sample_weight).ravel()
                b0 = float(np.sum(w * y) / np.clip(np.sum(w), EPS, None))
            y_hat = np.full_like(y, b0)
        else:
            b0, beta_S = _fit_nnls_with_intercept(X_unscaled[:, S], y, sample_weight)
            beta_i[S] = beta_S
            y_hat = b0 + X_unscaled[:, S] @ beta_S
        mse_tr = _safe_weighted_mse(y, y_hat, sample_weight)
        k_eff = len(S) + 1  # include intercept
        aic_i, bic_i = calculate_aic_bic(y, y_hat, k_eff, n_samples)
        path_rows.append({'k': int(steps_k[i]), 'coeffs': beta_i,
                          'mse_train': mse_tr, 'aic': aic_i, 'bic': bic_i})
    df_path = pd.DataFrame(path_rows).drop_duplicates(subset='k').sort_values('k').reset_index(drop=True)

    # ---------- 3) CV on the SAME supports (NNLS refit with intercept in each fold) ----------
    print("  Cross-validating on unified OMP supports…")
    start_time_cv = datetime.datetime.now()
    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
    cv_mse_mean, cv_mse_std = [], []
    for i, S in enumerate(supports):
        fold_mse = []
        for tr_idx, va_idx in kf.split(X_unscaled):
            Xtr, Xva = X_unscaled[tr_idx][:, S], X_unscaled[va_idx][:, S]
            ytr, yva = y[tr_idx], y[va_idx]
            wtr = sample_weight[tr_idx] if sample_weight is not None else None
            wva = sample_weight[va_idx] if sample_weight is not None else None
            try:
                if len(S) == 0:
                    # intercept-only in fold
                    if wtr is None:
                        b0 = float(np.mean(ytr))
                    else:
                        sw = np.asarray(wtr).ravel()
                        b0 = float(np.sum(sw * ytr) / np.clip(np.sum(sw), EPS, None))
                    yhat_va = np.full_like(yva, b0)
                else:
                    b0, beta_S = _fit_nnls_with_intercept(Xtr, ytr, wtr)
                    yhat_va = b0 + Xva @ beta_S
                mse_va = _safe_weighted_mse(yva, yhat_va, wva)
            except Exception:
                mse_va = np.inf
            fold_mse.append(mse_va)
        fold_mse = np.array(fold_mse, dtype=float)
        cv_mse_mean.append(np.nanmean(fold_mse))
        cv_mse_std.append(np.nanstd(fold_mse))
    cv_mse_mean = np.array(cv_mse_mean); cv_mse_std = np.array(cv_mse_std)

    # robust min–max normalization to [0,1]
    mm_min = np.nanmin(cv_mse_mean); mm_max = np.nanmax(cv_mse_mean)
    scale = (mm_max - mm_min)
    if not np.isfinite(scale) or scale < EPS:
        cv_nmse_mean = np.zeros_like(cv_mse_mean)
        cv_nmse_std = np.zeros_like(cv_mse_std)
    else:
        cv_nmse_mean = (cv_mse_mean - mm_min) / scale
        cv_nmse_std = cv_mse_std / scale

    best_idx = int(np.nanargmin(cv_mse_mean))
    best_k = int(steps_k[best_idx])

    # ---------- 4) results (coeffs already nonnegative) ----------
    duration_cv = (datetime.datetime.now() - start_time_cv).total_seconds()
    duration_path = (datetime.datetime.now() - start_time_path).total_seconds()

    row_cv = df_path.iloc[best_idx]
    results['OMPCV'].update({
        'coeffs': row_cv['coeffs'], 'n_nonzero': int(row_cv['k']),
        'aic': row_cv['aic'], 'bic': row_cv['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_cv['coeffs'][j] > 1e-6 ],
        'time': duration_cv, 'optimal_k': best_k, 'k': best_k
    })
    idx_aic = int(df_path['aic'].values.argmin())
    idx_bic = int(df_path['bic'].values.argmin())
    row_aic = df_path.iloc[idx_aic]; row_bic = df_path.iloc[idx_bic]
    results['OMPAIC'].update({
        'coeffs': row_aic['coeffs'], 'n_nonzero': int(row_aic['k']),
        'aic': row_aic['aic'], 'bic': row_aic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_aic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_aic['k']), 'k': int(row_aic['k'])
    })
    results['OMPBIC'].update({
        'coeffs': row_bic['coeffs'], 'n_nonzero': int(row_bic['k']),
        'aic': row_bic['aic'], 'bic': row_bic['bic'],
        'selected_features': [ (fn['term_label'] if isinstance(fn, dict) and 'term_label' in fn else str(fn))
                               for j, fn in enumerate(feature_names) if row_bic['coeffs'][j] > 1e-6 ],
        'time': duration_path, 'optimal_k': int(row_bic['k']), 'k': int(row_bic['k'])
    })

    # ---------- 5) plots ----------
    try:
        chosen = {
            'CV':  {'index': best_idx, 'label_val': best_k,               'color': plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': idx_aic,  'label_val': int(row_aic['k']),    'color': plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': idx_bic,  'label_val': int(row_bic['k']),    'color': plasma(0.85), 'linestyle': '-.'}
        }
        plot_activation_heatmap(
            steps_k, coefs_path, feature_names, chosen,
            title="OMP Activation Path (positive)", x_axis_label="Iteration (k)",
            filename=os.path.join(save_dir, f"{title_prefix}_OMP_ActivationHeatmap.pdf"),
            log_x=False, reverse_x=False, is_omp=True
        )
    except Exception as e:
        print(f"WARN: OMP heatmap plotting failed: {e}")

    try:
        fig, axs = plt.subplots(3, 1, figsize=SMALL_FIG_LASSO_LARS_OMP, sharex=True)
        # Row 0: CV NMSE mean ± std
        axs[0].plot(steps_k, cv_nmse_mean, 'o-', ms=3, lw=1.5, color=plasma(0.25))
        try:
            axs[0].fill_between(steps_k, cv_nmse_mean - cv_nmse_std, cv_nmse_mean + cv_nmse_std,
                                alpha=0.15, linewidth=0, color=plasma(0.25))
        except Exception:
            pass
        axs[0].set_ylabel('CV NMSE', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-10, color=plasma(0.25))
        axs[0].set_ylim(-0.05, 1.05); axs[0].set_yticks([0.0, 1.0])
        axs[0].axvline(best_k, color=plasma(0.25), ls='--', lw=1.5)
        _annotate_iter(axs[0], best_k, plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_path['k'], df_path['aic'], 'o-', ms=3, color=plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[1].axvline(int(row_aic['k']), color=plasma(0.55), ls='-', lw=1.5)
        _annotate_iter(axs[1], int(row_aic['k']), plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_path['k'], df_path['bic'], 'o-', ms=3, color=plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[2].axvline(int(row_bic['k']), color=plasma(0.85), ls='-.', lw=1.5)
        _annotate_iter(axs[2], int(row_bic['k']), plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])

        k_min, k_max = int(df_path['k'].min()), int(df_path['k'].max())
        for ax in axs:
            ax.set_xlim(k_min - 0.5, k_max + 0.5)
            ax.set_xticks([k_min, k_max])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        axs[2].set_xlabel('OMP Iteration', fontsize=PLOT_LABEL_FONTSIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f"{title_prefix}_OMP_Criteria.pdf"),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f"WARN: OMP plotting failed: {e}")

    return results

def build_equal_per_mode_weights(mode_labels, eps=1e-12):
    """
    Equalizes contribution per deformation mode:
      w_i = 1 / count(mode_i), normalized so mean(w)=1.
    """
    counts = {}
    for m in mode_labels:
        counts[m] = counts.get(m, 0) + 1
    w = np.array([1.0 / max(counts[m], eps) for m in mode_labels], dtype=float)
    s = np.sum(w)
    if s > 0:
        w *= (len(w) / s)
    return w

def prepare_weighted_normalized_design(X, y, w, eps=1e-12):
    """
    Row-weight X,y by sqrt(w), then column-normalize X by weighted L2 norm.
    Returns:
      Xw_norm, yw, col_scales
    """
    sw = np.asarray(w, dtype=float)
    sw_sqrt = np.sqrt(np.clip(sw, 0.0, np.inf))
    Xw = X * sw_sqrt[:, None]
    yw = y * sw_sqrt
    col_scales = np.sqrt(np.maximum((Xw**2).sum(axis=0), eps))
    Xw_norm = Xw / col_scales[None, :]
    return Xw_norm, yw, col_scales

# --- Main Loop ---
overall_start_time = datetime.datetime.now()
print(f"\nStarting Anisotropic Model Discovery at: {overall_start_time.strftime('%Y-%m-%d %H:%M:%S')}")
os.makedirs(BASE_SAVE_DIR, exist_ok=True)

all_results_by_noise = {} 

# --- 3. Construct Feature Matrix X (Stage 1) and Target y (Original Data) ---
num_data_points_orig = len(experimental_data_list)
X_features_stage1 = np.zeros((num_data_points_orig, num_psi_terms))
y_target_orig = np.zeros(num_data_points_orig)
data_point_test_names_orig = [] 
valid_rows = np.ones(num_data_points_orig, dtype=bool) # Initialize valid_rows before use
experimental_data_list_filtered = experimental_data_list # Initialize before potential filtering
n_samples_clean = num_data_points_orig # Initialize before potential filtering

print(f"Building Stage 1 feature matrix (fixed w1=1.0) using original data...")
for i, data_point in enumerate(experimental_data_list):
    y_target_orig[i] = data_point['target_stress']
    data_point_test_names_orig.append(data_point['test_name']) 
    invariants = get_invariants(data_point['F'])
    for j, term_def in enumerate(psi_terms_definitions): 
        psi_derivs = get_psi_deriv_for_term(term_def, invariants, w1=1.0) 
        X_features_stage1[i, j] = calculate_stress_contribution_linearized(data_point, psi_derivs, term_def)


valid_rows = ~np.isnan(X_features_stage1).any(axis=1) & ~np.isinf(X_features_stage1).any(axis=1) & ~np.isnan(y_target_orig) & ~np.isinf(y_target_orig)
if np.sum(~valid_rows) > 0:
    print(f"Warning: Removing {np.sum(~valid_rows)} rows with NaN/inf values from original X_features/y_target.")
    X_features_stage1 = X_features_stage1[valid_rows]
    y_target_original_clean = y_target_orig[valid_rows] # Store the cleaned original target
    data_point_test_names = [data_point_test_names_orig[i] for i, v_row in enumerate(valid_rows) if v_row]
    experimental_data_list_filtered = [dp for i, dp in enumerate(experimental_data_list) if valid_rows[i]]
else:
    y_target_original_clean = y_target_orig
    data_point_test_names = data_point_test_names_orig
    experimental_data_list_filtered = experimental_data_list

n_samples_clean = X_features_stage1.shape[0]
if n_samples_clean == 0:
    exit("CRITICAL ERROR: No valid data points remaining after cleaning feature matrix. Cannot proceed.")
print(f"Stage 1 Feature matrix built. Cleaned data points: {n_samples_clean}")

# Precompute invariants for the final filtered data list
all_invariants_filtered = [get_invariants(dp['F']) for dp in experimental_data_list_filtered]
# Calculate PDF predictions using the final filtered invariants
pdf_model_predictions = calculate_predictions(pdf_model_coeffs_w2, pdf_model_w1_map, all_invariants_filtered, experimental_data_list_filtered)

for noise_level in RELATIVE_NOISE_LEVELS:
    noise_prefix = f"RelNoise{noise_level*100:.0f}pct"
    print("\n" + "#"*80)
    print(f"### PROCESSING NOISE LEVEL: {noise_prefix} ###")
    print("#"*80)

    # Apply noise to the CLEANED original target data
    if noise_level == 0.0:
        y_target_current = y_target_original_clean.copy()
    else:
        noise_scale = noise_level * np.abs(y_target_original_clean)
        noise = np.random.normal(0, noise_scale, size=y_target_original_clean.shape)
        noise[np.abs(y_target_original_clean) < WEIGHTING_EPSILON] = 0.0 
        y_target_current = y_target_original_clean + noise

    # Update target in filtered list (for Stage 2 objective)
    for i, dp in enumerate(experimental_data_list_filtered):
        if i < len(y_target_current):
            dp['target_stress'] = y_target_current[i]

    current_save_dir = os.path.join(BASE_SAVE_DIR, noise_prefix)
    os.makedirs(current_save_dir, exist_ok=True)

    stress_based_weights = build_stress_magnitude_weights(experimental_data_list_filtered)
    sample_weights_current = stress_based_weights.copy()
    print(f"[{noise_prefix}] Using inverse stress magnitude sample weights.")
    
    # Build weighted-normalized design used by ALL Stage-1 methods
    X_stage1_weighted, y_stage1_weighted, col_scales = prepare_weighted_normalized_design(
        X_features_stage1, y_target_current, sample_weights_current
    )

    # --- Run Stage 1 Analysis on weighted-normalized design (no sample_weight) ---
    scenario_initial_results = {}
    model_selection_results_path = []
    analysis_functions = {'Lasso': run_lasso_analysis, 'LARS': run_lars_analysis, 'OMP': run_omp_analysis}

    for method_base_name, analysis_func in analysis_functions.items():
        results_dict = None
        print("\n" + "="*30 + f" Running {method_base_name} Variants for {noise_prefix} " + "="*30)
        try:
            if method_base_name == 'Lasso':
                # Lasso returns results + path data
                results_dict, lasso_path_data = analysis_func(
                    X_stage1_weighted, y_stage1_weighted, psi_terms_definitions, CV_FOLDS,
                    noise_prefix, current_save_dir, sample_weight=None
                )
                model_selection_results_path = lasso_path_data
            else:
                results_dict = analysis_func(
                    X_stage1_weighted, y_stage1_weighted, psi_terms_definitions, CV_FOLDS,
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
            print(f"FATAL ERROR during {method_base_name} Analysis block: {outer_e}")
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
                    print(f"  INFO: Applying NNLS pre-refinement for {method_key} coefficients.")
                    selected_feature_indices_stage1 = np.where(np.abs(current_coeffs_for_refinement) > 1e-9)[0]
                    if len(selected_feature_indices_stage1) > 0:
                        # Use ORIGINAL (unweighted) X for NNLS prior to Stage 2
                        X_selected_for_nnls = X_features_stage1[:, selected_feature_indices_stage1]
                        try:
                            nnls_model = LinearRegression(positive=True, fit_intercept=False)
                            nnls_model.fit(X_selected_for_nnls, y_target_current, sample_weight=sample_weights_current)
                            nnls_coeffs_full_vector = np.zeros_like(current_coeffs_for_refinement)
                            nnls_coeffs_full_vector[selected_feature_indices_stage1] = nnls_model.coef_
                            nnls_coeffs_full_vector = np.maximum(nnls_coeffs_full_vector, 0)
                            current_coeffs_for_refinement = nnls_coeffs_full_vector
                            n_positive_after_nnls = np.sum(current_coeffs_for_refinement > 1e-6)
                            print(f"    NNLS for {method_key}: {len(selected_feature_indices_stage1)} -> {n_positive_after_nnls} positive coefficients.")
                        except Exception as e:
                            print(f"    WARNING: NNLS pre-refinement failed for {method_key}: {e}. Falling back to clamping.")
                            current_coeffs_for_refinement = np.maximum(current_coeffs_for_refinement, 0)
                    else:
                        print(f"    No features originally selected by {method_key} in Stage 1. Coefficients will be zero for refinement.")
                        current_coeffs_for_refinement = np.zeros_like(current_coeffs_for_refinement)

                temp_stage1_dict_for_refinement = stage1_result.copy()
                temp_stage1_dict_for_refinement['coeffs'] = current_coeffs_for_refinement

                refined_model = refine_model(
                    temp_stage1_dict_for_refinement,
                    f"{noise_prefix} {method_key}",
                    y_target_current,
                    sample_weights_current,   # weights used inside Stage 2 loss
                    all_invariants_filtered
                )
                noise_level_refined_results[method_key] = refined_model

                if refined_model.get('success', False):
                    generate_stress_strain_plot_refined(
                        refined_model['coeffs'], refined_model['w1_map'],
                        pdf_model_predictions, y_target_current,
                        f"{method_key}", save_dir=current_save_dir,
                        file_prefix=f"{noise_prefix}_{method_key}",
                        all_invariants_current=all_invariants_filtered
                    )
                else:
                    print(f"Skipping plot for {method_key} due to refinement failure or no terms selected.")
            elif 'error' in stage1_result:
                print(f"Skipping refinement for {method_key} due to Stage 1 error: {stage1_result['error']}")
                noise_level_refined_results[method_key] = {'success': False, 'message': f"Stage 1 Error: {stage1_result['error']}"}
            else:
                print(f"Skipping refinement for {method_key} due to missing coefficients in Stage 1 result.")
                noise_level_refined_results[method_key] = {'success': False, 'message': 'Missing coefficients in Stage 1 result'}
        else:
            print(f"No Stage 1 results found for {method_key} in scenario_initial_results.")
            noise_level_refined_results[method_key] = {'success': False, 'message': 'No Stage 1 results'}

    all_results_by_noise[noise_prefix] = {'initial': scenario_initial_results, 'refined': noise_level_refined_results}

# --- 8. Print Final Summary Tables ---
print("\n\n" + "="*80)
print("### FINAL DETAILED PERFORMANCE SUMMARY ACROSS NOISE LEVELS ###")
print("="*80)
summary_table_data = []

# Always evaluate overall metrics against the CLEAN target
y_true_eval_for_r2 = y_target_original_clean.copy()

# Calculate overall metrics for PDF model ONCE (using original y_target for reference)
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
temp_plot_series_pdf = {k: v.copy() for k, v in experimental_plot_series.items()} 
pred_map_pdf_only = {}
for i, dp_filtered in enumerate(experimental_data_list_filtered): 
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
    **{f"{mode} R2": pdf_metrics_per_mode.get(mode,{}).get('R2',np.nan) for mode in experimental_plot_series.keys()},
    **{f"{mode} RMSE": pdf_metrics_per_mode.get(mode,{}).get('RMSE',np.nan) for mode in experimental_plot_series.keys()}
})

# Add refined model metrics for each noise level
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
                all_invariants_filtered, experimental_data_list_filtered
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
            temp_plot_series_ref = {k: v.copy() for k, v in experimental_plot_series.items()} 
            pred_map_ref = {}
            for i, dp_filtered in enumerate(experimental_data_list_filtered): 
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
            **{f"{mode} R2": refined_metrics_per_mode.get(mode,{}).get('R2',np.nan) for mode in experimental_plot_series.keys()},
            **{f"{mode} RMSE": refined_metrics_per_mode.get(mode,{}).get('RMSE',np.nan) for mode in experimental_plot_series.keys()}
        })

# Create and print the final DataFrame
summary_df_final = pd.DataFrame(summary_table_data)
overall_cols = ['Noise Level', 'Method', 'Selection', 'N Terms', 'Overall R2', 'Overall RMSE']
mode_cols_r2 = sorted([f"{mode} R2" for mode in experimental_plot_series.keys()])
mode_cols_rmse = sorted([f"{mode} RMSE" for mode in experimental_plot_series.keys()])
final_cols = overall_cols + mode_cols_r2 + mode_cols_rmse
# Ensure all expected columns exist before reindexing
for col in final_cols:
    if col not in summary_df_final.columns:
        summary_df_final[col] = np.nan # Add missing columns with NaN
summary_df_final = summary_df_final[final_cols] # Use reindex to handle potentially missing columns

print("\n\n--- FINAL DETAILED PERFORMANCE SUMMARY TABLE ---")
with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 2000): 
     print(summary_df_final.to_string(index=False, float_format="%.3f", na_rep="N/A"))

print(f"\nCompleted all scenarios at: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Total analysis duration: {datetime.datetime.now() - overall_start_time}")
