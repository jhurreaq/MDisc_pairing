"""Load the original heart-tissue sheet and preserve its shear/biaxial column mapping."""

import numpy as np
import pandas as pd


def load_experimental_data(*, context):
    df_experimental = None
    try:
        df_temp = pd.read_excel(context.EXCEL_FILE_NAME, sheet_name=context.SHEET_NAME, header=None)
        if isinstance(df_temp, dict):
            df_experimental = df_temp.get(context.SHEET_NAME)
            if df_experimental is None and df_temp:
                first_sheet_key = list(df_temp.keys())[0]
                df_experimental = df_temp[first_sheet_key]
                df_experimental.attrs['sheet_name'] = first_sheet_key
                print(f"Warning: Sheet '{context.SHEET_NAME}' not found. Loaded sheet '{first_sheet_key}' instead.")
            elif df_experimental is not None:
                 df_experimental.attrs['sheet_name'] = context.SHEET_NAME
        else:
            df_experimental = df_temp
            df_experimental.attrs['sheet_name'] = context.SHEET_NAME
        if df_experimental is not None:
            print(f"Successfully loaded Excel: '{context.EXCEL_FILE_NAME}', Sheet: '{df_experimental.attrs['sheet_name']}'")
    except FileNotFoundError: exit(f"Error: '{context.EXCEL_FILE_NAME}' not found.")
    except ValueError as e:
        if "Worksheet named" in str(e) and f"'{context.SHEET_NAME}'" in str(e):
            print(f"Error: Worksheet named '{context.SHEET_NAME}' not found in '{context.EXCEL_FILE_NAME}'.")
            try:
                print(f'Attempting to load the first sheet...')
                xls = pd.ExcelFile(context.EXCEL_FILE_NAME); first_sheet_name_actual = xls.sheet_names[0]
                df_experimental = pd.read_excel(xls, sheet_name=first_sheet_name_actual, header=None)
                df_experimental.attrs['sheet_name'] = first_sheet_name_actual
                print(f"Successfully loaded the first sheet ('{first_sheet_name_actual}').")
            except Exception as e_fallback: exit(f'Could not load the first sheet either: {e_fallback}')
        else: exit(f'An error occurred while reading the Excel file: {e}')
    except Exception as e_generic: exit(f'A generic error occurred during file loading: {e_generic}')
    if df_experimental is None: exit("CRITICAL ERROR: Data loading failed.")
    shear_test_definitions = {
        'fs': {'P_col': 1, 'gamma_col': 0, 'plot_title': r'Shear $\sigma_{fs}$', 'stress_label': r'$\sigma_{fs}$', 'F_func': lambda g: np.array([[1,0,0],[g,1,0],[0,0,1]]), 'stress_to_calc': 'fs'},
        'fn': {'P_col': 3, 'gamma_col': 2, 'plot_title': r'Shear $\sigma_{fn}$', 'stress_label': r'$\sigma_{fn}$', 'F_func': lambda g: np.array([[1,0,0],[0,1,0],[g,0,1]]), 'stress_to_calc': 'fn'},
        'sf': {'P_col': 6, 'gamma_col': 5, 'plot_title': r'Shear $\sigma_{sf}$', 'stress_label': r'$\sigma_{sf}$', 'F_func': lambda g: np.array([[1,g,0],[0,1,0],[0,0,1]]), 'stress_to_calc': 'sf'},
        'sn': {'P_col': 8, 'gamma_col': 7, 'plot_title': r'Shear $\sigma_{sn}$', 'stress_label': r'$\sigma_{sn}$', 'F_func': lambda g: np.array([[1,0,0],[0,1,0],[0,g,1]]), 'stress_to_calc': 'sn'},
        'nf': {'P_col': 11, 'gamma_col': 10, 'plot_title': r'Shear $\sigma_{nf}$', 'stress_label': r'$\sigma_{nf}$', 'F_func': lambda g: np.array([[1,0,g],[0,1,0],[0,0,1]]), 'stress_to_calc': 'nf'},
        'ns': {'P_col': 13, 'gamma_col': 12, 'plot_title': r'Shear $\sigma_{ns}$', 'stress_label': r'$\sigma_{ns}$', 'F_func': lambda g: np.array([[1,0,0],[0,1,g],[0,0,1]]), 'stress_to_calc': 'ns'}
    }
    experimental_data_list = []
    context.experimental_plot_series = {}
    for mode, D in shear_test_definitions.items():
        gamma_vals_series = pd.to_numeric(df_experimental.iloc[context.DATA_START_ROW:, D['gamma_col']], errors='coerce')
        P_vals_series = pd.to_numeric(df_experimental.iloc[context.DATA_START_ROW:, D['P_col']], errors='coerce')
        valid_indices = gamma_vals_series.notna() & P_vals_series.notna()
        gamma_vals = gamma_vals_series[valid_indices]; P_vals = P_vals_series[valid_indices]
        if not gamma_vals.empty:
            context.experimental_plot_series[mode] = {'x': gamma_vals.values, 'y_exp': P_vals.values, 'type': 'shear','xlabel': r'Shear Strain $\gamma$ [-]', 'ylabel': f"Stress {D['stress_label']} [kPa]", 'title': D['plot_title']}
            for i in range(len(gamma_vals)):
                g = gamma_vals.iloc[i]; F = D['F_func'](g)
                experimental_data_list.append({'F': F, 'type': 'shear', 'target_stress': P_vals.iloc[i], 'test_name': mode, 'gamma_val': g, 'stress_to_calc': D['stress_to_calc'], 'original_x': g})
    biaxial_test_definitions = {
        '1:1':    {'MFs':15,'MFp':16,'CFs':17,'CFp':18, 'plot_title_f':r'Biax. $\sigma_{ff}$ (1:1)','plot_title_n':r'Biax. $\sigma_{nn}$ (1:1)'},
        '1:0.75': {'MFs':20,'MFp':21,'CFs':22,'CFp':23, 'plot_title_f':r'Biax. $\sigma_{ff}$ (1:0.75)','plot_title_n':r'Biax. $\sigma_{nn}$ (1:0.75)'},
        '0.75:1': {'MFs':25,'MFp':26,'CFs':27,'CFp':28, 'plot_title_f':r'Biax. $\sigma_{ff}$ (0.75:1)','plot_title_n':r'Biax. $\sigma_{nn}$ (0.75:1)'},
        '1:0.5':  {'MFs':30,'MFp':31,'CFs':32,'CFp':33, 'plot_title_f':r'Biax. $\sigma_{ff}$ (1:0.5)','plot_title_n':r'Biax. $\sigma_{nn}$ (1:0.5)'},
        '0.5:1':  {'MFs':35,'MFp':36,'CFs':37,'CFp':38, 'plot_title_f':r'Biax. $\sigma_{ff}$ (0.5:1)','plot_title_n':r'Biax. $\sigma_{nn}$ (0.5:1)'}
    }
    for ratio_name, D in biaxial_test_definitions.items():
        lambda_f_col_data = pd.to_numeric(df_experimental.iloc[context.DATA_START_ROW:, D['MFs']], errors='coerce')
        P_ff_col_data = pd.to_numeric(df_experimental.iloc[context.DATA_START_ROW:, D['MFp']], errors='coerce')
        lambda_n_col_data = pd.to_numeric(df_experimental.iloc[context.DATA_START_ROW:, D['CFs']], errors='coerce')
        P_nn_col_data = pd.to_numeric(df_experimental.iloc[context.DATA_START_ROW:, D['CFp']], errors='coerce')
        valid_ff = lambda_f_col_data.notna() & P_ff_col_data.notna() & lambda_n_col_data.notna()
        lf_vals_ff, ln_vals_ff, p_ff_vals = lambda_f_col_data[valid_ff], lambda_n_col_data[valid_ff], P_ff_col_data[valid_ff]
        plot_key_f = ratio_name + '_f'
        if not lf_vals_ff.empty:
            context.experimental_plot_series[plot_key_f] = {'x': lf_vals_ff.values, 'y_exp': p_ff_vals.values, 'type': 'biaxial_f','xlabel': r'Stretch $\lambda_f$ [-]', 'ylabel': r'Stress $\sigma_{ff}$ [kPa]', 'title': D['plot_title_f']}
            for i in range(len(lf_vals_ff)):
                lf, ln = lf_vals_ff.iloc[i], ln_vals_ff.iloc[i]
                if lf * ln == 0: continue
                ls = 1.0 / (lf * ln); F = np.diag([lf, ls, ln])
                experimental_data_list.append({'F': F, 'type': 'biaxial', 'target_stress': p_ff_vals.iloc[i], 'test_name': plot_key_f, 'lambda_f_val': lf, 'lambda_n_val': ln, 'stress_to_calc': 'ff', 'original_x': lf})
        valid_nn = lambda_n_col_data.notna() & P_nn_col_data.notna() & lambda_f_col_data.notna()
        ln_vals_nn, lf_vals_nn, p_nn_vals = lambda_n_col_data[valid_nn], lambda_f_col_data[valid_nn], P_nn_col_data[valid_nn]
        plot_key_n = ratio_name + '_n'
        if not ln_vals_nn.empty:
            context.experimental_plot_series[plot_key_n] = {'x': ln_vals_nn.values, 'y_exp': p_nn_vals.values, 'type': 'biaxial_n','xlabel': r'Stretch $\lambda_n$ [-]', 'ylabel': r'Stress $\sigma_{nn}$ [kPa]', 'title': D['plot_title_n']}
            for i in range(len(ln_vals_nn)):
                ln, lf = ln_vals_nn.iloc[i], lf_vals_nn.iloc[i]
                if lf * ln == 0: continue
                ls = 1.0 / (lf * ln); F = np.diag([lf, ls, ln])
                experimental_data_list.append({'F': F, 'type': 'biaxial', 'target_stress': p_nn_vals.iloc[i], 'test_name': plot_key_n, 'lambda_f_val': lf, 'lambda_n_val': ln, 'stress_to_calc': 'nn', 'original_x': ln})
    y_target_original = np.array([dp['target_stress'] for dp in experimental_data_list])
    return experimental_data_list, y_target_original
