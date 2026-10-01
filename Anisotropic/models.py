"""Cardiac models; original numerical and presentation choices."""

import numpy as np
import warnings


def get_invariants(F):
    C = F.T @ F; I1 = np.trace(C); I2 = 0.5 * (I1**2 - np.trace(C @ C))
    I4f, I4s, I4n = C[0,0], C[1,1], C[2,2]; I8fs, I8fn, I8sn = C[0,1], C[0,2], C[1,2]
    return {'I1':I1,'I2':I2,'I4f':I4f,'I4s':I4s,'I4n':I4n,'I4f_bar':max(I4f,1.0),'I4s_bar':max(I4s,1.0),
            'I4n_bar':max(I4n,1.0),'I8fs':I8fs,'I8fn':I8fn,'I8sn':I8sn}

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

def calculate_single_prediction_full_tensor(data_point, invariants, coeffs_w2, w1_map, *, context):
    """
    Calculates a single stress component for one data point using the full,
    physically correct stress tensor formulation. Used for Stage 2 refinement and final predictions.
    """
    F = data_point['F']
    f0, s0, n0 = np.array([1.,0.,0.]), np.array([0.,1.,0.]), np.array([0.,0.,1.])

    try:
        B = F @ F.T
        I1_val = invariants.get('I1', np.trace(B))
        # Use pushed-forward structural directions in the current configuration.
        mf = F @ f0
        ms = F @ s0
        mn = F @ n0
    except np.linalg.LinAlgError:
        return np.nan

    sigma_hyper_tensor = np.zeros((3, 3))

    for term_idx, term_def in enumerate(context.psi_terms_definitions):
        w2 = coeffs_w2[term_idx]
        if abs(w2) < context.EPS: continue

        # --- SIMPLE SOLUTION TO NULLIFY I4/I8 LINEAR TERMS ---
        inv_key = term_def['inv_key']
        term_type = term_def['type']
        if ('I4' in inv_key or 'I8' in inv_key) and term_type in ['lin', 'exp_lin']:
            continue # Skip this term's contribution entirely
        # --- END OF SIMPLE SOLUTION ---

        w1 = w1_map.get(term_idx, 1.0)
        psi_derivs = get_psi_deriv_for_term(term_def, invariants, w1)

        if 'I1' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I1'] * 2 * B
        if 'I2' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I2'] * 2 * (I1_val * B - B @ B)
        if 'I4f_bar' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I4f_bar'] * 2 * np.outer(mf, mf)
        if 'I4s_bar' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I4s_bar'] * 2 * np.outer(ms, ms)
        if 'I4n_bar' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I4n_bar'] * 2 * np.outer(mn, mn)
        if 'I8fs' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I8fs'] * (np.outer(mf, ms) + np.outer(ms, mf))
        if 'I8fn' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I8fn'] * (np.outer(mf, mn) + np.outer(mn, mf))
        if 'I8sn' in psi_derivs: sigma_hyper_tensor += w2 * psi_derivs['I8sn'] * (np.outer(ms, mn) + np.outer(mn, ms))

    # Enforce sigma_ss = 0 by defining pressure p = sigma_hyper_ss
    pressure_p = sigma_hyper_tensor[1, 1]
    sigma_total_tensor = sigma_hyper_tensor - pressure_p * np.eye(3)

    component_map = {'ff': (0, 0), 'ss': (1, 1), 'nn': (2, 2), 'fs': (0, 1), 'sf': (1, 0), 'fn': (0, 2), 'nf': (2, 0), 'sn': (1, 2), 'ns': (2, 1)}
    stress_comp = data_point['stress_to_calc']
    return sigma_total_tensor[component_map[stress_comp]] if stress_comp in component_map else np.nan

def calculate_predictions(coeffs_w2_full, w1_map, invar_list, data_list, *, context):
    """ Calculates final predictions for plotting/evaluation. Uses the full tensor calculation. """
    y_pred = np.zeros(len(data_list))
    for i, dp in enumerate(data_list):
        invariants = invar_list[i]
        y_pred[i] = calculate_single_prediction_full_tensor(dp, invariants, coeffs_w2_full, w1_map, context=context)
    return y_pred


def make_term_library():
    """Build a fresh copy of the original 32-term invariant/exponential library."""
    return [
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
