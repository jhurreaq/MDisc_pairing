"""Shared preprocessing operations, preserving the original formulas."""

import numpy as np
from sklearn.preprocessing import StandardScaler


def fit_standard_scaler(values):
    """Fit the original StandardScaler defaults, leaving transform timing to the case."""
    scaler = StandardScaler()
    scaler.fit(values)
    return scaler

def build_stress_magnitude_weights(data_list, *, eps=1e-10):
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
    all_rms_values = [stats['rms'] for stats in stress_stats.values() if stats['rms'] > eps]
    if not all_rms_values:
        print("WARN: No valid stress data for weighting. Returning equal weights.")
        return np.ones(len(data_list))

    global_rms = np.sqrt(np.mean(np.array(all_rms_values)**2))
    print(f"    Global RMS Stress (for normalization): {global_rms:.3f} kPa")

    # 4. Calculate the final weight factor for each mode
    mode_weights = {}
    for mode, stats in stress_stats.items():
        mode_weights[mode] = global_rms / stats['rms'] if stats['rms'] > eps else 1.0
        print(f"    Mode '{mode}': Final Weight Factor = {mode_weights[mode]:.3f}")

    # 5. Create the final weight array, mapping the correct weight to each data point
    final_weights = np.array([mode_weights.get(dp.get('test_name'), 1.0) for dp in data_list])

    return final_weights

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
