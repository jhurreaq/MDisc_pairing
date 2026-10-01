"""Shared numerics operations, preserving the original formulas."""

import numpy as np

def safe_log(x, *, eps=1e-10):
    if isinstance(x, (np.ndarray)):
        res = np.full_like(x, -700.0, dtype=float) # Use large negative for log(small)
        mask = x > eps
        res[mask] = np.log(x[mask])
        return res
    else: # Scalar case
        return np.log(x) if x > eps else -700.0

def exact_safe_power(base, exponent, *, tol_int=1e-12, limit=1e18):
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

def clipped_safe_power(base, exponent, *, eps=1e-10):
    base = np.maximum(base, 0)
    if isinstance(base, (np.ndarray)):
        result = np.zeros_like(base, dtype=float)
        mask_safe = base > eps
        # Use np.power for element-wise operation
        if np.isscalar(exponent) or len(exponent) == 1:
            result[mask_safe] = np.power(base[mask_safe], exponent)
        else:
             result[mask_safe] = np.power(base[mask_safe], exponent[mask_safe])

        # Careful with exponent type check if it could be array
        is_zero_exp = np.all(exponent == 0) if isinstance(exponent, np.ndarray) else (exponent == 0)
        is_neg_exp = np.all(exponent < 0) if isinstance(exponent, np.ndarray) else (exponent < 0)

        if is_zero_exp: result[base <= eps] = 1.0
        elif is_neg_exp: result[base <= eps] = 0.0
        return result
    else: # Scalar case
        if base > eps: return base ** exponent
        elif exponent > 0: return 0.0
        elif exponent == 0: return 1.0
        else: return 0.0
