"""Shared isotropic operations, preserving the original formulas."""

import re

def mooney_rivlin_terms(max_total_power, *, max_i1=None, max_i2=None):
    """Generate the original ordered MR terms with explicit per-case power limits."""
    for total_power in range(1, max_total_power + 1):
        upper = total_power + 1 if max_i1 is None else min(total_power + 1, max_i1 + 1)
        for i in range(upper):
            j = total_power - i
            if j < 0 or (max_i2 is not None and j > max_i2):
                continue
            term_i1 = f"(I₁-3)^{i}" if i > 1 else ("(I₁-3)" if i == 1 else "")
            term_i2 = f"(I₂-3)^{j}" if j > 1 else ("(I₂-3)" if j == 1 else "")
            if term_i1 and term_i2:
                yield f"MR: {term_i1}{term_i2}"
            elif term_i1:
                yield f"MR: {term_i1}"
            else:
                yield f"MR: {term_i2}"

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
