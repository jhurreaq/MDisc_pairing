"""Synthetic incompressible nominal stress data from the common energy model."""
import numpy as np
import pandas as pd


def generate_data(library, ground_truth_pa, *, noise, points, stretch_range, seed):
    rng = np.random.default_rng(seed)
    coefficients = np.zeros(len(library.active_terms))
    for parameter in ground_truth_pa:
        positions = [i for i, term in enumerate(library.active_terms)
                     if term.family == parameter.family and
                     (term.powers == parameter.powers if term.family == 'MR'
                      else term.exponent == parameter.exponent)]
        if len(positions) != 1:
            raise ValueError(f'Ground-truth term must match exactly one library term: {parameter}')
        coefficients[positions[0]] += parameter.coefficient_pa * 1e-6
    rows = []
    for mode in ('UT', 'PS', 'EBT'):
        for lam in np.linspace(*stretch_range, points):
            l2 = lam ** -0.5 if mode == 'UT' else (1.0 if mode == 'PS' else lam)
            F = np.diag([lam, l2, 1 / (lam * l2)])
            nominal = library.stress(F, coefficients, measure='nominal')
            true = np.array([nominal[0, 0], nominal[1, 1]])
            noisy = true + rng.normal(0, noise * np.abs(true))
            rows.append({'lambda1': lam, 'lambda2': l2, 'lambda3': F[2, 2],
                         'P11': noisy[0], 'P22': noisy[1], 'P11_true': true[0], 'P22_true': true[1],
                         'mode': mode, 'dataset': 'Synthetic', 'strain_pct': (lam - 1) * 100})
    return pd.DataFrame(rows)
