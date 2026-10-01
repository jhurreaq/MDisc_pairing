"""Evaluation is independent of plotting and states its scope explicitly."""
import numpy as np


def metrics(y, prediction):
    y, prediction = np.asarray(y, float), np.asarray(prediction, float)
    if y.shape != prediction.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(prediction)):
        raise ValueError('Metric inputs must be matching finite arrays')
    mse = float(np.mean((y - prediction)**2))
    denominator = float(np.sum((y - np.mean(y))**2))
    return {'n': len(y), 'rmse': float(np.sqrt(mse)), 'mse': mse,
            'r2': 1 - float(np.sum((y-prediction)**2)) / denominator if denominator > 0 else None}


def mode_metrics(observations, prediction):
    return {mode: metrics(observations.y[observations.modes == mode], prediction[observations.modes == mode])
            for mode in np.unique(observations.modes)}


def material_checks(library, coefficients, nonlinear, Fs):
    """Fundamental constitutive checks plus a sampled ellipticity diagnostic.

    Neither positive coefficients nor sampled tests certify global polyconvexity
    or strong ellipticity. Report sampled rank-one curvature honestly, including
    semidefinite reference responses in high-order models.
    """
    identity = np.eye(3)
    reference = library.stress(identity, coefficients, nonlinear)
    reference_energy = float(library.energy(identity, coefficients, nonlinear))
    states = np.unique(np.concatenate([identity[None], Fs]), axis=0)
    energy = library.energy(states, coefficients, nonlinear)
    stress = library.stress(states, coefficients, nonlinear)
    h = 1e-4
    plus, minus, base_indices = [], [], []
    # Incompressible rank-one directions satisfy b.F^-1.a=0, giving det(F+tH)=1.
    rng = np.random.default_rng(81)
    for index, F in enumerate(states):
        for _ in range(6):
            a, b = rng.normal(size=(2, 3))
            a /= np.linalg.norm(a)
            v = np.linalg.solve(F, a)
            b -= v * np.dot(b, v) / np.dot(v, v)
            b /= np.linalg.norm(b)
            H = np.outer(a, b)
            plus.append(F+h*H)
            minus.append(F-h*H)
            base_indices.append(index)
    ep = library.energy(np.array(plus), coefficients, nonlinear)
    em = library.energy(np.array(minus), coefficients, nonlinear)
    base = energy[base_indices]
    values = (ep - 2*base + em) / h**2
    numerical_tolerance = 32 * np.finfo(float).eps * (np.abs(ep) + 2*np.abs(base) + np.abs(em)) / h**2 + 1e-8
    energy_scale = max(1.0, float(np.max(np.abs(energy))))
    return {'reference_energy': reference_energy,
            'zero_response_model': bool(not np.any(coefficients)),
            'reference_stress_norm': float(np.linalg.norm(reference)),
            'nonnegative_coefficients': bool(np.all(coefficients >= 0)),
            'sampled_energy_nonnegative': bool(np.all(energy >= -1e-10 * energy_scale)),
            'cauchy_stress_symmetric': bool(np.allclose(stress, np.swapaxes(stress, -1, -2), atol=1e-8)),
            'sampled_rank_one_curvature_min': float(min(values)),
            'sampled_rank_one_nonnegative': bool(np.all(values >= -numerical_tolerance)),
            'global_stability_certified': False,
            'diagnostic_scope': f'{len(states)} distinct measured/reference states, six isochoric rank-one directions each'}


def admissible(checks):
    return (checks['reference_stress_norm'] < 1e-8
            and abs(checks['reference_energy']) < 1e-8
            and all(checks[key] for key in ('nonnegative_coefficients', 'sampled_energy_nonnegative',
                                          'cauchy_stress_symmetric', 'sampled_rank_one_nonnegative')))
