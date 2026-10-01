"""Nonnegative cardiac amplitude/shape calibration using the SAME energy model."""
import numpy as np
from scipy.optimize import least_squares
from mdisc.discovery import linear_refit
from mdisc.types import ModelFit


def refine(library, observations, X, support, weights, scaling, settings):
    linear = linear_refit(library, observations, X, support, weights, scaling, settings)
    if not support:
        return linear
    support = np.asarray(support, dtype=int)
    exp_positions = [i for i, compact in enumerate(support) if library.active_terms[compact].nonlinear]
    if not exp_positions:
        return linear
    original_indices = [library.active_terms[support[i]].index for i in exp_positions]
    F = observations.F
    C = np.swapaxes(F, -1, -2) @ F
    I1 = np.trace(C, axis1=-2, axis2=-1)
    I2 = 0.5 * (I1**2 - np.trace(C @ C, axis1=-2, axis2=-1))
    q = {'I1': I1 - 3, 'I2': I2 - 3,
         'I4f': np.maximum(C[:, 0, 0] - 1, 0), 'I4s': np.maximum(C[:, 1, 1] - 1, 0),
         'I4n': np.maximum(C[:, 2, 2] - 1, 0),
         'I8fs': C[:, 0, 1], 'I8fn': C[:, 0, 2], 'I8sn': C[:, 1, 2]}
    upper = []
    arguments = []
    for i in exp_positions:
        term = library.active_terms[support[i]]
        argument = q[term.invariant] ** (2 if term.activation == 'exp_square' else 1)
        # Numerical domain, not exponent clipping: the energy and its analytic
        # derivative always describe the same function. Bounds are recorded.
        upper.append(settings.refinement.max_exponential_argument / max(float(np.max(np.abs(argument))), 1e-6))
        arguments.append(argument)
    scales = scaling.columns[support]
    sw = np.sqrt(weights)
    penalty = np.sqrt(settings.ridge * len(observations.y))
    cache = {}
    base_design = X[:, support]
    arguments = np.column_stack(arguments)

    def evaluate(parameters):
        if 'parameters' in cache and np.array_equal(parameters, cache['parameters']):
            return cache['residual'], cache['jacobian']
        amplitudes = parameters[:len(support)]
        shape_values = parameters[len(support):]
        factors = np.exp((shape_values[None, :] - 1) * arguments)
        design = base_design.copy()
        design[:, exp_positions] *= shape_values[None, :] * factors
        physical = amplitudes * scaling.target / scales
        prediction = design @ physical
        residual = (prediction - observations.y) / scaling.target * sw
        jac_amp = design / scales * sw[:, None]
        derivative = base_design[:, exp_positions] * factors * (1 + shape_values[None, :] * arguments)
        jac_shape = derivative * physical[exp_positions][None, :] / scaling.target * sw[:, None]
        jacobian = np.column_stack([jac_amp, jac_shape])
        if penalty:
            residual = np.concatenate([residual, penalty * amplitudes])
            regularizer = np.zeros((len(support), len(parameters)))
            regularizer[:, :len(support)] = penalty * np.eye(len(support))
            jacobian = np.vstack([jacobian, regularizer])
        cache.update(parameters=parameters.copy(), residual=residual, jacobian=jacobian)
        return residual, jacobian

    initial_amplitudes = scaling.coefficients_to_scaled(linear.coefficients)[support]
    bounds = (np.r_[np.zeros(len(support)), np.full(len(exp_positions), 1e-8)],
              np.r_[np.full(len(support), np.inf), upper])
    attempts = []
    # Deterministic multistart reduces dependence on the arbitrary fixed-w1
    # discovery surrogate. A local optimizer does not certify global optimality.
    for start in settings.refinement.starts:
        initial = np.r_[np.maximum(initial_amplitudes, 1e-12), np.minimum(start, np.array(upper) * 0.5)]
        fit = least_squares(lambda p: evaluate(p)[0], initial, jac=lambda p: evaluate(p)[1],
                            bounds=bounds, x_scale='jac', max_nfev=settings.refinement.max_evaluations,
                            ftol=settings.refinement.tolerance, xtol=settings.refinement.tolerance,
                            gtol=settings.refinement.tolerance)
        if fit.success and np.isfinite(fit.cost):
            attempts.append(fit)
    if not attempts:
        raise RuntimeError(f'Cardiac nonlinear calibration failed in all {len(settings.refinement.starts)} starts')
    fit = min(attempts, key=lambda result: result.cost)
    coefficients = np.zeros(len(library.active_terms))
    coefficients[support] = fit.x[:len(support)] * scaling.target / scales
    shapes = dict(zip(original_indices, fit.x[len(support):].tolist()))
    prediction = library.predict(observations, coefficients, shapes)
    jacobian = evaluate(fit.x)[1][:len(observations.y)]
    active_amp = fit.x[:len(support)] > settings.paths.coefficient_tolerance
    active_shape = np.array([active_amp[i] for i in exp_positions])
    active = np.r_[active_amp, active_shape]
    J = jacobian[:, active]
    R = np.diag(np.r_[np.full(len(support), penalty**2), np.zeros(len(exp_positions))][active])
    gram = J.T @ J
    df = float(np.trace(gram @ np.linalg.pinv(gram + R))) if len(gram) else 0.0
    return ModelFit(coefficients, prediction, df, shapes, True,
                    f'bounded analytical-Jacobian multistart least_squares; {fit.message}',
                    {'shape_upper_bounds': dict(zip(original_indices, upper)), 'optimality': fit.optimality,
                     'evaluations': fit.nfev, 'successful_starts': len(attempts), 'global_optimum_certified': False})
