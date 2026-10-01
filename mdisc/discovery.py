"""Shared, leakage-free sparse support selection and physical coefficient refits."""
from dataclasses import dataclass, field
import numpy as np
from .types import ModelFit, Candidate, DiscoveryResult
from .paths import PathSettings, discovery_path
from .preprocessing import WeightPolicy, DesignScaling
from .regression import fit_nonnegative
from .criteria import information_criteria
from .evaluation import material_checks, admissible

@dataclass(frozen=True)
class RefinementSettings:
    starts: tuple = (1.0, 0.25, 4.0)
    max_evaluations: int = 1000
    tolerance: float = 1e-8
    max_exponential_argument: float = 100.0

    def __post_init__(self):
        if not self.starts or any(s <= 0 or not np.isfinite(s) for s in self.starts):
            raise ValueError('Nonlinear starts must be finite and positive')
        if self.max_evaluations < 1 or not (np.finfo(float).eps < self.tolerance < 1):
            raise ValueError('Invalid nonlinear iteration count or tolerance')
        if not (0 < self.max_exponential_argument < 500):
            raise ValueError('Exponential parameter domain must stay within floating-point range')

@dataclass(frozen=True)
class DiscoverySettings:
    folds: int = 5
    seed: int = 42
    ridge: float = 1e-7  # in dimensionless, normalized coordinates
    paths: PathSettings = field(default_factory=PathSettings)
    weights: WeightPolicy = field(default_factory=WeightPolicy)
    refinement: RefinementSettings = field(default_factory=RefinementSettings)

    def __post_init__(self):
        if self.folds < 2 or self.ridge < 0 or not np.isfinite(self.ridge) or self.seed < 0:
            raise ValueError('Require at least two folds and a nonnegative ridge penalty')


def grouped_mode_folds(observations, folds, seed):
    """Split independent deformation measurements; paired stresses stay together.

    Stratify by loading experiment, retaining every mode in each training fold.
    A group may contain multiple measured components of the same deformation.
    """
    group_modes = {}
    for group, mode in zip(observations.groups, observations.modes):
        group_modes.setdefault(group, set()).add(mode)
    strata = {}
    for group, modes in group_modes.items():
        strata.setdefault(tuple(sorted(modes)), []).append(group)
    if folds < 2 or any(len(groups) < folds for groups in strata.values()):
        raise ValueError("Each experiment needs at least folds independent measurement groups")
    rng = np.random.default_rng(seed)
    bins = [set() for _ in range(folds)]
    for groups in strata.values():
        groups = np.asarray(groups)
        rng.shuffle(groups)
        for k, chunk in enumerate(np.array_split(groups, folds)):
            bins[k].update(chunk.tolist())
    splits = []
    for groups in bins:
        validation = np.array([g in groups for g in observations.groups])
        splits.append((np.flatnonzero(~validation), np.flatnonzero(validation)))
    return splits


def linear_refit(library, observations, X, support, weights, scaling, settings):
    coefficients = np.zeros(X.shape[1])
    support = np.asarray(support, dtype=int)
    A, b = scaling.transform(X, observations.y, weights)
    fitted = fit_nonnegative(A[:, support], b, ridge=settings.ridge * len(b))
    scaled_coefficients = np.zeros(X.shape[1])
    scaled_coefficients[support] = fitted.coefficients
    coefficients = scaling.coefficients_to_physical(scaled_coefficients)
    return ModelFit(coefficients, X @ coefficients, fitted.effective_df, message=fitted.solver)


def discover(library, observations, method, settings=None, *, refiner=None):
    """Repeat weighting, scaling, the discovery path and refitting within folds.

    CV selects a data-independent alpha or normalized path position. Every fold
    builds its OWN supports; full-data supports never enter validation fits.
    Cardiac nonlinear refinement is included inside these folds too.
    """
    settings = settings or DiscoverySettings()
    underlying_refit = linear_refit if refiner is None else refiner

    def refit(library, obs, design, support, weights, scaling, settings):
        # One tolerance, in nondimensional coordinates. Prune numerical-zero
        # amplitudes and recalibrate the reduced physical model to convergence.
        while True:
            fitted = underlying_refit(library, obs, design, support, weights, scaling, settings)
            active = tuple(np.flatnonzero(scaling.coefficients_to_scaled(fitted.coefficients)
                                          > settings.paths.coefficient_tolerance).tolist())
            if active == support:
                return fitted
            if not set(active).issubset(support):
                raise RuntimeError('Refit activated a term outside its discovery support')
            support = active
    X = library.design(observations)
    weights_model = settings.weights.fit(observations.y, observations.modes)
    weights = weights_model.transform(observations.modes)
    scaling = DesignScaling.fit(X, observations.y, weights)
    A, b = scaling.transform(X, observations.y, weights)
    diagnostics = {'full_data': {}, 'folds': []}
    full_path = discovery_path(A, b, method, settings.paths, diagnostics=diagnostics['full_data'])
    parameters = settings.paths.parameters(method)
    candidates = []
    # Identical full-data supports share a refit, but retain separate CV
    # parameters: their fold-specific supports need not be identical.
    cache = {}
    for parameter, beta in zip(parameters, full_path.T):
        support = tuple(np.flatnonzero(np.abs(beta) > settings.paths.coefficient_tolerance).tolist())
        if support not in cache:
            fitted = refit(library, observations, X, support, weights, scaling, settings)
            aic, bic = information_criteria(observations.y, fitted.prediction, fitted.effective_df, weights=weights)
            checks = material_checks(library, fitted.coefficients, fitted.nonlinear, observations.F)
            cache[support] = fitted, aic, bic, checks
        fitted, aic, bic, checks = cache[support]
        candidates.append(Candidate(float(parameter), support, fitted, aic, bic,
                                    path_coefficients=beta.copy(), mechanics=checks))
    folds = grouped_mode_folds(observations, settings.folds, settings.seed)
    losses = [[] for _ in candidates]
    predictions = {i: np.full(len(observations.y), np.nan) for i in range(len(candidates))}
    for train, validation in folds:
        tr, va = observations.subset(train), observations.subset(validation)
        Xtr = X[train]
        wm = settings.weights.fit(tr.y, tr.modes)
        wtr, wva = wm.transform(tr.modes), wm.transform(va.modes)
        sc = DesignScaling.fit(Xtr, tr.y, wtr)
        Atr, btr = sc.transform(Xtr, tr.y, wtr)
        fold_diagnostics = {'weights': wm.factors, 'column_scales': sc.columns, 'target_scale': sc.target}
        fold_path = discovery_path(Atr, btr, method, settings.paths, diagnostics=fold_diagnostics)
        diagnostics['folds'].append(fold_diagnostics)
        fold_cache = {}
        for i, beta in enumerate(fold_path.T):
            support = tuple(np.flatnonzero(np.abs(beta) > settings.paths.coefficient_tolerance).tolist())
            if support not in fold_cache:
                fit = refit(library, tr, Xtr, support, wtr, sc, settings)
                if not fit.converged:
                    raise RuntimeError(f"{method} validation refit failed: {fit.message}")
                checks = material_checks(library, fit.coefficients, fit.nonlinear, tr.F)
                fold_cache[support] = library.predict(va, fit.coefficients, fit.nonlinear) if admissible(checks) else None
            predicted = fold_cache[support]
            if predicted is None:
                losses[i].append(float('inf'))
                continue
            predictions[i][validation] = predicted
            # Dimensionless validation loss on TRAINING stress scales. Weights
            # and target scale are never re-estimated on held-out targets.
            losses[i].append(float(np.mean(wva * ((va.y - predicted) / sc.target) ** 2)))
    for candidate, errors in zip(candidates, losses):
        candidate.cv_mse = float(np.mean(errors))
        candidate.cv_std = float(np.std(errors, ddof=1)) if np.isfinite(errors).all() else float('nan')
    eligible = np.array([admissible(c.mechanics) and c.fit.converged for c in candidates])
    chosen = {}
    for criterion, values in (('CV', [c.cv_mse for c in candidates]),
                              ('AIC', [c.aic for c in candidates]), ('BIC', [c.bic for c in candidates])):
        scores = np.where(eligible, values, np.inf)
        if not np.isfinite(scores).any():
            raise RuntimeError(f'No converged, mechanically admissible candidate for {method}/{criterion}')
        chosen[criterion] = int(np.argmin(scores))
    return DiscoveryResult(method, candidates, chosen,
                           [{'train': tr.tolist(), 'validation': va.tolist()} for tr, va in folds],
                           {criterion: predictions[i] for criterion, i in chosen.items()}, diagnostics)
