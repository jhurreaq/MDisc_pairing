"""Shared experiment reporting, serialization and reproducibility provenance."""
from dataclasses import asdict, is_dataclass
from importlib.metadata import version
from pathlib import Path
import hashlib
import json
import platform
import numpy as np
from .discovery import discover
from .evaluation import metrics, mode_metrics
from .rendering import render_selection


def json_value(value):
    if is_dataclass(value):
        return json_value(asdict(value))
    if isinstance(value, np.ndarray):
        return json_value(value.tolist())
    if isinstance(value, np.generic):
        return json_value(value.item())
    if isinstance(value, dict):
        return {str(k): json_value(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def provenance(input_dir, configuration):
    input_dir = Path(input_dir)
    paths = sorted(p for p in input_dir.glob('*') if p.suffix in ('.csv', '.xlsx'))
    inputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    source_root = Path(__file__).resolve().parent.parent
    source_files = sorted(p for folder in ('mdisc', 'Isotropic', 'Anisotropic') for p in (source_root/folder).rglob('*.py'))
    digest = hashlib.sha256()
    for path in source_files:
        digest.update(str(path.relative_to(source_root)).encode())
        digest.update(path.read_bytes())
    return {'configuration': configuration, 'input_sha256': inputs, 'source_sha256': digest.hexdigest(),
            'python': platform.python_version(),
            'dependencies': {name: version(name) for name in ('numpy','scipy','scikit-learn','pandas','matplotlib','openpyxl')},
            'information_criteria': 'Conditional Gaussian working likelihood; local effective df plus residual variance',
            'cv_scope': 'Grouped, mode-stratified model-selection CV including preprocessing, discovery and refitting; not an independent test estimate'}


def write_report(output_dir, results):
    path = Path(output_dir) / 'results.json'
    path.write_text(json.dumps(json_value(results), indent=2, sort_keys=True, allow_nan=False) + '\n')


def log_line(log, message):
    if log is not None:
        print(message, file=log, flush=True)


def run_scenario(library, training, evaluation, configuration, output_dir, prefix,
                 *, log, style, render_predictions, plots=True, refiner=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'library': [asdict(term) for term in library.terms], 'models': {}, 'paths': {},
               'training': {'targets': training.y, 'groups': training.groups, 'modes': training.modes,
                            'measure': training.measure, 'unit': training.unit},
               'evaluation': {'targets': evaluation.y, 'groups': evaluation.groups, 'modes': evaluation.modes}}
    for method in ('Lasso', 'LARS', 'OMP'):
        log_line(log, f'\n--- Running {method} Analysis (CV, AIC, BIC) ({prefix}) ---')
        print(f'{prefix}: {method}', flush=True)
        discovered = discover(library, training, method, configuration.discovery, refiner=refiner)
        results['paths'][method] = discovered
        for location, diagnostics in [('full data', discovered.diagnostics['full_data'])] + [
                (f'fold {i+1}', d) for i, d in enumerate(discovered.diagnostics['folds'])]:
            for warning in diagnostics.get('warnings', []):
                log_line(log, f'  NUMERICAL DIAGNOSTIC ({location}): {warning}')
        log_line(log, f'  Design numerical rank: {discovered.diagnostics["full_data"]["numerical_rank"]}/'
                 f'{discovered.diagnostics["full_data"]["features"]}; coefficients may be nonunique in dependent libraries.')
        if plots:
            render_selection(discovered, library, output_dir, prefix, style=style)
        for criterion, index in discovered.chosen.items():
            candidate = discovered.candidates[index]
            fit = candidate.fit
            method_key = ('Lars' if method == 'LARS' else method) + criterion
            prediction = library.predict(evaluation, fit.coefficients, fit.nonlinear)
            summary = {'coefficients': library.expand(fit.coefficients),
                       'nonlinear': {i+1: value for i, value in fit.nonlinear.items()},
                       'selected_term_ids': [term.term_id for term, beta in zip(library.active_terms, fit.coefficients) if beta > 0],
                       'selection_parameter': candidate.parameter,
                       'training_metrics': metrics(training.y, fit.prediction),
                       'evaluation_metrics': metrics(evaluation.y, prediction),
                       'evaluation_by_mode': mode_metrics(evaluation, prediction),
                       'prediction': prediction, 'selection_oof_prediction': discovered.cv_predictions[criterion],
                       'mechanics': candidate.mechanics, 'aic': candidate.aic, 'bic': candidate.bic,
                       'selection_cv_normalized_mse': candidate.cv_mse, 'converged': fit.converged,
                       'solver_message': fit.message}
            results['models'][method_key] = summary
            log_line(log, f'  {method_key}: parameter={candidate.parameter:.6g}; terms={summary["selected_term_ids"]}; '
                     f'CV normalized MSE={candidate.cv_mse:.6g}; AIC={candidate.aic:.6g}; BIC={candidate.bic:.6g}')
            if not any(fit.coefficients > 0):
                log_line(log, '  Selected zero-response model; this is not a strictly stable material.')
            if plots:
                render_predictions(fit, method_key, prediction, summary, output_dir, prefix)
    return results
