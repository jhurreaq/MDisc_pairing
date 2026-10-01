"""Cardiac discovery with original term IDs and fold-local nonlinear refinement."""
from pathlib import Path
import numpy as np
from mdisc.evaluation import mode_metrics, metrics
from mdisc.mechanics import cardiac_library
from mdisc.reporting import provenance, run_scenario, write_report, log_line
from mdisc.rendering import plot_style
from .configuration import Configuration
from .data import load_data
from .refinement import refine
from .plotting import render_predictions


def run(*, input_dir, output_dir, configuration=None, log=None, plots=True):
    configuration = configuration or Configuration()
    clean, records = load_data(input_dir, sheet=configuration.sheet)
    library = cardiac_library()
    style = plot_style(configuration.plot)
    output_dir = Path(output_dir)
    reference_full = np.zeros(32)
    for index, coefficient in configuration.reference_coefficients:
        reference_full[index] = coefficient
    reference = library.predict(clean, library.compact(reference_full), dict(configuration.reference_nonlinear))
    results = {'provenance': provenance(input_dir, configuration), 'scenarios': {}}
    base_noise = np.random.default_rng(configuration.discovery.seed).normal(size=len(clean.y))
    for noise in configuration.noise_levels:
        prefix = f'RelNoise{noise*100:.0f}pct'
        observations = clean.with_targets(clean.y + noise * np.abs(clean.y) * base_noise)
        log_line(log, f'\n### PROCESSING NOISE LEVEL: {prefix} ###')
        def render(fit, method_key, prediction, summary, destination, scenario):
            render_predictions(records, prediction, reference, observations.y, clean.y,
                               mode_metrics(clean, prediction), method_key, destination, scenario, context=style)
        scenario = run_scenario(library, observations, observations, configuration, output_dir / prefix, prefix,
                                log=log, style=style, render_predictions=render, plots=plots, refiner=refine)
        for model in scenario['models'].values():
            model['clean_reference_metrics'] = metrics(clean.y, model['prediction'])
            model['evaluation_scope'] = 'training-data reconstruction; clean experimental targets are not an independent test set'
        scenario['published_reference_metrics'] = metrics(clean.y, reference)
        results['scenarios'][prefix] = scenario
    write_report(output_dir, results)
    return results
