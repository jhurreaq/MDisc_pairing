"""Fit configured datasets and explicitly evaluate training and held-out scopes."""
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from mdisc.data import isotropic_observations
from mdisc.evaluation import metrics
from mdisc.reporting import provenance, run_scenario, write_report, log_line
from mdisc.rendering import plot_style, save_figure
from .configuration import Configuration
from .models import make_library
from .data import load_data
from .plotting import plot_treloar_prediction, plot_Kawabata_prediction, plot_treloar_data, plot_Kawabata_data


def run(*, input_dir, output_dir, configuration=None, log=None, plots=True):
    configuration = configuration or Configuration()
    frame = load_data(input_dir)
    fit_frame = frame[frame['dataset'].isin(configuration.fit_datasets)]
    if fit_frame.empty or set(configuration.fit_datasets) - set(frame['dataset']):
        raise ValueError('Configured fitting datasets are missing')
    training, evaluation = isotropic_observations(fit_frame), isotropic_observations(frame)
    library = make_library(configuration)
    style = plot_style(configuration.plot)
    output_dir = Path(output_dir)
    prefix = 'Fit_' + '_'.join(configuration.fit_datasets) + '_Only'
    log_line(log, f'### PROCESSING FITTING SCENARIO: {prefix} ###')
    if plots:
        datasets = list(frame['dataset'].unique())
        fig, axes = plt.subplots(1, len(datasets), figsize=(6*len(datasets), 5), squeeze=False)
        for ax, dataset in zip(axes.ravel(), datasets):
            helper = plot_treloar_data if dataset == 'Treloar' else plot_Kawabata_data
            helper(frame[frame['dataset'] == dataset], ax=ax, context=style)
        fig.tight_layout()
        save_figure(fig, output_dir / 'Combined_RawData_Reference_RawData.pdf')
    def render(fit, method_key, prediction, summary, destination, scenario):
        df = frame.copy()
        Fs = np.array([np.diag(row[['lambda1','lambda2','lambda3']].to_numpy(float)) for _,row in frame.iterrows()])
        stresses = library.stress(Fs, fit.coefficients, fit.nonlinear, measure='nominal')
        df['P11_pred'], df['P22_pred'] = stresses[:, 0, 0], stresses[:, 1, 1]
        has_kawabata = 'Kawabata' in set(frame['dataset'])
        fig, axes = plt.subplots(2 if has_kawabata else 1, 2, figsize=(8, 8) if has_kawabata else (8, 5))
        axes = axes.ravel()
        treloar = df[df['dataset'] == 'Treloar']
        score = metrics(treloar['P11'].to_numpy(), treloar['P11_pred'].to_numpy())
        plot_treloar_prediction(df, axes[:2], r2_overall_precomputed=score['r2'], context=style)
        if has_kawabata:
            kawabata = df[df['dataset'] == 'Kawabata']
            s11 = metrics(kawabata['P11'].to_numpy(), kawabata['P11_pred'].to_numpy())
            s22 = metrics(kawabata['P22'].to_numpy(), kawabata['P22_pred'].to_numpy())
            plot_Kawabata_prediction(df, axes[2:], r2_p11_precomputed=s11['r2'], r2_p22_precomputed=s22['r2'], context=style)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        save_figure(fig, destination / f'{scenario}_{method_key}_RefitSubset_EvalFull_Predictions.pdf')
    scenario = run_scenario(library, training, evaluation, configuration, output_dir / prefix, prefix,
                            log=log, style=style, render_predictions=render, plots=plots)
    held_out = ~frame['dataset'].isin(configuration.fit_datasets)
    if held_out.any():
        test = isotropic_observations(frame[held_out])
        for method_key, model in scenario['models'].items():
            result = scenario['paths']['LARS' if method_key.startswith('Lars') else ('Lasso' if method_key.startswith('Lasso') else 'OMP')]
            criterion = method_key[-3:] if method_key.endswith(('AIC','BIC')) else 'CV'
            fit = result.candidates[result.chosen[criterion]].fit
            model['held_out_metrics'] = metrics(test.y, library.predict(test, fit.coefficients, fit.nonlinear))
            model['held_out_datasets'] = sorted(set(frame.loc[held_out, 'dataset']))
    results = {'provenance': provenance(input_dir, configuration), 'scenarios': {prefix: scenario}}
    write_report(output_dir, results)
    return results
