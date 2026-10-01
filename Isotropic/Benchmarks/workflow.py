"""Synthetic experiments configured around the common discovery engine."""
from pathlib import Path
import matplotlib.pyplot as plt
from mdisc.data import isotropic_observations
from mdisc.evaluation import metrics
from mdisc.mechanics import isotropic_library
from mdisc.reporting import provenance, run_scenario, write_report, log_line
from mdisc.rendering import plot_style, save_figure
from .configuration import Configuration
from .data import generate_data
from .plotting import plot_predictions


def run(*, input_dir, output_dir, configuration=None, log=None, plots=True):
    configuration = configuration or Configuration()
    style = plot_style(configuration.plot)
    output_dir = Path(output_dir)
    results = {'provenance': provenance(input_dir, configuration), 'scenarios': {}}
    for truth in configuration.ground_truths:
        library = isotropic_library(mr_order=configuration.mr_order if truth.mr else 0,
                                    ogden_exponents=configuration.ogden_exponents if truth.ogden else ())
        for noise in configuration.noise_levels:
            prefix = f'{truth.name}_RelNoise{noise*100:.0f}pct'
            log_line(log, f'\n### PROCESSING SCENARIO: {prefix} (Fit UT+PS+EBT P11) ###')
            # Repeat the same underlying standard-normal draws across noise
            # levels to isolate the effect of noise amplitude on rediscovery.
            frame = generate_data(library, truth.coefficients_pa, noise=noise,
                                  points=configuration.points, stretch_range=configuration.stretch_range,
                                  seed=configuration.discovery.seed)
            observations = isotropic_observations(frame, components=('P11',))
            clean = observations.with_targets(frame['P11_true'].to_numpy())
            directory = output_dir / prefix
            if plots:
                fig, axes = plt.subplots(1, 3, figsize=(18, 5))
                for ax, mode, color in zip(axes, ('UT','PS','EBT'), ('tab:blue','darkorange','green')):
                    raw = frame[frame['mode'] == mode]
                    ax.scatter(raw['lambda1'], raw['P11'], s=15, alpha=0.7, color=color,
                               label=f'Noisy P11 (RelNoise {noise*100:.0f}%)')
                    ax.plot(raw['lambda1'], raw['P11_true'], 'k--' if mode == 'UT' else 'k:', label='True P11')
                    ax.set(xlabel='$λ$', ylabel='$P_{11}$ (MPa)', title=f'Generated {mode} Data P11')
                    ax.legend(frameon=False)
                fig.suptitle(f'Raw Synthetic Data: {prefix}')
                fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                save_figure(fig, directory / f'{prefix}_RawSynthData_P11_MPa.pdf')
            def render(fit, method_key, prediction, summary, destination, scenario):
                df = frame.copy()
                # Existing synthetic helpers display stresses in kPa.
                df['P11_pred'] = prediction * 1000
                fig, axes = plt.subplots(1, 3, figsize=(12, 3))
                for ax, mode in zip(axes, ('UT', 'PS', 'EBT')):
                    subset = df[df['mode'] == mode]
                    plot_predictions(subset, ax, context=style)
                fig.tight_layout(rect=[0, 0.03, 1, 0.94])
                save_figure(fig, destination / f'{scenario}_{method_key}_Predictions_Final_Checked_kPa.pdf')
            scenario = run_scenario(library, observations, observations, configuration, directory, prefix,
                                    log=log, style=style, render_predictions=render, plots=plots)
            for model in scenario['models'].values():
                model['clean_reference_metrics'] = metrics(clean.y, model['prediction'])
                model['evaluation_scope'] = 'training-data reconstruction; clean target comparison is a synthetic benchmark'
            results['scenarios'][prefix] = scenario
    write_report(output_dir, results)
    return results
