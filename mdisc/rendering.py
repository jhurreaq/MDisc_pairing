"""Shared report rendering. No fitting, prediction or metric computation here."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from .configuration import PlotSettings


def plot_style(settings=None):
    style = SimpleNamespace(**asdict(settings or PlotSettings()))
    style.plasma = plt.get_cmap('plasma')
    style.TRUNCATED_PLASMA_CMAP = style.plasma
    style.HEATMAP_CMAP = plt.get_cmap('Blues')
    style.color_original = 'black'
    style.color_refined = style.plasma(0.15)
    style.color_pdf = style.plasma(0.55)
    style.color_noisy_data = style.plasma(0.7)
    return style


def save_figure(figure, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        figure.savefig(path, format='pdf', bbox_inches='tight', metadata={'CreationDate': None, 'ModDate': None})
    finally:
        plt.close(figure)


def render_selection(result, library, save_dir, prefix, *, style):
    x = np.array([c.parameter for c in result.candidates])
    fig, axes = plt.subplots(3, 1, figsize=style.SMALL_FIG_LASSO_LARS_OMP, sharex=True)
    curves = [([c.cv_mse for c in result.candidates], 'CV normalized MSE'),
              ([c.aic for c in result.candidates], 'Conditional AIC'),
              ([c.bic for c in result.candidates], 'Conditional BIC')]
    for ax, (values, label), criterion, shade in zip(axes, curves, ('CV', 'AIC', 'BIC'), (0.25, 0.55, 0.85)):
        ax.plot(x, values, color=style.plasma(shade), linewidth=style.PLOT_LINEWIDTH)
        chosen = result.candidates[result.chosen[criterion]].parameter
        ax.axvline(chosen, linestyle='--', color=style.plasma(shade), linewidth=1)
        if criterion == 'CV':
            mean, sd = np.array(values), np.array([c.cv_std for c in result.candidates])
            ax.fill_between(x, np.maximum(mean-sd, 0), mean+sd, alpha=0.2, color=style.plasma(shade))
        ax.set_ylabel(label, fontsize=style.PLOT_LABEL_FONTSIZE)
        ax.tick_params(labelsize=style.PLOT_TICK_FONTSIZE)
        if result.method == 'Lasso':
            ax.set_xscale('log')
            ax.invert_xaxis()
    axes[-1].set_xlabel('α' if result.method == 'Lasso' else 'Path fraction', fontsize=style.PLOT_LABEL_FONTSIZE)
    fig.tight_layout()
    display_method = 'Lars' if result.method == 'LARS' else result.method
    save_figure(fig, Path(save_dir) / f'{prefix}_{display_method}_Criteria.pdf')
    # Keep the original Blues/plasma vocabulary and original 32 IDs, including
    # blank disabled rows. Coefficients are refitted physical model amplitudes.
    coefficients = np.column_stack([library.expand(c.fit.coefficients) for c in result.candidates])
    rowmax = np.max(np.abs(coefficients), axis=1, keepdims=True)
    intensity = np.divide(coefficients, rowmax, out=np.zeros_like(coefficients), where=rowmax > 0)
    fig, ax = plt.subplots(figsize=(7, max(4, len(library.terms) * 0.28)))
    ax.imshow(intensity, aspect='auto', cmap=style.HEATMAP_CMAP, vmin=0, vmax=1, interpolation='nearest')
    labels = [(t.label if t.family == 'cardiac' else f'{t.term_id}: {t.label}')
              + (' [excluded]' if not t.enabled else '') for t in library.terms]
    ax.set_yticks(np.arange(len(labels)), labels=labels, fontsize=style.PLOT_HEATMAP_YLABEL_SIZE)
    for row, term in enumerate(library.terms):
        if not term.enabled:
            ax.axhspan(row-0.5, row+0.5, facecolor='lightgray', alpha=0.5)
    for criterion, color in zip(('CV', 'AIC', 'BIC'), (0.25, 0.55, 0.85)):
        ax.axvline(result.chosen[criterion], color=style.plasma(color), linestyle='--', label=criterion)
    ax.set_xlabel('Candidate path position', fontsize=style.PLOT_LABEL_FONTSIZE)
    ax.legend(fontsize=style.PLOT_LEGEND_FONTSIZE)
    fig.tight_layout()
    save_figure(fig, Path(save_dir) / f'{prefix}_{display_method}_ActivationHeatmap.pdf')
