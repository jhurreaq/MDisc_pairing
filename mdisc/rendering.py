"""Shared report rendering. No fitting, prediction or metric computation here."""
from pathlib import Path
from dataclasses import asdict
from types import SimpleNamespace
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from .configuration import PlotSettings


def plot_style(settings=None):
    style = SimpleNamespace(**asdict(settings or PlotSettings()))
    style.plasma = plt.get_cmap('plasma')
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


def _number(value, precision=3):
    """Compact scientific notation without Matplotlib's separate offset text."""
    label = f'{value:.{precision}g}'
    if 'e' in label:
        mantissa, exponent = label.split('e')
        label = f'{mantissa}e{int(exponent)}'
    return label


def _endpoint_ticks(ax):
    low, high = ax.get_ylim()
    # Labels must remain distinct even for a nearly constant criterion.
    for precision in range(3, 17):
        labels = [_number(value, precision) for value in (low, high)]
        if labels[0] != labels[1]:
            break
    ax.set_yticks([low, high], labels=labels)


def selection_figure(result, *, style):
    """Actual CV/conditional IC values on the discovery parameter grid."""
    x = np.array([c.parameter for c in result.candidates])
    fig, axes = plt.subplots(3, 1, figsize=style.SMALL_FIG_LASSO_LARS_OMP, sharex=True)
    curves = [([c.cv_mse for c in result.candidates], 'CV\nNMSE'),
              ([c.aic for c in result.candidates], 'AIC'),
              ([c.bic for c in result.candidates], 'BIC')]
    for ax, (values, label), criterion, shade in zip(axes, curves, ('CV', 'AIC', 'BIC'), (0.25, 0.55, 0.85)):
        ax.plot(x, values, color=style.plasma(shade), linewidth=style.PLOT_LINEWIDTH)
        chosen = result.candidates[result.chosen[criterion]].parameter
        ax.axvline(chosen, linestyle='--', color=style.plasma(shade), linewidth=1)
        symbol = r'$\alpha$' if result.method == 'Lasso' else 'fraction'
        ax.set_title(f'Selected {symbol} = {_number(chosen)}',
                     fontsize=style.PLOT_LEGEND_FONTSIZE, pad=5)
        if criterion == 'CV':
            mean, sd = np.array(values), np.array([c.cv_std for c in result.candidates])
            ax.fill_between(x, np.maximum(mean-sd, 0), mean+sd, alpha=0.2, color=style.plasma(shade))
            ax.set_ylim(bottom=0)
        ax.set_ylabel(label, fontsize=style.PLOT_LABEL_FONTSIZE)
        ax.tick_params(labelsize=style.PLOT_TICK_FONTSIZE)
        _endpoint_ticks(ax)
    if result.method == 'Lasso':
        axes[-1].set_xscale('log')
        axes[-1].set_xlim(x.max(), x.min())
        ticks = [x.max(), np.sqrt(x.min()*x.max()), x.min()]
        axes[-1].set_xticks(np.unique(ticks), labels=[_number(v) for v in np.unique(ticks)])
        axes[-1].minorticks_off()
    else:
        axes[-1].set_xlim(0, 1)
        axes[-1].set_xticks([0, 0.5, 1], labels=['0', '0.5', '1'])
    axes[-1].set_xlabel(r'Penalty $\alpha$' if result.method == 'Lasso' else 'Path fraction',
                        fontsize=style.PLOT_LABEL_FONTSIZE)
    fig.text(0.5, 0.015, 'CV band: ±1 fold SD', ha='center',
             fontsize=style.PLOT_LEGEND_FONTSIZE)
    fig.tight_layout(pad=0.6, h_pad=1.4, rect=(0, 0.07, 1, 1))
    fig.align_ylabels(axes)
    return fig


def _term_label(term):
    """Energy expressions come from metadata, never from parsing display text."""
    if term.family == 'MR':
        expression = ''.join(f'(I_{i+1}-3)' + (f'^{{{p}}}' if p > 1 else '')
                             for i, p in enumerate(term.powers) if p)
    elif term.family == 'Ogden':
        expression = rf'(\lambda_1^{{{term.exponent:g}}}+\lambda_2^{{{term.exponent:g}}}+\lambda_3^{{{term.exponent:g}}}-3)'
    else:
        invariant = rf'I_{{{term.invariant[1:]}}}'
        if term.invariant in ('I1', 'I2'):
            argument = f'({invariant}-3)'
        elif term.invariant.startswith('I4'):
            argument = rf'\langle {invariant}-1\rangle_+'
        else:
            argument = invariant
        if term.activation in ('square', 'exp_square'):
            argument += '^2'
        expression = (rf'[\exp(b_{{{term.term_id}}}{argument})-1]'
                      if term.nonlinear else argument)
    return rf'$c_{{{term.term_id}}}{expression}$'


def activation_figure(result, library, *, style):
    """Binary activation of final full-data refits, in original registry IDs.

    Discovery already prunes numerical zeros in dimensionless coordinates.
    Do not impose a second, unit-dependent tolerance on physical coefficients.
    """
    coefficients = np.column_stack([library.expand(c.fit.coefficients) for c in result.candidates])
    status = coefficients > 0
    rows = np.flatnonzero(status.any(axis=1))
    rows = sorted(rows, key=lambda row: np.flatnonzero(status[row])[0])
    fig, ax = plt.subplots(figsize=(7, max(5, len(rows)*0.4)))
    cmap = LinearSegmentedColormap.from_list('activation_blues', style.HEATMAP_CMAP(np.linspace(0, 0.75, 256)))
    if rows:
        ax.imshow(status[rows], aspect='auto', cmap=cmap, vmin=0, vmax=1, interpolation='nearest')
        ax.set_yticks(np.arange(len(rows)), labels=[_term_label(library.terms[row]) for row in rows],
                      fontsize=style.PLOT_HEATMAP_YLABEL_SIZE)
    else:
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.text(0.5, 0.5, 'No active terms: zero-response path', ha='center', va='center',
                transform=ax.transAxes, fontsize=style.PLOT_LEGEND_FONTSIZE)
    count = len(result.candidates)
    ax.set_xlim(-0.5, count-0.5)
    ticks = np.unique(np.linspace(0, count-1, min(5, count), dtype=int))
    x = np.array([c.parameter for c in result.candidates])
    labels = [_number(x[i]) for i in ticks]
    ax.set_xticks(ticks, labels=labels, fontsize=style.PLOT_TICK_FONTSIZE)
    ax.set_xlabel(r'Penalty $\alpha$' if result.method == 'Lasso' else 'Path fraction',
                  fontsize=style.PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Model Terms', fontsize=style.PLOT_LABEL_FONTSIZE)
    display_method = 'LAR' if result.method == 'LARS' else result.method.upper()
    ax.set_title(f'{display_method}: refitted term activation', fontsize=style.PLOT_TITLE_FONTSIZE)
    handles = [Patch(facecolor=cmap(1.0), label='Active'),
               Patch(facecolor=cmap(0.0), edgecolor='darkgray', label='Inactive')]
    # Group coincident choices so one line cannot conceal another criterion.
    choices = {}
    for criterion in ('CV', 'AIC', 'BIC'):
        choices.setdefault(result.chosen[criterion], []).append(criterion)
    for index, criteria in choices.items():
        first = ('CV', 'AIC', 'BIC').index(criteria[0])
        color, linestyle = style.plasma((0.25, 0.55, 0.85)[first]), ('--', ':', '-.')[first]
        ax.axvline(index, color=color, linestyle=linestyle, linewidth=style.PLOT_LINEWIDTH)
        value = _number(x[index])
        label = f'{"/".join(criteria)} ({value})'
        handles.append(Line2D([0], [0], color=color, linestyle=linestyle, label=label))
    # A separate legend row for activation and for choices avoids cell overlays.
    fig.legend(handles=handles[:2], loc='lower center', bbox_to_anchor=(0.5, 0.08),
               ncol=2, fontsize=style.PLOT_LEGEND_FONTSIZE)
    fig.legend(handles=handles[2:], loc='lower center', bbox_to_anchor=(0.5, 0.02),
               ncol=len(choices), fontsize=style.PLOT_LEGEND_FONTSIZE)
    fig.text(0.5, 0.155, 'Only terms active somewhere on the refitted path are shown.',
             ha='center', fontsize=style.PLOT_LEGEND_FONTSIZE)
    fig.tight_layout(pad=0.6, rect=(0, 0.21, 1, 1))
    return fig


def render_selection(result, library, save_dir, prefix, *, style):
    display_method = 'Lars' if result.method == 'LARS' else result.method
    save_figure(selection_figure(result, style=style),
                Path(save_dir) / f'{prefix}_{display_method}_Criteria.pdf')
    fig = activation_figure(result, library, style=style)
    save_figure(fig, Path(save_dir) / f'{prefix}_{display_method}_ActivationHeatmap.pdf')
