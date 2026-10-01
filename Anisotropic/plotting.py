"""Cardiac plotting; original numerical and presentation choices."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker
import matplotlib.colors as mcolors
from matplotlib.patches import Patch
from sklearn.metrics import mean_squared_error, r2_score
import os
import re
from .models import calculate_predictions

def generate_stress_strain_plot_refined(coeffs_w2_full, w1_map, pdf_preds, y_target_noisy,
                                        plot_title_suffix, save_dir, file_prefix,
                                        all_invariants_current, *, context):
    """
    Plot refined model vs. data across modes with consistent layout and labels.
    """
    # Predictions
    y_pred_plot_specific = calculate_predictions(
        coeffs_w2_full, w1_map, all_invariants_current, context.experimental_data_list_filtered,
        context=context,
    )

    plot_metrics = {}
    temp_plot_series = {k: v.copy() for k, v in context.experimental_plot_series.items()}
    for k in temp_plot_series:
        base_len = len(temp_plot_series[k].get('x', []))
        for arr_key in ['y_exp', 'y_noisy', 'y_pred_refined', 'y_pred_pdf']:
            if arr_key not in temp_plot_series[k]:
                temp_plot_series[k][arr_key] = np.full(base_len, np.nan, dtype=float)

    # Map predictions / noisy
    pred_map_refined, pred_map_pdf, noisy_target_map = {}, {}, {}
    for i, dp in enumerate(context.experimental_data_list_filtered):
        key = (dp.get('test_name'), dp.get('original_x'))
        if None in key: continue
        if i < len(y_pred_plot_specific): pred_map_refined[key] = y_pred_plot_specific[i]
        if i < len(pdf_preds):            pred_map_pdf[key]     = pdf_preds[i]
        if i < len(y_target_noisy):       noisy_target_map[key] = y_target_noisy[i]

    for test_key, series in temp_plot_series.items():
        x_arr = np.array(series.get('x', []))
        series['y_pred_refined'] = np.full_like(x_arr, np.nan, dtype=float)
        series['y_pred_pdf']     = np.full_like(x_arr, np.nan, dtype=float)
        series['y_noisy']        = np.full_like(x_arr, np.nan, dtype=float)
        for j, xv in enumerate(x_arr):
            mk = (test_key, xv)
            if mk in pred_map_refined: series['y_pred_refined'][j] = pred_map_refined[mk]
            if mk in pred_map_pdf:     series['y_pred_pdf'][j]     = pred_map_pdf[mk]
            if mk in noisy_target_map: series['y_noisy'][j]        = noisy_target_map[mk]

    # Figure (4:3 ratio)
    fig_w, fig_h = 9.5, 12.5  # 4:3
    fig, axs = plt.subplots(6, 3, figsize=(fig_w, fig_h))

    # Legend prototypes
    ax_proto = axs.flat[0]
    line_data,   = ax_proto.plot([], [], 'o', markersize=5, markerfacecolor='none',
                                 color=context.color_original, label='Original Data')
    line_noise,  = ax_proto.plot([], [], 'o', markersize=5, color=context.color_noisy_data,
                                 alpha=0.7, label='Noisy Data')
    line_refined,= ax_proto.plot([], [], '-',  lw=2, color=context.color_refined,
                                 label=f'Our Disc. Model ({plot_title_suffix})')
    line_pdf,    = ax_proto.plot([], [], '--', lw=2, color=context.color_pdf,
                                 label='Martonová et al. (2024)')

    shear_modes_sorted    = ['fs', 'fn', 'sf', 'sn', 'nf', 'ns']
    biaxial_ratios_sorted = ['1:1', '1:0.75', '0.75:1', '1:0.5', '0.5:1']
    lambda_f_keys         = [f'{r}_f' for r in biaxial_ratios_sorted]
    lambda_n_keys         = [f'{r}_n' for r in biaxial_ratios_sorted]

    def plot_single(ax, mode_key, data_dict, noise_prefix, active=True):
        if not active:
            ax.axis('off'); return
        d = data_dict.get(mode_key)
        if d is None or d.get('x') is None or d.get('y_exp') is None or len(d['x']) == 0:
            ax.set_title("No Data", fontsize=context.PLOT_LABEL_FONTSIZE)
            return
        x_raw = np.array(d['x']); y_exp = np.array(d['y_exp'])
        idx = np.argsort(x_raw)
        x = x_raw[idx]; y_exp = y_exp[idx]
        y_noisy = np.array(d.get('y_noisy', np.full_like(x, np.nan)))[idx]
        y_ref   = np.array(d.get('y_pred_refined', np.full_like(x, np.nan)))[idx]
        y_pdf   = np.array(d.get('y_pred_pdf', np.full_like(x, np.nan)))[idx]

        ax.plot(x, y_exp, 'o', markersize=5, markerfacecolor='none', color=context.color_original)
        if noise_prefix.split('_')[0] != "RelNoise0pct" and not np.all(np.isnan(y_noisy)):
            ax.plot(x, y_noisy, 'o', markersize=5, color=context.color_noisy_data, alpha=0.7)
        if not np.all(np.isnan(y_ref)): ax.plot(x, y_ref, '-', lw=2, color=context.color_refined)
        if not np.all(np.isnan(y_pdf)): ax.plot(x, y_pdf, '--', lw=2, color=context.color_pdf)

        if x.size:
            xmin, xmax = np.nanmin(x), np.nanmax(x)
            if xmin != xmax:
                ax.set_xticks([xmin, xmax])
                ax.xaxis.set_major_formatter(
                    plt.FormatStrFormatter('%.2f' if (xmax - xmin) < 0.2 else '%.1f')
                )
        y_all = np.concatenate([
            y_exp[~np.isnan(y_exp)],
            y_noisy[~np.isnan(y_noisy)],
            y_ref[~np.isnan(y_ref)],
            y_pdf[~np.isnan(y_pdf)]
        ]) if any([~np.all(np.isnan(a)) for a in [y_exp, y_noisy, y_ref, y_pdf]]) else np.array([])
        if y_all.size:
            ymin, ymax = np.nanmin(y_all), np.nanmax(y_all)
            if ymin != ymax:
                ax.set_yticks([ymin, ymax])
                ax.yaxis.set_major_formatter(
                    plt.FormatStrFormatter('%.2f' if (ymax - ymin) < 0.2 else '%.1f')
                )

        r2_val = rmse_val = np.nan
        mask = ~np.isnan(y_ref) & ~np.isnan(y_exp)
        if np.sum(mask) > 1:
            try:
                r2_val = r2_score(y_exp[mask], y_ref[mask])
                rmse_val = np.sqrt(mean_squared_error(y_exp[mask], y_ref[mask]))
            except ValueError:
                pass
        base_title = d.get('title', mode_key)
        if not np.isnan(r2_val):
            title_full = f'{base_title}\n$R^2$: {r2_val:.3f}, RMSE: {rmse_val:.3f}'
        else:
            title_full = f'{base_title}\n$R^2$/RMSE: N/A'
        ax.set_title(title_full, fontsize=context.PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=context.PLOT_TICK_FONTSIZE)
        plot_metrics[mode_key] = {'R2': r2_val, 'RMSE': rmse_val}

    # Populate
    for r in range(6):
        plot_single(axs[r, 0], shear_modes_sorted[r], temp_plot_series, file_prefix,
                    active=(r < len(shear_modes_sorted)))
    for r in range(6):
        plot_single(axs[r, 1], lambda_f_keys[r] if r < len(lambda_f_keys) else '',
                    temp_plot_series, file_prefix, active=(r < len(lambda_f_keys)))
    for r in range(6):
        plot_single(axs[r, 2], lambda_n_keys[r] if r < len(lambda_n_keys) else '',
                    temp_plot_series, file_prefix, active=(r < len(lambda_n_keys)))

    # Clear labels (custom placement)
    for ax in axs.ravel():
        ax.set_xlabel(""); ax.set_ylabel("")

    # Assign labels to ALL subplots now
    for r in range(6):
        axs[r, 0].set_ylabel("Stress [kPa]", fontsize=context.PLOT_LABEL_FONTSIZE)#, rotation=0,
                            #  ha='right', va='center')

    for r in range(6):
        axs[r, 0].set_xlabel("Shear Strain $\\gamma$ [-]", fontsize=context.PLOT_LABEL_FONTSIZE)
        axs[r, 1].set_xlabel("Stretch $\\lambda_f$ [-]", fontsize=context.PLOT_LABEL_FONTSIZE)
        axs[r, 2].set_xlabel("Stretch $\\lambda_n$ [-]", fontsize=context.PLOT_LABEL_FONTSIZE)

    # Draw for positioning
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Incrust Y labels
    for r in range(6):
        ax = axs[r, 0]
        tick_labels = ax.get_yticklabels()
        widths = []
        for tl in tick_labels:
            if tl.get_text():
                widths.append(tl.get_window_extent(renderer=renderer).width)
        if not widths: continue
        max_w = max(widths)
        fig_w_pixels = fig_w * fig.dpi
        gap = 150
        offset_fig_frac = (max_w + gap) / fig_w_pixels
        ax.yaxis.set_label_coords(-offset_fig_frac, 0.5)

    # Incrust X labels (all subplots)
    for ax in axs.ravel():
        tick_labels_x = ax.get_xticklabels()
        heights = []
        for tl in tick_labels_x:
            if tl.get_text():
                heights.append(tl.get_window_extent(renderer=renderer).height)
        if not heights: continue
        max_h = max(heights)
        fig_h_pixels = fig_h * fig.dpi
        gap_pix = 150
        offset = (max_h + gap_pix) / fig_h_pixels
        ax.xaxis.set_label_coords(0.5, -offset)

    # Spacing
    title_lines = [ax.get_title().count('\n') + 1 for ax in axs.ravel() if ax.get_title()]
    max_lines = max(title_lines) if title_lines else 1
    line_h_in = (context.PLOT_LABEL_FONTSIZE / 72.0) * 0.9
    needed_in = max_lines * line_h_in
    hspace = min(0.9, max(0.15, 1.05 * needed_in / fig_h))
    wspace = 0.35
    fig.subplots_adjust(left=0.065, right=0.985, top=0.93, bottom=0.055,
                        hspace=hspace, wspace=wspace)

    # Legend (top center)
    handles = [line_data]
    if file_prefix.split('_')[0] != "RelNoise0pct":
        handles.append(line_noise)
    handles += [line_refined, line_pdf]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 0.965),
               ncol=4, fontsize=context.PLOT_LEGEND_FONTSIZE + 4)

    # Tighten (no suptitle region needed now)
    fig.tight_layout(rect=[0, 0.035, 1, 0.93], pad=0.25)

    # Save
    filename = f"{file_prefix}_{plot_title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}_Comparison.pdf"
    save_path = os.path.join(save_dir, filename)
    try:
        os.makedirs(save_dir, exist_ok=True)
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(f'Saved plot: {save_path}')
    except Exception as e:
        print(f'Error saving plot {save_path}: {e}')
    plt.close(fig)
    return plot_metrics

def format_term_for_latex(term_name):
    """Formats a term string like 'c2exp(w1(I1-3))' into a LaTeX string."""
    term = term_name

    # Handle special cases first
    term = term.replace('_bar', '')  # remove _bar from internal key name

    # Add subscripts to c, w, I using regex (be more careful with boundaries)
    term = re.sub(r'\bc(\d+)', r'c_{\1}', term)
    term = re.sub(r'\bw(\d+)', r'w_{\1}', term)
    term = re.sub(r'\bI(\d+[a-z]*)', r'I_{\1}', term)

    # Handle bar for I4 invariants after subscripting
    term = re.sub(r'(I_{4[fsn]})', r'\\bar{\1}', term)

    # Format exp - be more careful to not break it
    term = re.sub(r'\bexp\b', r'\\exp', term)

    # Format powers
    term = re.sub(r'\)\^(\d+)', r')^{\1}', term)

    return f'${term}$'

def plot_activation_heatmap(path_variable, coef_path, feature_names_defs, chosen_values_dict,
                            title, x_axis_label, filename=None, log_x=False,
                            reverse_x=True, activation_thresh=1e-5, is_omp=False, *, context):
    """
    Generates a single activation heatmap with multiple vertical lines for different selection criteria.
    """
    n_features, n_steps = coef_path.shape
    if n_steps == 0 or path_variable is None or len(path_variable) != n_steps:
        print(f'Warning: Cannot plot {title}, path or path variable invalid.')
        return

    # --- Create Binary Feature Activation Matrix ---
    feature_status = (np.abs(coef_path) > activation_thresh).astype(float)

    # --- Determine feature entry order (step index) ---
    feature_entry = {}
    for fidx in range(n_features):
        active_indices = np.where(feature_status[fidx, :] > 0)[0]
        feature_entry[fidx] = active_indices[0] if len(active_indices) > 0 else n_steps

    # --- Sort features by entry order ---
    active_feature_indices = [fidx for fidx in range(n_features) if feature_entry[fidx] < n_steps]
    if not active_feature_indices:
         print(f'Warning: No features became active for {title}. Skipping heatmap.')
         return

    sorted_active_features = sorted(active_feature_indices, key=lambda f: feature_entry[f])
    try:
         ordered_basis_names = [format_term_for_latex(feature_names_defs[i]['term_label']) for i in sorted_active_features]
    except Exception as e:
         print(f'WARN: LaTeX formatting failed for heatmap labels: {e}. Using raw names.')
         ordered_basis_names = [feature_names_defs[i]['term_label'] for i in sorted_active_features]

    sorted_status = feature_status[sorted_active_features, :]
    num_sorted_features = len(sorted_active_features)

    # --- Handle X-axis and Reversal ---
    plot_x_values = path_variable
    plot_status = sorted_status
    if reverse_x:
        plot_status = plot_status[:, ::-1]
        plot_x_values = plot_x_values[::-1]

    # --- Plotting ---
    height_per_feature = 0.4
    fig_height = max(5, num_sorted_features * height_per_feature)
    fig, ax = plt.subplots(figsize=(7, fig_height))

    # Use the same colormap as synthetic data
    truncated_cmap = mcolors.LinearSegmentedColormap.from_list(
        "truncated_heatmap_cmap", context.HEATMAP_CMAP(np.linspace(0, 0.75, 256))
    )

    im = ax.imshow(plot_status, cmap=truncated_cmap, aspect='auto', interpolation='nearest', vmin=0, vmax=1)

    # --- Y-axis Ticks and Labels ---
    ax.set_yticks(np.arange(num_sorted_features))
    ax.set_yticklabels(ordered_basis_names, fontsize=context.PLOT_HEATMAP_YLABEL_SIZE)
    ax.set_ylabel('Model Terms', fontsize=context.PLOT_LABEL_FONTSIZE)

    # --- X-axis Ticks and Labels ---
    if log_x:
        tick_indices = np.linspace(0, n_steps - 1, min(8, n_steps), dtype=int)
        ax.set_xticks(tick_indices)
        tick_labels = []
        for i in tick_indices:
            log_val = np.log10(plot_x_values[i] + context.EPS)
            if abs(log_val - round(log_val)) < 0.3:
                tick_labels.append(f'10$^{{{int(round(log_val))}}}$')
            else:
                tick_labels.append(f'10$^{{{log_val:.1f}}}$')
        ax.set_xticklabels(tick_labels, fontsize=context.PLOT_TICK_FONTSIZE)
    else:
        tick_indices = np.linspace(0, n_steps - 1, min(10, n_steps), dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([f'{int(plot_x_values[i])}' for i in tick_indices], fontsize=context.PLOT_TICK_FONTSIZE)
    ax.set_xlabel(x_axis_label, fontsize=context.PLOT_LABEL_FONTSIZE)

    # --- Mark the chosen values ---
    legend_elements = [
        Patch(facecolor=context.HEATMAP_CMAP(0.75), label='Active'),
        Patch(facecolor=context.HEATMAP_CMAP(0.0), edgecolor='darkgray', label='Inactive')
    ]

    for method, info in chosen_values_dict.items():
        chosen_index = info.get('index')  # Now using 'index' instead of 'value'
        label_val_for_display = info.get('label_val', chosen_index)
        color = info.get('color', 'red')
        linestyle = info.get('linestyle', '--')

        if chosen_index is not None and 0 <= chosen_index < n_steps:
            # Direct index mapping with reversal handling
            if reverse_x:
                chosen_x_coord = (n_steps - 1) - chosen_index
            else:
                chosen_x_coord = chosen_index

            ax.axvline(chosen_x_coord, color=color, linestyle=linestyle, linewidth=1.5, alpha=0.9)

            # Format the label for the legend
            if log_x:
                label_val_str = f'10$^{{{np.log10(max(label_val_for_display, context.EPS)):.1f}}}$'
            else:
                label_val_str = f'{int(label_val_for_display)}' if is_omp else f'{label_val_for_display:.2f}'

            vline_label = f'{method}'
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=1.5, ls=linestyle, label=vline_label))
        else:
            print(f'WARN: Chosen index for {method} is invalid ({chosen_index}), cannot plot vertical line.')

    # --- Grid and Legend ---
    ax.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=5, fontsize=context.PLOT_LEGEND_FONTSIZE)

    # --- Final Touches ---
    ax.set_title(title, fontsize=context.PLOT_TITLE_FONTSIZE)
    fig.subplots_adjust(bottom=0.25, left=0.3, right=0.95)

    if filename:
        try:
            os.makedirs(os.path.dirname(filename), exist_ok=True)
            fig.savefig(filename, format='pdf', bbox_inches='tight')
            print(f'Saved activation heatmap: {filename}')
        except Exception as e: print(f'Error saving heatmap: {e}')
    plt.close(fig)


def plot_lasso_selection(*, _clip_nonneg, alphas_cv_plot, alphas_lasso, best_alpha_from_cv, coefs_lasso, context, cv_nmse_mean_plot, cv_nmse_std_plot, df_path, feature_names, save_dir, title_prefix):
    if not df_path.empty and alphas_lasso is not None and coefs_lasso is not None:
        print("\n  Generating plots from unified path data...")

        # Exact models for markers
        cv_exact_model = None; aic_exact_model = None; bic_exact_model = None
        if pd.notna(best_alpha_from_cv):
            cv_idx = (np.abs(df_path['alpha'] - best_alpha_from_cv)).argsort()[0]
            cv_exact_model = df_path.iloc[cv_idx]
        if not df_path.empty:
            aic_exact_model = df_path.loc[df_path['aic'].idxmin()]
            bic_exact_model = df_path.loc[df_path['bic'].idxmin()]

        def find_alpha_index_in_path(target_alpha, alphas_array):
            return np.argmin(np.abs(alphas_array - target_alpha))

        chosen_values_heatmap = {
            'CV':  {'index': find_alpha_index_in_path(cv_exact_model['alpha'], alphas_lasso) if cv_exact_model is not None else None,
                    'label_val': cv_exact_model['k'] if cv_exact_model is not None else None,
                    'color': context.plasma(0.25), 'linestyle': '--'},
            'AIC': {'index': find_alpha_index_in_path(aic_exact_model['alpha'], alphas_lasso) if aic_exact_model is not None else None,
                    'label_val': aic_exact_model['k'] if aic_exact_model is not None else None,
                    'color': context.plasma(0.55), 'linestyle': '-'},
            'BIC': {'index': find_alpha_index_in_path(bic_exact_model['alpha'], alphas_lasso) if bic_exact_model is not None else None,
                    'label_val': bic_exact_model['k'] if bic_exact_model is not None else None,
                    'color': context.plasma(0.85), 'linestyle': '-.'}
        }

        plot_activation_heatmap(
            alphas_lasso,
            # pass the already clipped nonnegative coefs (recompute just in case)
            np.column_stack([_clip_nonneg(coefs_lasso[:, i]) for i in range(coefs_lasso.shape[1])]),
            feature_names, chosen_values_heatmap,
            title=f'Lasso Activation Path', x_axis_label=r'Regularization Strength',
            filename=os.path.join(save_dir, f'{title_prefix}_Lasso_ActivationHeatmap.pdf'),
            log_x=True, reverse_x=True, is_omp=False,
            context=context,
        )

        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)

        alpha_for_cv  = cv_exact_model['alpha']  if cv_exact_model  is not None else None
        alpha_for_aic = aic_exact_model['alpha'] if aic_exact_model is not None else None
        alpha_for_bic = bic_exact_model['alpha'] if bic_exact_model is not None else None

        # Row 0: CV NMSE (mean ± std) and Norm. L1
        ax0_twin = axs[0].twinx()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0 and np.isfinite(cv_nmse_mean_plot).any():
            axs[0].plot(alphas_cv_plot, cv_nmse_mean_plot, 'o-', ms=3, color=context.plasma(0.25))
            if isinstance(cv_nmse_std_plot, np.ndarray) and cv_nmse_std_plot.size == cv_nmse_mean_plot.size:
                try:
                    axs[0].fill_between(alphas_cv_plot,
                                        np.clip(cv_nmse_mean_plot - cv_nmse_std_plot, 0.0, 1.0),
                                        np.clip(cv_nmse_mean_plot + cv_nmse_std_plot, 0.0, 1.0),
                                        color=context.plasma(0.25), alpha=0.15, linewidth=0)
                except Exception:
                    pass
        else:
            axs[0].text(0.5, 0.5, "CV NMSE unavailable", transform=axs[0].transAxes, ha='center', va='center')

        axs[0].set_ylabel('CV\nNMSE', fontsize=context.PLOT_LABEL_FONTSIZE, color=context.plasma(0.25), labelpad=-20)
        ax0_twin.set_ylabel('Norm. L1', fontsize=context.PLOT_LABEL_FONTSIZE, color=context.plasma(0.65), labelpad=-10)

        ax0_twin.plot(df_path['alpha'], df_path['norm_l1'], 's-', ms=3, color=context.plasma(0.65), alpha=0.6)
        if pd.notna(alpha_for_cv):
            axs[0].axvline(alpha_for_cv, color=context.plasma(0.25), ls='--', lw=1.5)
            axs[0].text(alpha_for_cv * 1.5, 0.5, f'$\\lambda_L={alpha_for_cv:.1e}$',
                        rotation=90, va='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.25))

        axs[0].set_ylim(-0.05, 1.05); ax0_twin.set_ylim(-0.05, 1.05)
        axs[0].set_yticks([0.0, 1.0]); ax0_twin.set_yticks([0.0, 1.0])
        ax0_twin.tick_params(axis='y', labelcolor=context.plasma(0.65))
        axs[0].tick_params(axis='y', labelcolor=context.plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_path['alpha'], df_path['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_aic):
            axs[1].axvline(alpha_for_aic, color=context.plasma(0.55), ls='-', lw=1.5)
            axs[1].text(alpha_for_aic * 1.5, np.mean(axs[1].get_ylim()), f'$\\lambda_L={alpha_for_aic:.1e}$',
                        rotation=90, va='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_path['alpha'], df_path['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-20)
        if pd.notna(alpha_for_bic):
            axs[2].axvline(alpha_for_bic, color=context.plasma(0.85), ls='-.', lw=1.5)
            axs[2].text(alpha_for_bic * 1.5, np.mean(axs[2].get_ylim()), f'$\\lambda_L={alpha_for_bic:.1e}$',
                        rotation=90, va='center', fontsize=context.PLOT_LEGEND_FONTSIZE-1, color=context.plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])

        axs[2].set_xlabel('Regularization Strength', fontsize=context.PLOT_LABEL_FONTSIZE)
        x_min = df_path['alpha'].min(); x_max = df_path['alpha'].max()
        if isinstance(alphas_cv_plot, np.ndarray) and alphas_cv_plot.size > 0:
            x_min = min(x_min, np.min(alphas_cv_plot))
            x_max = max(x_max, np.max(alphas_cv_plot))
        for ax in axs:
            ax.set_xscale('log')
            ax.xaxis.set_major_locator(matplotlib.ticker.FixedLocator([x_min, x_max]))
            ax.xaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1e'))
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_Lasso_Criteria.pdf'), format='pdf', bbox_inches='tight')
        plt.close(fig)


def plot_lars_selection(*, _annotate_iter, best_k, context, cv_nmse_mean, cv_nmse_std, df_path, row_aic, row_bic, save_dir, steps_k, title_prefix):
    try:
        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)
        # Row 0: CV NMSE mean ± std
        axs[0].plot(steps_k, cv_nmse_mean, 'o-', ms=3, lw=1.5, color=context.plasma(0.25))
        try:
            axs[0].fill_between(steps_k, cv_nmse_mean - cv_nmse_std, cv_nmse_mean + cv_nmse_std,
                                alpha=0.15, linewidth=0, color=context.plasma(0.25))
        except Exception:
            pass
        axs[0].set_ylabel('CV NMSE', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-10, color=context.plasma(0.25))
        axs[0].set_ylim(-0.05, 1.05); axs[0].set_yticks([0.0, 1.0])
        axs[0].axvline(best_k, color=context.plasma(0.25), ls='--', lw=1.5)
        _annotate_iter(axs[0], best_k, context.plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_path['k'], df_path['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[1].axvline(int(row_aic['k']), color=context.plasma(0.55), ls='-', lw=1.5)
        _annotate_iter(axs[1], int(row_aic['k']), context.plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_path['k'], df_path['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[2].axvline(int(row_bic['k']), color=context.plasma(0.85), ls='-.', lw=1.5)
        _annotate_iter(axs[2], int(row_bic['k']), context.plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])

        # X ticks min–max
        k_min, k_max = int(df_path['k'].min()), int(df_path['k'].max())
        for ax in axs:
            ax.set_xlim(k_min - 0.5, k_max + 0.5)
            ax.set_xticks([k_min, k_max])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        axs[2].set_xlabel('LARS Iteration', fontsize=context.PLOT_LABEL_FONTSIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_LARS_Criteria.pdf'),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'WARN: LARS plotting failed: {e}')


def plot_omp_selection(*, _annotate_iter, best_k, context, cv_nmse_mean, cv_nmse_std, df_path, row_aic, row_bic, save_dir, steps_k, title_prefix):
    try:
        fig, axs = plt.subplots(3, 1, figsize=context.SMALL_FIG_LASSO_LARS_OMP, sharex=True)
        # Row 0: CV NMSE mean ± std
        axs[0].plot(steps_k, cv_nmse_mean, 'o-', ms=3, lw=1.5, color=context.plasma(0.25))
        try:
            axs[0].fill_between(steps_k, cv_nmse_mean - cv_nmse_std, cv_nmse_mean + cv_nmse_std,
                                alpha=0.15, linewidth=0, color=context.plasma(0.25))
        except Exception:
            pass
        axs[0].set_ylabel('CV NMSE', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-10, color=context.plasma(0.25))
        axs[0].set_ylim(-0.05, 1.05); axs[0].set_yticks([0.0, 1.0])
        axs[0].axvline(best_k, color=context.plasma(0.25), ls='--', lw=1.5)
        _annotate_iter(axs[0], best_k, context.plasma(0.25))

        # Row 1: AIC
        axs[1].plot(df_path['k'], df_path['aic'], 'o-', ms=3, color=context.plasma(0.55))
        axs[1].set_ylabel('AIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[1].axvline(int(row_aic['k']), color=context.plasma(0.55), ls='-', lw=1.5)
        _annotate_iter(axs[1], int(row_aic['k']), context.plasma(0.55))
        axs[1].set_yticks([df_path['aic'].min(), df_path['aic'].max()])

        # Row 2: BIC
        axs[2].plot(df_path['k'], df_path['bic'], 'o-', ms=3, color=context.plasma(0.85))
        axs[2].set_ylabel('BIC', fontsize=context.PLOT_LABEL_FONTSIZE, labelpad=-10)
        axs[2].axvline(int(row_bic['k']), color=context.plasma(0.85), ls='-.', lw=1.5)
        _annotate_iter(axs[2], int(row_bic['k']), context.plasma(0.85))
        axs[2].set_yticks([df_path['bic'].min(), df_path['bic'].max()])

        k_min, k_max = int(df_path['k'].min()), int(df_path['k'].max())
        for ax in axs:
            ax.set_xlim(k_min - 0.5, k_max + 0.5)
            ax.set_xticks([k_min, k_max])
            ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
        axs[2].set_xlabel('OMP Iteration', fontsize=context.PLOT_LABEL_FONTSIZE)

        fig.tight_layout(rect=[0, 0, 1, 0.95], pad=0.5)
        plt.savefig(os.path.join(save_dir, f'{title_prefix}_OMP_Criteria.pdf'),
                    format='pdf', bbox_inches='tight')
        plt.close(fig)
    except Exception as e:
        print(f'WARN: OMP plotting failed: {e}')
