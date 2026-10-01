"""Case-specific rendering templates; numerical results are supplied by evaluation."""
import os
import numpy as np
import matplotlib.pyplot as plt
from mdisc.rendering import save_figure

def render_predictions(records, prediction, reference, noisy, clean, mode_scores,
                       plot_title_suffix, save_dir, file_prefix, *, context):
    temp_plot_series = {}
    for mode in records['mode'].unique():
        indices = np.flatnonzero(records['mode'].to_numpy() == mode)
        temp_plot_series[mode] = {'x': records.iloc[indices]['x'].to_numpy(),
                                 'y_exp': clean[indices], 'y_noisy': noisy[indices],
                                 'y_pred_refined': prediction[indices], 'y_pred_pdf': reference[indices],
                                 'title': mode}
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
            ax.axis('off')
            return
        d = data_dict.get(mode_key)
        if d is None or d.get('x') is None or d.get('y_exp') is None or len(d['x']) == 0:
            ax.set_title("No Data", fontsize=context.PLOT_LABEL_FONTSIZE)
            return
        x_raw = np.array(d['x'])
        y_exp = np.array(d['y_exp'])
        idx = np.argsort(x_raw)
        x = x_raw[idx]
        y_exp = y_exp[idx]
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

        score = mode_scores.get(mode_key, {})
        r2_val = score.get('r2')
        r2_val = np.nan if r2_val is None else r2_val
        rmse_val = score.get('rmse', np.nan)
        base_title = d.get('title', mode_key)
        if not np.isnan(r2_val):
            title_full = f'{base_title}\n$R^2$: {r2_val:.3f}, RMSE: {rmse_val:.3f}'
        else:
            title_full = f'{base_title}\n$R^2$/RMSE: N/A'
        ax.set_title(title_full, fontsize=context.PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', labelsize=context.PLOT_TICK_FONTSIZE)


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
        ax.set_xlabel("")
        ax.set_ylabel("")

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
    save_figure(fig, os.path.join(save_dir, f'{file_prefix}_{plot_title_suffix}_Comparison.pdf'))
