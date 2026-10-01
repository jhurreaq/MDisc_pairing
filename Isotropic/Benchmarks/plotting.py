"""Case-specific rendering templates; numerical results are supplied by evaluation."""
import numpy as np
import pandas as pd
import matplotlib.ticker

def style_y_axis(ax, *, context):
    y_lim = ax.get_ylim()

    ticks_to_set = sorted(list(set([y_lim[0], y_lim[1]])))
    if y_lim[0] < 0 < y_lim[1]:
        ticks_to_set.append(0)
    elif np.abs(y_lim[0]) < 0.05 * (y_lim[1] - y_lim[0]) and y_lim[0] > -1e-10 and y_lim[0] * y_lim[1] >=0 : # Min is positive and close to 0
        ticks_to_set.append(0)
    elif np.abs(y_lim[1]) < 0.05 * (y_lim[1] - y_lim[0]) and y_lim[1] < 1e-10 and y_lim[0] * y_lim[1] >=0 : # Max is negative and close to 0
         ticks_to_set.append(0)

    ticks_to_set = sorted(list(set(ticks_to_set)))

    if len(ticks_to_set) > 1:
        processed_ticks = []
        # Heuristic to avoid overly crowded ticks if min/0/max are too close after rounding
        # This simplified logic might need adjustment based on typical data ranges
        if len(ticks_to_set) == 3: # min, 0, max
            if abs(ticks_to_set[0] - ticks_to_set[1]) < 0.05 * abs(y_lim[1]-y_lim[0]) and abs(ticks_to_set[0]) < 0.1: # min and 0 are too close
                processed_ticks = [ticks_to_set[0], ticks_to_set[2]] # Show min and max
            elif abs(ticks_to_set[1] - ticks_to_set[2]) < 0.05 * abs(y_lim[1]-y_lim[0]) and abs(ticks_to_set[2]) < 0.1: # 0 and max are too close
                processed_ticks = [ticks_to_set[0], ticks_to_set[2]] # Show min and max
            else:
                processed_ticks = ticks_to_set
        else: # Should be 2 ticks (min, max) or 1 if min=max
             processed_ticks = ticks_to_set
        ax.set_yticks(processed_ticks)
    else:
        ax.set_yticks(ticks_to_set)

    ax.yaxis.set_major_formatter(matplotlib.ticker.FormatStrFormatter('%.1f'))
    ax.tick_params(axis='y', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)


def plot_predictions(df, ax, *, context):
    """Render UT, PS or EBT nominal stress predictions in kPa on supplied axes."""
    kpa_conversion = 1e3

    mode = df['mode'].iloc[0]
    df_plot = df.copy()

    for col in ['P11', 'P11_true']:
        if col in df_plot.columns:
            df_plot[col] = pd.to_numeric(df_plot[col], errors='coerce') * kpa_conversion

    if 'P11_pred' in df_plot.columns:
         df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')
    else:
         df_plot['P11_pred'] = np.nan

    df_plot['lambda1'] = pd.to_numeric(df_plot['lambda1'], errors='coerce')

    shades = {'UT': (0.4, 0.2, 0.4), 'PS': (0.8, 0.3, 0.7), 'EBT': (0.0, 0.3, 0.1)}[mode]
    color_synth_data, color_true_curve, color_model_fit = [context.plasma(shade) for shade in shades]

    ax.scatter(df_plot['lambda1'], df_plot['P11'], color=color_synth_data, marker='o',
               s=context.PLOT_MARKER_SIZE, label='Synth. Data', alpha=0.3,
               edgecolor='black', linewidth=0.5)

    valid_pred_mask = df_plot['P11_pred'].notna()
    if valid_pred_mask.any():
        df_sorted_pred = df_plot[valid_pred_mask].dropna(subset=['lambda1', 'P11_pred']).sort_values(by='lambda1').drop_duplicates(subset=['lambda1'])
        ax.plot(df_sorted_pred['lambda1'], df_sorted_pred['P11_pred'], color=color_model_fit,
                linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label='Disc. Model')

    if 'P11_true' in df_plot.columns and df_plot['P11_true'].notna().all():
        df_sorted_true = df_plot.dropna(subset=['lambda1', 'P11_true']).sort_values(by='lambda1').drop_duplicates(subset=['lambda1'])
        ax.plot(df_sorted_true['lambda1'], df_sorted_true['P11_true'], color=color_true_curve,
                linestyle=':', linewidth=context.PLOT_PRED_LINEWIDTH*1.2, label='True Model')

    ax.set_xlabel('Stretch $\lambda$ [-]', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Stress $P_{11}$ [kPa]', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.tick_params(axis='x', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)

    first_line_title = {'UT': 'Uniaxial', 'PS': 'Pure Shear', 'EBT': 'Equi-biaxial'}[mode] + ' $P_{11}$'
    ax.set_title(first_line_title, fontsize=context.PLOT_TITLE_FONTSIZE)

    ax.legend(fontsize=context.PLOT_LEGEND_FONTSIZE, frameon=False) # Legend box turned off

    y_min_list = []
    y_max_list = []
    min_valid_y, max_valid_y = np.nan, np.nan

    if 'P11' in df_plot and df_plot['P11'].notna().any():
        y_min_list.append(df_plot['P11'].min())
        y_max_list.append(df_plot['P11'].max())
    if 'P11_true' in df_plot and df_plot['P11_true'].notna().any():
        y_min_list.append(df_plot['P11_true'].min())
        y_max_list.append(df_plot['P11_true'].max())
    if 'P11_pred' in df_plot and df_plot['P11_pred'].notna().any():
        y_min_list.append(df_plot['P11_pred'].min())
        y_max_list.append(df_plot['P11_pred'].max())

    if y_min_list:
        min_valid_y = np.nanmin(y_min_list)
        max_valid_y = np.nanmax(y_max_list)

    if pd.notna(min_valid_y) and pd.notna(max_valid_y):
        y_range_abs = max_valid_y - min_valid_y
        padding = y_range_abs * 0.075 if y_range_abs > 1e-10 else abs(min_valid_y) * 0.1 + (0.01 * kpa_conversion)
        final_y_min = min_valid_y - padding
        final_y_max = max_valid_y + padding
        ax.set_ylim(final_y_min, final_y_max)
        style_y_axis(ax, context=context)
    else:
        ax.tick_params(axis='y', which='major', labelsize=context.PLOT_LABEL_FONTSIZE)

    if df_plot['lambda1'].notna().any():
        min_l = df_plot['lambda1'].min()
        max_l = df_plot['lambda1'].max()
        if pd.notna(min_l) and pd.notna(max_l):
            x_padding = (max_l - min_l) * 0.05 if (max_l - min_l) > 1e-10 else 0.05
            ax.set_xlim(min_l - x_padding, max_l + x_padding)

    return ax
