"""Case-specific rendering templates; numerical results are supplied by evaluation."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

def plot_treloar_prediction(df, axes, r2_overall_precomputed=np.nan, *, context):
    """Plot Treloar data with model predictions (Helper). Needs axes=[ax_full, ax_zoom]. Uses precomputed R2 for title."""
    ax_full, ax_zoom = axes
    colors = {'UT': context.TRUNCATED_PLASMA_CMAP(0.2), 'PS': context.TRUNCATED_PLASMA_CMAP(0.5), 'EBT': context.TRUNCATED_PLASMA_CMAP(0.8)}
    markers = {'UT': 'o', 'PS': 's', 'EBT': '^'}

    df_plot = df[df['dataset']=='Treloar'].copy()
    df_plot['P11'] = pd.to_numeric(df_plot['P11'], errors='coerce')
    df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')

    r2_overall = r2_overall_precomputed

    max_lambda_full = 0
    min_stress_overall = 0
    max_stress_overall = 0
    # --- First pass for limits ---
    for mode in ['UT', 'PS', 'EBT']:
        mode_data = df_plot[df_plot['mode'] == mode]
        if not mode_data.empty:
            valid_mask = ~np.isnan(mode_data['P11']) & ~np.isnan(mode_data['P11_pred'])
            if valid_mask.sum() == 0: continue
            lambda1_vals = mode_data['lambda1'][valid_mask]
            exp_stress = mode_data['P11'][valid_mask]
            pred_stress = mode_data['P11_pred'][valid_mask]
            if len(lambda1_vals) > 0: max_lambda_full = max(max_lambda_full, np.nanmax(lambda1_vals))
            if len(exp_stress) > 0:
                min_stress_overall = min(min_stress_overall, np.nanmin(exp_stress))
                max_stress_overall = max(max_stress_overall, np.nanmax(exp_stress))
            if len(pred_stress) > 0:
                min_stress_overall = min(min_stress_overall, np.nanmin(pred_stress))
                max_stress_overall = max(max_stress_overall, np.nanmax(pred_stress))

    # --- Second pass for plotting ---
    max_y_zoom_overall = -np.inf
    min_y_zoom_overall = np.inf
    for i, ax in enumerate([ax_full, ax_zoom]):
        legend_handles = []
        for mode in ['UT', 'PS', 'EBT']:
            mode_data = df_plot[df_plot['mode'] == mode]
            if not mode_data.empty:
                valid_mask = ~np.isnan(mode_data['P11']) & ~np.isnan(mode_data['P11_pred'])
                if valid_mask.sum() == 0: continue
                lambda1_vals = mode_data['lambda1'][valid_mask]
                exp_stress = mode_data['P11'][valid_mask]
                pred_stress = mode_data['P11_pred'][valid_mask]
                sort_idx = np.argsort(lambda1_vals)
                lambda1_sorted = lambda1_vals.iloc[sort_idx]
                exp_sorted = exp_stress.iloc[sort_idx]
                pred_sorted = pred_stress.iloc[sort_idx]
                sc = ax.scatter(lambda1_sorted, exp_sorted, color=colors[mode], marker=markers[mode], s=context.PLOT_MARKER_SIZE, label=f'{mode} (Exp. data)' if i==0 else None, alpha=0.7, edgecolor='black', linewidth=0.5)
                pl, = ax.plot(lambda1_sorted, pred_sorted, color=colors[mode], linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'{mode} (Model)' if i==0 else None)
                if i == 0: legend_handles.extend([sc, pl])
                if i == 1:
                    mask_zoom_mode = (lambda1_sorted >= 1.0) & (lambda1_sorted <= 4.5)
                    if mask_zoom_mode.sum() > 0:
                         max_y_zoom_overall = max(max_y_zoom_overall, np.nanmax(exp_sorted[mask_zoom_mode]), np.nanmax(pred_sorted[mask_zoom_mode]))
                         min_y_zoom_overall = min(min_y_zoom_overall, np.nanmin(exp_sorted[mask_zoom_mode]), np.nanmin(pred_sorted[mask_zoom_mode]))

        ax.set_xlabel('Stretch λ', fontsize=context.PLOT_LABEL_FONTSIZE)
        ax.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
        ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)

        if i == 0:
            title_r2_str = f'{r2_overall:.3f}' if pd.notna(r2_overall) else "N/A"
            ax.set_title(f'Treloar (Full Range, $R^2 \\approx {title_r2_str}$)', fontsize=context.PLOT_TITLE_FONTSIZE)
            ax.legend(handles=legend_handles, loc='best', fontsize=context.PLOT_LEGEND_FONTSIZE)
            ax.set_xlim(left=1.0, right=max(4.5, max_lambda_full * 1.05))
            ax.set_ylim(bottom=min(0, min_stress_overall - 0.2), top=max(0.2, max_stress_overall * 1.1))
        else:
            ax.set_title(f'Treloar (Zoomed)', fontsize=context.PLOT_TITLE_FONTSIZE)
            ax.set_xlim(1.0, 5.5)
            if not np.isfinite(min_y_zoom_overall) or not np.isfinite(max_y_zoom_overall):
                ax.set_ylim(0, 1)
            else:
                ax.set_ylim(bottom=min(0, min_y_zoom_overall - 0.1), top=max(0.2, max_y_zoom_overall * 1.1))


def plot_Kawabata_prediction(df, axes, r2_p11_precomputed=np.nan, r2_p22_precomputed=np.nan, *, context):
    """Plot Kawabata data with model predictions (Helper). Uses precomputed R2 values for titles."""
    ax_p11, ax_p22 = axes
    df_plot = df[df['dataset']=='Kawabata'].copy()
    df_plot['P11'] = pd.to_numeric(df_plot['P11'], errors='coerce')
    df_plot['P11_pred'] = pd.to_numeric(df_plot['P11_pred'], errors='coerce')
    df_plot['P22'] = pd.to_numeric(df_plot['P22'], errors='coerce')
    df_plot['P22_pred'] = pd.to_numeric(df_plot['P22_pred'], errors='coerce')
    lambda1_values = sorted(df_plot['lambda1'].unique())
    cmap = context.TRUNCATED_PLASMA_CMAP
    colors = cmap(np.linspace(0.1, 0.9, len(lambda1_values)))

    r2_P11 = r2_p11_precomputed
    r2_P22 = r2_p22_precomputed

    valid_p11_plot = ~np.isnan(df_plot['P11']) & ~np.isnan(df_plot['P11_pred'])
    valid_p22_plot = ~np.isnan(df_plot['P22']) & ~np.isnan(df_plot['P22_pred'])

    handles_p11 = []
    max_y_p11 = 0
    for i, lambda1 in enumerate(lambda1_values):
        subset = df_plot[df_plot['lambda1'] == lambda1].sort_values(by='lambda2')
        v_p11 = valid_p11_plot[subset.index]
        if v_p11.sum() == 0: continue
        sc = ax_p11.scatter(subset['lambda2'][v_p11], subset['P11'][v_p11], color=colors[i], s=context.PLOT_MARKER_SIZE, alpha=0.7, marker='o', edgecolor='k', linewidth=0.5)
        pl, = ax_p11.plot(subset['lambda2'][v_p11], subset['P11_pred'][v_p11], color=colors[i], linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}')
        if i == 0:
            handles_p11.extend([Patch(color='gray', alpha=0.7, label='Exp'), Patch(color='black', label='Model')])
        handles_p11.append(pl)
        if v_p11.sum() > 0:
            max_y_p11 = max(max_y_p11, np.nanmax(subset['P11'][v_p11]), np.nanmax(subset['P11_pred'][v_p11]))

    ax_p11.set_xlabel('Stretch λ₂', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax_p11.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    title_r2_p11_str = f'{r2_P11:.3f}' if pd.notna(r2_P11) else "N/A"
    ax_p11.set_title(f'Kawabata P₁₁ ($R^2 \\approx {title_r2_p11_str}$)', fontsize=context.PLOT_TITLE_FONTSIZE)
    ax_p11.legend(handles=handles_p11[:10], fontsize=context.PLOT_LEGEND_FONTSIZE*0.8, loc='best', ncol=2, frameon=False)
    ax_p11.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    ax_p11.set_ylim(bottom= min(0, np.nanmin(df_plot['P11'][valid_p11_plot])-0.1) if valid_p11_plot.sum() > 0 else 0, top=max(0.2, max_y_p11*1.1))

    handles_p22 = []
    max_y_p22 = 0
    for i, lambda1 in enumerate(lambda1_values):
        subset = df_plot[df_plot['lambda1'] == lambda1].sort_values(by='lambda2')
        v_p22 = valid_p22_plot[subset.index]
        if v_p22.sum() == 0: continue
        sc = ax_p22.scatter(subset['lambda2'][v_p22], subset['P22'][v_p22], color=colors[i], s=context.PLOT_MARKER_SIZE, alpha=0.7, marker='s', edgecolor='k', linewidth=0.5)
        pl, = ax_p22.plot(subset['lambda2'][v_p22], subset['P22_pred'][v_p22], color=colors[i], linestyle='--', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}')
        if i == 0:
            handles_p22.extend([Patch(color='gray', alpha=0.7, label='Exp'), Patch(color='black', linestyle='--', label='Model')])
        handles_p22.append(pl)
        if v_p22.sum() > 0:
            max_y_p22 = max(max_y_p22, np.nanmax(subset['P22'][v_p22]), np.nanmax(subset['P22_pred'][v_p22]))

    ax_p22.set_xlabel('Stretch λ₂', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax_p22.set_ylabel('Nominal Stress P₂₂ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    title_r2_p22_str = f'{r2_P22:.3f}' if pd.notna(r2_P22) else "N/A"
    ax_p22.set_title(f'Kawabata P₂₂ ($R^2 \\approx {title_r2_p22_str}$)', fontsize=context.PLOT_TITLE_FONTSIZE)
    ax_p22.legend(handles=handles_p22[:10], fontsize=context.PLOT_LEGEND_FONTSIZE*0.8, loc='best', ncol=2, frameon=False)
    ax_p22.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    ax_p22.set_ylim(bottom= min(0, np.nanmin(df_plot['P22'][valid_p22_plot])-0.1) if valid_p22_plot.sum() > 0 else 0, top=max(0.2, max_y_p22*1.1))


def plot_treloar_data(df, ax=None, *, context):
    """Plot Treloar dataset - Helper for raw data plot"""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    colors = {'UT': 'blue', 'PS': 'green', 'EBT': 'red'}
    markers = {'UT': 'o', 'PS': 's', 'EBT': '^'}
    for mode in ['UT', 'PS', 'EBT']:
        mode_data = df[df['mode'] == mode]
        p11_data = pd.to_numeric(mode_data['P11'], errors='coerce')
        valid_mask = ~np.isnan(p11_data)
        if valid_mask.sum() > 0:
            strain = (mode_data['lambda1'][valid_mask] - 1) * 100
            stress = p11_data[valid_mask]
            sort_idx = np.argsort(strain)
            ax.scatter(strain.iloc[sort_idx], stress.iloc[sort_idx], color=colors[mode], marker=markers[mode], label=mode, alpha=0.8, s=context.PLOT_MARKER_SIZE, edgecolor='k', linewidth=0.5)
    ax.set_xlabel('Strain (%)', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Nominal Stress P₁₁ (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_title('Treloar Data', fontsize=context.PLOT_TITLE_FONTSIZE)
    ax.legend(fontsize=context.PLOT_LEGEND_FONTSIZE)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    return ax


def plot_Kawabata_data(df, ax=None, *, context):
    """Plot Kawabata biaxial dataset - Helper for raw data plot"""
    if ax is None: fig, ax = plt.subplots(figsize=(6, 5))
    lambda1_values = sorted(df['lambda1'].unique())
    cmap = context.TRUNCATED_PLASMA_CMAP
    colors = cmap(np.linspace(0.1, 0.9, len(lambda1_values)))
    handles = [] # For legend
    for i, lambda1 in enumerate(lambda1_values):
        subset = df[df['lambda1'] == lambda1].copy()
        subset['P11'] = pd.to_numeric(subset['P11'], errors='coerce')
        subset['P22'] = pd.to_numeric(subset['P22'], errors='coerce')
        subset = subset.sort_values(by='lambda2')
        valid_p11 = ~np.isnan(subset['P11'])
        valid_p22 = ~np.isnan(subset['P22'])
        line_p11, = ax.plot(subset['lambda2'][valid_p11], subset['P11'][valid_p11], marker='o', markersize=4, color=colors[i], linestyle='-', linewidth=context.PLOT_PRED_LINEWIDTH, label=f'λ₁={lambda1:.2f}') if valid_p11.sum() > 0 else (None,)
        line_p22, = ax.plot(subset['lambda2'][valid_p22], subset['P22'][valid_p22], marker='s', markersize=4, color=colors[i], linestyle='--', linewidth=context.PLOT_PRED_LINEWIDTH) if valid_p22.sum() > 0 else (None,)
        if i==0: handles.extend([Patch(color='gray', label='P₁₁'), Patch(color='gray', linestyle='--', label='P₂₂')]) # General labels
        if line_p11: handles.append(line_p11) # Add specific lambda line handle
    ax.set_xlabel('λ₂', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_ylabel('Nominal Stress (MPa)', fontsize=context.PLOT_LABEL_FONTSIZE)
    ax.set_title('Kawabata Biaxial Data', fontsize=context.PLOT_TITLE_FONTSIZE)
    ax.legend(handles=handles[:12], fontsize=context.PLOT_LEGEND_FONTSIZE*0.9)
    ax.grid(alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=context.PLOT_TICK_FONTSIZE)
    return ax
