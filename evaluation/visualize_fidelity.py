# %%

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.cm import ScalarMappable
from matplotlib.colors import LinearSegmentedColormap, Normalize, to_rgba
from matplotlib.ticker import MaxNLocator, PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import seaborn as sns
import os
from load_data import *
from lifelines import KaplanMeierFitter
from lifelines import CoxPHFitter
from scipy.spatial.distance import jensenshannon

plt.style.use("default")

# Configuration
cohort_folder = "demo_0"

eval_set = "generation_train"

cohort = cohort_folder.split("_")[0]
cfg = load_config()
cohort_config = cfg["cohort_configs"][cohort]

fidelity_dir = get_fidelity_evaluation_dir(cohort_folder=cohort_folder, cfg=cfg)
output_dir = os.path.join(fidelity_dir, "figures")
os.makedirs(output_dir, exist_ok=True)

model_order = cfg["model_order"]
model_display_names = cfg["model_display_names"]
font_size = cfg.get("plotting", {}).get("font_size", 20)
annotation_font_size = cfg.get("plotting", {}).get("annotation_font_size", 20)
fig_width = 12


# %% joint survival time distribution
def plot_joint_survival_time_distribution(eval_set):
    """Arrange survival-time histograms with panels by row and methods by column."""
    cfg, datasets, _ = load_cohort_data(cohort_folder, verbose=True)
    cohort_config = cfg["cohort_configs"][cohort]

    treatment_col = cohort_config["treatment"]
    event_col = cohort_config.get(
        "outcome_event", cohort_config.get("outcome_censoring")
    )
    time_col = cohort_config["outcome_time"]
    treatment_labels = get_label_map(cfg, cohort, treatment_col)
    event_labels = get_label_map(cfg, cohort, event_col)

    df_ref = datasets[eval_set]
    colors_dict = cfg.get("plotting", {}).get("colors", {})
    ref_color = colors_dict[eval_set]
    panel_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]
    n_models = len(model_order)

    fig, axes = plt.subplots(
        len(panel_pairs),
        n_models,
        figsize=(fig_width + 1, 16),
        sharex=True,
        sharey="row",
    )
    axes = np.atleast_2d(axes)

    for col_idx, model_name in enumerate(model_order):
        model_display = model_display_names[model_name]
        df_synth = datasets[model_name]
        synth_color = colors_dict[model_display]

        for row_idx, (treatment_value, event_value) in enumerate(panel_pairs):
            ax = axes[row_idx, col_idx]

            ref_mask = (df_ref[treatment_col] == treatment_value) & (
                df_ref[event_col] == event_value
            )
            synth_mask = (df_synth[treatment_col] == treatment_value) & (
                df_synth[event_col] == event_value
            )

            t_ref = pd.to_numeric(
                df_ref.loc[ref_mask, time_col], errors="coerce"
            ).dropna()
            t_synth = pd.to_numeric(
                df_synth.loc[synth_mask, time_col], errors="coerce"
            ).dropna()

            if len(t_ref) > 0:
                ax.hist(
                    t_ref,
                    bins=30,
                    alpha=0.5,
                    color=ref_color,
                    density=False,
                    edgecolor="black",
                    linewidth=0.4,
                    label="Original",
                )
            if len(t_synth) > 0:
                ax.hist(
                    t_synth,
                    bins=30,
                    alpha=0.5,
                    color=synth_color,
                    density=False,
                    edgecolor="black",
                    linewidth=0.4,
                    label=model_display,
                )

            if len(t_ref) > 1 and len(t_synth) > 1:
                bins = np.histogram_bin_edges(
                    np.concatenate([t_ref.to_numpy(), t_synth.to_numpy()]),
                    bins=30,
                )
                p_ref, _ = np.histogram(t_ref, bins=bins, density=False)
                p_synth, _ = np.histogram(t_synth, bins=bins, density=False)
                p_ref = p_ref.astype(float)
                p_synth = p_synth.astype(float)
                if p_ref.sum() > 0 and p_synth.sum() > 0:
                    p_ref /= p_ref.sum()
                    p_synth /= p_synth.sum()
                    jsd_value = float(jensenshannon(p_ref, p_synth, base=2))
                    ax.text(
                        0.03,
                        0.95,
                        f"JSD = {jsd_value:.3f}",
                        transform=ax.transAxes,
                        fontsize=annotation_font_size,
                        verticalalignment="top",
                        bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
                    )

            tr_label = treatment_labels.get(treatment_value, treatment_value)
            ev_label = event_labels.get(event_value, event_value)
            if row_idx == len(panel_pairs) - 1:
                ax.set_xlabel("Survival time", fontsize=font_size)
            if col_idx == 0:
                ax.set_ylabel("Number of patients", fontsize=font_size)
            ax.tick_params(labelsize=font_size - 2)
            ax.grid(True, alpha=0.2, axis="y")

    handles = [
        Patch(
            facecolor=ref_color,
            edgecolor="black",
            linewidth=0.4,
            alpha=0.5,
            label="Original",
        )
    ]
    handles.extend(
        [
            Patch(
                facecolor=colors_dict[model_display_names[m]],
                edgecolor="black",
                linewidth=0.4,
                alpha=0.5,
                label=model_display_names[m],
            )
            for m in model_order
        ]
    )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=min(len(handles), 4),
        fontsize=font_size - 1,
        frameon=True,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=0.18, hspace=0.28, left=0.1, right=0.99, top=0.88, bottom=0.07
    )
    row_center_x = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2
    for row_idx, (treatment_value, event_value) in enumerate(panel_pairs):
        tr_label = treatment_labels.get(treatment_value, treatment_value)
        ev_label = event_labels.get(event_value, event_value)
        row_top = axes[row_idx, 0].get_position().y1
        if row_idx == 0:
            y_text = 0.02 + row_top + 0.9 * (fig.subplotpars.top - row_top)
        else:
            prev_bottom = axes[row_idx - 1, 0].get_position().y0
            y_text = row_top + (prev_bottom - row_top) / 2
        fig.text(
            row_center_x,
            y_text,
            f"W={tr_label}, C={ev_label}",
            ha="center",
            va="center",
            fontsize=font_size,
        )
    output_file = os.path.join(
        output_dir,
        f"joint_survival_time_distribution_{cohort}_ref[{eval_set}].png",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.show()


plot_joint_survival_time_distribution(eval_set=eval_set)


# %% joint survival time by year
def plot_joint_survival_time_by_year(eval_set, show_outliers=True):
    """Arrange year-of-diagnosis boxplots with panels by row and methods by column."""
    cfg, datasets, _ = load_cohort_data(cohort_folder, verbose=True)
    cohort_config = cfg["cohort_configs"][cohort]

    treatment_col = cohort_config["treatment"]
    event_col = cohort_config.get(
        "outcome_event", cohort_config.get("outcome_censoring")
    )
    time_col = cohort_config["outcome_time"]
    year_col = cohort_config.get("year_variable", "Year_of_Diagnosis")

    df_ref = datasets[eval_set]
    if year_col not in df_ref.columns:
        print(
            f"Skipping joint Year_of_Diagnosis plot: column '{year_col}' not found in ref data."
        )
        return

    treatment_labels = get_label_map(cfg, cohort, treatment_col)
    event_labels = get_label_map(cfg, cohort, event_col)
    colors_dict = cfg["plotting"]["colors"]
    ref_color = colors_dict[eval_set]
    available_models = [m for m in model_order if year_col in datasets[m].columns]
    panel_pairs = [(0, 0), (0, 1), (1, 0), (1, 1)]

    if not available_models:
        print(
            f"Skipping joint Year_of_Diagnosis plot: column '{year_col}' not found in synthetic data."
        )
        return

    if cohort_folder.startswith("lung"):
        figsize = (fig_width + 8, 18)
    elif cohort_folder.startswith("breast"):
        figsize = (fig_width + 4, 18)
    elif cohort_folder.startswith("demo"):
        figsize = (fig_width - 2, 14)
    else:
        raise ValueError(
            f"Unexpected cohort folder: {cohort_folder}. Please adjust figsize accordingly."
        )
    fig, axes = plt.subplots(
        len(panel_pairs),
        len(available_models),
        figsize=figsize,
        sharey="row",
    )
    axes = np.atleast_2d(axes)

    for col_idx, model_name in enumerate(available_models):
        model_display = model_display_names[model_name]
        df_synth = datasets[model_name]
        synth_color = colors_dict[model_display]

        for row_idx, (treatment_value, event_value) in enumerate(panel_pairs):
            ax = axes[row_idx, col_idx]

            ref_subset = df_ref[
                (df_ref[treatment_col] == treatment_value)
                & (df_ref[event_col] == event_value)
            ][[year_col, time_col]].copy()
            synth_subset = df_synth[
                (df_synth[treatment_col] == treatment_value)
                & (df_synth[event_col] == event_value)
            ][[year_col, time_col]].copy()

            ref_subset[year_col] = pd.to_numeric(ref_subset[year_col], errors="coerce")
            ref_subset[time_col] = pd.to_numeric(ref_subset[time_col], errors="coerce")
            synth_subset[year_col] = pd.to_numeric(
                synth_subset[year_col], errors="coerce"
            )
            synth_subset[time_col] = pd.to_numeric(
                synth_subset[time_col], errors="coerce"
            )

            ref_subset = ref_subset.dropna()
            synth_subset = synth_subset.dropna()

            if not ref_subset.empty:
                ref_subset[year_col] = ref_subset[year_col].astype(int)
            if not synth_subset.empty:
                synth_subset[year_col] = synth_subset[year_col].astype(int)

            year_order = sorted(
                set(ref_subset[year_col].tolist())
                | set(synth_subset[year_col].tolist())
            )

            tr_label = treatment_labels.get(treatment_value, treatment_value)
            ev_label = event_labels.get(event_value, event_value)

            if not year_order:
                ax.text(
                    0.5,
                    0.5,
                    "No data",
                    transform=ax.transAxes,
                    ha="center",
                    va="center",
                    fontsize=font_size - 4,
                )
                ax.tick_params(labelsize=font_size - 6)
                continue

            ref_subset["Dataset"] = "Original"
            synth_subset["Dataset"] = model_display
            plot_df = pd.concat([ref_subset, synth_subset], ignore_index=True)

            sns.boxplot(
                data=plot_df,
                x=year_col,
                y=time_col,
                hue="Dataset",
                order=year_order,
                hue_order=["Original", model_display],
                palette={"Original": ref_color, model_display: synth_color},
                ax=ax,
                linewidth=1,
                showfliers=show_outliers,
            )

            if row_idx == len(panel_pairs) - 1:
                ax.set_xlabel("Year of diagnosis", fontsize=font_size)
            else:
                ax.set_xlabel(None)
            if col_idx == 0:
                ax.set_ylabel("Survival time", fontsize=font_size)
            else:
                ax.set_ylabel(None)
            ax.tick_params(axis="x", rotation=30, labelsize=font_size - 3)
            ax.tick_params(axis="y", labelsize=font_size - 3)
            ax.grid(True, alpha=0.2, axis="y")

            if ax.get_legend() is not None:
                ax.get_legend().remove()

    legend_handles = [
        Patch(facecolor=ref_color, edgecolor="black", linewidth=1, label="Original")
    ]
    legend_handles.extend(
        [
            Patch(
                facecolor=colors_dict[model_display_names[m]],
                edgecolor="black",
                linewidth=1,
                label=model_display_names[m],
            )
            for m in available_models
        ]
    )
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=min(len(legend_handles), 4),
        fontsize=font_size - 1,
        frameon=True,
    )

    plt.tight_layout()
    fig.subplots_adjust(
        wspace=0.08, hspace=0.48, left=0.11, right=0.995, top=0.88, bottom=0.08
    )
    row_center_x = (axes[0, 0].get_position().x0 + axes[0, -1].get_position().x1) / 2
    for row_idx, (treatment_value, event_value) in enumerate(panel_pairs):
        tr_label = treatment_labels.get(treatment_value, treatment_value)
        ev_label = event_labels.get(event_value, event_value)
        row_top = axes[row_idx, 0].get_position().y1
        if row_idx == 0:
            y_text = 0.015 + row_top + 0.65 * (fig.subplotpars.top - row_top)
        else:
            prev_bottom = axes[row_idx - 1, 0].get_position().y0
            y_text = row_top + (prev_bottom - row_top) / 2 - 0.02
        fig.text(
            row_center_x,
            y_text,
            f"W={tr_label}, C={ev_label}",
            ha="center",
            va="center",
            fontsize=font_size,
        )
    output_file = os.path.join(
        output_dir,
        f"joint_survival_time_by_year_{cohort}_ref[{eval_set}].png",
    )
    plt.savefig(output_file, dpi=300, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.show()


plot_joint_survival_time_by_year(eval_set=eval_set, show_outliers=False)


# %% functions
def _censor_at_horizon(t, e, h):
    t = np.asarray(t)
    e = np.asarray(e)
    t_horizon = np.minimum(t, h)
    e_horizon = np.where(t > h, 0, e)
    return t_horizon, e_horizon


def _plot_km_curve(ax, row):
    kmf = KaplanMeierFitter()
    kmf.fit(row["times"], row["events"], label=row["label"])
    kmf.plot_survival_function(
        ax=ax,
        color=row["color"],
        linewidth=row.get("linewidth", 2),
        ci_show=False,
        linestyle=row.get("linestyle", "-"),
    )
    line = ax.get_lines()[-1]
    line.set_alpha(row.get("alpha", 1.0))
    line.set_zorder(row.get("zorder", line.get_zorder()))
    return line


def _km_ylim_from_curve_rows(curve_rows, xlim=None):
    y_candidates = []
    for row in curve_rows:
        times = np.asarray(row["times"], dtype=float)
        events = np.asarray(row["events"], dtype=int)
        kmf = KaplanMeierFitter()
        kmf.fit(times, events)
        surv_df = kmf.survival_function_
        if xlim is not None:
            mask = (surv_df.index.to_numpy(dtype=float) >= xlim[0]) & (
                surv_df.index.to_numpy(dtype=float) <= xlim[1]
            )
            y_vals = surv_df.iloc[mask, 0].to_numpy(dtype=float)
            if y_vals.size == 0:
                y_vals = surv_df.iloc[:, 0].to_numpy(dtype=float)
        else:
            y_vals = surv_df.iloc[:, 0].to_numpy(dtype=float)
        y_candidates.extend(y_vals.tolist())

    if not y_candidates:
        return 0.0, 1.02

    y_min = max(0.0, min(y_candidates) - 0.03)
    y_max = min(1.02, max(y_candidates) + 0.03)
    if y_max <= y_min:
        y_max = min(1.02, y_min + 0.1)
    return y_min, y_max


def _risk_counts_by_treatment(
    df, treatment_col, time_col, treatment_groups, time_points
):
    treatment_values = pd.to_numeric(df[treatment_col], errors="coerce")
    observed_times = pd.to_numeric(df[time_col], errors="coerce")

    counts = {}
    for treatment_value in treatment_groups:
        arm_times = observed_times[
            (treatment_values == treatment_value) & observed_times.notna()
        ]
        counts[treatment_value] = [
            int(np.sum(arm_times.to_numpy(dtype=float) >= time_point))
            for time_point in time_points
        ]

    return pd.DataFrame(counts, index=time_points).T


def _format_time_point(time_point):
    if float(time_point).is_integer():
        return f"{int(time_point)}"
    return f"{time_point:g}"


def _get_treatment_label(treatment_labels, treatment_value):
    label = treatment_labels.get(treatment_value)
    if label is None:
        try:
            label = treatment_labels.get(int(treatment_value))
        except (TypeError, ValueError):
            label = None
    if label is not None:
        return str(label)
    if isinstance(treatment_value, float) and treatment_value.is_integer():
        treatment_value = int(treatment_value)
    return f"Treatment {treatment_value}"


# %% Updated mixed fidelity


def _build_method_km_panel_data(
    df_ref,
    df_method,
    treatment_col,
    time_col,
    event_col,
    treatment_groups,
    time_points,
    max_horizon,
    ref_display,
    method_display,
    treatment_labels,
    ref_color,
    method_color,
    ref_curve_linewidth,
    ref_curve_alpha,
    method_curve_linewidth,
):
    ref_treatment_values = pd.to_numeric(df_ref[treatment_col], errors="coerce")
    method_treatment_values = pd.to_numeric(df_method[treatment_col], errors="coerce")
    ref_counts_by_treatment = _risk_counts_by_treatment(
        df_ref, treatment_col, time_col, treatment_groups, time_points
    )
    method_counts_by_treatment = _risk_counts_by_treatment(
        df_method, treatment_col, time_col, treatment_groups, time_points
    )

    ref_curve_rows = []
    method_curve_rows = []
    heatmap_values = []
    annotations = []
    row_labels = []
    column_labels = [_format_time_point(time_point) for time_point in time_points]

    for treatment_idx, treatment in enumerate(treatment_groups):
        linestyle = "-" if treatment_idx == 0 else "--"
        treatment_label = treatment_labels.get(treatment, f"Treatment {treatment}")

        ref_mask = ref_treatment_values == treatment
        method_mask = method_treatment_values == treatment
        ref_t_horizon, ref_e_horizon = _censor_at_horizon(
            df_ref.loc[ref_mask, time_col],
            df_ref.loc[ref_mask, event_col],
            max_horizon,
        )
        method_t_horizon, method_e_horizon = _censor_at_horizon(
            df_method.loc[method_mask, time_col],
            df_method.loc[method_mask, event_col],
            max_horizon,
        )

        ref_curve_rows.append(
            {
                "label": ref_display,
                "color": ref_color,
                "times": np.asarray(ref_t_horizon, dtype=float),
                "events": np.asarray(ref_e_horizon, dtype=int),
                "linewidth": ref_curve_linewidth,
                "alpha": ref_curve_alpha,
                "linestyle": linestyle,
                "zorder": 0,
            }
        )
        method_curve_rows.append(
            {
                "label": method_display,
                "color": method_color,
                "times": np.asarray(method_t_horizon, dtype=float),
                "events": np.asarray(method_e_horizon, dtype=int),
                "linewidth": method_curve_linewidth,
                "linestyle": linestyle,
                "zorder": 1,
            }
        )

        ref_counts = ref_counts_by_treatment.loc[treatment]
        method_counts = method_counts_by_treatment.loc[treatment]
        original_values = [
            int(ref_counts.loc[time_point]) for time_point in time_points
        ]
        diff_values = [
            int(method_counts.loc[time_point] - ref_counts.loc[time_point])
            for time_point in time_points
        ]

        heatmap_values.append(original_values)
        annotations.append([f"{value}" for value in original_values])
        row_labels.append(f"{treatment_label} (orig)")
        heatmap_values.append(diff_values)
        annotations.append([f"{value:+d}" for value in diff_values])
        row_labels.append(f"{treatment_label} (synth)")

    curve_rows = ref_curve_rows + method_curve_rows
    heatmap_df = pd.DataFrame(
        heatmap_values, index=row_labels, columns=column_labels, dtype=float
    )
    annotation_df = pd.DataFrame(annotations, index=row_labels, columns=column_labels)
    return curve_rows, heatmap_df, annotation_df


def plot_updated_joint_fidelity_summary(eval_set, remove_average: bool = True):
    """Plot an updated mixed fidelity summary with KM at-risk-count heatmaps."""

    joint_models = ["CTGAN", "TabDiff", "OncoSynth"]
    joint_figsize = (10.2, 18.5)
    joint_dpi = 300
    joint_text_fontsize = font_size
    joint_small_fontsize = font_size - 5
    row_gap_12 = 0.6
    row_gap_23 = 0.55
    outer_left = 0.08
    outer_right = 0.95
    outer_top = 0.97
    outer_bottom = 0.06
    mixed_heatmap_width_ratios = [0.92, 0.92, 0.92, 0.05]
    mixed_heatmap_wspace = 0.18
    mixed_prevalence_wspace = 0.2
    mixed_prevalence_bar_width = 0.62
    mixed_prevalence_fill_alpha = 0.5
    mixed_prevalence_edge_linewidth = 2
    mixed_prevalence_hline_linewidth = 3
    updated_km_ref_curve_linewidth = 3
    updated_km_model_curve_linewidth = 3
    updated_covariate_heatmap_vmin = 0.0
    updated_covariate_heatmap_vmax = None
    updated_covariate_heatmap_vmax_percentile = 90
    updated_method_color_margin_linewidth = 8.0
    updated_covariate_heatmap_cmap = LinearSegmentedColormap.from_list(
        "covariate_blue_to_early_red",
        [
            (0.00, "#97ADCB"),
            (0.20, "#97ADCB"),
            (1.00, "#d17373"),
        ],
    )

    updated_km_count_heatmap_vmin = 0.0
    if cohort.startswith("demo"):
        updated_km_count_heatmap_vmax = 250
    updated_km_count_heatmap_cmap = LinearSegmentedColormap.from_list(
        "km_count_blue_to_early_red",
        [
            (0.00, "#97ADCB"),
            (0.25, "#97ADCB"),
            (1.00, "#d17373"),
        ],
    )
    updated_km_count_heatmap_cmap.set_bad("white")
    updated_km_heatmap_height_ratio = 1.45
    updated_km_height_ratios = [2.0, updated_km_heatmap_height_ratio]
    updated_km_hspace = 0.42
    updated_km_wspace = 0.2
    updated_km_heatmap_annotation_size = joint_small_fontsize
    updated_km_heatmap_annotation_size_bottom = max(
        6, updated_km_heatmap_annotation_size - 2
    )
    updated_km_heatmap_left_margin_x = -0.18
    updated_km_heatmap_row_gap_width = 3.0
    updated_km_heatmap_y_tick_label_pad = 12

    if cohort.startswith("breast"):
        mixed_km_shown_range = (0, 120)
    elif cohort.startswith("lung"):
        mixed_km_shown_range = (0, 60)
    elif cohort.startswith("demo"):
        mixed_km_shown_range = (0, 120)
    else:
        raise ValueError(f"Unsupported cohort: {cohort}")

    mixed_km_main_max_x_ticks = 4
    updated_km_time_points = np.linspace(
        mixed_km_shown_range[0],
        mixed_km_shown_range[1],
        mixed_km_main_max_x_ticks,
        dtype=float,
    )
    if len(updated_km_time_points) > 1:
        km_tick_spacing = updated_km_time_points[1] - updated_km_time_points[0]
    else:
        km_tick_spacing = mixed_km_shown_range[1] - mixed_km_shown_range[0]
    updated_km_aligned_xlim = (
        updated_km_time_points[0] - km_tick_spacing / 2,
        updated_km_time_points[-1] + km_tick_spacing / 2,
    )

    updated_km_ref_curve_alpha = 0.9

    selected_models = [m for m in joint_models if m in model_order]
    if len(selected_models) != 3:
        raise ValueError(
            f"Expected exactly 3 methods in joint_models, found {len(selected_models)}: {selected_models}"
        )

    selected_displays = [model_display_names[m] for m in selected_models]
    colors_dict = cfg.get("plotting", {}).get("colors", {})
    ref_display = model_display_names[eval_set]
    ref_color = colors_dict[eval_set]
    fig = plt.figure(figsize=joint_figsize)
    outer_gs = gridspec.GridSpec(
        5,
        1,
        figure=fig,
        height_ratios=[1.2, row_gap_12, 0.85, row_gap_23, 1.85],
        hspace=0.0,
        left=outer_left,
        right=outer_right,
        top=outer_top,
        bottom=outer_bottom,
    )

    def _style_joint_colorbar(colorbar, label):
        colorbar.set_label(
            label,
            fontsize=joint_small_fontsize,
            rotation=90,
            labelpad=0,
        )
        colorbar.ax.yaxis.set_label_position("left")
        colorbar.ax.tick_params(labelsize=joint_small_fontsize)
        colorbar.outline.set_visible(True)
        colorbar.outline.set_linewidth(0.8)

    # Row 1: covariate fidelity heatmaps with blue-to-red color scale
    combined_file = os.path.join(
        fidelity_dir, f"univariate_fidelity_combined_ref[{eval_set}].csv"
    )
    df_combined = pd.read_csv(combined_file, index_col=0)
    df_combined = df_combined.rename(index=lambda x: get_display_name(cfg, cohort, x))

    model_data = {}
    observed_covariate_min, observed_covariate_vmax = np.inf, -np.inf
    for model_name in selected_models:
        model_display = model_display_names[model_name]
        model_cols = [
            col for col in df_combined.columns if col.startswith(model_display)
        ]
        df_model = df_combined[model_cols].copy()
        df_model.columns = [
            col.replace(f"{model_display}_", "").replace("_", " ") for col in model_cols
        ]
        plot_data = (
            df_model[df_model.index != "Average"] if remove_average else df_model
        )
        model_data[model_display] = plot_data
        observed_covariate_min = min(observed_covariate_min, plot_data.min().min())
        observed_covariate_vmax = max(
            observed_covariate_vmax,
            np.percentile(plot_data, updated_covariate_heatmap_vmax_percentile),
        )

    covariate_heatmap_vmin = (
        observed_covariate_min
        if updated_covariate_heatmap_vmin is None
        else updated_covariate_heatmap_vmin
    )
    covariate_heatmap_vmax = (
        observed_covariate_vmax
        if updated_covariate_heatmap_vmax is None
        else updated_covariate_heatmap_vmax
    )

    row1_gs = gridspec.GridSpecFromSubplotSpec(
        1,
        4,
        subplot_spec=outer_gs[0],
        width_ratios=mixed_heatmap_width_ratios,
        wspace=mixed_heatmap_wspace,
    )
    heatmap_axes = []
    for idx, (model_name, model_display) in enumerate(
        zip(selected_models, selected_displays)
    ):
        ax = fig.add_subplot(
            row1_gs[0, idx], sharey=heatmap_axes[0] if idx > 0 else None
        )
        sns.heatmap(
            model_data[model_display],
            ax=ax,
            cmap=updated_covariate_heatmap_cmap,
            annot=True,
            fmt=".2f",
            annot_kws={"size": joint_small_fontsize, "color": "black"},
            cbar=False,
            vmin=covariate_heatmap_vmin,
            vmax=covariate_heatmap_vmax,
            linewidths=0.5,
            linecolor="gray",
        )
        ax.set_title(model_display, fontsize=joint_text_fontsize, pad=15)
        ax.set_xticklabels(
            ax.get_xticklabels(), rotation=30, ha="right", fontsize=joint_small_fontsize
        )
        ax.tick_params(axis="y", labelsize=joint_small_fontsize)
        ax.set_ylabel("")
        method_color = colors_dict.get(
            model_display, colors_dict.get(model_name, "black")
        )
        n_covariate_cols = model_data[model_display].shape[1]
        ax.hlines(
            0,
            0,
            n_covariate_cols,
            colors="white",
            linewidth=4.0,
            clip_on=False,
            zorder=29,
        )
        ax.hlines(
            -0.28,
            0,
            n_covariate_cols,
            colors=method_color,
            linewidth=updated_method_color_margin_linewidth,
            clip_on=False,
            zorder=30,
        )
        heatmap_axes.append(ax)

    cax = fig.add_subplot(row1_gs[0, 3])
    norm = Normalize(vmin=covariate_heatmap_vmin, vmax=covariate_heatmap_vmax)
    sm = ScalarMappable(norm=norm, cmap=updated_covariate_heatmap_cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, cax=cax, orientation="vertical")
    _style_joint_colorbar(cbar, "Distance")

    # Row 2: treatment and outcome prevalence
    cfg_loaded, datasets, _ = load_cohort_data(cohort_folder, verbose=True)
    cohort_config = cfg_loaded["cohort_configs"][cohort]
    treatment_col = cohort_config["treatment"]
    event_col = cohort_config.get(
        "outcome_event", cohort_config.get("outcome_censoring")
    )
    treatment_labels = get_label_map(cfg_loaded, cohort, treatment_col)
    event_labels = get_label_map(cfg_loaded, cohort, event_col)
    df_ref = datasets[eval_set]

    treatment_prevalences = {
        eval_set: df_ref[treatment_col].value_counts(normalize=True).sort_index()
    }
    event_prevalences = {
        eval_set: df_ref[event_col].value_counts(normalize=True).sort_index()
    }
    for model_name in selected_models:
        model_display = model_display_names[model_name]
        df_synth = datasets[model_name]
        treatment_prevalences[model_display] = (
            df_synth[treatment_col].value_counts(normalize=True).sort_index()
        )
        event_prevalences[model_display] = (
            df_synth[event_col].value_counts(normalize=True).sort_index()
        )

    treatment_groups = sorted(treatment_prevalences[eval_set].index)
    treatment_names = [
        treatment_labels.get(t, f"Treatment {t}") for t in treatment_groups
    ]
    event_groups = sorted(event_prevalences[eval_set].index)
    event_names = [event_labels.get(e, f"Event {e}") for e in event_groups]
    prevalence_dataset_keys = [eval_set] + selected_displays
    prevalence_method_labels = [ref_display] + selected_displays
    prevalence_method_colors = [ref_color] + [
        colors_dict.get(model_display, colors_dict.get(model_name, "0.5"))
        for model_name, model_display in zip(selected_models, selected_displays)
    ]
    prevalence_method_facecolors = [
        to_rgba(method_color, mixed_prevalence_fill_alpha)
        for method_color in prevalence_method_colors
    ]
    prevalence_hatches = ["", "////"]

    row2_gs = gridspec.GridSpecFromSubplotSpec(
        1, 2, subplot_spec=outer_gs[2], wspace=mixed_prevalence_wspace
    )
    prevalence_axes = [
        fig.add_subplot(row2_gs[0, 0]),
        fig.add_subplot(row2_gs[0, 1]),
    ]
    width = mixed_prevalence_bar_width

    def _plot_stacked_prevalence(
        ax, prevalences, groups, group_names, title, show_ylabel
    ):
        x = np.arange(len(prevalence_dataset_keys))
        bottoms = np.zeros(len(prevalence_dataset_keys), dtype=float)
        for group_idx, (group_value, group_name) in enumerate(zip(groups, group_names)):
            values = np.array(
                [
                    prevalences[dataset_key].get(group_value, 0)
                    for dataset_key in prevalence_dataset_keys
                ],
                dtype=float,
            )
            for dataset_idx, value in enumerate(values):
                ax.bar(
                    x[dataset_idx],
                    value,
                    width,
                    bottom=bottoms[dataset_idx],
                    color=prevalence_method_facecolors[dataset_idx],
                    edgecolor=prevalence_method_colors[dataset_idx],
                    linewidth=mixed_prevalence_edge_linewidth,
                    hatch="" if group_idx == 0 else prevalence_hatches[1],
                )
            bottoms += values

        if groups:
            original_prevalence = prevalences[eval_set].get(groups[0], 0)
            ax.axhline(
                original_prevalence,
                color=ref_color,
                linestyle="--",
                linewidth=mixed_prevalence_hline_linewidth,
                zorder=8,
            )

        ax.set_title(title, fontsize=joint_text_fontsize)
        if show_ylabel:
            ax.set_ylabel("Prevalence", fontsize=joint_text_fontsize)
        ax.set_xticks(x)
        ax.set_xticklabels(prevalence_method_labels, fontsize=joint_small_fontsize)
        ax.set_yticks(np.arange(0, 1.1, 0.2))
        ax.tick_params(axis="y", labelsize=joint_small_fontsize)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0))
        ax.set_ylim(0, 1)
        ax.grid(axis="y", alpha=0.3, linestyle="--")
        ax.set_axisbelow(True)

        stack_handles = [
            Patch(
                facecolor=to_rgba("0.7", mixed_prevalence_fill_alpha),
                edgecolor="0.25",
                hatch="" if i == 0 else prevalence_hatches[1],
                label=group_name,
            )
            for i, group_name in enumerate(group_names)
        ]
        ax.legend(
            handles=stack_handles,
            loc="upper center",
            bbox_to_anchor=(0.5, -0.15),
            ncol=min(len(stack_handles), 3),
            fontsize=joint_small_fontsize,
            handlelength=1.0,
            frameon=True,
            facecolor="white",
            framealpha=0.9,
            borderaxespad=0.0,
        )

    _plot_stacked_prevalence(
        prevalence_axes[0],
        treatment_prevalences,
        treatment_groups,
        treatment_names,
        "Treatment group",
        show_ylabel=True,
    )
    _plot_stacked_prevalence(
        prevalence_axes[1],
        event_prevalences,
        event_groups,
        event_names,
        "Outcome",
        show_ylabel=False,
    )
    prevalence_axes[1].tick_params(axis="y", left=False)

    # Row 3: Kaplan-Meier curves with at-risk-count heatmaps
    time_col = cohort_config["outcome_time"]
    max_horizon = pd.to_numeric(df_ref[time_col], errors="coerce").dropna().max()
    km_treatment_groups = (
        pd.to_numeric(df_ref[treatment_col], errors="coerce")
        .dropna()
        .sort_values()
        .unique()
        .tolist()
    )
    if not km_treatment_groups:
        raise ValueError(f"No treatment groups found in '{treatment_col}'.")

    km_method_models = selected_models
    km_method_displays = [model_display_names[m] for m in km_method_models]
    km_method_colors = [colors_dict[m] for m in km_method_models]

    method_panel_data = {}
    risk_count_abs_values = []
    for method_name, method_display, method_color in zip(
        km_method_models, km_method_displays, km_method_colors
    ):
        method_curve_rows, heatmap_df, annotation_df = _build_method_km_panel_data(
            df_ref,
            datasets[method_name],
            treatment_col,
            time_col,
            event_col,
            km_treatment_groups,
            updated_km_time_points,
            max_horizon,
            ref_display,
            method_display,
            treatment_labels,
            ref_color,
            method_color,
            updated_km_ref_curve_linewidth,
            updated_km_ref_curve_alpha,
            updated_km_model_curve_linewidth,
        )
        method_panel_data[method_display] = (
            method_curve_rows,
            heatmap_df,
            annotation_df,
        )
        flat_values = heatmap_df.to_numpy(dtype=float).ravel()
        risk_count_abs_values.extend(
            np.abs(flat_values[np.isfinite(flat_values)]).tolist()
        )

    risk_count_max_abs = max(risk_count_abs_values) if risk_count_abs_values else 1.0
    if risk_count_max_abs == 0:
        risk_count_max_abs = 1.0
    risk_count_heatmap_vmax = (
        risk_count_max_abs
        if updated_km_count_heatmap_vmax is None
        else updated_km_count_heatmap_vmax
    )

    n_methods = len(km_method_models)
    top_heatmap_total_width = sum(mixed_heatmap_width_ratios)
    top_colorbar_width_fraction = (
        mixed_heatmap_width_ratios[-1] / top_heatmap_total_width
    )
    updated_km_count_heatmap_cbar_width = (
        n_methods * top_colorbar_width_fraction / (1 - top_colorbar_width_fraction)
    )
    km_panel_width_ratios = [mixed_heatmap_width_ratios[0]] * n_methods
    row3_gs = gridspec.GridSpecFromSubplotSpec(
        2,
        n_methods + 1,
        subplot_spec=outer_gs[4],
        height_ratios=updated_km_height_ratios,
        width_ratios=km_panel_width_ratios + [updated_km_count_heatmap_cbar_width],
        hspace=updated_km_hspace,
        wspace=updated_km_wspace,
    )
    km_axes = [fig.add_subplot(row3_gs[0, i]) for i in range(n_methods)]
    km_heatmap_axes = [fig.add_subplot(row3_gs[1, i]) for i in range(n_methods)]
    km_cbar_space_ax = fig.add_subplot(row3_gs[0, n_methods])
    km_cbar_space_ax.axis("off")
    km_heatmap_cax = fig.add_subplot(row3_gs[1, n_methods])

    km_axes_bounds = [ax.get_position() for ax in km_axes]
    km_heatmap_axes_bounds = [ax.get_position() for ax in km_heatmap_axes]
    fig.text(
        (km_axes_bounds[0].x0 + km_axes_bounds[-1].x1) / 2,
        min(bounds.y1 for bounds in km_heatmap_axes_bounds) + 0.01,
        "Numbers at risk",
        ha="center",
        va="center",
        fontsize=joint_small_fontsize,
    )

    if n_methods > 1:
        for ax in km_axes[1:]:
            ax.sharey(km_axes[0])
        for ax in km_heatmap_axes[1:]:
            ax.sharey(km_heatmap_axes[0])

    for idx, (method_name, method_display, method_color) in enumerate(
        zip(km_method_models, km_method_displays, km_method_colors)
    ):
        ax = km_axes[idx]
        heatmap_ax = km_heatmap_axes[idx]
        curve_rows, heatmap_df, annotation_df = method_panel_data[method_display]

        for curve_row in curve_rows:
            _plot_km_curve(ax, curve_row)

        ax.set_title("")
        ax.set_xlabel("Time (months)", fontsize=joint_small_fontsize, labelpad=6)
        if idx == 0:
            ax.set_ylabel("Survival probability", fontsize=joint_text_fontsize)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="both", labelsize=joint_small_fontsize)
        ax.grid(True, alpha=0.3)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(*updated_km_aligned_xlim)
        ax.set_ylim(*_km_ylim_from_curve_rows(curve_rows, xlim=updated_km_aligned_xlim))
        tick_times = updated_km_time_points
        ax.set_xticks(tick_times)
        ax.set_xticklabels([f"{int(tick):d}" for tick in tick_times])
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()
        if idx == 0 and len(km_treatment_groups) >= 2:
            treatment_legend_handles = [
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="-",
                    linewidth=2.5,
                    label=_get_treatment_label(
                        treatment_labels, km_treatment_groups[0]
                    ),
                ),
                Line2D(
                    [0],
                    [0],
                    color="black",
                    linestyle="--",
                    linewidth=2.5,
                    label=_get_treatment_label(
                        treatment_labels, km_treatment_groups[1]
                    ),
                ),
            ]
            ax.legend(
                handles=treatment_legend_handles,
                loc="upper right",
                fontsize=joint_small_fontsize,
                handlelength=0.8,
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                borderaxespad=0.2,
            )

        color_df = heatmap_df.abs().copy()
        color_df.iloc[0::2, :] = np.nan
        sns.heatmap(
            color_df,
            ax=heatmap_ax,
            cmap=updated_km_count_heatmap_cmap,
            vmin=updated_km_count_heatmap_vmin,
            vmax=risk_count_heatmap_vmax,
            annot=False,
            linewidths=0,
            cbar=False,
        )
        heatmap_ax.set_facecolor("white")
        for row_idx, row_label in enumerate(annotation_df.index):
            for col_idx, annotation_text in enumerate(annotation_df.loc[row_label]):
                heatmap_ax.text(
                    col_idx + 0.5,
                    row_idx + 0.5,
                    annotation_text,
                    ha="center",
                    va="center",
                    fontsize=updated_km_heatmap_annotation_size_bottom,
                    color="black",
                )
        n_heatmap_cols = len(heatmap_df.columns)
        border_pad = 0.12
        for row_idx in range(len(heatmap_df.index)):
            row_color = ref_color if row_idx % 2 == 0 else method_color
            heatmap_ax.vlines(
                updated_km_heatmap_left_margin_x,
                row_idx + border_pad,
                row_idx + 1 - border_pad,
                colors=row_color,
                linewidth=updated_method_color_margin_linewidth,
                clip_on=False,
                zorder=30,
            )
        for row_boundary in range(1, len(heatmap_df.index)):
            heatmap_ax.hlines(
                row_boundary,
                updated_km_heatmap_left_margin_x,
                n_heatmap_cols,
                colors="white",
                linewidth=updated_km_heatmap_row_gap_width,
                clip_on=False,
                zorder=32,
            )
        heatmap_ax.set_xlabel("Time (months)", fontsize=joint_small_fontsize)
        heatmap_ax.set_ylabel("")
        heatmap_ax.set_yticklabels(
            heatmap_df.index, rotation=0, fontsize=joint_small_fontsize
        )
        if idx != 0:
            heatmap_ax.tick_params(axis="y", left=False, labelleft=False)
        heatmap_ax.tick_params(axis="y", pad=updated_km_heatmap_y_tick_label_pad)
        heatmap_ax.set_xticklabels(
            heatmap_ax.get_xticklabels(), rotation=0, fontsize=joint_small_fontsize
        )
        heatmap_ax.tick_params(axis="both", length=0)
        for spine in heatmap_ax.spines.values():
            spine.set_visible(False)

    km_cbar_norm = Normalize(
        vmin=updated_km_count_heatmap_vmin,
        vmax=risk_count_heatmap_vmax,
    )
    km_cbar_sm = ScalarMappable(norm=km_cbar_norm, cmap=updated_km_count_heatmap_cmap)
    km_cbar_sm.set_array([])
    km_cbar = fig.colorbar(km_cbar_sm, cax=km_heatmap_cax, orientation="vertical")
    _style_joint_colorbar(km_cbar, "Deviation")
    km_heatmap_cax.yaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))

    output_file = os.path.join(
        output_dir, f"updated_joint_fidelity_summary_{cohort}_ref[{eval_set}].png"
    )
    plt.savefig(output_file, dpi=joint_dpi)
    print(f"Saved: {output_file}")
    plt.show()


plot_updated_joint_fidelity_summary(eval_set=eval_set)
