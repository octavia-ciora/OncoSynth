# %%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import seaborn as sns
from scipy.spatial.distance import jensenshannon
import os
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_curve,
    precision_recall_curve,
    roc_auc_score,
    auc,
    log_loss,
)
from load_data import *
from matplotlib.gridspec import GridSpec

plt.style.use("default")

#####
cohort_folder = "demo_0"
horizon = 96
#####

train_set = "downstream_train"
pred_set = "heldout_test"

classifier = "logistic"
correlation_display = "mae"
fi_display = "pearson"
calibration_display = "ece"


cohort = cohort_folder.split("_")[0]
cfg = load_config()
cohort_config = cfg["cohort_configs"][cohort]

eval_dir = get_utility_evaluation_dir(cohort_folder=cohort_folder, cfg=cfg)
output_dir = os.path.join(eval_dir, "figures")

os.makedirs(output_dir, exist_ok=True)

model_order = cfg["model_order"]

model_display_names = cfg["model_display_names"]
font_size = cfg["plotting"]["font_size"]
annotation_font_size = cfg["plotting"]["annotation_font_size"]
fig_width = 12

correlation_display = correlation_display.lower()

if correlation_display not in {"pearson", "spearman", "mae"}:
    raise ValueError(
        f"Invalid correlation_display: {correlation_display}. Expected 'pearson', 'spearman', or 'mae'."
    )
calibration_display = calibration_display.lower()
if calibration_display not in {"brier", "ece"}:
    raise ValueError(
        f"Invalid calibration_display: {calibration_display}. Expected 'brier' or 'ece'."
    )


def format_display_metric(pearson_corr, spearman_corr, mae, decimals=3):
    """Format the selected stored metric for plot annotations."""
    if correlation_display == "pearson":
        return f"r = {pearson_corr:.{decimals}f}"
    if correlation_display == "spearman":
        return f"ρ = {spearman_corr:.{decimals}f}"
    return f"MAE = {mae:.{decimals}f}"


def format_calibration_metric(brier, ece, decimals=3):
    """Format the selected calibration metric for plot annotations."""
    if calibration_display == "brier":
        return f"Brier = {brier:.{decimals}f}"
    return f"ECE = {ece:.{decimals}f}"


def smooth_display_curve(x, y, window=31):
    """Smooth a plotted curve for display without changing the underlying CSV."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(y) < 5:
        return y
    window = min(window, len(y) if len(y) % 2 == 1 else len(y) - 1)
    window = max(window, 3)
    if window % 2 == 0:
        window -= 1
    y_smooth = pd.Series(y).rolling(window=window, center=True, min_periods=1).mean()
    y_smooth = y_smooth.to_numpy(dtype=float)
    y_smooth[0] = y[0]
    y_smooth[-1] = y[-1]
    return y_smooth


# %% Shortened utility without propensity panels
def plot_joint_utility_shortened(train_set, pred_set, horizon):
    """Plot a 3-row utility summary without the propensity panels."""

    joint_models = ["CTGAN", "TabDiff", "OncoSynth"]
    joint_figsize = (9.8, 12.5)
    joint_dpi = 300
    joint_text_fontsize = font_size
    joint_small_fontsize = joint_text_fontsize - 2
    ate_marker_size = 12
    ate_cap_size = 8
    ate_line_width = 2.4
    ate_reference_band_alpha = 0.2
    inset_legend_border_color = "0.6"
    ite_decile_dot_size = 15
    ite_decile_sd_line_width = 1
    qini_line_width = 2.4
    random_line_width = 1.2
    qini_reference_line_width = qini_line_width - 0.8
    qini_fill_alpha = 0.2
    qini_smooth_window = 0
    if cohort_folder.startswith("breast"):
        x_model_annot = 0.7
        y_model_annot = 0.8
        x_ref_annot = 0.2
        y_ref_annot = 0.15
        ite_metric_annot_x = 0.97
        ite_metric_annot_y = 0.03
        ite_metric_annot_ha = "right"
        ite_metric_annot_va = "bottom"
    elif cohort_folder.startswith("lung"):
        x_model_annot = 0.3
        y_model_annot = 0.9
        x_ref_annot = 0.8
        y_ref_annot = 0.3
        ite_metric_annot_x = 0.05
        ite_metric_annot_y = 0.95
        ite_metric_annot_ha = "left"
        ite_metric_annot_va = "top"
    elif cohort_folder.startswith("demo"):
        x_model_annot = 0.7
        y_model_annot = 0.8
        x_ref_annot = 0.2
        y_ref_annot = 0.15
        ite_metric_annot_x = 0.97
        ite_metric_annot_y = 0.03
        ite_metric_annot_ha = "right"
        ite_metric_annot_va = "bottom"
    else:
        raise ValueError(f"Unknown cohort folder: {cohort_folder}")
    row_gap_12 = 0.05
    row_gap_23 = 0.1
    inset_anchor_x = 0.97
    inset_anchor_y = 0.03
    ate_legend_anchor_x = 0.99
    ate_legend_anchor_y = 0.95
    bottom_legend_left = 0.08
    bottom_legend_bottom = 0.01
    bottom_legend_width = 0.84
    bottom_legend_height = 0.055
    shared_ylabel_x = 0.03

    colors_dict = cfg["plotting"]["colors"]
    selected_models = [m for m in joint_models if m in model_order]
    if len(selected_models) != 3:
        raise ValueError(
            f"Expected exactly 3 methods in joint_models, found {len(selected_models)}: {selected_models}"
        )

    selected_displays = [model_display_names[m] for m in selected_models]
    ref_display = model_display_names[train_set]
    ref_color = colors_dict[train_set]

    ate_file = os.path.join(
        eval_dir, f"ATE_csf_train[{train_set}]_horizon[{horizon}].csv"
    )
    ite_file = os.path.join(
        eval_dir, f"ITE_csf_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv"
    )
    qini_curves_file = os.path.join(
        eval_dir,
        f"qini_curves_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    qini_metrics_file = os.path.join(
        eval_dir,
        f"qini_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    ite_decile_metrics_file = os.path.join(
        eval_dir,
        f"ITE_decile_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )

    df_ate = pd.read_csv(ate_file, index_col=0)
    df_ite = pd.read_csv(ite_file)
    df_qini = pd.read_csv(qini_curves_file)
    df_qini_metrics = pd.read_csv(qini_metrics_file)
    df_ite_decile_metrics = pd.read_csv(ite_decile_metrics_file, index_col=0)

    if ref_display not in df_ate.columns:
        raise ValueError(f"Missing ATE column for reference model: {ref_display}")
    if ref_display not in df_ite.columns:
        raise ValueError(f"Missing ITE column for reference model: {ref_display}")
    if ref_display not in df_qini["model"].unique():
        raise ValueError(f"Missing Qini curve for reference model: {ref_display}")

    metric_map = dict(zip(df_qini_metrics["model"], df_qini_metrics["auqini"]))
    base_curve = df_qini[df_qini["model"] == ref_display].copy()
    base_area = metric_map[ref_display]
    base_curve["cum_gain_smooth"] = smooth_display_curve(
        base_curve["fraction"], base_curve["cum_gain"], window=qini_smooth_window
    )

    ite_real_all = df_ite[ref_display].to_numpy(dtype=float)
    n_bins = 10
    valid_real = np.isfinite(ite_real_all)
    if valid_real.sum() < n_bins:
        raise ValueError(
            f"Not enough finite real ITE values ({valid_real.sum()}) for {n_bins} bins"
        )
    quantile_idx = pd.Series(np.nan, index=df_ite.index, dtype=float)
    quantile_idx.loc[valid_real] = pd.qcut(
        pd.Series(ite_real_all[valid_real]).rank(method="first"),
        q=n_bins,
        labels=False,
    ).to_numpy()

    def mean_by_bin(values):
        means = []
        for b in range(n_bins):
            mask_b = (quantile_idx == b).to_numpy() & np.isfinite(values)
            vals_b = values[mask_b]
            means.append(np.mean(vals_b) if len(vals_b) > 0 else np.nan)
        return np.array(means, dtype=float)

    ite_real_decile_mean = mean_by_bin(ite_real_all)
    ite_decile_global_min = np.inf
    ite_decile_global_max = -np.inf
    ite_decile_stats = {}
    for model_display in selected_displays:
        model_vals = df_ite[model_display].to_numpy(dtype=float)
        x_positions = []
        y_means = []
        y_stds = []
        for b in range(n_bins):
            if not np.isfinite(ite_real_decile_mean[b]):
                continue
            mask_b = (quantile_idx == b).to_numpy() & np.isfinite(model_vals)
            vals_b = model_vals[mask_b]
            if len(vals_b) == 0:
                continue
            x_positions.append(ite_real_decile_mean[b])
            y_means.append(np.mean(vals_b))
            y_stds.append(np.std(vals_b, ddof=1) if len(vals_b) > 1 else 0.0)
        x_vals = np.array(x_positions, dtype=float)
        y_vals = np.array(y_means, dtype=float)
        y_std = np.array(y_stds, dtype=float)
        ite_decile_stats[model_display] = (x_vals, y_vals, y_std)
        if len(x_vals) > 0 and len(y_vals) > 0:
            y_all = np.concatenate([y_vals - y_std, y_vals + y_std])
            local_min = min(np.min(x_vals), np.min(y_all))
            local_max = max(np.max(x_vals), np.max(y_all))
            ite_decile_global_min = min(ite_decile_global_min, local_min)
            ite_decile_global_max = max(ite_decile_global_max, local_max)

    fig = plt.figure(figsize=joint_figsize)
    gs = GridSpec(
        5,
        len(selected_displays),
        figure=fig,
        height_ratios=[1, row_gap_12, 1, row_gap_23, 1],
        hspace=0.0,
        wspace=0.0,
        left=0.12,
        right=0.95,
        top=0.96,
        bottom=0.09,
    )

    ite_decile_axes = np.empty(len(selected_displays), dtype=object)
    qini_axes = np.empty(len(selected_displays), dtype=object)
    for idx in range(len(selected_displays)):
        ite_decile_axes[idx] = fig.add_subplot(
            gs[2, idx], sharey=ite_decile_axes[0] if idx > 0 else None
        )
        qini_axes[idx] = fig.add_subplot(
            gs[4, idx], sharey=qini_axes[0] if idx > 0 else None
        )
    ax_ate = fig.add_subplot(gs[0, :])
    for ax in ite_decile_axes:
        ax.set_box_aspect(1)
    ax_ate.set_box_aspect(1 / 3)

    model_x = np.arange(len(selected_displays), dtype=float) + 0.5
    reference_ci_lower = df_ate.loc["ci_lower", ref_display]
    reference_ci_upper = df_ate.loc["ci_upper", ref_display]
    reference_estimate = df_ate.loc["estimate", ref_display]
    ax_ate.axhspan(
        reference_ci_lower,
        reference_ci_upper,
        color=ref_color,
        alpha=ate_reference_band_alpha,
        zorder=0,
    )
    ax_ate.axhline(
        reference_estimate,
        color=ref_color,
        linewidth=ate_line_width - 1,
        linestyle=":",
        label="ATE (original)",
        zorder=2,
    )
    ate_y_min = reference_ci_lower
    ate_y_max = reference_ci_upper
    for divider_x in range(1, len(selected_displays)):
        ax_ate.axvline(divider_x, color="black", alpha=0.15, linewidth=1.0, zorder=1)

    for idx, model_display in enumerate(selected_displays):
        model_color = colors_dict[model_display]
        if model_display not in df_ate.columns:
            raise ValueError(f"Missing ATE column for model: {model_display}")
        model_estimate = df_ate.loc["estimate", model_display]
        model_ci_lower = df_ate.loc["ci_lower", model_display]
        model_ci_upper = df_ate.loc["ci_upper", model_display]
        ate_y_min = min(ate_y_min, model_ci_lower)
        ate_y_max = max(ate_y_max, model_ci_upper)
        ax_ate.errorbar(
            model_x[idx],
            model_estimate,
            yerr=[
                [model_estimate - model_ci_lower],
                [model_ci_upper - model_estimate],
            ],
            fmt="o",
            markersize=ate_marker_size,
            capsize=ate_cap_size,
            linewidth=ate_line_width,
            elinewidth=ate_line_width,
            color=model_color,
            ecolor=model_color,
            markeredgecolor=model_color,
            markeredgewidth=ate_line_width,
            zorder=3,
        )

    ate_y_pad = 0.08 * (ate_y_max - ate_y_min) if ate_y_max > ate_y_min else 0.2
    ax_ate.set_ylim(ate_y_min - ate_y_pad, ate_y_max + ate_y_pad)
    ax_ate.set_xlim(0.0, len(selected_displays))
    ax_ate.set_xticks([])
    ax_ate.set_xticklabels([])
    ax_ate.tick_params(axis="x", length=0)
    ax_ate.set_xlabel("Synthetic approach", fontsize=joint_text_fontsize, labelpad=10)
    ax_ate.tick_params(axis="y", labelsize=joint_small_fontsize)
    ax_ate.grid(True, alpha=0.2, axis="y")
    ate_legend = ax_ate.legend(
        loc="upper right",
        bbox_to_anchor=(ate_legend_anchor_x, ate_legend_anchor_y),
        bbox_transform=ax_ate.transAxes,
        fontsize=joint_small_fontsize,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handleheight=0.8,
        borderaxespad=0.0,
    )
    ate_legend.get_frame().set_edgecolor(inset_legend_border_color)
    ate_legend.get_frame().set_boxstyle("round,pad=0.25")

    for idx, model_display in enumerate(selected_displays):
        model_color = colors_dict[model_display]
        if model_display not in df_ite.columns:
            raise ValueError(f"Missing ITE column for model: {model_display}")

        ax_ite_decile = ite_decile_axes[idx]
        x_vals, y_vals, y_std = ite_decile_stats[model_display]
        ax_ite_decile.errorbar(
            x_vals,
            y_vals,
            yerr=y_std,
            fmt=".",
            color=model_color,
            ecolor=model_color,
            markersize=ite_decile_dot_size,
            elinewidth=ite_decile_sd_line_width,
            capsize=0,
            zorder=3,
        )
        if np.isfinite(ite_decile_global_min) and np.isfinite(ite_decile_global_max):
            decile_pad = (
                0.05 * (ite_decile_global_max - ite_decile_global_min)
                if ite_decile_global_max > ite_decile_global_min
                else 0.1
            )
            decile_lo = ite_decile_global_min - decile_pad
            decile_hi = ite_decile_global_max + decile_pad
            ax_ite_decile.plot(
                [decile_lo, decile_hi],
                [decile_lo, decile_hi],
                "k--",
                alpha=0.5,
                linewidth=1.5,
                label="_nolegend_",
            )
            ax_ite_decile.set_xlim(decile_lo, decile_hi)
            ax_ite_decile.set_ylim(decile_lo, decile_hi)
        if model_display not in df_ite_decile_metrics.columns:
            raise ValueError(f"Missing ITE decile metrics for model: {model_display}")
        ax_ite_decile.text(
            ite_metric_annot_x,
            ite_metric_annot_y,
            format_calibration_metric(
                df_ite_decile_metrics.loc["ITE_decile_brier", model_display],
                df_ite_decile_metrics.loc["ITE_decile_ece", model_display],
                decimals=3,
            ),
            transform=ax_ite_decile.transAxes,
            fontsize=joint_small_fontsize,
            verticalalignment=ite_metric_annot_va,
            horizontalalignment=ite_metric_annot_ha,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        if idx == 0:
            ax_ite_decile.tick_params(axis="y", labelsize=joint_small_fontsize)
        else:
            ax_ite_decile.tick_params(axis="y", left=False, labelleft=False)
        ax_ite_decile.tick_params(axis="x", labelsize=joint_small_fontsize, pad=1)
        ax_ite_decile.grid(True, alpha=0.2)

        ax_qini = qini_axes[idx]
        ax_qini.set_box_aspect(1)
        curve_df = df_qini[df_qini["model"] == model_display].copy()
        if curve_df.empty:
            raise ValueError(f"Missing Qini curve for model: {model_display}")
        area = metric_map[model_display]
        curve_df["cum_gain_smooth"] = smooth_display_curve(
            curve_df["fraction"], curve_df["cum_gain"], window=qini_smooth_window
        )
        ax_qini.plot(
            base_curve["fraction"],
            base_curve["cum_gain_smooth"],
            color=ref_color,
            linewidth=qini_reference_line_width,
            alpha=0.9,
        )
        ax_qini.fill_between(
            base_curve["fraction"],
            base_curve["random"],
            base_curve["cum_gain_smooth"],
            color=ref_color,
            alpha=0.08,
            zorder=-1,
        )
        ax_qini.plot(
            curve_df["fraction"],
            curve_df["cum_gain_smooth"],
            color=model_color,
            linewidth=qini_line_width,
            label="_nolegend_",
        )
        ax_qini.fill_between(
            curve_df["fraction"],
            curve_df["random"],
            curve_df["cum_gain_smooth"],
            color=model_color,
            alpha=qini_fill_alpha,
            zorder=0,
        )
        ax_qini.plot(
            curve_df["fraction"],
            curve_df["random"],
            "k--",
            linewidth=random_line_width,
            alpha=0.6,
            label="random",
        )
        ax_qini.text(
            x_model_annot,
            y_model_annot,
            f"{area:.2f}",
            color=model_color,
            fontsize=joint_small_fontsize,
            ha="center",
            va="center",
            transform=ax_qini.transAxes,
        )
        ax_qini.text(
            x_ref_annot,
            y_ref_annot,
            f"{base_area:.2f}",
            color=ref_color,
            fontsize=joint_small_fontsize,
            ha="center",
            va="center",
            transform=ax_qini.transAxes,
        )
        if idx == 0:
            ax_qini.tick_params(axis="y", labelsize=joint_small_fontsize)
        else:
            ax_qini.tick_params(axis="y", left=False, labelleft=False)
        ax_qini.tick_params(axis="x", labelsize=joint_small_fontsize, pad=1)
        ax_qini.set_xticks([0, 0.5, 1])
        ax_qini.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_qini.grid(True, alpha=0.2)
        if idx == len(selected_displays) - 1:
            qini_legend = ax_qini.legend(
                loc="lower right",
                bbox_to_anchor=(inset_anchor_x, inset_anchor_y),
                bbox_transform=ax_qini.transAxes,
                fontsize=joint_small_fontsize,
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                borderpad=0.2,
                labelspacing=0.2,
                handlelength=0.8,
                handleheight=0.8,
                borderaxespad=0.0,
            )
            qini_legend.get_frame().set_edgecolor("0.6")
            qini_legend.get_frame().set_boxstyle("round,pad=0.25")

    fig.canvas.draw()
    row1_y = ax_ate.get_position().y0 + ax_ate.get_position().height / 2
    row2_bottom = min(ax.get_position().y0 for ax in ite_decile_axes)
    row2_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in ite_decile_axes]
    )
    row3_bottom = min(ax.get_position().y0 for ax in qini_axes)
    row3_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in qini_axes]
    )
    subplot_block_center_x = 0.5 * (
        ite_decile_axes[0].get_position().x0 + ite_decile_axes[-1].get_position().x1
    )

    fig.text(
        shared_ylabel_x,
        row1_y,
        "ATE with 95% CI",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        shared_ylabel_x,
        row2_y,
        "Synthetic ITE",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        shared_ylabel_x,
        row3_y,
        "Cumulative benefit",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        subplot_block_center_x,
        row2_bottom - 0.04,
        "Original mean ITE per decile",
        ha="center",
        va="center",
        fontsize=joint_text_fontsize,
    )
    fig.text(
        subplot_block_center_x,
        row3_bottom - 0.025,
        "Ranked patient fraction",
        ha="center",
        va="top",
        fontsize=joint_text_fontsize,
    )

    bottom_legend_handles = [
        Patch(
            facecolor=ref_color, edgecolor="none", label=model_display_names[train_set]
        )
    ] + [
        Patch(
            facecolor=colors_dict[model_display],
            edgecolor="none",
            label=model_display,
        )
        for model_display in selected_displays
    ]
    legend = fig.legend(
        handles=bottom_legend_handles,
        loc="lower left",
        bbox_to_anchor=(
            bottom_legend_left,
            bottom_legend_bottom,
            bottom_legend_width,
            bottom_legend_height,
        ),
        ncol=len(bottom_legend_handles),
        frameon=False,
        fontsize=joint_text_fontsize,
        handlelength=1,
        handleheight=1,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.2,
        mode="expand",
        bbox_transform=fig.transFigure,
    )
    legend.get_frame().set_edgecolor(inset_legend_border_color)
    legend.get_frame().set_boxstyle("round,pad=0.25")

    output_file = os.path.join(
        output_dir,
        f"joint_utility_shortened_{cohort}_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].png",
    )
    plt.savefig(output_file, dpi=joint_dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.show()


plot_joint_utility_shortened(train_set=train_set, pred_set=pred_set, horizon=horizon)


# %% Mixed utility
def plot_joint_utility_summary(train_set, pred_set, horizon):
    """Plot a 5-row utility summary using the main plots from earlier blocks."""

    # Major settings for quick editing
    joint_models = ["CTGAN", "TabDiff", "OncoSynth"]
    joint_figsize = (9.8, 22)
    joint_dpi = 300
    joint_text_fontsize = font_size
    joint_small_fontsize = joint_text_fontsize - 2
    panel_x_margin = 0.05
    propensity_bins = 30
    propensity_hist_alpha = 0.6
    propensity_scatter_size = 20
    propensity_scatter_alpha = 0.04
    ate_marker_size = 12
    ate_cap_size = 8
    ate_line_width = 2.4
    ate_reference_band_alpha = 0.2
    inset_legend_border_color = "0.6"
    ite_decile_dot_size = 15
    ite_decile_sd_line_width = 1
    qini_line_width = 2.4
    random_line_width = 1.2
    qini_reference_line_width = qini_line_width - 0.8
    qini_fill_alpha = 0.2
    qini_smooth_window = 0
    if cohort_folder.startswith("breast"):
        x_model_annot = 0.7
        y_model_annot = 0.8
        x_ref_annot = 0.2
        y_ref_annot = 0.15
        ite_metric_annot_x = 0.97
        ite_metric_annot_y = 0.03
        ite_metric_annot_ha = "right"
        ite_metric_annot_va = "bottom"
    elif cohort_folder.startswith("lung"):
        x_model_annot = 0.3
        y_model_annot = 0.9
        x_ref_annot = 0.8
        y_ref_annot = 0.3
        ite_metric_annot_x = 0.05
        ite_metric_annot_y = 0.95
        ite_metric_annot_ha = "left"
        ite_metric_annot_va = "top"
    elif cohort_folder.startswith("demo"):
        x_model_annot = 0.7
        y_model_annot = 0.8
        x_ref_annot = 0.2
        y_ref_annot = 0.15
        ite_metric_annot_x = 0.97
        ite_metric_annot_y = 0.03
        ite_metric_annot_ha = "right"
        ite_metric_annot_va = "bottom"
    else:
        raise ValueError(f"Unknown cohort folder: {cohort_folder}")
    row_gap_12 = 0.15
    row_gap_23 = 0.25
    row_gap_34 = 0.1
    row_gap_45 = 0.25
    inset_anchor_x = 0.97
    inset_anchor_y = 0.03
    ate_legend_anchor_x = 0.99
    ate_legend_anchor_y = 0.95
    bottom_legend_left = 0.08
    bottom_legend_bottom = 0.01
    bottom_legend_width = 0.84
    bottom_legend_height = 0.055
    shared_ylabel_x = 0.03

    colors_dict = cfg["plotting"]["colors"]
    selected_models = [m for m in joint_models if m in model_order]
    if len(selected_models) != 3:
        raise ValueError(
            f"Expected exactly 3 methods in joint_models, found {len(selected_models)}: {selected_models}"
        )

    selected_displays = [model_display_names[m] for m in selected_models]
    ref_display = model_display_names[train_set]
    ref_color = colors_dict[train_set]

    ate_file = os.path.join(
        eval_dir, f"ATE_csf_train[{train_set}]_horizon[{horizon}].csv"
    )
    ite_file = os.path.join(
        eval_dir, f"ITE_csf_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv"
    )
    qini_curves_file = os.path.join(
        eval_dir,
        f"qini_curves_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    qini_metrics_file = os.path.join(
        eval_dir,
        f"qini_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    propensity_pred_file = os.path.join(
        eval_dir, f"propensities_{classifier}_train[{train_set}]_pred[{pred_set}].csv"
    )
    propensity_metrics_file = os.path.join(
        eval_dir,
        f"propensity_metrics_{classifier}_train[{train_set}]_pred[{pred_set}].csv",
    )
    ite_decile_metrics_file = os.path.join(
        eval_dir,
        f"ITE_decile_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )

    df_ate = pd.read_csv(ate_file, index_col=0)
    df_ite = pd.read_csv(ite_file)
    df_qini = pd.read_csv(qini_curves_file)
    df_qini_metrics = pd.read_csv(qini_metrics_file)
    df_propensity = pd.read_csv(propensity_pred_file)
    df_ite_decile_metrics = pd.read_csv(ite_decile_metrics_file, index_col=0)

    propensity_jsd_dict = {}
    propensity_metrics_map = None
    if os.path.exists(propensity_metrics_file):
        df_propensity_metrics = pd.read_csv(propensity_metrics_file)
        propensity_metrics_map = df_propensity_metrics.set_index("model")
        for _, row in df_propensity_metrics.iterrows():
            propensity_jsd_dict[row["model"]] = row["distance"]

    if ref_display not in df_ate.columns:
        raise ValueError(f"Missing ATE column for reference model: {ref_display}")
    if ref_display not in df_ite.columns:
        raise ValueError(f"Missing ITE column for reference model: {ref_display}")
    if ref_display not in df_qini["model"].unique():
        raise ValueError(f"Missing Qini curve for reference model: {ref_display}")

    metric_map = dict(zip(df_qini_metrics["model"], df_qini_metrics["auqini"]))
    base_curve = df_qini[df_qini["model"] == ref_display].copy()
    base_area = metric_map[ref_display]
    base_curve["cum_gain_smooth"] = smooth_display_curve(
        base_curve["fraction"], base_curve["cum_gain"], window=qini_smooth_window
    )
    propensity_real = df_propensity[f"{ref_display}_proba"].values
    hist_max_count = 0
    for model_display in selected_displays:
        synth_values = df_propensity[f"{model_display}_proba"].values
        hist_max_count = max(
            hist_max_count,
            np.histogram(propensity_real, bins=propensity_bins, range=(0, 1))[0].max(),
            np.histogram(synth_values, bins=propensity_bins, range=(0, 1))[0].max(),
        )

    ite_real_all = df_ite[ref_display].to_numpy(dtype=float)
    ite_global_min = np.nanmin(df_ite[selected_displays + [ref_display]].to_numpy())
    ite_global_max = np.nanmax(df_ite[selected_displays + [ref_display]].to_numpy())
    ite_pad = (
        0.05 * (ite_global_max - ite_global_min)
        if ite_global_max > ite_global_min
        else 0.1
    )
    ite_lo = ite_global_min - ite_pad
    ite_hi = ite_global_max + ite_pad

    n_bins = 10
    valid_real = np.isfinite(ite_real_all)
    if valid_real.sum() < n_bins:
        raise ValueError(
            f"Not enough finite real ITE values ({valid_real.sum()}) for {n_bins} bins"
        )
    quantile_idx = pd.Series(np.nan, index=df_ite.index, dtype=float)
    quantile_idx.loc[valid_real] = pd.qcut(
        pd.Series(ite_real_all[valid_real]).rank(method="first"),
        q=n_bins,
        labels=False,
    ).to_numpy()

    def mean_by_bin(values):
        means = []
        for b in range(n_bins):
            mask_b = (quantile_idx == b).to_numpy() & np.isfinite(values)
            vals_b = values[mask_b]
            means.append(np.mean(vals_b) if len(vals_b) > 0 else np.nan)
        return np.array(means, dtype=float)

    ite_real_decile_mean = mean_by_bin(ite_real_all)
    ite_decile_global_min = np.inf
    ite_decile_global_max = -np.inf
    ite_decile_stats = {}
    for model_display in selected_displays:
        model_vals = df_ite[model_display].to_numpy(dtype=float)
        x_positions = []
        y_means = []
        y_stds = []
        for b in range(n_bins):
            if not np.isfinite(ite_real_decile_mean[b]):
                continue
            mask_b = (quantile_idx == b).to_numpy() & np.isfinite(model_vals)
            vals_b = model_vals[mask_b]
            if len(vals_b) == 0:
                continue
            x_positions.append(ite_real_decile_mean[b])
            y_means.append(np.mean(vals_b))
            y_stds.append(np.std(vals_b, ddof=1) if len(vals_b) > 1 else 0.0)
        x_vals = np.array(x_positions, dtype=float)
        y_vals = np.array(y_means, dtype=float)
        y_std = np.array(y_stds, dtype=float)
        ite_decile_stats[model_display] = (x_vals, y_vals, y_std)
        if len(x_vals) > 0 and len(y_vals) > 0:
            y_all = np.concatenate([y_vals - y_std, y_vals + y_std])
            local_min = min(np.min(x_vals), np.min(y_all))
            local_max = max(np.max(x_vals), np.max(y_all))
            ite_decile_global_min = min(ite_decile_global_min, local_min)
            ite_decile_global_max = max(ite_decile_global_max, local_max)

    fig = plt.figure(figsize=joint_figsize)
    gs = GridSpec(
        9,
        len(selected_displays),
        figure=fig,
        height_ratios=[
            1,
            row_gap_12,
            1,
            row_gap_23,
            1,
            row_gap_34,
            1,
            row_gap_45,
            1,
        ],
        hspace=0.0,
        wspace=0.0,
        left=0.12,
        right=0.95,
        top=0.95,
        bottom=0.08,
    )

    propensity_axes = np.empty(len(selected_displays), dtype=object)
    propensity_scatter_axes = np.empty(len(selected_displays), dtype=object)
    ite_decile_axes = np.empty(len(selected_displays), dtype=object)
    qini_axes = np.empty(len(selected_displays), dtype=object)
    for idx in range(len(selected_displays)):
        propensity_axes[idx] = fig.add_subplot(
            gs[0, idx], sharey=propensity_axes[0] if idx > 0 else None
        )
        propensity_scatter_axes[idx] = fig.add_subplot(
            gs[2, idx],
            sharex=propensity_scatter_axes[0] if idx > 0 else None,
            sharey=propensity_scatter_axes[0] if idx > 0 else None,
        )
        ite_decile_axes[idx] = fig.add_subplot(
            gs[6, idx], sharey=ite_decile_axes[0] if idx > 0 else None
        )
        qini_axes[idx] = fig.add_subplot(
            gs[8, idx], sharey=qini_axes[0] if idx > 0 else None
        )
    ax_ate = fig.add_subplot(gs[4, :])
    for ax in ite_decile_axes:
        ax.set_box_aspect(1)
    ax_ate.set_box_aspect(1 / 3)

    # Row 1: Propensity distributions
    for idx, model_display in enumerate(selected_displays):
        model_color = colors_dict[model_display]
        ax = propensity_axes[idx]
        synth_values = df_propensity[f"{model_display}_proba"].values
        ax.hist(
            propensity_real,
            bins=propensity_bins,
            range=(0, 1),
            alpha=propensity_hist_alpha,
            color=ref_color,
            label=train_set,
            density=False,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.hist(
            synth_values,
            bins=propensity_bins,
            range=(0, 1),
            alpha=propensity_hist_alpha,
            color=model_color,
            label=model_display,
            density=False,
            edgecolor="black",
            linewidth=0.5,
        )
        if model_display in propensity_jsd_dict:
            ax.text(
                0.05,
                0.95,
                f"JSD = {propensity_jsd_dict[model_display]:.3f}",
                transform=ax.transAxes,
                fontsize=joint_small_fontsize,
                verticalalignment="top",
                horizontalalignment="left",
                bbox=dict(
                    boxstyle="round",
                    facecolor="white",
                    edgecolor=inset_legend_border_color,
                    alpha=0.8,
                ),
            )
        ax.set_box_aspect(1)
        ax.set_title(model_display, fontsize=joint_text_fontsize)
        # ax.set_xlabel("Predicted propensity", fontsize=joint_text_fontsize)
        if idx == 0:
            ax.tick_params(axis="y", labelsize=joint_small_fontsize)
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
        ax.tick_params(axis="x", labelsize=joint_small_fontsize)
        ax.set_xticks([0, 0.5, 1])
        ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax.grid(True, alpha=0.2, axis="y")
        ax.set_xlim(-panel_x_margin, 1 + panel_x_margin)
        ax.set_ylim(0, hist_max_count * 1.08 if hist_max_count > 0 else None)

    # Row 2: Propensity scatter
    for idx, model_display in enumerate(selected_displays):
        model_color = colors_dict[model_display]
        ax_scatter = propensity_scatter_axes[idx]
        synth_prob = df_propensity[f"{model_display}_proba"].to_numpy(dtype=float)
        real_prob = df_propensity[f"{ref_display}_proba"].to_numpy(dtype=float)
        ax_scatter.set_box_aspect(1)
        ax_scatter.scatter(
            synth_prob,
            real_prob,
            s=propensity_scatter_size,
            alpha=propensity_scatter_alpha,
            color=model_color,
            marker="o",
            edgecolors="none",
            linewidths=0,
            rasterized=False,
        )
        ax_scatter.plot([0, 1], [0, 1], "k--", alpha=0.5, linewidth=1.3)
        if (
            propensity_metrics_map is None
            or model_display not in propensity_metrics_map.index
        ):
            raise ValueError(f"Missing propensity metrics for model: {model_display}")
        ax_scatter.text(
            0.05,
            0.95,
            format_display_metric(
                propensity_metrics_map.loc[model_display, "correlation_pearson"],
                propensity_metrics_map.loc[model_display, "correlation_spearman"],
                propensity_metrics_map.loc[model_display, "MAE"],
                decimals=2,
            ),
            transform=ax_scatter.transAxes,
            fontsize=joint_small_fontsize,
            verticalalignment="top",
            horizontalalignment="left",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        if idx == 0:
            ax_scatter.tick_params(axis="y", labelsize=joint_small_fontsize)
        else:
            ax_scatter.tick_params(axis="y", left=False, labelleft=False)
        ax_scatter.tick_params(axis="x", labelsize=joint_small_fontsize, pad=1)
        ax_scatter.set_xticks([0, 0.5, 1])
        ax_scatter.set_yticks([0, 0.5, 1])
        ax_scatter.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_scatter.yaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_scatter.grid(True, alpha=0.2)
        ax_scatter.set_xlim(-panel_x_margin, 1 + panel_x_margin)
        ax_scatter.set_ylim(-panel_x_margin, 1 + panel_x_margin)

    # Row 3: ATE
    reference_x = 0.18
    model_x = np.arange(len(selected_displays), dtype=float) + 0.5
    reference_ci_lower = df_ate.loc["ci_lower", ref_display]
    reference_ci_upper = df_ate.loc["ci_upper", ref_display]
    reference_estimate = df_ate.loc["estimate", ref_display]
    ax_ate.axhspan(
        reference_ci_lower,
        reference_ci_upper,
        color=ref_color,
        alpha=ate_reference_band_alpha,
        zorder=0,
    )
    ax_ate.axhline(
        reference_estimate,
        color=ref_color,
        linewidth=ate_line_width - 1,
        linestyle=":",
        label="ATE (original)",
        zorder=2,
    )
    ate_zero_visible = (reference_ci_lower <= 0) and (reference_ci_upper >= 0)
    ate_y_min = reference_ci_lower
    ate_y_max = reference_ci_upper
    for divider_x in range(1, len(selected_displays)):
        ax_ate.axvline(divider_x, color="black", alpha=0.15, linewidth=1.0, zorder=1)

    for idx, model_display in enumerate(selected_displays):
        model_color = colors_dict[model_display]
        if model_display not in df_ate.columns:
            raise ValueError(f"Missing ATE column for model: {model_display}")
        model_estimate = df_ate.loc["estimate", model_display]
        model_ci_lower = df_ate.loc["ci_lower", model_display]
        model_ci_upper = df_ate.loc["ci_upper", model_display]
        ate_zero_visible = ate_zero_visible or (
            (model_ci_lower <= 0) and (model_ci_upper >= 0)
        )
        ate_y_min = min(ate_y_min, model_ci_lower)
        ate_y_max = max(ate_y_max, model_ci_upper)
        ax_ate.errorbar(
            model_x[idx],
            model_estimate,
            yerr=[
                [model_estimate - model_ci_lower],
                [model_ci_upper - model_estimate],
            ],
            fmt="o",
            markersize=ate_marker_size,
            capsize=ate_cap_size,
            linewidth=ate_line_width,
            elinewidth=ate_line_width,
            color=model_color,
            ecolor=model_color,
            markeredgecolor=model_color,
            markeredgewidth=ate_line_width,
            zorder=3,
        )

    ate_y_pad = 0.08 * (ate_y_max - ate_y_min) if ate_y_max > ate_y_min else 0.2
    ax_ate.set_ylim(ate_y_min - ate_y_pad, ate_y_max + ate_y_pad)
    ax_ate.set_xlim(0.0, len(selected_displays))
    ax_ate.set_xticks([])
    ax_ate.set_xticklabels([])
    ax_ate.tick_params(axis="x", length=0)
    ax_ate.set_xlabel("Synthetic approach", fontsize=joint_text_fontsize, labelpad=10)
    ax_ate.tick_params(axis="y", labelsize=joint_small_fontsize)
    ax_ate.grid(True, alpha=0.2, axis="y")
    ate_legend = ax_ate.legend(
        loc="upper right",
        bbox_to_anchor=(ate_legend_anchor_x, ate_legend_anchor_y),
        bbox_transform=ax_ate.transAxes,
        fontsize=joint_small_fontsize,
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        borderpad=0.2,
        labelspacing=0.2,
        handlelength=1.0,
        handleheight=0.8,
        borderaxespad=0.0,
    )
    ate_legend.get_frame().set_edgecolor(inset_legend_border_color)
    ate_legend.get_frame().set_boxstyle("round,pad=0.25")

    for idx, model_display in enumerate(selected_displays):
        model_color = colors_dict[model_display]
        if model_display not in df_ite.columns:
            raise ValueError(f"Missing ITE column for model: {model_display}")
        # Row 4: ITE decile calibration
        ax_ite_decile = ite_decile_axes[idx]
        x_vals, y_vals, y_std = ite_decile_stats[model_display]
        ax_ite_decile.errorbar(
            x_vals,
            y_vals,
            yerr=y_std,
            fmt=".",
            color=model_color,
            ecolor=model_color,
            markersize=ite_decile_dot_size,
            elinewidth=ite_decile_sd_line_width,
            capsize=0,
            zorder=3,
        )
        if np.isfinite(ite_decile_global_min) and np.isfinite(ite_decile_global_max):
            decile_pad = (
                0.05 * (ite_decile_global_max - ite_decile_global_min)
                if ite_decile_global_max > ite_decile_global_min
                else 0.1
            )
            decile_lo = ite_decile_global_min - decile_pad
            decile_hi = ite_decile_global_max + decile_pad
            ax_ite_decile.plot(
                [decile_lo, decile_hi],
                [decile_lo, decile_hi],
                "k--",
                alpha=0.5,
                linewidth=1.5,
                label="_nolegend_",
            )
            ax_ite_decile.set_xlim(decile_lo, decile_hi)
            ax_ite_decile.set_ylim(decile_lo, decile_hi)
        if model_display not in df_ite_decile_metrics.columns:
            raise ValueError(f"Missing ITE decile metrics for model: {model_display}")
        ax_ite_decile.text(
            ite_metric_annot_x,
            ite_metric_annot_y,
            format_calibration_metric(
                df_ite_decile_metrics.loc["ITE_decile_brier", model_display],
                df_ite_decile_metrics.loc["ITE_decile_ece", model_display],
                decimals=3,
            ),
            transform=ax_ite_decile.transAxes,
            fontsize=joint_small_fontsize,
            verticalalignment=ite_metric_annot_va,
            horizontalalignment=ite_metric_annot_ha,
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8),
        )
        if idx == 0:
            ax_ite_decile.tick_params(axis="y", labelsize=joint_small_fontsize)
        else:
            ax_ite_decile.tick_params(axis="y", left=False, labelleft=False)
        ax_ite_decile.tick_params(axis="x", labelsize=joint_small_fontsize, pad=1)
        ax_ite_decile.grid(True, alpha=0.2)
        # Row 5: Qini
        ax_qini = qini_axes[idx]
        ax_qini.set_box_aspect(1)
        curve_df = df_qini[df_qini["model"] == model_display].copy()
        if curve_df.empty:
            raise ValueError(f"Missing Qini curve for model: {model_display}")
        area = metric_map[model_display]
        curve_df["cum_gain_smooth"] = smooth_display_curve(
            curve_df["fraction"], curve_df["cum_gain"], window=qini_smooth_window
        )

        ax_qini.plot(
            base_curve["fraction"],
            base_curve["cum_gain_smooth"],
            color=ref_color,
            linewidth=qini_reference_line_width,
            alpha=0.9,
        )
        ax_qini.fill_between(
            base_curve["fraction"],
            base_curve["random"],
            base_curve["cum_gain_smooth"],
            color=ref_color,
            alpha=0.08,
            zorder=-1,
        )
        ax_qini.plot(
            curve_df["fraction"],
            curve_df["cum_gain_smooth"],
            color=model_color,
            linewidth=qini_line_width,
            label="_nolegend_",
        )
        ax_qini.fill_between(
            curve_df["fraction"],
            curve_df["random"],
            curve_df["cum_gain_smooth"],
            color=model_color,
            alpha=qini_fill_alpha,
            zorder=0,
        )
        ax_qini.plot(
            curve_df["fraction"],
            curve_df["random"],
            "k--",
            linewidth=random_line_width,
            alpha=0.6,
            label="random",
            color="black",
        )

        ax_qini.text(
            x_model_annot,
            y_model_annot,
            f"{area:.2f}",
            color=model_color,
            fontsize=joint_small_fontsize,
            ha="center",
            va="center",
            transform=ax_qini.transAxes,
        )
        ax_qini.text(
            x_ref_annot,
            y_ref_annot,
            f"{base_area:.2f}",
            color=ref_color,
            fontsize=joint_small_fontsize,
            ha="center",
            va="center",
            transform=ax_qini.transAxes,
        )
        if idx == 0:
            ax_qini.tick_params(axis="y", labelsize=joint_small_fontsize)
        else:
            ax_qini.tick_params(axis="y", left=False, labelleft=False)
        ax_qini.tick_params(axis="x", labelsize=joint_small_fontsize, pad=1)
        ax_qini.set_xticks([0, 0.5, 1])
        ax_qini.xaxis.set_major_formatter(FormatStrFormatter("%g"))
        ax_qini.grid(True, alpha=0.2)
        if idx == len(selected_displays) - 1:
            qini_legend = ax_qini.legend(
                loc="lower right",
                bbox_to_anchor=(inset_anchor_x, inset_anchor_y),
                bbox_transform=ax_qini.transAxes,
                fontsize=joint_small_fontsize,
                frameon=True,
                facecolor="white",
                framealpha=0.9,
                borderpad=0.2,
                labelspacing=0.2,
                handlelength=0.8,
                handleheight=0.8,
                borderaxespad=0.0,
            )
            qini_legend.get_frame().set_edgecolor("0.6")
            qini_legend.get_frame().set_boxstyle("round,pad=0.25")

    fig.canvas.draw()
    row1_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in propensity_axes]
    )
    row1_bottom = min(ax.get_position().y0 for ax in propensity_axes)
    row2_bottom = min(ax.get_position().y0 for ax in propensity_scatter_axes)
    row2_y = np.mean(
        [
            ax.get_position().y0 + ax.get_position().height / 2
            for ax in propensity_scatter_axes
        ]
    )
    row3_bottom = ax_ate.get_position().y0
    row3_y = np.mean([ax_ate.get_position().y0 + ax_ate.get_position().height / 2])
    row4_bottom = min(ax.get_position().y0 for ax in ite_decile_axes)
    row4_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in ite_decile_axes]
    )
    row5_bottom = min(ax.get_position().y0 for ax in qini_axes)
    row5_y = np.mean(
        [ax.get_position().y0 + ax.get_position().height / 2 for ax in qini_axes]
    )
    subplot_block_center_x = 0.5 * (
        propensity_axes[0].get_position().x0 + propensity_axes[-1].get_position().x1
    )
    fig.text(
        shared_ylabel_x,
        row1_y,
        "Number of patients",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        shared_ylabel_x,
        row2_y,
        "Propensity on original data",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        shared_ylabel_x,
        row3_y,
        "ATE with 95% CI",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        shared_ylabel_x,
        row4_y,
        "Synthetic ITE",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        shared_ylabel_x,
        row5_y,
        "Cumulative benefit",
        ha="center",
        va="center",
        rotation=90,
        fontsize=joint_text_fontsize,
    )
    fig.text(
        subplot_block_center_x,
        row1_bottom - 0.03,
        "Predicted propensity",
        ha="center",
        va="center",
        fontsize=joint_text_fontsize,
    )
    fig.text(
        subplot_block_center_x,
        row2_bottom - 0.025,
        "Propensity on synthetic data",
        ha="center",
        va="center",
        fontsize=joint_text_fontsize,
    )
    fig.text(
        subplot_block_center_x,
        row4_bottom - 0.025,
        "Original mean ITE per decile",
        ha="center",
        va="center",
        fontsize=joint_text_fontsize,
    )
    fig.text(
        subplot_block_center_x,
        row5_bottom - 0.025,
        "Ranked patient fraction",
        ha="center",
        va="top",
        fontsize=joint_text_fontsize,
    )
    bottom_legend_handles = [
        Patch(
            facecolor=ref_color, edgecolor="none", label=model_display_names[train_set]
        )
    ] + [
        Patch(
            facecolor=colors_dict[model_display],
            edgecolor="none",
            label=model_display,
        )
        for model_display in selected_displays
    ]
    propensity_legend = fig.legend(
        handles=bottom_legend_handles,
        loc="lower left",
        bbox_to_anchor=(
            bottom_legend_left,
            bottom_legend_bottom,
            bottom_legend_width,
            bottom_legend_height,
        ),
        ncol=len(bottom_legend_handles),
        frameon=False,
        fontsize=joint_text_fontsize,
        handlelength=1,
        handleheight=1,
        handletextpad=0.6,
        columnspacing=1.2,
        borderaxespad=0.2,
        mode="expand",
        bbox_transform=fig.transFigure,
    )
    propensity_legend.get_frame().set_edgecolor(inset_legend_border_color)
    propensity_legend.get_frame().set_boxstyle("round,pad=0.25")
    output_file = os.path.join(
        output_dir,
        f"joint_utility_summary_{cohort}_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].png",
    )
    plt.savefig(output_file, dpi=joint_dpi, bbox_inches="tight")
    print(f"Saved: {output_file}")
    plt.show()


plot_joint_utility_summary(train_set=train_set, pred_set=pred_set, horizon=horizon)
