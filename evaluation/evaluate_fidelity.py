import argparse
import importlib
import os
import numpy as np
import pandas as pd
import load_data
import fidelity_metrics
from load_data import *
from fidelity_metrics import *

importlib.reload(fidelity_metrics)
importlib.reload(load_data)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--cohort_folder",
    type=str,
    required=True,
    help="Cohort folder identifier, e.g. 'breast_0'.",
)
args = parser.parse_args()

cohort_folder = args.cohort_folder
eval_set = "generation_train"


def survival_time_jsd(df_ref, df_synth, time_col, n_bins=30):
    """Compute survival-time JSD exactly as in visualize_fidelity.py."""
    ref_times = df_ref[time_col].to_numpy(dtype=float)
    synth_times = df_synth[time_col].to_numpy(dtype=float)

    ref_times = ref_times[np.isfinite(ref_times)]
    synth_times = synth_times[np.isfinite(synth_times)]

    if len(ref_times) <= 1 or len(synth_times) <= 1:
        return np.nan

    bins = np.histogram_bin_edges(
        np.concatenate([ref_times, synth_times]),
        bins=n_bins,
    )
    ref_hist, _ = np.histogram(ref_times, bins=bins, density=False)
    synth_hist, _ = np.histogram(synth_times, bins=bins, density=False)

    if ref_hist.sum() == 0 or synth_hist.sum() == 0:
        return np.nan

    ref_hist = ref_hist.astype(float)
    synth_hist = synth_hist.astype(float)
    ref_hist /= ref_hist.sum()
    synth_hist /= synth_hist.sum()

    return float(jensenshannon(ref_hist, synth_hist, base=2))


def get_dataset_for_display(datasets, display_name_map, display_name):
    for dataset_name, dataset_display_name in display_name_map.items():
        if dataset_display_name == display_name:
            return datasets[dataset_name]
    raise KeyError(f"Could not find dataset for display name '{display_name}'")


cfg, datasets, features = load_cohort_data(cohort_folder, verbose=True)
cohort = cohort_folder.split("_")[0]
cohort_config = cfg["cohort_configs"][cohort]
output_dir = get_fidelity_evaluation_dir(cohort_folder=cohort_folder, cfg=cfg)

df_ref_full = datasets[eval_set]
print_data_summary(cohort, features)

model_order = cfg["model_order"]
treatment_col = cohort_config["treatment"]
treatment_labels = get_label_map(cfg, cohort, treatment_col)

treatment_0_label = treatment_labels.get(0, "Treatment 0")
treatment_1_label = treatment_labels.get(1, "Treatment 1")

subsets = {
    "All Patients": (df_ref_full, datasets),
    treatment_0_label: (
        df_ref_full[df_ref_full[treatment_col] == 0],
        {k: v[v[treatment_col] == 0] for k, v in datasets.items() if k in model_order},
    ),
    treatment_1_label: (
        df_ref_full[df_ref_full[treatment_col] == 1],
        {k: v[v[treatment_col] == 1] for k, v in datasets.items() if k in model_order},
    ),
}


print("\n" + "=" * 70)
print("RANGE COMPARISON: Ref vs Synthetic Data")
print("=" * 70)
range_comparison = compute_range_comparison(
    df_ref=df_ref_full,
    datasets_dict=datasets,
    continuous_features=cohort_config["continuous_covariates"],
    categorical_features=cohort_config["categorical_covariates"],
    model_order=model_order,
    model_display_names=cfg["model_display_names"],
    eval_set=eval_set,
)
print(range_comparison)
print()

range_filepath = os.path.join(
    output_dir, f"range_comparison_all_patients_ref[{eval_set}].csv"
)
range_comparison.to_csv(range_filepath)
print(f"Saved range comparison: {range_filepath}\n")

# Univariate fidelity analysis

fidelity_by_subset = {}
for subset_name, (df_ref_subset, datasets_dict) in subsets.items():
    fidelity_df = compute_univariate_fidelity_on_subset(
        df_ref=df_ref_subset,
        datasets_dict=datasets_dict,
        continuous_features=cohort_config["continuous_covariates"],
        categorical_features=cohort_config["categorical_covariates"],
        model_order=model_order,
        model_display_names=cfg["model_display_names"],
        subset_name=subset_name,
    )
    fidelity_by_subset[subset_name] = fidelity_df

    filename = f"univariate_fidelity_{subset_name.replace(' ', '_').lower()}_ref[{eval_set}].csv"
    filepath = os.path.join(output_dir, filename)
    fidelity_df.to_csv(filepath)
    print(f"Saved: {filepath}")

combined_columns = []
for model_name in model_order:
    model_display = cfg["model_display_names"][model_name]
    for subset_name in subsets.keys():
        col_name = f"{model_display}_{subset_name.replace(' ', '_')}"
        print(col_name)
        combined_columns.append((col_name, model_display, subset_name))

fidelity_combined = pd.DataFrame(index=fidelity_by_subset["All Patients"].index)
for col_name, model_display, subset_name in combined_columns:
    fidelity_combined[col_name] = fidelity_by_subset[subset_name][model_display]

print("\n" + "=" * 120)
print("COMBINED UNIVARIATE FIDELITY TABLE")
print("=" * 120)
print(fidelity_combined)
print()

combined_filepath = os.path.join(
    output_dir, f"univariate_fidelity_combined_ref[{eval_set}].csv"
)
fidelity_combined.to_csv(combined_filepath)
print(f"Saved combined table: {combined_filepath}")

# Bivariate correlation analysis
print("\n" + "=" * 70)
print("BIVARIATE CORRELATION: Pairwise Correlation Analysis")
print("=" * 70)

all_features = (
    cohort_config["continuous_covariates"] + cohort_config["categorical_covariates"]
)

bivariate_by_subset = {}
for subset_name, (df_ref_subset, datasets_dict) in subsets.items():
    df_bivariate, correlation_dict = compute_bivariate_correlation_on_subset(
        df_ref=df_ref_subset,
        datasets_dict=datasets_dict,
        features=all_features,
        continuous_features=cohort_config["continuous_covariates"],
        categorical_features=cohort_config["categorical_covariates"],
        model_order=model_order,
        model_display_names=cfg["model_display_names"],
        subset_name=subset_name,
        eval_set=eval_set,
    )
    bivariate_by_subset[subset_name] = df_bivariate

    filename = f"bivariate_correlation_{subset_name.replace(' ', '_').lower()}_ref[{eval_set}].csv"
    filepath = os.path.join(output_dir, filename)
    df_bivariate.to_csv(filepath)
    print(f"Saved: {filepath}")

    for model_name, df_corr in correlation_dict.items():
        corr_filepath = os.path.join(
            output_dir,
            f"bivariate_correlation_matrix_{model_name.replace(' ', '_').lower()}_{subset_name.replace(' ', '_').lower()}_ref[{eval_set}].csv",
        )
        df_corr.to_csv(corr_filepath)
        print(f"Saved correlation matrix: {corr_filepath}")

# Create combined table with MAE and Pearson as rows, columns like univariate
combined_columns = []
for model_name in model_order:
    model_display = cfg["model_display_names"][model_name]
    for subset_name in subsets.keys():
        col_name = f"{model_display}_{subset_name.replace(' ', '_')}"
        combined_columns.append((col_name, model_display, subset_name))

bivariate_combined = pd.DataFrame(index=["MAE", "Pearson r"])
for col_name, model_display, subset_name in combined_columns:
    subset_data = bivariate_by_subset[subset_name]
    if model_display in subset_data.columns:
        bivariate_combined[col_name] = subset_data[model_display]

print("\n" + "=" * 120)
print("COMBINED BIVARIATE CORRELATION TABLE")
print("=" * 120)
print(bivariate_combined)
print()

bivariate_combined_filepath = os.path.join(
    output_dir, f"bivariate_correlation_combined_ref[{eval_set}].csv"
)
bivariate_combined.to_csv(bivariate_combined_filepath)
print(f"Saved combined bivariate table: {bivariate_combined_filepath}")


# Treatment and outcome statisticss
print("\n" + "=" * 70)
print("TREATMENT AND OUTCOME STATISTICS")
print("=" * 70)

treatment_col = cohort_config["treatment"]
event_col = cohort_config.get("outcome_event", cohort_config.get("outcome_censoring"))
time_col = cohort_config["outcome_time"]
event_labels = get_label_map(cfg, cohort, event_col)
ref_name = cfg["model_display_names"][eval_set]

model_display_list = [cfg["model_display_names"][m] for m in model_order]
dataset_columns = [ref_name] + model_display_list
diff_columns = [f"{m}-{ref_name}" for m in model_display_list]
all_columns = dataset_columns + diff_columns
treatment_outcome_stats = pd.DataFrame(columns=all_columns)

for col in dataset_columns:
    df = (
        df_ref_full
        if col == ref_name
        else datasets[[k for k, v in cfg["model_display_names"].items() if v == col][0]]
    )

    treatment_outcome_stats.loc["Total (count)", col] = len(df)

    treatment_counts = df[treatment_col].value_counts().sort_index()
    treatment_outcome_stats.loc[f"{treatment_labels[1]} (count)", col] = (
        treatment_counts.get(1, 0)
    )
    treatment_outcome_stats.loc[f"{treatment_labels[1]} (prob)", col] = (
        treatment_counts.get(1, 0) / len(df)
    )

    event_counts = df[event_col].value_counts().sort_index()
    treatment_outcome_stats.loc[f"{event_labels[1]} (count)", col] = event_counts.get(
        1, 0
    )
    treatment_outcome_stats.loc[f"{event_labels[1]} (prob)", col] = event_counts.get(
        1, 0
    ) / len(df)

    horizon = cfg["evaluation"][cohort]["rmst_horizon"]
    rmst_val = compute_rmst(df, time_col, event_col, horizon)
    treatment_outcome_stats.loc[f"RMST_{horizon}", col] = rmst_val

for model_display in model_display_list:
    diff_col = f"{model_display}-{ref_name}"
    for row in treatment_outcome_stats.index:
        treatment_outcome_stats.loc[row, diff_col] = np.abs(
            treatment_outcome_stats.loc[row, model_display]
            - treatment_outcome_stats.loc[row, ref_name]
        )

print(treatment_outcome_stats)
print()

treatment_outcome_filepath = os.path.join(
    output_dir, f"treatment_outcome_statistics_ref[{eval_set}].csv"
)
treatment_outcome_stats.to_csv(treatment_outcome_filepath)
print(f"Saved treatment and outcome statistics: {treatment_outcome_filepath}")


# Summary table
print("\n" + "=" * 70)
print("FIDELITY SUMMARY TABLE")
print("=" * 70)

summary_rows = [
    "Average univariate fidelity (all)",
    f"Average univariate fidelity ({treatment_col}=0)",
    f"Average univariate fidelity ({treatment_col}=1)",
    "Bivariate fidelity MAE (all)",
    f"Bivariate fidelity MAE ({treatment_col}=0)",
    f"Bivariate fidelity MAE ({treatment_col}=1)",
    "Bivariate fidelity Pearson (all)",
    f"Bivariate fidelity Pearson ({treatment_col}=0)",
    f"Bivariate fidelity Pearson ({treatment_col}=1)",
    f"Treatment prevalence ({treatment_col}=1) [prob]",
    "Treatment prevalence diff from ref [abs prob]",
    f"Event prevalence ({event_col}=1) [prob]",
    "Event prevalence diff from ref [abs prob]",
    "RMST diff to 36",
    "RMST diff to 60",
    "RMST diff to 84",
    "RMST diff to 120",
    f"Survival time JSD ({treatment_col}=0, {event_col}=0)",
    f"Survival time JSD ({treatment_col}=0, {event_col}=1)",
    f"Survival time JSD ({treatment_col}=1, {event_col}=0)",
    f"Survival time JSD ({treatment_col}=1, {event_col}=1)",
    "Average survival time JSD",
]

summary_df = pd.DataFrame(index=summary_rows, columns=model_display_list, dtype=float)

subset_name_t0 = treatment_0_label
subset_name_t1 = treatment_1_label

for model_display in model_display_list:
    dataset_key = [
        k for k, v in cfg["model_display_names"].items() if v == model_display
    ][0]
    df_synth = datasets[dataset_key]

    summary_df.loc["Average univariate fidelity (all)", model_display] = (
        fidelity_by_subset["All Patients"].loc["Average", model_display]
    )
    summary_df.loc[
        f"Average univariate fidelity ({treatment_col}=0)", model_display
    ] = fidelity_by_subset[subset_name_t0].loc["Average", model_display]
    summary_df.loc[
        f"Average univariate fidelity ({treatment_col}=1)", model_display
    ] = fidelity_by_subset[subset_name_t1].loc["Average", model_display]

    summary_df.loc["Bivariate fidelity MAE (all)", model_display] = bivariate_by_subset[
        "All Patients"
    ].loc["MAE", model_display]
    summary_df.loc[f"Bivariate fidelity MAE ({treatment_col}=0)", model_display] = (
        bivariate_by_subset[subset_name_t0].loc["MAE", model_display]
    )
    summary_df.loc[f"Bivariate fidelity MAE ({treatment_col}=1)", model_display] = (
        bivariate_by_subset[subset_name_t1].loc["MAE", model_display]
    )
    summary_df.loc["Bivariate fidelity Pearson (all)", model_display] = (
        bivariate_by_subset["All Patients"].loc["Pearson r", model_display]
    )
    summary_df.loc[f"Bivariate fidelity Pearson ({treatment_col}=0)", model_display] = (
        bivariate_by_subset[subset_name_t0].loc["Pearson r", model_display]
    )
    summary_df.loc[f"Bivariate fidelity Pearson ({treatment_col}=1)", model_display] = (
        bivariate_by_subset[subset_name_t1].loc["Pearson r", model_display]
    )

    treatment_prev_ref = (df_ref_full[treatment_col].to_numpy(dtype=float) == 1).mean()
    treatment_prev_synth = (df_synth[treatment_col].to_numpy(dtype=float) == 1).mean()
    summary_df.loc[
        f"Treatment prevalence ({treatment_col}=1) [prob]", model_display
    ] = treatment_prev_synth
    summary_df.loc["Treatment prevalence diff from ref [abs prob]", model_display] = (
        abs(treatment_prev_synth - treatment_prev_ref)
    )

    event_prev_ref = (df_ref_full[event_col].to_numpy(dtype=float) == 1).mean()
    event_prev_synth = (df_synth[event_col].to_numpy(dtype=float) == 1).mean()
    summary_df.loc[f"Event prevalence ({event_col}=1) [prob]", model_display] = (
        event_prev_synth
    )
    summary_df.loc["Event prevalence diff from ref [abs prob]", model_display] = abs(
        event_prev_synth - event_prev_ref
    )

    for horizon in [36, 60, 84, 120]:
        rmst_ref = compute_rmst(df_ref_full, time_col, event_col, horizon)
        rmst_synth = compute_rmst(df_synth, time_col, event_col, horizon)
        summary_df.loc[f"RMST diff to {horizon}", model_display] = abs(
            rmst_synth - rmst_ref
        )

    jsd_values = []
    for treatment_value in [0, 1]:
        for event_value in [0, 1]:
            row_name = f"Survival time JSD ({treatment_col}={treatment_value}, {event_col}={event_value})"
            df_ref_stratum = df_ref_full[
                (df_ref_full[treatment_col] == treatment_value)
                & (df_ref_full[event_col] == event_value)
            ]
            df_synth_stratum = df_synth[
                (df_synth[treatment_col] == treatment_value)
                & (df_synth[event_col] == event_value)
            ]
            jsd_value = survival_time_jsd(
                df_ref=df_ref_stratum,
                df_synth=df_synth_stratum,
                time_col=time_col,
                n_bins=30,
            )
            summary_df.loc[row_name, model_display] = jsd_value
            if np.isfinite(jsd_value):
                jsd_values.append(jsd_value)

    summary_df.loc["Average survival time JSD", model_display] = (
        float(np.mean(jsd_values)) if jsd_values else np.nan
    )

print(summary_df)
print()

summary_filepath = os.path.join(output_dir, f"fidelity_summary_ref[{eval_set}].csv")
summary_df.to_csv(summary_filepath)
print(f"Saved fidelity summary table: {summary_filepath}")
