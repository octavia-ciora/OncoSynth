import pandas as pd
import numpy as np
from scipy.spatial.distance import jensenshannon
from scipy.stats import wasserstein_distance, spearmanr, pearsonr
from typing import Dict, List, Tuple
from itertools import combinations
from lifelines import KaplanMeierFitter
from lifelines.utils import restricted_mean_survival_time


def compute_range_comparison(
    df_ref: pd.DataFrame,
    datasets_dict: Dict[str, pd.DataFrame],
    continuous_features: List[str],
    categorical_features: List[str],
    model_order: List[str],
    model_display_names: Dict[str, str],
    eval_set: str,
) -> pd.DataFrame:

    all_features = continuous_features + categorical_features
    range_data = {}

    # Compute ranges/values for ref data
    ref_ranges = []
    for feature in all_features:
        if feature in continuous_features:
            min_val = df_ref[feature].min()
            max_val = df_ref[feature].max()
            ref_ranges.append(f"({min_val:.2f}, {max_val:.2f})")
        else:  # categorical
            unique_vals = sorted(df_ref[feature].unique())
            unique_vals = [
                int(v) if isinstance(v, (np.integer, np.int64)) else v
                for v in unique_vals
            ]
            ref_ranges.append(str(unique_vals))
    range_data[eval_set] = ref_ranges

    # Compute ranges/values for each synthetic dataset
    for model_name in model_order:
        model_display = model_display_names[model_name]
        synth_ranges = []
        for feature in all_features:
            if feature in continuous_features:
                min_val = datasets_dict[model_name][feature].min()
                max_val = datasets_dict[model_name][feature].max()
                synth_ranges.append(f"({min_val:.2f}, {max_val:.2f})")
            else:  # categorical
                unique_vals = sorted(datasets_dict[model_name][feature].unique())
                unique_vals = [
                    int(v) if isinstance(v, (np.integer, np.int64)) else v
                    for v in unique_vals
                ]
                synth_ranges.append(str(unique_vals))
        range_data[model_display] = synth_ranges

    range_df = pd.DataFrame(range_data, index=all_features)

    return range_df


def univariate_fidelity(
    df_ref: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    continuous_features: List[str],
    categorical_features: List[str],
) -> pd.DataFrame:
    results = []

    # Compute Wasserstein distance for continuous features
    for feature in continuous_features:
        ref_values = df_ref[feature].values
        synth_values = df_synthetic[feature].values

        min_val = ref_values.min()
        max_val = ref_values.max()
        range_val = max_val - min_val

        if range_val > 0:
            ref_normalized = (ref_values - min_val) / range_val
            synth_normalized = (synth_values - min_val) / range_val

        else:
            print(feature)
            print(df_ref[feature].describe())
            print(min_val, max_val)
            print(range_val)
            print(ref_values)
            raise ValueError(
                f"Feature '{feature}' has zero variance in ref data; cannot normalize."
            )

        distance = wasserstein_distance(ref_normalized, synth_normalized)

        results.append({"feature": feature, "distance": distance, "type": "continuous"})

    # Compute Jensen-Shannon divergence for categorical features
    for feature in categorical_features:
        ref_values = df_ref[feature].values
        synth_values = df_synthetic[feature].values

        if pd.isnull(ref_values).any() or pd.isnull(synth_values).any():
            raise ValueError(f"NaN values found in feature '{feature}'.")

        ref_counts = pd.Series(ref_values).value_counts(normalize=True)
        synth_counts = pd.Series(synth_values).value_counts(normalize=True)

        all_categories = ref_counts.index.union(synth_counts.index)

        ref_probs = ref_counts.reindex(all_categories, fill_value=0).to_numpy()
        synth_probs = synth_counts.reindex(all_categories, fill_value=0).to_numpy()

        distance = jensenshannon(ref_probs, synth_probs)

        results.append(
            {"feature": feature, "distance": distance, "type": "categorical"}
        )

    results_df = pd.DataFrame(results)

    avg_distance = results_df["distance"].mean()
    avg_row = pd.DataFrame(
        [{"feature": "Average", "distance": avg_distance, "type": "average"}]
    )

    results_df = pd.concat([results_df, avg_row], ignore_index=True)

    return results_df


def compute_univariate_fidelity_on_subset(
    df_ref: pd.DataFrame,
    datasets_dict: Dict[str, pd.DataFrame],
    continuous_features: List[str],
    categorical_features: List[str],
    model_order: List[str],
    model_display_names: Dict[str, str],
    subset_name: str,
    verbose: bool = True,
) -> pd.DataFrame:
    fidelity_results = {}
    for model_name in model_order:
        fidelity_df = univariate_fidelity(
            df_ref=df_ref,
            df_synthetic=datasets_dict[model_name],
            continuous_features=continuous_features,
            categorical_features=categorical_features,
        )
        fidelity_results[model_name] = fidelity_df.set_index("feature")["distance"]

    fidelity_comparison = pd.DataFrame(fidelity_results)
    fidelity_comparison.columns = [model_display_names[m] for m in model_order]

    if verbose:
        print(f"\nUnivariate Fidelity - {subset_name}")
        print("=" * 70)
        print(fidelity_comparison)
        print()

    return fidelity_comparison


def compute_cramers_v(x: np.ndarray, y: np.ndarray) -> float:
    confusion_matrix = pd.crosstab(x, y)
    chi2 = 0
    n = len(x)

    row_sums = confusion_matrix.sum(axis=1)
    col_sums = confusion_matrix.sum(axis=0)

    for i in range(len(row_sums)):
        for j in range(len(col_sums)):
            expected = (row_sums.iloc[i] * col_sums.iloc[j]) / n
            if expected > 0:
                observed = confusion_matrix.iloc[i, j]
                chi2 += (observed - expected) ** 2 / expected

    min_dim = min(len(row_sums) - 1, len(col_sums) - 1)
    if min_dim > 0:
        cramers_v = np.sqrt(chi2 / (n * min_dim))
    else:
        cramers_v = 0.0

    return cramers_v


def compute_correlation_ratio(categorical: np.ndarray, continuous: np.ndarray) -> float:
    df_temp = pd.DataFrame({"cat": categorical, "cont": continuous})

    overall_mean = continuous.mean()
    overall_var = continuous.var()

    if overall_var == 0:
        return 0.0

    between_var = 0
    for category in df_temp["cat"].unique():
        group_data = df_temp[df_temp["cat"] == category]["cont"]
        group_mean = group_data.mean()
        group_size = len(group_data)
        between_var += group_size * (group_mean - overall_mean) ** 2

    between_var /= len(continuous)

    eta = np.sqrt(between_var / overall_var)

    return eta


def bivariate_correlation(
    df_ref: pd.DataFrame,
    df_synthetic: pd.DataFrame,
    features: List[str],
    continuous_features: List[str],
    categorical_features: List[str],
) -> Tuple[float, float, pd.DataFrame, pd.DataFrame]:

    n_features = len(features)

    corr_original = pd.DataFrame(
        np.ones((n_features, n_features)), index=features, columns=features
    )
    corr_synthetic = pd.DataFrame(
        np.ones((n_features, n_features)), index=features, columns=features
    )

    for i, feat1 in enumerate(features):
        for j, feat2 in enumerate(features):
            if i >= j:
                continue

            orig_val1 = df_ref[feat1].values
            orig_val2 = df_ref[feat2].values
            synth_val1 = df_synthetic[feat1].values
            synth_val2 = df_synthetic[feat2].values

            if feat1 in continuous_features and feat2 in continuous_features:
                # Continuous-Continuous: Spearman correlation
                orig_corr, _ = spearmanr(orig_val1, orig_val2)
                synth_corr, _ = spearmanr(synth_val1, synth_val2)

            elif feat1 not in continuous_features and feat2 not in continuous_features:
                # Categorical-Categorical: Cramer's V
                orig_corr = compute_cramers_v(orig_val1, orig_val2)
                synth_corr = compute_cramers_v(synth_val1, synth_val2)

            else:
                # Categorical-Continuous: Correlation ratio
                if feat1 in continuous_features:
                    orig_corr = compute_correlation_ratio(orig_val2, orig_val1)
                    synth_corr = compute_correlation_ratio(synth_val2, synth_val1)
                else:
                    orig_corr = compute_correlation_ratio(orig_val1, orig_val2)
                    synth_corr = compute_correlation_ratio(synth_val1, synth_val2)

            corr_original.iloc[i, j] = orig_corr
            corr_original.iloc[j, i] = orig_corr
            corr_synthetic.iloc[i, j] = synth_corr
            corr_synthetic.iloc[j, i] = synth_corr

    mask = np.triu(np.ones_like(corr_original, dtype=bool), k=1)

    corr_original_upper = corr_original.values[mask]
    corr_synthetic_upper = corr_synthetic.values[mask]

    mae = np.mean(np.abs(corr_original_upper - corr_synthetic_upper))
    pearson_r, _ = pearsonr(corr_original_upper, corr_synthetic_upper)

    return mae, pearson_r, corr_original, corr_synthetic


def compute_bivariate_correlation_on_subset(
    df_ref: pd.DataFrame,
    datasets_dict: Dict[str, pd.DataFrame],
    features: List[str],
    continuous_features: List[str],
    categorical_features: List[str],
    model_order: List[str],
    model_display_names: Dict[str, str],
    subset_name: str,
    eval_set: str,
    verbose: bool = True,
) -> pd.DataFrame:
    results = []
    correlation_dict = {}

    for model_name in model_order:
        mae, pearson_r, df_corr_orig, df_corr_synth = bivariate_correlation(
            df_ref=df_ref,
            df_synthetic=datasets_dict[model_name],
            features=features,
            continuous_features=continuous_features,
            categorical_features=categorical_features,
        )

        results.append(
            {
                "Model": model_display_names[model_name],
                "MAE": mae,
                "Pearson r": pearson_r,
            }
        )

        correlation_dict[model_display_names[model_name]] = df_corr_synth

    correlation_dict[eval_set] = df_corr_orig

    df_bivariate = pd.DataFrame(results).set_index("Model")

    if verbose:
        print(f"\nBivariate Correlation - {subset_name}")
        print("=" * 70)
        print(df_bivariate)
        print()

    # Return transposed version: MAE and Pearson r as rows, models as columns
    return df_bivariate.T, correlation_dict


def compute_rmst(df, time_col, event_col, horizon):
    """Compute Restricted Mean Survival Time up to a given horizon."""
    kmf = KaplanMeierFitter()
    kmf.fit(df[time_col], df[event_col])
    rmst_value = restricted_mean_survival_time(kmf, t=horizon)
    return rmst_value
