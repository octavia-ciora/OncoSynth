import argparse
import os
import numpy as np
import pandas as pd
import importlib
from lifelines import KaplanMeierFitter
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr

from sklearn.metrics import log_loss, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
import utility_metrics
import CSF_in_R
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from load_data import *
from CSF_in_R import *
from utility_metrics import *

importlib.reload(utility_metrics)
importlib.reload(CSF_in_R)


def expected_calibration_error(y_true, y_prob, n_bins=10, strategy="quantile"):
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    valid = np.isfinite(y_true) & np.isfinite(y_prob)
    y_true = y_true[valid]
    y_prob = y_prob[valid]

    if len(y_true) == 0:
        return np.nan

    if strategy == "quantile":
        edges = np.quantile(y_prob, np.linspace(0, 1, n_bins + 1))
    elif strategy == "uniform":
        edges = np.linspace(0, 1, n_bins + 1)
    else:
        raise ValueError(f"Unknown calibration binning strategy: {strategy}")

    ece = 0.0
    n_total = len(y_true)
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        if b < n_bins - 1:
            mask = (y_prob >= lo) & (y_prob < hi)
        else:
            mask = (y_prob >= lo) & (y_prob <= hi)
        if not np.any(mask):
            continue
        observed = np.mean(y_true[mask])
        predicted = np.mean(y_prob[mask])
        ece += (np.sum(mask) / n_total) * abs(observed - predicted)

    return float(ece)


# Preprocess data
def preprocess_datasets(
    datasets,
    continuous_features,
    categorical_features,
    treatment_col,
    outcome_event_col,
    outcome_time_col,
    train_set,
    verbose=True,
):
    all_covariates = continuous_features + categorical_features
    df_train = datasets[train_set]

    categorical_cardinality = {
        feature: df_train[feature].nunique(dropna=True)
        for feature in categorical_features
    }
    binary_categorical_features = [
        feature
        for feature, cardinality in categorical_cardinality.items()
        if cardinality <= 2
    ]
    multiclass_categorical_features = [
        feature
        for feature, cardinality in categorical_cardinality.items()
        if cardinality > 2
    ]

    num_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    binary_cat_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "ordinal",
                OrdinalEncoder(
                    handle_unknown="use_encoded_value",
                    unknown_value=-1,
                    encoded_missing_value=-1,
                ),
            ),
        ]
    )

    transformers = [("num", num_pipe, continuous_features)]
    if binary_categorical_features:
        transformers.append(
            ("cat_binary", binary_cat_pipe, binary_categorical_features)
        )
    if multiclass_categorical_features:
        transformers.append(
            (
                "cat_multiclass",
                OneHotEncoder(drop=None, sparse_output=False, handle_unknown="ignore"),
                multiclass_categorical_features,
            )
        )

    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")

    preprocessor.fit(df_train[all_covariates])
    all_covariates_transformed = np.array(
        [
            feature_name.replace("cat_binary__", "cat__").replace(
                "cat_multiclass__", "cat__"
            )
            for feature_name in preprocessor.get_feature_names_out()
        ]
    )

    preprocessed_datasets = {}
    for dataset_name, df_dataset in datasets.items():
        if verbose:
            print(f"Preprocessing dataset: {dataset_name} (n={df_dataset.shape[0]})")

        T = df_dataset[outcome_time_col].values
        T = np.maximum(T, 0)
        preprocessed_datasets[dataset_name] = {
            "X": preprocessor.transform(df_dataset[all_covariates]),
            "W": df_dataset[treatment_col].to_numpy(dtype=int),
            "C": df_dataset[outcome_event_col].to_numpy(dtype=int),
            "T": T,
        }

    print("\nPreprocessing completed.")
    print(preprocessed_datasets.keys())
    return preprocessed_datasets, all_covariates_transformed


# Train treatment assignment models and predict propensities
def predict_propensities(
    preprocessed_datasets,
    output_dir,
    cfg,
    cohort_folder,
    train_set,
    pred_set,
    classifiers=["logistic"],
    verbose=True,
):
    X_test = preprocessed_datasets[pred_set]["X"]
    W_test = preprocessed_datasets[pred_set]["W"]

    for classifier_type in classifiers:
        df_propensities = pd.DataFrame()
        df_propensities["True_treatment"] = W_test

        for dataset_name in preprocessed_datasets.keys():
            if train_set == "downstream_train":
                if dataset_name in [pred_set, "generation_train"]:
                    continue
            elif train_set == "generation_train":
                if dataset_name in [pred_set, "downstream_train"]:
                    continue
            else:
                raise ValueError(f"Invalid train_set: {train_set}")
            if verbose:
                print(f"\nTraining propensity scores on: {dataset_name}")

            X = preprocessed_datasets[dataset_name]["X"]
            W = preprocessed_datasets[dataset_name]["W"]

            model = create_classifier(
                classifier_type, cfg=cfg, cohort_folder=cohort_folder, target="W"
            )
            model.fit(X, W)

            proba_test = model.predict_proba(X_test)[:, 1]
            pred_test = model.predict(X_test)

            dataset_display_name = cfg["model_display_names"][dataset_name]
            df_propensities[f"{dataset_display_name}_proba"] = proba_test
            df_propensities[f"{dataset_display_name}_pred"] = pred_test

        predictions_filepath = os.path.join(
            output_dir,
            f"propensities_{classifier_type}_train[{train_set}]_pred[{pred_set}].csv",
        )
        df_propensities.to_csv(predictions_filepath, index=False)


def evaluate_propensity(
    model_names,
    classifier,
    cfg,
    output_dir,
    train_set,
    pred_set,
):
    results = []
    reference_name = cfg["model_display_names"][train_set]

    propensity_file = os.path.join(
        output_dir, f"propensities_{classifier}_train[{train_set}]_pred[{pred_set}].csv"
    )
    df_propensity = pd.read_csv(propensity_file)
    true_treatment = df_propensity["True_treatment"].values

    eps = 1e-6

    for model_name in model_names:
        if model_name == reference_name:
            continue
        proba_ref = df_propensity[f"{reference_name}_proba"].values
        proba_synth = df_propensity[f"{model_name}_proba"].values

        similarity, distance = jsd_pi_similarity(proba_ref, proba_synth)
        mae = np.mean(np.abs(proba_ref - proba_synth))
        pearson_corr = np.corrcoef(proba_ref, proba_synth)[0, 1]
        spearman_corr, _ = spearmanr(proba_ref, proba_synth)
        brier_synth = np.mean((true_treatment - proba_synth) ** 2)
        brier_ref = np.mean((true_treatment - proba_ref) ** 2)
        ece_synth = expected_calibration_error(
            true_treatment, proba_synth, n_bins=10, strategy="quantile"
        )
        ece_ref = expected_calibration_error(
            true_treatment, proba_ref, n_bins=10, strategy="quantile"
        )
        auroc_synth = roc_auc_score(true_treatment, proba_synth)
        auroc_ref = roc_auc_score(true_treatment, proba_ref)
        log_loss_synth = log_loss(true_treatment, np.clip(proba_synth, eps, 1 - eps))
        log_loss_ref = log_loss(true_treatment, np.clip(proba_ref, eps, 1 - eps))

        results.append(
            {
                "model": model_name,
                "similarity": similarity,
                "distance": distance,
                "MAE": mae,
                "correlation": pearson_corr,
                "correlation_pearson": pearson_corr,
                "correlation_spearman": spearman_corr,
                "brier_score_synth": brier_synth,
                "brier_score_ref": brier_ref,
                "ece_synth": ece_synth,
                "ece_ref": ece_ref,
                "auroc_synth": auroc_synth,
                "auroc_ref": auroc_ref,
                "brier_check": brier_score_loss(true_treatment, proba_synth),
                "log_loss_synth": log_loss_synth,
                "log_loss_ref": log_loss_ref,
                "mean_prop_synth": float(np.mean(proba_synth)),
                "mean_prop_ref": float(np.mean(proba_ref)),
                "treat_rate": float(np.mean(true_treatment)),  # observed rate
            }
        )
    df_jsd = pd.DataFrame(results)
    jsd_filepath = os.path.join(
        output_dir,
        f"propensity_metrics_{classifier}_train[{train_set}]_pred[{pred_set}].csv",
    )
    df_jsd.to_csv(jsd_filepath, index=False)
    print("Propensity evaluation completed.")
    print(
        df_jsd[
            [
                "model",
                "MAE",
                "brier_score_synth",
                "ece_synth",
                "auroc_synth",
                "correlation_pearson",
                "correlation_spearman",
                "distance",
            ]
        ]
    )


def evaluate_propensity_deciles(
    model_names,
    classifier,
    cfg,
    output_dir,
    train_set,
    pred_set,
    n_deciles=10,
    verbose=True,
):

    reference_name = cfg["model_display_names"][train_set]

    propensity_file = os.path.join(
        output_dir, f"propensities_{classifier}_train[{train_set}]_pred[{pred_set}].csv"
    )
    df_propensity = pd.read_csv(propensity_file)

    true_treatment = df_propensity["True_treatment"].values
    proba_ref = df_propensity[f"{reference_name}_proba"].values

    decile_edges = np.percentile(proba_ref, np.linspace(0, 100, n_deciles + 1))
    decile_labels = [f"D{i+1}" for i in range(n_deciles)]

    decile_ids = np.digitize(proba_ref, decile_edges[1:-1])  # Returns 0 to n_deciles-1

    results = []

    for d in range(n_deciles):
        decile_mask = decile_ids == d

        if not np.any(decile_mask):
            if verbose:
                print(f"Warning: Decile {d+1} is empty")
            continue

        observed_rate = np.mean(true_treatment[decile_mask])

        mean_proba_ref = np.mean(proba_ref[decile_mask])

        calib_gap_ref = abs(observed_rate - mean_proba_ref)

        results.append(
            {
                "decile": decile_labels[d],
                "decile_num": d + 1,
                "model": reference_name,
                "n_patients": int(np.sum(decile_mask)),
                "observed_rate": observed_rate,
                "mean_propensity": mean_proba_ref,
                "calibration_gap": calib_gap_ref,
            }
        )

        for model_name in model_names:
            if model_name == reference_name:
                continue

            proba_synth = df_propensity[f"{model_name}_proba"].values
            mean_proba_synth = np.mean(proba_synth[decile_mask])
            calib_gap_synth = abs(observed_rate - mean_proba_synth)

            results.append(
                {
                    "decile": decile_labels[d],
                    "decile_num": d + 1,
                    "model": model_name,
                    "n_patients": int(np.sum(decile_mask)),
                    "observed_rate": observed_rate,
                    "mean_propensity": mean_proba_synth,
                    "calibration_gap": calib_gap_synth,
                }
            )

    df_deciles = pd.DataFrame(results)
    deciles_filepath = os.path.join(
        output_dir,
        f"propensity_deciles_{classifier}_train[{train_set}]_pred[{pred_set}].csv",
    )
    df_deciles.to_csv(deciles_filepath, index=False)

    if verbose:
        print(f"\nPropensity decile evaluation completed.")
        print(f"Saved to: {deciles_filepath}")

        print("\nMean calibration gap by model:")
        summary = df_deciles.groupby("model")["calibration_gap"].mean().sort_values()
        print(summary)

    return df_deciles


# Predict treatment effects with causal survival forest
def predict_effects(
    preprocessed_datasets,
    cfg,
    cohort,
    output_dir,
    horizon,
    all_covariates_transformed,
    train_set,
    pred_set,
    random_seed=42,
    verbose=True,
):

    X_test = preprocessed_datasets[pred_set]["X"]

    df_ATEs = pd.DataFrame(index=["estimate", "std_error", "ci_lower", "ci_upper"])
    df_ITEs = pd.DataFrame()

    for dataset_name in preprocessed_datasets.keys():
        if train_set == "downstream_train":
            if dataset_name in [pred_set, "generation_train"]:
                continue
        elif train_set == "generation_train":
            if dataset_name in [pred_set, "downstream_train"]:
                continue
        else:
            raise ValueError(f"Invalid train_set: {train_set}")
        dataset_display_name = cfg["model_display_names"][dataset_name]
        if verbose:
            print(f"\nPredicting effects on: {dataset_display_name}")

        X = preprocessed_datasets[dataset_name]["X"]
        W = preprocessed_datasets[dataset_name]["W"]
        C = preprocessed_datasets[dataset_name]["C"]
        T = preprocessed_datasets[dataset_name]["T"]

        csf_model = csf_fit(
            X=X,
            W=W,
            T=T,
            C=C,
            horizon=horizon,
            num_trees=cfg["evaluation"][cohort]["n_trees_csf"],
            min_node_size=cfg["evaluation"][cohort]["min_node_size_csf"],
            sample_fraction=cfg["evaluation"][cohort]["sample_fraction_csf"],
            mtry=cfg["evaluation"][cohort]["mtry_csf"],
            honesty=cfg["evaluation"][cohort]["honesty_csf"],
            honesty_fraction=cfg["evaluation"][cohort]["honesty_fraction_csf"],
            alpha=cfg["evaluation"][cohort]["alpha_csf"],
            imbalance_penalty=cfg["evaluation"][cohort]["imbalance_penalty_csf"],
            seed=random_seed,
        )

        # ate
        ate_results = get_ate_from_csf(csf_model, wald_ci=True)
        print(f"ATE results: {ate_results}")
        ate_dict = {
            "estimate": ate_results["estimate"],
            "std_error": ate_results["std.err"],
            "ci_lower": ate_results["ci.lower.wald"],
            "ci_upper": ate_results["ci.upper.wald"],
        }
        if dataset_display_name in df_ATEs.columns:
            raise ValueError(f"ATEs already stored for '{dataset_display_name}'")
        df_ATEs[dataset_display_name] = pd.Series(
            ate_dict, index=df_ATEs.index, dtype=float
        )

        # ite
        ite_test = csf_predict_ite(csf_model, X_test)
        df_ITEs[dataset_display_name] = ite_test

    ate_filepath = os.path.join(
        output_dir, f"ATE_csf_train[{train_set}]_horizon[{horizon}].csv"
    )
    df_ATEs.to_csv(ate_filepath, index=True)
    ite_filepath = os.path.join(
        output_dir,
        f"ITE_csf_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    df_ITEs.to_csv(ite_filepath, index=False)

    if verbose:
        print("Treatment effect prediction completed.")


# Evaluate effect predictions
def evaluate_predictions(
    model_names,
    cfg,
    output_dir,
    train_set,
    pred_set,
    horizon,
    verbose=True,
):
    df_ITE = pd.read_csv(
        os.path.join(
            output_dir,
            f"ITE_csf_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
        )
    )
    df_ATE = pd.read_csv(
        os.path.join(output_dir, f"ATE_csf_train[{train_set}]_horizon[{horizon}].csv"),
        index_col=0,
    )

    results = {}
    reference_name = cfg["model_display_names"][train_set]
    ite_bins = 30

    for model_name in model_names:
        if model_name == reference_name:
            continue

        tau_ref = df_ITE[reference_name].values
        tau_synth = df_ITE[model_name].values
        upehe = u_pehe(tau_ref, tau_synth)
        ite_mae = np.mean(np.abs(tau_ref - tau_synth))

        pearson_corr = np.corrcoef(tau_ref, tau_synth)[0, 1]
        spearman_corr, _ = spearmanr(tau_ref, tau_synth)

        ite_global_min = np.nanmin(np.concatenate([tau_ref, tau_synth]))
        ite_global_max = np.nanmax(np.concatenate([tau_ref, tau_synth]))
        ref_hist, _ = np.histogram(
            tau_ref, bins=ite_bins, range=(ite_global_min, ite_global_max)
        )
        synth_hist, _ = np.histogram(
            tau_synth, bins=ite_bins, range=(ite_global_min, ite_global_max)
        )
        if ref_hist.sum() > 0 and synth_hist.sum() > 0:
            ite_jsd = jensenshannon(
                ref_hist / ref_hist.sum(),
                synth_hist / synth_hist.sum(),
            )
        else:
            ite_jsd = np.nan

        ate_ref = df_ATE.loc["estimate", reference_name]
        ate_synth = df_ATE.loc["estimate", model_name]
        ate_dist = abs(ate_ref - ate_synth)

        results[model_name] = {
            "ATE_dist": ate_dist,
            "U-PEHE": upehe,
            "ITE_JSD": ite_jsd,
            "ITE_MAE": ite_mae,
            "ITE_correlation": pearson_corr,
            "ITE_correlation_pearson": pearson_corr,
            "ITE_correlation_spearman": spearman_corr,
        }

    df_results = pd.DataFrame.from_dict(results)
    results_filepath = os.path.join(
        output_dir,
        f"effect_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    df_results.to_csv(results_filepath, index=True)
    if verbose:
        print("Treatment effect evaluation completed.")
        print(
            df_results.loc[
                [
                    "ITE_JSD",
                    "ITE_MAE",
                    "ITE_correlation_pearson",
                    "ITE_correlation_spearman",
                ]
            ]
        )


def fit_km_censoring(time, event):
    km = KaplanMeierFitter()
    censor_event = 1 - np.asarray(event).astype(int)
    km.fit(np.asarray(time, dtype=float), event_observed=censor_event)
    return km


def ghat(km_cens, t):
    vals = np.asarray(km_cens.survival_function_at_times(t)).reshape(-1)
    return np.clip(vals, 1e-8, None)


def ipcw_eventfree_outcome(time, event, tau, km_cens):
    time = np.asarray(time, dtype=float)
    y = np.zeros_like(time, dtype=float)
    success = time >= tau
    if np.any(success):
        y[success] = 1.0 / ghat(km_cens, np.full(success.sum(), tau, dtype=float))
    return y


def transformed_outcome(y_ipcw, treatment, p=None, e=None):
    treatment = np.asarray(treatment).astype(int)
    y_ipcw = np.asarray(y_ipcw, dtype=float)

    if e is None:
        p = float(treatment.mean()) if p is None else float(p)
        e = np.full_like(y_ipcw, fill_value=p, dtype=float)
    else:
        e = np.clip(np.asarray(e, dtype=float), 1e-6, 1 - 1e-6)

    return y_ipcw * (treatment / e - (1 - treatment) / (1 - e))


def qini_curve_binned(score, z, n_bins=100):
    df = pd.DataFrame(
        {
            "score": np.asarray(score, float),
            "z": np.asarray(z, float),
        }
    )

    df = df[np.isfinite(df["score"]) & np.isfinite(df["z"])].copy()
    df = df.sort_values("score", ascending=False).reset_index(drop=True)

    df["bin"] = pd.qcut(df.index, q=n_bins, labels=False)

    grouped = df.groupby("bin", as_index=False).agg(z_sum=("z", "sum"), n=("z", "size"))

    grouped["cum_gain"] = grouped["z_sum"].cumsum()
    grouped["n_cum"] = grouped["n"].cumsum()
    grouped["fraction"] = grouped["n_cum"] / len(df)

    total_gain = grouped["z_sum"].sum()
    grouped["random"] = grouped["fraction"] * total_gain
    grouped["qini"] = grouped["cum_gain"] - grouped["random"]

    return grouped


def qini_auc(curve_df):
    return float(
        np.trapezoid(curve_df["qini"].to_numpy(), curve_df["fraction"].to_numpy())
    )


def survival_qini(time, event, treatment, score, tau, propensity=None):
    time = np.asarray(time, dtype=float)
    event = np.asarray(event, dtype=int)
    treatment = np.asarray(treatment, dtype=int)
    score = np.asarray(score, dtype=float)

    valid = (
        np.isfinite(time)
        & np.isfinite(event)
        & np.isfinite(treatment)
        & np.isfinite(score)
    )
    if propensity is not None:
        propensity = np.asarray(propensity, dtype=float)
        valid &= np.isfinite(propensity)

    time = time[valid]
    event = event[valid]
    treatment = treatment[valid]
    score = score[valid]
    propensity = propensity[valid] if propensity is not None else None

    km_cens = fit_km_censoring(time, event)
    y_ipcw = ipcw_eventfree_outcome(time, event, tau, km_cens)
    z = transformed_outcome(
        y_ipcw,
        treatment,
        p=float(treatment.mean()) if propensity is None else None,
        e=propensity,
    )
    curve = qini_curve_binned(score, z, n_bins=50)

    return curve, qini_auc(curve)


def evaluate_qini(
    preprocessed_datasets,
    model_names,
    cfg,
    output_dir,
    train_set,
    pred_set,
    horizon,
    classifier="lightgbm",
    verbose=True,
):
    reference_name = cfg["model_display_names"][train_set]
    df_ite = pd.read_csv(
        os.path.join(
            output_dir,
            f"ITE_csf_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
        )
    )

    time = preprocessed_datasets[pred_set]["T"]
    event = preprocessed_datasets[pred_set]["C"]
    treatment = preprocessed_datasets[pred_set]["W"]

    if len(df_ite) != len(time):
        raise ValueError(
            f"Length mismatch between ITE scores ({len(df_ite)}) and {pred_set} data ({len(time)})"
        )

    propensity = None
    prop_file = os.path.join(
        output_dir, f"propensities_{classifier}_train[{train_set}]_pred[{pred_set}].csv"
    )
    propensity_col = f"{reference_name}_proba"
    if os.path.exists(prop_file):
        df_prop = pd.read_csv(prop_file)
        if len(df_prop) == len(df_ite) and propensity_col in df_prop.columns:
            propensity = df_prop[propensity_col].to_numpy(dtype=float)

    all_models = [reference_name] + [m for m in model_names if m != reference_name]
    curve_tables = []
    metrics = []

    for model_name in all_models:
        if model_name not in df_ite.columns:
            raise ValueError(f"Missing required ITE column: {model_name}")

        curve_df, area = survival_qini(
            time=time,
            event=event,
            treatment=treatment,
            score=df_ite[model_name].to_numpy(dtype=float),
            tau=horizon,
            propensity=propensity,
        )
        curve_tables.append(curve_df.assign(model=model_name, tau=float(horizon)))
        metrics.append(
            {
                "model": model_name,
                "auqini": area,
                "n": len(curve_df),
                "tau": float(horizon),
            }
        )

    pd.concat(curve_tables, ignore_index=True).to_csv(
        os.path.join(
            output_dir,
            f"qini_curves_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
        ),
        index=False,
    )
    pd.DataFrame(metrics).to_csv(
        os.path.join(
            output_dir,
            f"qini_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
        ),
        index=False,
    )
    if verbose:
        print("Qini evaluation completed.")
        for metric in metrics:
            print(f"  {metric['model']}: {metric['auqini']:.4f}")


# Compute median ITEs within quantile bins
def evaluate_ite_quantiles_median(
    cfg,
    output_dir,
    train_set,
    pred_set,
    horizon,
    n_quantiles=4,
    include_iqr=False,
    include_counts=False,
    verbose=True,
):
    ite_file = os.path.join(
        output_dir,
        f"ITE_csf_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    df_ite = pd.read_csv(ite_file)

    reference_name = cfg["model_display_names"][train_set]
    if reference_name not in df_ite.columns:
        raise ValueError(
            f"Reference ITE column '{reference_name}' not found in {ite_file}. "
            f"Available columns: {list(df_ite.columns)}"
        )

    ref_ite = df_ite[reference_name].to_numpy()

    if n_quantiles < 2:
        raise ValueError("n_quantiles must be >= 2")

    quantiles = np.linspace(0.0, 1.0, n_quantiles + 1)

    edges = np.nanpercentile(ref_ite, quantiles * 100)

    if verbose:
        unique_edges = np.unique(edges)
        if len(unique_edges) < len(edges):
            print(
                "Warning: some quantile edges are identical (ties in reference ITE). "
                "Some bins may be empty or merged."
            )
        print(f"Quantile edges ({n_quantiles} bins): {edges}")

    bin_id = np.full(ref_ite.shape[0], fill_value=-1, dtype=int)
    for b in range(n_quantiles):
        lo, hi = edges[b], edges[b + 1]
        if b < n_quantiles - 1:
            mask = (ref_ite >= lo) & (ref_ite < hi)
        else:
            mask = (ref_ite >= lo) & (ref_ite <= hi)
        bin_id[mask] = b
    ###

    if np.any(bin_id == -1):
        raise Exception(f"{np.sum(bin_id == -1)} rows could not be binned")

    bin_labels = [f"Q{b+1}" for b in range(n_quantiles)]
    df_median = pd.DataFrame(index=bin_labels, columns=df_ite.columns, dtype=float)
    df_metrics = pd.DataFrame(
        index=["ITE_decile_brier", "ITE_decile_ece"],
        columns=df_ite.columns,
        dtype=float,
    )

    df_count = None
    df_iqr = None
    if include_counts:
        df_count = pd.DataFrame(index=bin_labels, columns=df_ite.columns, dtype=float)
    if include_iqr:
        df_iqr = pd.DataFrame(index=bin_labels, columns=df_ite.columns, dtype=float)

    for col in df_ite.columns:
        vals = df_ite[col].to_numpy()
        for b in range(n_quantiles):
            m = bin_id == b
            if not np.any(m):
                df_median.loc[bin_labels[b], col] = np.nan
                if include_counts:
                    df_count.loc[bin_labels[b], col] = 0
                if include_iqr:
                    df_iqr.loc[bin_labels[b], col] = np.nan
                continue

            v = vals[m]
            df_median.loc[bin_labels[b], col] = np.median(v)
            if include_counts:
                df_count.loc[bin_labels[b], col] = int(np.sum(m))
            if include_iqr:
                q75 = np.percentile(v, 75)
                q25 = np.percentile(v, 25)
                df_iqr.loc[bin_labels[b], col] = q75 - q25

    ref_bin_means = []
    bin_weights = []
    for b in range(n_quantiles):
        m = bin_id == b
        if not np.any(m):
            ref_bin_means.append(np.nan)
            bin_weights.append(0.0)
            continue
        ref_bin_means.append(np.mean(ref_ite[m]))
        bin_weights.append(np.sum(m) / len(ref_ite))
    ref_bin_means = np.array(ref_bin_means, dtype=float)
    bin_weights = np.array(bin_weights, dtype=float)

    for col in df_ite.columns:
        vals = df_ite[col].to_numpy()
        model_bin_means = []
        for b in range(n_quantiles):
            m = bin_id == b
            if not np.any(m):
                model_bin_means.append(np.nan)
                continue
            model_bin_means.append(np.mean(vals[m]))
        model_bin_means = np.array(model_bin_means, dtype=float)
        valid_bins = (
            np.isfinite(ref_bin_means)
            & np.isfinite(model_bin_means)
            & (bin_weights > 0)
        )
        if not np.any(valid_bins):
            df_metrics.loc["ITE_decile_brier", col] = np.nan
            df_metrics.loc["ITE_decile_ece", col] = np.nan
            continue
        errors = model_bin_means[valid_bins] - ref_bin_means[valid_bins]
        weights = bin_weights[valid_bins]
        weights = weights / weights.sum()
        df_metrics.loc["ITE_decile_brier", col] = np.sum(weights * (errors**2))
        df_metrics.loc["ITE_decile_ece", col] = np.sum(weights * np.abs(errors))

    # Save
    out_med = os.path.join(
        output_dir,
        f"ITE_quantile_medians_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    df_median.to_csv(out_med, index=True)
    out_metrics = os.path.join(
        output_dir,
        f"ITE_decile_metrics_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
    )
    df_metrics.to_csv(out_metrics, index=True)

    if include_counts:
        out_cnt = os.path.join(
            output_dir,
            f"ITE_quantile_counts_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
        )
        df_count.to_csv(out_cnt, index=True)

    if include_iqr:
        out_iqr = os.path.join(
            output_dir,
            f"ITE_quantile_iqr_train[{train_set}]_pred[{pred_set}]_horizon[{horizon}].csv",
        )
        df_iqr.to_csv(out_iqr, index=True)

    if verbose:
        print(f"Saved median ITEs to: {out_med}")
        print(f"Saved ITE decile metrics to: {out_metrics}")
        if include_counts:
            print(f"Saved counts to: {out_cnt}")
        if include_iqr:
            print(f"Saved IQRs to: {out_iqr}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cohort_folder",
        type=str,
        required=True,
        help="Cohort folder identifier, e.g. 'breast_0'.",
    )
    parser.add_argument(
        "--horizon",
        type=float,
        required=True,
        help="RMST horizon to use for evaluation.",
    )
    args = parser.parse_args()

    cohort_folder = args.cohort_folder
    horizon = int(args.horizon)

    ################################
    train_set = "downstream_train"
    pred_set = "heldout_test"
    classifier = "logistic"
    train_propensity = True
    n_quantiles = 6
    ################################

    cfg, datasets, features = load_cohort_data(cohort_folder, verbose=True)
    cohort = cohort_folder.split("_")[0]

    cohort_config = cfg["cohort_configs"][cohort]
    random_seed = cfg["evaluation"][cohort]["random_seed"]
    model_order = cfg["model_order"]
    model_names = [
        cfg["model_display_names"][m] if m in cfg["model_display_names"] else m
        for m in model_order
    ]
    treatment_col = cohort_config["treatment"]
    outcome_event_col = cohort_config["outcome_censoring"]
    outcome_time_col = cohort_config["outcome_time"]
    continuous_features = cohort_config["continuous_covariates"]
    categorical_features = cohort_config["categorical_covariates"]
    all_covariates = continuous_features + categorical_features
    output_dir = get_utility_evaluation_dir(cohort_folder=cohort_folder, cfg=cfg)
    os.makedirs(output_dir, exist_ok=True)

    preprocessed_datasets, all_covariates_transformed = preprocess_datasets(
        datasets=datasets,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        treatment_col=treatment_col,
        outcome_event_col=outcome_event_col,
        outcome_time_col=outcome_time_col,
        train_set=train_set,
        verbose=False,
    )
    if train_propensity:

        predict_propensities(
            preprocessed_datasets=preprocessed_datasets,
            classifiers=[classifier],
            cfg=cfg,
            cohort_folder=cohort_folder,
            output_dir=output_dir,
            train_set=train_set,
            pred_set=pred_set,
        )
        evaluate_propensity(
            model_names=model_names,
            classifier=classifier,
            cfg=cfg,
            output_dir=output_dir,
            train_set=train_set,
            pred_set=pred_set,
        )
        evaluate_propensity_deciles(
            model_names=model_names,
            classifier=classifier,
            cfg=cfg,
            output_dir=output_dir,
            train_set=train_set,
            pred_set=pred_set,
            n_deciles=10,
            verbose=False,
        )

    predict_effects(
        preprocessed_datasets=preprocessed_datasets,
        cfg=cfg,
        cohort=cohort,
        output_dir=output_dir,
        horizon=horizon,
        all_covariates_transformed=all_covariates_transformed,
        random_seed=random_seed,
        train_set=train_set,
        pred_set=pred_set,
    )

    evaluate_predictions(
        model_names=model_names,
        cfg=cfg,
        output_dir=output_dir,
        horizon=horizon,
        train_set=train_set,
        pred_set=pred_set,
        verbose=True,
    )
    evaluate_qini(
        preprocessed_datasets=preprocessed_datasets,
        model_names=model_names,
        cfg=cfg,
        output_dir=output_dir,
        horizon=horizon,
        classifier=classifier,
        train_set=train_set,
        pred_set=pred_set,
        verbose=True,
    )

    evaluate_ite_quantiles_median(
        cfg,
        output_dir,
        train_set=train_set,
        pred_set=pred_set,
        horizon=horizon,
        n_quantiles=n_quantiles,
        include_iqr=False,
        include_counts=False,
        verbose=True,
    )
