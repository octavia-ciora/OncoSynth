# %%

import os
import numpy as np
import pandas as pd
import optuna

os.environ["RPY2_CFFI_MODE"] = "ABI"

from rpy2.robjects import numpy2ri
from rpy2.robjects.conversion import localconverter
from rpy2.robjects import pandas2ri
import rpy2.robjects as ro
from rpy2.robjects.packages import importr

from sklearn.model_selection import train_test_split

from load_data import *
from CSF_in_R import csf_fit, csf_predict_ite, get_ate_from_csf
from evaluate_utility import preprocess_datasets

grf = importr("grf")


r_get_csf_scores = ro.r("""
    function(forest) {
      as.numeric(grf::get_scores(forest))
    }
    """)

r_get_csf_w_hat = ro.r("""
    function(forest) {
      if (!is.null(forest$W.hat)) {
        as.numeric(forest$W.hat)
      } else {
        numeric(0)
      }
    }
    """)


def get_csf_oob_scores(csf) -> np.ndarray:
    with localconverter(ro.default_converter + numpy2ri.converter):
        scores = np.array(r_get_csf_scores(csf), dtype=float)
    return scores


def get_csf_propensity_estimates(csf) -> np.ndarray:
    with localconverter(ro.default_converter + numpy2ri.converter):
        w_hat = np.array(r_get_csf_w_hat(csf), dtype=float)
    return w_hat


def tune_csf_optuna(
    cohort_folder: str,
    n_trials: int = 40,
    horizons: list[float] | None = None,
    tune_fraction: float = 0.2,
):

    if horizons is None:
        horizons = [
            96.0,
        ]

    cfg, datasets, _ = load_cohort_data(cohort_folder, verbose=True)
    cohort = cohort_folder.split("_")[0]
    cohort_config = cfg["cohort_configs"][cohort]

    seed = int(cfg["evaluation"][cohort]["random_seed"])

    treatment_col = cohort_config["treatment"]
    outcome_event_col = cohort_config.get(
        "outcome_event", cohort_config.get("outcome_censoring")
    )
    outcome_time_col = cohort_config["outcome_time"]

    continuous_features = cohort_config["continuous_covariates"]
    categorical_features = cohort_config["categorical_covariates"]

    output_dir = get_utility_evaluation_dir(cohort_folder=cohort_folder, cfg=cfg)
    os.makedirs(output_dir, exist_ok=True)

    df_train_full = datasets["downstream_train"].copy()
    stratify_labels = (
        df_train_full[treatment_col].astype(str)
        + "__"
        + df_train_full[outcome_event_col].astype(str)
    )

    df_tune, _ = train_test_split(
        df_train_full,
        train_size=tune_fraction,
        random_state=seed,
        stratify=stratify_labels,
    )

    preprocessed_datasets, _ = preprocess_datasets(
        datasets={"downstream_train": df_tune},
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        treatment_col=treatment_col,
        outcome_event_col=outcome_event_col,
        outcome_time_col=outcome_time_col,
        train_set="downstream_train",
        verbose=False,
    )
    x_tune = preprocessed_datasets["downstream_train"]["X"]
    w_tune = preprocessed_datasets["downstream_train"]["W"].astype(float)
    t_tune = np.maximum(
        preprocessed_datasets["downstream_train"]["T"].astype(float), 1e-8
    )
    c_tune = preprocessed_datasets["downstream_train"]["C"].astype(int)

    def _safe_spearman(x: np.ndarray, y: np.ndarray) -> float:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)

        mask = np.isfinite(x) & np.isfinite(y)
        if mask.sum() < 3:
            return 0.0

        xr = pd.Series(x[mask]).rank(method="average").to_numpy(dtype=float)
        yr = pd.Series(y[mask]).rank(method="average").to_numpy(dtype=float)

        if np.std(xr) <= 1e-12 or np.std(yr) <= 1e-12:
            return 0.0

        corr = float(np.corrcoef(xr, yr)[0, 1])
        if not np.isfinite(corr):
            return 0.0
        return corr

    def objective(trial: optuna.Trial) -> float:
        num_trees = trial.suggest_int("num_trees", 500, 3000, step=500)
        min_node_size = trial.suggest_int("min_node_size", 10, 100, step=10)

        sample_fraction = 0.5

        mtry = trial.suggest_int(
            "mtry",
            max(1, int(np.sqrt(x_tune.shape[1]))),
            x_tune.shape[1],
            step=max(1, x_tune.shape[1] // 10),
        )

        honesty = trial.suggest_categorical("honesty", [True, False])

        honesty_fraction = (
            trial.suggest_categorical("honesty_fraction", [0.5, 0.6, 0.7])
            if honesty
            else 0.5
        )

        alpha = 0.05

        imbalance_penalty = 0

        losses = []

        for horizon in horizons:
            csf = csf_fit(
                X=x_tune,
                W=w_tune,
                T=t_tune,
                C=c_tune,
                horizon=float(horizon),
                num_trees=num_trees,
                min_node_size=min_node_size,
                sample_fraction=sample_fraction,
                mtry=mtry,
                honesty=honesty,
                honesty_fraction=honesty_fraction,
                alpha=alpha,
                imbalance_penalty=imbalance_penalty,
                seed=seed,
            )

            # Orthogonal loss (MAIN TERM)
            oob_scores = get_csf_oob_scores(csf)
            oob_scores = oob_scores[np.isfinite(oob_scores)]

            if len(oob_scores) < 5:
                return 1e12

            orthogonal_loss = float(np.mean(oob_scores**2))

            # Overlap penalty
            w_hat = get_csf_propensity_estimates(csf)

            if len(w_hat) == 0:
                return 1e12

            w_hat = w_hat[np.isfinite(w_hat)]
            if len(w_hat) < 5:
                return 1e12

            overlap_penalty = float(np.mean((w_hat < 0.05) | (w_hat > 0.95)))

            # ITE collapse penalty
            ite = np.asarray(csf_predict_ite(csf, x_tune), dtype=float)
            ite = ite[np.isfinite(ite)]

            if len(ite) < 5:
                return 1e12

            q90, q10 = np.quantile(ite, [0.9, 0.1])
            ite_spread = float(q90 - q10)

            target_spread = 0.01
            ite_collapse_penalty = max(0.0, target_spread - ite_spread)

            loss_h = (
                orthogonal_loss + 0.5 * overlap_penalty + 0.5 * ite_collapse_penalty
            )

            losses.append(loss_h)

        final_loss = float(np.mean(losses))

        return final_loss

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    print("\nBest params:", study.best_params)
    print("Best score:", study.best_value)

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )

    horizon_tag = "_".join(str(int(h)) for h in horizons)
    out_path = os.path.join(output_dir, f"csf_optuna_trials_h_{horizon_tag}.csv")
    trials_df.to_csv(out_path, index=False)
    print("Saved:", out_path)

    return study, trials_df


###########
cohort = "demo_0"
n_trials = 50
tune_fraction = 0.8
###########

study, df_trials = tune_csf_optuna(
    cohort, n_trials=n_trials, tune_fraction=tune_fraction
)
