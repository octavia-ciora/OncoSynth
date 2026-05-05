# %%
#!/usr/bin/env python3
import argparse
import json
import os

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold


from load_data import get_utility_evaluation_dir, load_cohort_data
from evaluate_utility import preprocess_datasets


def _build_classifier(trial: optuna.Trial, classifier: str, seed: int):
    if classifier == "logistic":
        solver = trial.suggest_categorical(
            "logreg_solver", ["liblinear", "lbfgs", "saga"]
        )
        if solver == "lbfgs":
            penalty = "l2"
        else:
            penalty = trial.suggest_categorical("logreg_penalty", ["l1", "l2"])
        c_value = trial.suggest_float("logreg_c", 1e-4, 100.0, log=True)
        class_weight = trial.suggest_categorical(
            "logreg_class_weight", [None, "balanced"]
        )
        return LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=c_value,
            class_weight=class_weight,
            max_iter=5000,
            random_state=seed,
        )
    raise ValueError(f"Unsupported classifier: {classifier}")


def tune_w_model(
    cohort: str,
    setting_id: str,
    classifier: str,
    n_trials: int,
    seed=0,
):
    n_folds = 3

    cfg, datasets, _ = load_cohort_data(
        setting_id=setting_id,
        verbose=True,
        load_synthetic_data=False,
    )

    cohort_cfg = cfg["cohort_configs"][cohort]
    treatment_col = cohort_cfg["treatment"]
    outcome_event_col = cohort_cfg.get(
        "outcome_event", cohort_cfg.get("outcome_censoring")
    )
    outcome_time_col = cohort_cfg["outcome_time"]
    continuous_features = cohort_cfg["continuous_covariates"]
    categorical_features = cohort_cfg["categorical_covariates"]

    preprocessed, _ = preprocess_datasets(
        datasets={"downstream_train": datasets["downstream_train"]},
        continuous_features=continuous_features,
        categorical_features=categorical_features,
        treatment_col=treatment_col,
        outcome_event_col=outcome_event_col,
        outcome_time_col=outcome_time_col,
        verbose=False,
        train_set="downstream_train",
    )
    x_train = preprocessed["downstream_train"]["X"]
    w_train = preprocessed["downstream_train"]["W"].astype(int)

    def objective(trial: optuna.Trial) -> float:
        print(f"\nTrial {trial.number + 1}/{n_trials}:")
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        fold_bces = []

        for fold_idx, (train_idx, valid_idx) in enumerate(cv.split(x_train, w_train)):
            x_tr, x_va = x_train[train_idx], x_train[valid_idx]
            w_tr, w_va = w_train[train_idx], w_train[valid_idx]

            model = _build_classifier(
                trial=trial,
                classifier=classifier,
                seed=seed + fold_idx,
            )

            model.fit(x_tr, w_tr)
            print(f"Fold {fold_idx + 1} trained. Evaluating BCE on validation set...")
            proba_va = np.clip(model.predict_proba(x_va)[:, 1], 1e-8, 1 - 1e-8)
            fold_bces.append(float(log_loss(w_va, proba_va, labels=[0, 1])))

        cv_bce_mean = float(np.mean(fold_bces))
        cv_bce_std = float(np.std(fold_bces))
        trial.set_user_attr("cv_bce_mean", cv_bce_mean)
        trial.set_user_attr("cv_bce_std", cv_bce_std)
        trial.set_user_attr("n_folds", n_folds)
        return cv_bce_mean

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(objective, n_trials=n_trials)

    utility_dir = get_utility_evaluation_dir(cohort_folder=setting_id, cfg=cfg)
    output_dir = os.path.join(utility_dir, "optuna_eval_model", classifier)
    os.makedirs(output_dir, exist_ok=True)

    best_payload = {
        "cohort": cohort,
        "setting_id": setting_id,
        "seed": seed,
        "classifier": classifier,
        "n_folds": n_folds,
        "best_cv_bce": float(study.best_value),
        "best_params": study.best_params,
        "n_trials": len(study.trials),
    }

    trials_df = study.trials_dataframe(
        attrs=("number", "value", "params", "user_attrs", "state")
    )
    trials_path = os.path.join(output_dir, f"optuna_eval_model_trials_{classifier}.csv")
    best_path = os.path.join(output_dir, f"optuna_eval_model_best_{classifier}.json")
    trials_df.to_csv(trials_path, index=False)
    with open(best_path, "w") as f:
        json.dump(best_payload, f, indent=2)

    print(f"Saved trials: {trials_path}")
    print(f"Saved best:   {best_path}")
    print(f"Best CV BCE: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    return study, trials_df, best_payload


def main():
    ###########
    cohort = "demo"
    seed = 0
    n_trials = 50
    ###########

    tune_w_model(
        cohort=cohort,
        setting_id="{}_{}".format(cohort, seed),
        classifier="logistic",
        n_trials=n_trials,
        seed=seed,
    )


if __name__ == "__main__":
    main()

# %%
