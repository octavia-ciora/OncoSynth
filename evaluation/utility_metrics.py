import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from scipy.spatial.distance import jensenshannon


def _build_from_optuna_best(best_params, seed=42):
    model_type = best_params["model_type"]

    print(f"\n--- Building {model_type} with optuna best params:\n")

    if model_type == "logistic_regression":
        solver = best_params.get("logreg_solver", "lbfgs")
        penalty = best_params.get("logreg_penalty", "l2")
        if solver == "lbfgs":
            penalty = "l2"
        return LogisticRegression(
            solver=solver,
            penalty=penalty,
            C=float(best_params.get("logreg_c", 1.0)),
            class_weight=best_params.get("logreg_class_weight", None),
            max_iter=5000,
            random_state=seed,
        )

    raise ValueError(f"Unsupported Optuna model_type: {model_type}")


def _load_optuna_best_params(cfg, cohort_folder):
    cohort = cohort_folder.split("_")[0]
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), cfg["paths"]["base_dir"])
    )
    model_root = cfg["cohort_configs"][cohort]["model_dir"]
    best_path = os.path.join(
        base_dir,
        model_root,
        cohort_folder,
        "w_model_optuna",
        "optuna_w_model_best.json",
    )

    if not os.path.exists(best_path):
        return None
    with open(best_path, "r") as f:
        payload = json.load(f)
        print(
            f"\nLoaded Optuna best params from {best_path}:\n{json.dumps(payload, indent=2)}\n"
        )
    return payload.get("best_params", None)


def _load_eval_optuna_best_params(cfg, cohort_folder, classifier_type, target="W"):
    cohort = cohort_folder.split("_")[0]
    base_dir = os.path.abspath(
        os.path.join(os.path.dirname(__file__), cfg["paths"]["base_dir"])
    )
    eval_root = cfg["cohort_configs"][cohort]["evaluation_dir"]
    target = str(target).upper()

    candidate_paths = [
        os.path.join(
            base_dir,
            eval_root,
            cohort_folder,
            "utility",
            "optuna_eval_model",
            target,
            classifier_type,
            f"optuna_eval_model_best_{target}_{classifier_type}.json",
        ),
        os.path.join(
            base_dir,
            eval_root,
            cohort_folder,
            "utility",
            "optuna_eval_model",
            target,
            classifier_type,
            f"optuna_eval_model_best_{classifier_type}.json",
        ),
        # Legacy global path (historically W-based tuning).
        os.path.join(
            base_dir,
            eval_root,
            cohort_folder,
            "utility",
            "optuna_eval_model",
            classifier_type,
            f"optuna_eval_model_best_{classifier_type}.json",
        ),
    ]
    best_path = next((p for p in candidate_paths if os.path.exists(p)), None)
    if best_path is None:
        return None
    with open(best_path, "r") as f:
        payload = json.load(f)
        print(
            f"\nLoaded Optuna best params from {best_path}:\n{json.dumps(payload, indent=2)}\n"
        )
    return payload["best_params"]


def create_classifier(
    classifier_type="logistic", cfg=None, cohort_folder=None, seed=42, target="W"
):
    if classifier_type in ["logistic", "random_forest", "xgboost", "lightgbm"]:
        tuned = None
        if cfg is not None and cohort_folder is not None:
            tuned = _load_eval_optuna_best_params(
                cfg=cfg,
                cohort_folder=cohort_folder,
                classifier_type=classifier_type,
                target=target,
            )
        if tuned is not None:
            if classifier_type == "logistic":
                return _build_from_optuna_best(
                    {"model_type": "logistic_regression", **tuned}, seed=seed
                )
        if classifier_type == "logistic":
            return LogisticRegression(max_iter=1000, random_state=seed)

    elif classifier_type == "optuna_w":
        if cfg is None or cohort_folder is None:
            raise ValueError(
                "create_classifier('optuna_w') requires cfg and cohort_folder."
            )
        best_params = _load_optuna_best_params(cfg=cfg, cohort_folder=cohort_folder)
        if best_params is None:
            return LogisticRegression(max_iter=1000, random_state=seed)
        return _build_from_optuna_best(best_params=best_params, seed=seed)
    else:
        raise ValueError(f"Unknown classifier type: {classifier_type}")


def jsd_pi_similarity(proba_real, proba_synth, base=2, eps=1e-12):

    p = np.clip(np.asarray(proba_real), eps, 1 - eps)
    q = np.clip(np.asarray(proba_synth), eps, 1 - eps)

    P = np.column_stack([p, 1 - p])
    Q = np.column_stack([q, 1 - q])

    d = np.array([jensenshannon(P[i], Q[i], base=base) for i in range(len(p))])
    return 1 - float(d.mean()), float(d.mean())


def u_pehe(tau_real, tau_synth) -> float:
    a = np.asarray(tau_real, dtype=float).reshape(-1)
    b = np.asarray(tau_synth, dtype=float).reshape(-1)

    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch: tau_real {a.shape} vs tau_synth {b.shape}")

    if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("Non-finite values found in tau_real or tau_synth")

    diff = a - b
    return float(np.sqrt(np.mean(diff * diff)))
