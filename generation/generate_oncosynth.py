#!/usr/bin/env python3
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime

import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from sksurv.ensemble import RandomSurvivalForest
from sksurv.util import Surv

MIN_ROWS_PER_ARM = 200
MIN_OBSERVED_PER_PROCESS = 10

C_COL_IS_CENSORING = False  # False -> c_col == 1 means event observed

APPLY_INTEGER_DAYS = True
APPLY_CLIP = False
CLIP_MAX = None

# default RSF params for latent time models
RSF_PARAMS = {
    "n_estimators": 600,
    "max_depth": None,
    "min_samples_split": 6,
    "min_samples_leaf": 5,
    "max_features": 0.7,
    "bootstrap": True,
    "max_leaf_nodes": None,
    "max_samples": None,
    "low_memory": False,
    "n_jobs": -1,
}

TIME_EPS = 1e-8


def _pick_best_ema(ckpt_dir: str) -> str | None:
    candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("best_ema_model")]

    def _epoch(name: str) -> int:
        return int(name.split("_")[-1].split(".")[0])

    candidates.sort(key=_epoch, reverse=True)
    return os.path.join(ckpt_dir, candidates[0]) if candidates else None


def _extract_epoch(ckpt_path: str) -> int:
    name = os.path.basename(ckpt_path)
    return int(name.split("_")[-1].split(".")[0])


def _preprocess(
    df: pd.DataFrame, continuous_features: list[str], categorical_features: list[str]
) -> tuple[np.ndarray, ColumnTransformer]:
    missing = [
        c for c in (continuous_features + categorical_features) if c not in df.columns
    ]
    if missing:
        raise ValueError(f"Missing columns for preprocessing: {missing}")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), continuous_features),
            (
                "cat",
                OneHotEncoder(
                    drop="first", sparse_output=False, handle_unknown="ignore"
                ),
                categorical_features,
            ),
        ],
        remainder="drop",
    )

    x = preprocessor.fit_transform(df[continuous_features + categorical_features])
    return x, preprocessor


def _fit_w_model(
    x_train: np.ndarray,
    w_train: np.ndarray,
    x_valid: np.ndarray,
    w_valid: np.ndarray,
    base_model,
):
    fitted = clone(base_model)
    fitted.fit(x_train, w_train)

    if len(np.unique(w_valid)) < 2:
        print("Validation W has <2 classes; skipping calibration.")
        return fitted

    calibrated = CalibratedClassifierCV(fitted, method="isotonic", cv="prefit")
    calibrated.fit(x_valid, w_valid)
    return calibrated


def _extract_best_w_model(model_root: str, seed: int):
    default_model = LogisticRegression(max_iter=1000, solver="lbfgs", random_state=seed)
    best_path = os.path.join(model_root, "w_model_optuna", "optuna_w_model_best.json")
    if not os.path.exists(best_path):
        print(f"No Optuna W-model found at {best_path}. Using logistic regression.")
        return default_model

    with open(best_path, "r") as f:
        best_payload = json.load(f)
    best_params = best_payload.get("best_params", {})
    model_type = best_params.get("model_type", "logistic_regression")
    print("--- Found Optuna W-model with parameters: ---")
    for param, value in best_params.items():
        print(f"{param}: {value}")

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
    raise ValueError(f"Unsupported model type in Optuna W-model: {model_type}")


def _normalize_optuna_rsf_params(best_params: dict) -> dict:
    params = dict(RSF_PARAMS)

    if not best_params:
        return params

    if "n_estimators" in best_params:
        params["n_estimators"] = int(best_params["n_estimators"])

    if "max_depth" in best_params:
        params["max_depth"] = (
            None if best_params["max_depth"] is None else int(best_params["max_depth"])
        )
    elif best_params.get("max_depth_mode") == "none":
        params["max_depth"] = None

    if "min_samples_split" in best_params:
        params["min_samples_split"] = int(best_params["min_samples_split"])
    if "min_samples_leaf" in best_params:
        params["min_samples_leaf"] = int(best_params["min_samples_leaf"])

    if "max_features" in best_params:
        val = best_params["max_features"]
        if isinstance(val, str):
            params["max_features"] = val
        else:
            params["max_features"] = float(val)

    if "bootstrap" in best_params:
        params["bootstrap"] = bool(best_params["bootstrap"])

    if "max_leaf_nodes" in best_params:
        params["max_leaf_nodes"] = (
            None
            if best_params["max_leaf_nodes"] is None
            else int(best_params["max_leaf_nodes"])
        )

    if "max_samples" in best_params:
        params["max_samples"] = (
            None
            if best_params["max_samples"] is None
            else float(best_params["max_samples"])
        )

    if "low_memory" in best_params:
        params["low_memory"] = bool(best_params["low_memory"])

    return params


def _load_best_rsf_params(
    model_root: str,
    process_name: str,
    arm_value: int,
) -> dict:

    best_path = os.path.join(
        model_root,
        "rsf_time_model_optuna",
        f"optuna_{process_name}_w{arm_value}_best.json",
    )

    if not os.path.exists(best_path):
        print(
            f"No Optuna RSF model found at {best_path}. "
            f"Using default RSF_PARAMS for {process_name}, W={arm_value}."
        )
        return dict(RSF_PARAMS)

    with open(best_path, "r") as f:
        payload = json.load(f)

    best_params = payload.get("best_params", {})
    final_params = _normalize_optuna_rsf_params(best_params)

    print(f"--- Found Optuna RSF params for {process_name}, W={arm_value} ---")
    for k, v in final_params.items():
        print(f"{k}: {v}")

    return final_params


def _get_feature_names(preprocessor, raw_cols, n_out):
    try:
        return list(preprocessor.get_feature_names_out(raw_cols))
    except Exception:
        return [f"x_{i}" for i in range(n_out)]


def _build_design_matrix_x_only(
    df: pd.DataFrame,
    *,
    x_preprocessor,
    continuous_features,
    categorical_features,
) -> pd.DataFrame:
    raw_cols = continuous_features + categorical_features
    Z = x_preprocessor.transform(df[raw_cols])
    feat_names = _get_feature_names(x_preprocessor, raw_cols, Z.shape[1])
    return pd.DataFrame(Z, columns=feat_names, index=df.index)


def _event_observed_from_c(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c).astype(int)
    if C_COL_IS_CENSORING:
        return 1 - c
    return c


def _censor_observed_from_c(c: np.ndarray) -> np.ndarray:
    c = np.asarray(c).astype(int)
    if C_COL_IS_CENSORING:
        return c
    return 1 - c


def _drop_low_variance_and_duplicate_columns(
    df: pd.DataFrame,
    variance_threshold: float = 1e-10,
) -> tuple[pd.DataFrame, list[str]]:
    df = df.copy()

    if df.shape[1] == 0:
        raise ValueError("Design matrix has no columns.")

    variances = df.var(axis=0, numeric_only=True)
    keep_var = variances[variances > variance_threshold].index.tolist()
    dropped = [c for c in df.columns if c not in keep_var]
    df = df[keep_var].copy()

    if df.shape[1] == 0:
        raise ValueError("All columns were removed due to low variance.")

    duplicated_mask = df.T.duplicated()
    duplicate_cols = df.columns[duplicated_mask].tolist()
    if duplicate_cols:
        df = df.loc[:, ~duplicated_mask].copy()
        dropped.extend(duplicate_cols)

    if df.shape[1] == 0:
        raise ValueError("All columns were removed after dropping duplicates.")

    return df, dropped


def _fit_rsf_time_model(
    X_df: pd.DataFrame,
    durations: np.ndarray,
    event_indicator: np.ndarray,
    seed: int,
    rsf_params: dict | None = None,
) -> tuple[RandomSurvivalForest, list[str], float, float, dict]:
    durations = np.asarray(durations, dtype=float)
    event_indicator = np.asarray(event_indicator, dtype=bool)

    if (durations <= 0).any():
        raise ValueError("All durations must be strictly positive.")

    if len(durations) != len(event_indicator):
        raise ValueError("durations and event_indicator must have the same length.")

    X_clean, dropped = _drop_low_variance_and_duplicate_columns(X_df)
    keep_cols = list(X_clean.columns)

    y = Surv.from_arrays(event=event_indicator, time=durations)

    params = dict(RSF_PARAMS if rsf_params is None else rsf_params)
    params["random_state"] = seed

    model = RandomSurvivalForest(**params)
    model.fit(X_clean.to_numpy(dtype=float), y)

    t_min = float(np.min(durations))
    t_max = float(np.max(durations))

    meta = {
        "n_estimators": int(params["n_estimators"]),
        "max_depth": params["max_depth"],
        "min_samples_split": int(params["min_samples_split"]),
        "min_samples_leaf": int(params["min_samples_leaf"]),
        "max_features": params["max_features"],
        "bootstrap": bool(params.get("bootstrap", True)),
        "max_leaf_nodes": params.get("max_leaf_nodes", None),
        "max_samples": params.get("max_samples", None),
        "low_memory": bool(params.get("low_memory", False)),
        "dropped_columns": dropped,
        "n_rows": int(len(durations)),
        "n_events": int(np.sum(event_indicator)),
    }
    return model, keep_cols, t_min, t_max, meta


def _sample_time_from_rsf_survival_functions(
    model: RandomSurvivalForest,
    X_df: pd.DataFrame,
    keep_cols: list[str],
    rng: np.random.Generator,
    fallback_time: float,
    tail_mode: str = "exponential",
    min_tail_hazard: float = 1e-6,
) -> np.ndarray:
    """
    Sample from RSF survival functions with proper handling of residual tail mass.

    tail_mode:
        - "exponential": extrapolate beyond last support point using exponential tail
        - "administrative": place unresolved draws at fallback_time
    """

    if len(X_df) == 0:
        return np.array([], dtype=float)

    X_use = X_df[keep_cols].to_numpy(dtype=float)
    surv_fns = model.predict_survival_function(X_use, return_array=False)

    sampled = np.empty(len(surv_fns), dtype=float)

    for i, fn in enumerate(surv_fns):
        times = np.asarray(fn.x, dtype=float)
        surv = np.asarray(fn.y, dtype=float)

        if times.size == 0:
            sampled[i] = max(float(fallback_time), TIME_EPS)
            continue

        surv = np.clip(surv, 0.0, 1.0)
        surv = np.minimum.accumulate(surv)

        u = rng.uniform()

        idx = np.searchsorted(-surv, -u, side="left")

        if idx < len(times):
            t = times[idx]
            sampled[i] = max(float(t), TIME_EPS)
            continue

        t_last = float(times[-1])
        s_last = float(surv[-1])

        if tail_mode == "administrative":
            t = fallback_time

        elif tail_mode == "exponential":
            if len(times) >= 2:
                t_prev = float(times[-2])
                s_prev = float(max(surv[-2], s_last + 1e-12))
                dt = max(t_last - t_prev, TIME_EPS)

                if s_last > 0 and s_prev > s_last:
                    h_tail = max(
                        (np.log(s_prev) - np.log(s_last)) / dt, min_tail_hazard
                    )
                else:
                    h_tail = min_tail_hazard
            else:
                h_tail = min_tail_hazard

            u_eff = min(u, s_last - 1e-15) if s_last > 1e-15 else 1e-15
            delta = np.log(max(s_last, 1e-15) / max(u_eff, 1e-15)) / h_tail
            t = t_last + delta

            if not np.isfinite(t):
                t = fallback_time

        else:
            raise ValueError(f"Unsupported tail_mode: {tail_mode}")

        sampled[i] = max(float(t), TIME_EPS)

    return sampled


def _estimate_admin_censoring_model(
    df: pd.DataFrame,
    *,
    year_col: str,
    time_col: str,
    c_col: str,
    quantile: float = 0.98,
    use_only_censored: bool = True,
    smoothing: str = "linear",
    min_rows_per_year: int = 5,
) -> dict:
    if year_col not in df.columns:
        raise ValueError(
            f"Administrative censoring year column '{year_col}' not found."
        )

    work = df[[year_col, time_col, c_col]].copy()
    work[year_col] = pd.to_numeric(work[year_col], errors="coerce")
    work[time_col] = pd.to_numeric(work[time_col], errors="coerce")
    work = work.dropna(subset=[year_col, time_col])

    if use_only_censored:
        censor_mask = _censor_observed_from_c(work[c_col].to_numpy()).astype(bool)
        work = work.loc[censor_mask].copy()
        if work.empty:
            raise ValueError(
                "No censored rows available to estimate administrative censoring."
            )

    grouped = (
        work.groupby(year_col)[time_col]
        .agg(["count", lambda s: float(np.quantile(s.to_numpy(dtype=float), quantile))])
        .reset_index()
    )
    grouped.columns = [year_col, "count", "cap_raw"]
    grouped = grouped[grouped["count"] >= min_rows_per_year].copy()

    if grouped.empty:
        raise ValueError(
            "No diagnosis-year groups had enough rows to estimate administrative censoring."
        )

    grouped = grouped.sort_values(year_col).reset_index(drop=True)
    years = grouped[year_col].to_numpy(dtype=float)
    caps_raw = grouped["cap_raw"].to_numpy(dtype=float)

    if smoothing == "none" or len(years) < 2:
        caps_smooth = caps_raw.copy()
        fit_info = None
    elif smoothing == "linear":
        slope, intercept = np.polyfit(years, caps_raw, deg=1)
        caps_smooth = intercept + slope * years
        fit_info = {
            "type": "linear",
            "slope": float(slope),
            "intercept": float(intercept),
        }
    else:
        raise ValueError(
            f"Unsupported admin censor smoothing mode: {smoothing}. "
            "Use 'linear' or 'none'."
        )

    caps_smooth = np.maximum(caps_smooth, TIME_EPS)

    out = grouped.copy()
    out["cap_smooth"] = caps_smooth

    print("--- Estimated administrative censoring boundary by year ---")
    for _, row in out.iterrows():
        print(
            f"{year_col}={row[year_col]} | n={int(row['count'])} | "
            f"raw_q={row['cap_raw']:.4f} | cap={row['cap_smooth']:.4f}"
        )

    return {
        "year_col": year_col,
        "quantile": float(quantile),
        "use_only_censored": bool(use_only_censored),
        "smoothing": smoothing,
        "min_rows_per_year": int(min_rows_per_year),
        "per_year_table": out,
        "fit_info": fit_info,
    }


def _predict_admin_censor_times(
    df: pd.DataFrame,
    admin_model: dict,
) -> np.ndarray:
    year_col = admin_model["year_col"]
    if year_col not in df.columns:
        raise ValueError(
            f"Administrative censoring year column '{year_col}' not found in synth data."
        )

    years = pd.to_numeric(df[year_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(years).any():
        raise ValueError(
            f"Synthetic data contains NaN/non-numeric values in '{year_col}'."
        )

    table = admin_model["per_year_table"]
    fit_info = admin_model["fit_info"]
    smoothing = admin_model["smoothing"]

    if smoothing == "linear" and fit_info is not None:
        caps = fit_info["intercept"] + fit_info["slope"] * years
        caps = np.maximum(caps, TIME_EPS)
        return caps.astype(float)

    known_years = table[year_col].to_numpy(dtype=float)
    known_caps = table["cap_smooth"].to_numpy(dtype=float)

    caps = np.empty(len(years), dtype=float)
    for i, y in enumerate(years):
        idx = int(np.argmin(np.abs(known_years - y)))
        caps[i] = known_caps[idx]

    caps = np.maximum(caps, TIME_EPS)
    return caps.astype(float)


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description=(
            "Generate synthetic data with latent event and censoring times using "
            "arm-specific Random Survival Forests, with explicit year-based "
            "administrative censoring."
        )
    )
    parser.add_argument(
        "--config",
        default=os.path.join("generation", "config.yaml"),
        help="Path to generation config.yaml",
    )
    parser.add_argument("--cohort", default="breast", help="Cohort name")
    parser.add_argument(
        "--cohort_seed",
        type=int,
        required=True,
        help="Cohort seed used to select the prepared cohort split",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )
    parser.add_argument(
        "--admin_censor_quantile",
        type=float,
        default=0.98,
        help=(
            "High quantile of follow-up among censored rows, by diagnosis year, "
            "used as the administrative censoring cap."
        ),
    )
    parser.add_argument(
        "--admin_censor_smoothing",
        type=str,
        default="linear",
        choices=["linear", "none"],
        help=("How to smooth the year-specific administrative censoring caps."),
    )
    parser.add_argument(
        "--admin_censor_min_rows_per_year",
        type=int,
        default=5,
        help=(
            "Minimum number of rows required in a diagnosis-year stratum "
            "to estimate the administrative censoring cap."
        ),
    )
    parser.add_argument(
        "--admin_censor_use_all_rows",
        action="store_true",
        help=(
            "If set, estimate administrative censoring from all rows instead of only censored rows. "
            "Default is to use only censored rows."
        ),
    )
    args = parser.parse_args()

    if not (0.5 < float(args.admin_censor_quantile) < 1.0):
        raise ValueError("--admin_censor_quantile must be between 0.5 and 1.0.")

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    base_root = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(args.config)),
            config["paths"]["base_dir"],
        )
    )
    cohort_cfg = config["cohort_configs"][args.cohort]
    real_dir = cohort_cfg["real_dir"]
    model_dir = cohort_cfg["model_dir"]
    synth_dir = cohort_cfg["synth_dir"]
    seed = args.cohort_seed
    synth_dir_override = config.get("paths", {}).get("synth_dir_override")

    cohort_split = f"{args.cohort}_{seed}"
    x_data_dir = os.path.join(base_root, real_dir, cohort_split, "views", "x")
    x_model_root = os.path.join(base_root, model_dir, cohort_split, "oncosynth", "x")
    x_ckpt_dir = os.path.join(x_model_root, "ckpt")
    x_ckpt_path = _pick_best_ema(x_ckpt_dir)
    if x_ckpt_path is None:
        print(f"No best_ema_model* found in {x_ckpt_dir}")
        return 1

    if synth_dir_override:
        base_out = (
            synth_dir_override
            if os.path.isabs(synth_dir_override)
            else os.path.join(base_root, synth_dir_override)
        )
    else:
        base_out = os.path.join(base_root, synth_dir, cohort_split)
    out_synth_dir = os.path.join(base_out, "oncosynth_fusion")
    os.makedirs(out_synth_dir, exist_ok=True)

    num_samples = None
    info_path = os.path.join(x_data_dir, "info.json")
    with open(info_path, "r") as f:
        info = yaml.safe_load(f)
    if isinstance(info, dict) and "train_num" in info and "test_num" in info:
        num_samples = int(info["train_num"]) + int(info["test_num"])

    cmd = [
        sys.executable,
        os.path.join(
            base_root, "generation", "third_party", "TabDiff", "main.py"
        ),
        "--mode",
        "test",
        "--data_dir",
        x_data_dir,
        "--synth_dir",
        out_synth_dir,
        "--ckpt_path",
        x_ckpt_path,
        "--gpu",
        str(args.gpu),
        "--seed",
        str(seed),
        "--deterministic",
        "--no_wandb",
    ]
    if num_samples is not None:
        cmd.extend(["--num_samples_to_generate", str(num_samples)])

    print("Running:", " ".join(cmd))
    if args.dry_run:
        print(
            f"Completed at {datetime.now().isoformat()} "
            f"(elapsed {time.time() - start_time:.2f}s)"
        )
        return 0

    tabdiff_env = os.environ.copy()
    tabdiff_env.setdefault("PYTHONHASHSEED", str(seed))
    tabdiff_env.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    code = subprocess.call(cmd, env=tabdiff_env, cwd=base_root)
    if code != 0:
        return code

    epoch = _extract_epoch(x_ckpt_path)
    samples_path = os.path.join(out_synth_dir, "results", str(epoch), "samples.csv")
    x_synth_path = os.path.join(out_synth_dir, "x_synthetic.csv")
    if os.path.exists(samples_path):
        shutil.copyfile(samples_path, x_synth_path)
        print(f"Wrote {x_synth_path}")
    else:
        print(f"Expected samples not found at {samples_path}")
        return 1

    continuous_features = cohort_cfg["continuous_covariates"]
    categorical_features = cohort_cfg["categorical_covariates"]
    w_col = cohort_cfg["treatment"]
    c_col = cohort_cfg["outcome_censoring"]
    t_col = cohort_cfg["outcome_time"]
    admin_year_col = cohort_cfg["year_variable"]

    xwct_train = pd.read_csv(
        os.path.join(base_root, real_dir, cohort_split, "views", "xwct", "train.csv")
    )
    xwct_valid = pd.read_csv(
        os.path.join(base_root, real_dir, cohort_split, "views", "xwct", "test.csv")
    )

    if (xwct_train[t_col] <= 0).any():
        raise ValueError(
            "Training data contains non-positive follow-up times; "
            "all durations must be strictly positive."
        )

    print(f"Using '{admin_year_col}' as administrative censoring year column.")

    admin_censor_model = _estimate_admin_censoring_model(
        xwct_train,
        year_col=admin_year_col,
        time_col=t_col,
        c_col=c_col,
        quantile=float(args.admin_censor_quantile),
        use_only_censored=not args.admin_censor_use_all_rows,
        smoothing=args.admin_censor_smoothing,
        min_rows_per_year=int(args.admin_censor_min_rows_per_year),
    )

    # -------------------------------------------------
    # Step 1: X generated by TabDiff
    # -------------------------------------------------
    x_train = xwct_train[continuous_features + categorical_features]
    x_valid = xwct_valid[continuous_features + categorical_features]

    w_train = xwct_train[w_col].to_numpy().astype(int)
    w_valid = xwct_valid[w_col].to_numpy().astype(int)

    x_train_proc, x_preprocessor = _preprocess(
        df=x_train,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    x_valid_proc = x_preprocessor.transform(x_valid)

    model_root = os.path.join(base_root, model_dir, cohort_split)
    best_w_base_model = _extract_best_w_model(model_root=model_root, seed=seed)
    w_model = _fit_w_model(
        x_train_proc, w_train, x_valid_proc, w_valid, best_w_base_model
    )

    event_rsf_params_by_arm = {
        0: _load_best_rsf_params(
            model_root=model_root, process_name="event", arm_value=0
        ),
        1: _load_best_rsf_params(
            model_root=model_root, process_name="event", arm_value=1
        ),
    }
    censor_rsf_params_by_arm = {
        0: _load_best_rsf_params(
            model_root=model_root, process_name="censor", arm_value=0
        ),
        1: _load_best_rsf_params(
            model_root=model_root, process_name="censor", arm_value=1
        ),
    }

    x_synth = pd.read_csv(x_synth_path)
    x_synth = x_synth[continuous_features + categorical_features]
    x_synth_proc = x_preprocessor.transform(x_synth)

    rng = np.random.default_rng(seed)
    w_prob = w_model.predict_proba(x_synth_proc)[:, 1]
    w_synth = rng.binomial(1, w_prob)
    print(
        "W assignment: probabilistic sampling "
        f"(synthetic prevalence={np.mean(w_synth):.4f})"
    )

    print(
        f"W prevalence train: {np.mean(w_train):.4f} | valid: {np.mean(w_valid):.4f} | synth: {np.mean(w_synth):.4f}"
    )

    xw_synth = x_synth.copy()
    xw_synth[w_col] = w_synth

    # -------------------------------------------------
    # Step 2: Build X-only design matrices for per-arm latent time models
    # -------------------------------------------------
    print("Building arm-specific RSF design matrices...")
    x_train_design = _build_design_matrix_x_only(
        xwct_train,
        x_preprocessor=x_preprocessor,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )
    x_synth_design = _build_design_matrix_x_only(
        xw_synth,
        x_preprocessor=x_preprocessor,
        continuous_features=continuous_features,
        categorical_features=categorical_features,
    )

    durations_train = xwct_train[t_col].astype(float).to_numpy()
    event_observed_train = _event_observed_from_c(xwct_train[c_col].to_numpy())
    censor_observed_train = _censor_observed_from_c(xwct_train[c_col].to_numpy())

    sampled_event_times = np.empty(len(xw_synth), dtype=float)
    sampled_censor_times_rsf = np.empty(len(xw_synth), dtype=float)

    event_model_info = {}
    censor_model_info = {}

    for w in [0, 1]:
        idx_train_w = xwct_train[w_col].astype(int).to_numpy() == w
        idx_synth_w = w_synth == w

        n_train_w = int(np.sum(idx_train_w))
        if n_train_w < MIN_ROWS_PER_ARM:
            raise ValueError(
                f"Too few total training rows in treatment arm W={w}: "
                f"{n_train_w} < {MIN_ROWS_PER_ARM}"
            )

        n_event_obs_w = int(np.sum(event_observed_train[idx_train_w]))
        n_censor_obs_w = int(np.sum(censor_observed_train[idx_train_w]))

        if n_event_obs_w < MIN_OBSERVED_PER_PROCESS:
            raise ValueError(
                f"Too few observed events in treatment arm W={w}: "
                f"{n_event_obs_w} < {MIN_OBSERVED_PER_PROCESS}"
            )

        if n_censor_obs_w < MIN_OBSERVED_PER_PROCESS:
            raise ValueError(
                f"Too few observed censorings in treatment arm W={w}: "
                f"{n_censor_obs_w} < {MIN_OBSERVED_PER_PROCESS}"
            )

        X_train_w = x_train_design.loc[idx_train_w].copy()
        X_synth_w = x_synth_design.loc[idx_synth_w].copy()
        d_train_w = durations_train[idx_train_w]

        print(f"Fitting RSF event model for treatment arm W={w}...")
        (
            event_model,
            keep_cols_event,
            event_t_min,
            event_t_max,
            event_meta,
        ) = _fit_rsf_time_model(
            X_df=X_train_w,
            durations=d_train_w,
            event_indicator=event_observed_train[idx_train_w],
            seed=seed + 1000 + w,
            rsf_params=event_rsf_params_by_arm[w],
        )

        event_model_info[w] = {
            "keep_cols": keep_cols_event,
            "t_min": event_t_min,
            "t_max": event_t_max,
            **event_meta,
        }

        print(f"Fitting RSF censoring model for treatment arm W={w}...")
        (
            censor_model,
            keep_cols_censor,
            censor_t_min,
            censor_t_max,
            censor_meta,
        ) = _fit_rsf_time_model(
            X_df=X_train_w,
            durations=d_train_w,
            event_indicator=censor_observed_train[idx_train_w],
            seed=seed + 2000 + w,
            rsf_params=censor_rsf_params_by_arm[w],
        )

        censor_model_info[w] = {
            "keep_cols": keep_cols_censor,
            "t_min": censor_t_min,
            "t_max": censor_t_max,
            **censor_meta,
        }

        print(
            f"Sampling synthetic event and censoring times for treatment arm W={w}..."
        )
        if idx_synth_w.any():
            sampled_event_times[idx_synth_w] = _sample_time_from_rsf_survival_functions(
                model=event_model,
                X_df=X_synth_w,
                keep_cols=keep_cols_event,
                rng=rng,
                fallback_time=event_t_max,
                tail_mode="exponential",
                min_tail_hazard=1e-6,
            )
            sampled_censor_times_rsf[idx_synth_w] = (
                _sample_time_from_rsf_survival_functions(
                    model=censor_model,
                    X_df=X_synth_w,
                    keep_cols=keep_cols_censor,
                    rng=rng,
                    fallback_time=censor_t_max,
                    tail_mode="exponential",
                    min_tail_hazard=1e-6,
                )
            )

    sampled_event_times = np.maximum(sampled_event_times, TIME_EPS)
    sampled_censor_times_rsf = np.maximum(sampled_censor_times_rsf, TIME_EPS)

    # -------------------------------------------------
    # Step 3: Estimate synthetic administrative censoring from diagnosis year
    # Final censoring time is min(RSF censoring, admin cap by year)
    # -------------------------------------------------
    print("Predicting administrative censoring times for synthetic rows...")
    admin_censor_times = _predict_admin_censor_times(
        df=xw_synth,
        admin_model=admin_censor_model,
    )
    admin_censor_times = np.maximum(admin_censor_times, TIME_EPS)

    sampled_censor_times_final = np.minimum(
        sampled_censor_times_rsf,
        admin_censor_times,
    )

    # -------------------------------------------------
    # Final observed time and status from latent T and censoring
    # -------------------------------------------------
    observed_time = np.minimum(sampled_event_times, sampled_censor_times_final)
    event_observed_synth = (sampled_event_times <= sampled_censor_times_final).astype(
        int
    )

    if C_COL_IS_CENSORING:
        c_synth = 1 - event_observed_synth
    else:
        c_synth = event_observed_synth

    if APPLY_CLIP:
        if CLIP_MAX is None:
            raise ValueError("APPLY_CLIP=True but CLIP_MAX is None.")
        clip_max = float(CLIP_MAX)
        admin_mask = observed_time > clip_max
        observed_time = np.minimum(observed_time, clip_max)

        if C_COL_IS_CENSORING:
            c_synth[admin_mask] = 1
        else:
            c_synth[admin_mask] = 0

    if APPLY_INTEGER_DAYS:
        observed_time = np.maximum(observed_time, 1.0)
        observed_time = np.round(observed_time).astype(int)

    xwc_synth = xw_synth.copy()
    xwc_synth[c_col] = c_synth
    xwc_synth_path = os.path.join(out_synth_dir, "xwc_synthetic.csv")
    xwc_synth.to_csv(xwc_synth_path, index=False)
    print(f"Wrote {xwc_synth_path}")

    xwct_synth = xwc_synth.copy()
    xwct_synth[t_col] = observed_time
    if APPLY_INTEGER_DAYS:
        xwct_synth[t_col] = pd.to_numeric(xwct_synth[t_col], errors="coerce").astype(
            "Int64"
        )
    else:
        xwct_synth[t_col] = pd.to_numeric(xwct_synth[t_col], errors="coerce")

    final_out = os.path.join(base_out, "oncosynth.csv")
    xwct_synth.to_csv(final_out, index=False)
    print(f"Wrote {final_out}")

    metadata_path = os.path.join(out_synth_dir, "run_metadata.json")
    metadata = {
        "cohort": args.cohort,
        "seed": seed,
        "gpu": args.gpu,
        "c_col_is_censoring": C_COL_IS_CENSORING,
        "w_synthetic_prevalence": float(np.mean(w_synth)),
        "c_synthetic_prevalence": float(np.mean(c_synth)),
        "x_model_ckpt": x_ckpt_path,
        "x_synthetic": x_synth_path,
        "xwc_synthetic": xwc_synth_path,
        "final_csv": final_out,
        "generated_at": datetime.now().isoformat(),
        "min_rows_per_arm": MIN_ROWS_PER_ARM,
        "min_observed_per_process": MIN_OBSERVED_PER_PROCESS,
        "time_model": "random_survival_forest_latent_event_plus_censor_with_admin_cap",
        "default_rsf_params": RSF_PARAMS,
        "event_rsf_params_by_arm": event_rsf_params_by_arm,
        "censor_rsf_params_by_arm": censor_rsf_params_by_arm,
        "event_model_trained_on_all_rows": True,
        "censor_model_trained_on_all_rows": True,
        "event_model_info": event_model_info,
        "censor_model_info": censor_model_info,
        "admin_censoring": {
            "year_col": admin_censor_model["year_col"],
            "quantile": admin_censor_model["quantile"],
            "use_only_censored": admin_censor_model["use_only_censored"],
            "smoothing": admin_censor_model["smoothing"],
            "min_rows_per_year": admin_censor_model["min_rows_per_year"],
            "fit_info": admin_censor_model["fit_info"],
            "per_year_table": admin_censor_model["per_year_table"].to_dict(
                orient="records"
            ),
        },
        "sampled_event_time_summary": {
            "min": float(np.min(sampled_event_times)),
            "median": float(np.median(sampled_event_times)),
            "max": float(np.max(sampled_event_times)),
        },
        "sampled_censor_time_rsf_summary": {
            "min": float(np.min(sampled_censor_times_rsf)),
            "median": float(np.median(sampled_censor_times_rsf)),
            "max": float(np.max(sampled_censor_times_rsf)),
        },
        "admin_censor_time_summary": {
            "min": float(np.min(admin_censor_times)),
            "median": float(np.median(admin_censor_times)),
            "max": float(np.max(admin_censor_times)),
        },
        "final_censor_time_summary": {
            "min": float(np.min(sampled_censor_times_final)),
            "median": float(np.median(sampled_censor_times_final)),
            "max": float(np.max(sampled_censor_times_final)),
        },
    }
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)
    print(f"Wrote {metadata_path}")

    print(
        f"Completed at {datetime.now().isoformat()} "
        f"(elapsed {time.time() - start_time:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
