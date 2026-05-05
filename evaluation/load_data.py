import pandas as pd
import os
import numpy as np
import yaml
from typing import Dict, Tuple


def load_config(config_path: str = None) -> Dict:
    """Load configuration from YAML file."""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), "config.yaml")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def load_datasets(
    synth_data_path: str = None, real_data_path: str = None, cfg: Dict = None, seed=None
) -> Dict[str, pd.DataFrame]:
    """Load all datasets from the specified path."""
    datasets = {}
    file_config = cfg["dataset_files"]
    display_names = cfg.get("model_display_names", {})

    for name, filename in file_config.items():
        if name in ["generation_train", "downstream_train", "heldout_test"]:
            filepath = os.path.join(real_data_path, filename)
        else:
            if synth_data_path is None:
                continue
            else:
                filepath = os.path.join(synth_data_path, filename)
        datasets[name] = pd.read_csv(filepath)
        display_name = display_names.get(name, name.upper())
        print(f"{display_name}: {datasets[name].shape}")
    return datasets


def round_time_to_int(T):
    T_int = np.round(T).astype(int)

    return T_int


def _get_outcome_event_col(config: Dict) -> str:
    if "outcome_event" in config:
        return config["outcome_event"]
    if "outcome_censoring" in config:
        return config["outcome_censoring"]
    raise KeyError("Missing outcome event/censoring key in cohort config")


def extract_features(df: pd.DataFrame, config: Dict) -> Dict[str, pd.DataFrame]:
    all_covs = config["continuous_covariates"] + config["categorical_covariates"]

    features = {
        "X_all": df[all_covs].copy(),
        "X_continuous": df[config["continuous_covariates"]].copy(),
        "X_categorical": df[config["categorical_covariates"]].copy(),
        "W": df[config["treatment"]].copy(),
        "C": df[_get_outcome_event_col(config)].copy(),
        "T": df[config["outcome_time"]].copy(),
    }
    return features


def get_label_map(cfg: Dict, cohort: str, variable: str) -> Dict:
    """Get label mapping for a categorical variable."""
    return cfg["cohort_configs"][cohort].get("label_maps", {}).get(variable, {})


def get_display_name(cfg: Dict, cohort: str, variable: str) -> str:
    """Get human-readable display name for a variable."""
    return (
        cfg["cohort_configs"][cohort]
        .get("column_display_names", {})
        .get(variable, variable)
    )


def load_cohort_data(
    setting_id: str,
    config_path: str = None,
    verbose: bool = True,
    load_synthetic_data=True,
) -> Tuple[Dict, Dict, Dict]:
    cfg = load_config(config_path)

    cohort = setting_id.split("_")[0]

    if setting_id.count("_") == 1:
        seed = setting_id.split("_")[1]
    else:
        raise ValueError(
            f"setting_id must be in the format '<cohort>_<seed>', not {setting_id}"
        )
    real_root = get_real_dir(cfg, cohort)
    if os.path.exists(os.path.join(real_root, setting_id)):
        real_data_path = os.path.join(real_root, setting_id)
    else:
        raise FileNotFoundError(
            f"Real data path not found: {os.path.join(real_root, setting_id)}"
        )

    if load_synthetic_data:
        synth_root = get_synth_dir(cfg, cohort)

        if os.path.exists(os.path.join(synth_root, setting_id)):
            synth_data_path = os.path.join(synth_root, setting_id)
        else:
            raise FileNotFoundError(
                f"Synthetic data path not found: {os.path.join(synth_root, setting_id)}"
            )

    if verbose:
        print(f"Loading {cohort.upper()} cancer cohort")
        if load_synthetic_data:
            print(f"Synth data path: {synth_data_path}\n")
        print(f"Real data path: {real_data_path}\n")

    # Load datasets
    if load_synthetic_data:
        datasets = load_datasets(
            synth_data_path=synth_data_path,
            real_data_path=real_data_path,
            cfg=cfg,
            seed=seed,
        )
    else:
        datasets = load_datasets(
            synth_data_path=None,
            real_data_path=real_data_path,
            cfg=cfg,
            seed=seed,
        )

    cohort_config = cfg["cohort_configs"][cohort]

    all_covariates = (
        cohort_config["continuous_covariates"] + cohort_config["categorical_covariates"]
    )

    if verbose:
        print(f"\n--- Column Configuration for {cohort.upper()} ---")
        print(f"Total Covariates (X): {len(all_covariates)} features")
        print(
            f"  - Continuous ({len(cohort_config['continuous_covariates'])}): "
            f"{cohort_config['continuous_covariates']}"
        )
        print(
            f"  - Categorical ({len(cohort_config['categorical_covariates'])}): "
            f"{cohort_config['categorical_covariates']}"
        )
        print(f"Treatment (W): {cohort_config['treatment']}")
        print(f"Censoring (C): {_get_outcome_event_col(cohort_config)}")
        print(f"Time-to-Event (T): {cohort_config['outcome_time']}")

    features = {
        name: extract_features(df, cohort_config) for name, df in datasets.items()
    }

    return cfg, datasets, features


def print_data_summary(cohort: str, features: Dict):
    """Print a summary of the loaded data."""
    print("=" * 70)
    print(f"FEATURE SUMMARY: {cohort.upper()} Cancer Cohort")
    print("=" * 70)

    for name, feats in features.items():
        print(f"\n{name.upper()}:")
        print(f"  X_all shape:        {feats['X_all'].shape}")
        print(f"  X_continuous:       {feats['X_continuous'].shape}")
        print(f"  X_categorical:      {feats['X_categorical'].shape}")
        print(f"  W (Treatment):      {feats['W'].shape}")
        print(f"    Distribution:     {feats['W'].value_counts().to_dict()}")
        print(f"  C (Censoring):      {feats['C'].shape}")
        print(f"    Events:           {feats['C'].sum()}")
        print(f"    Censored:         {len(feats['C']) - feats['C'].sum()}")
        print(f"  T (Time):           {feats['T'].shape}")
        print(f"    Mean:             {feats['T'].mean():.2f}")
        print(f"    Range:            [{feats['T'].min():.1f}, {feats['T'].max():.1f}]")


def get_base_dir(cfg: Dict) -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), cfg["paths"]["base_dir"])
    )


def _get_cohort_config(cfg: Dict, cohort: str) -> Dict:
    if "cohort_configs" not in cfg or cohort not in cfg["cohort_configs"]:
        raise KeyError(f"Missing cohort config for '{cohort}'")
    return cfg["cohort_configs"][cohort]


def _get_general_evaluation_dir(cohort_folder: str, cfg: Dict) -> str:
    cohort = cohort_folder.split("_")[0]
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort)
    if "evaluation_dir" in cohort_config:
        base_eval_dir = os.path.join(BASE_DIR, cohort_config["evaluation_dir"])
    else:
        raise KeyError("Missing evaluation_dir in cohort config")
    output_dir = os.path.join(base_eval_dir, cohort_folder)
    os.makedirs(output_dir, exist_ok=True)
    return output_dir


def get_fidelity_evaluation_dir(cohort_folder: str, cfg: Dict) -> str:
    eval_dir = _get_general_evaluation_dir(cohort_folder, cfg)
    fidelity_dir = os.path.join(eval_dir, "fidelity")
    os.makedirs(fidelity_dir, exist_ok=True)
    return fidelity_dir


def get_utility_evaluation_dir(cohort_folder: str, cfg: Dict) -> str:
    eval_dir = _get_general_evaluation_dir(cohort_folder, cfg)
    utility_dir = os.path.join(eval_dir, "utility")
    os.makedirs(utility_dir, exist_ok=True)
    return utility_dir


def get_hyperparam_utility_dir(cohort_folder: str, cfg: Dict) -> str:
    cohort = cohort_folder.split("_")[0]
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort)
    if "hyperparam_evaluation_dir" in cohort_config:
        base_eval_dir = os.path.join(
            BASE_DIR, cohort_config["hyperparam_evaluation_dir"]
        )
    else:
        raise KeyError("Missing hyperparam_evaluation_dir in cohort config")
    utility_dir = os.path.join(base_eval_dir, cohort_folder, "utility")
    os.makedirs(utility_dir, exist_ok=True)
    return utility_dir


def get_synth_dir(cfg: Dict, cohort: str) -> str:
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort)
    if "synth_dir" in cohort_config:
        return os.path.join(BASE_DIR, cohort_config["synth_dir"])
    raise KeyError("Missing synth_dir in config")


def get_real_dir(cfg: Dict, cohort: str = None) -> str:
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort)
    if "real_dir" in cohort_config:
        return os.path.join(BASE_DIR, cohort_config["real_dir"])
    raise KeyError("Missing real_dir in config")


def get_hyperparam_search_dir(cfg: Dict, cohort: str) -> str:
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort)
    if "hyperparam_search_dir" in cohort_config:
        return os.path.join(BASE_DIR, cohort_config["hyperparam_search_dir"])
    raise KeyError("Missing hyperparam_search_dir in config")


def get_cohort_group_evaluation_dir(cohort_group: str, cfg: Dict) -> str:
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort_group)
    if "evaluation_dir" in cohort_config:
        path = os.path.join(BASE_DIR, cohort_config["evaluation_dir"])
    else:
        raise KeyError("Missing evaluation_dir in cohort config")
    os.makedirs(path, exist_ok=True)
    return path


def get_hyperparam_evaluation_dir(cohort_group: str, cfg: Dict) -> str:
    BASE_DIR = get_base_dir(cfg)
    cohort_config = _get_cohort_config(cfg, cohort_group)
    if "hyperparam_evaluation_dir" in cohort_config:
        path = os.path.join(BASE_DIR, cohort_config["hyperparam_evaluation_dir"])
    else:
        raise KeyError("Missing hyperparam_evaluation_dir in cohort config")
    os.makedirs(path, exist_ok=True)
    return path
