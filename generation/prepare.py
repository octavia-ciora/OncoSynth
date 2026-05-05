#!/usr/bin/env python3

import os
import argparse
import json
import subprocess
import sys
from typing import List, Dict, Any, Tuple

import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def write_json(path: str, obj: Dict[str, Any]) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def impute_continuous_with_train_median(
    df_generation_train: pd.DataFrame,
    df_downstream_train: pd.DataFrame,
    df_heldout_test: pd.DataFrame,
    continuous_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    available_cont_cols = [
        c for c in continuous_cols if c in df_generation_train.columns
    ]
    for col in available_cont_cols:
        has_missing = (
            df_generation_train[col].isna().any()
            or (
                col in df_downstream_train.columns
                and df_downstream_train[col].isna().any()
            )
            or (col in df_heldout_test.columns and df_heldout_test[col].isna().any())
        )
        if not has_missing:
            continue

        train_median = df_generation_train[col].median()
        if pd.isna(train_median):
            print(
                f"Skipping median imputation for '{col}' because train median is NaN."
            )
            continue

        df_generation_train[col] = df_generation_train[col].fillna(train_median)
        if col in df_downstream_train.columns:
            df_downstream_train[col] = df_downstream_train[col].fillna(train_median)
        if col in df_heldout_test.columns:
            df_heldout_test[col] = df_heldout_test[col].fillna(train_median)

    return df_generation_train, df_downstream_train, df_heldout_test


def _print_binary_prevalence_summary(
    label: str,
    df: pd.DataFrame,
    treatment_col: str,
    censor_col: str,
) -> None:
    treat_prev = pd.to_numeric(df[treatment_col], errors="coerce").mean()
    censor_prev = pd.to_numeric(df[censor_col], errors="coerce").mean()
    print(
        f"{label}: n={len(df)} | "
        f"W prevalence={treat_prev:.4f} | "
        f"C prevalence={censor_prev:.4f}"
    )


def main():
    parser = argparse.ArgumentParser(
        description="Prepare generation data splits and TabDiff views."
    )
    parser.add_argument(
        "--cohort",
        default="lung",
        help="Cohort name to prepare.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used for split generation.",
    )
    args = parser.parse_args()

    cohort_name = args.cohort
    seed = args.seed

    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "config.yaml"
    )
    with open(config_file, "r") as f:
        config = yaml.safe_load(f)

    BASE_DIR = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(config_file)),
            config["paths"]["base_dir"],
        )
    )
    cohort_config = config["cohort_configs"][cohort_name]

    PREPARATION_DIR = os.path.join(BASE_DIR, cohort_config["real_dir"])

    subprocess.run(
        [
            sys.executable,
            os.path.join(BASE_DIR, "generation", "split_data.py"),
            "--cohort",
            cohort_name,
            "--cohort_seed",
            str(seed),
        ],
        check=True,
        cwd=BASE_DIR,
    )
    splits_dir = os.path.join(PREPARATION_DIR, f"{cohort_name}_{seed}/splits")

    continuous_covariates = cohort_config["continuous_covariates"]
    categorical_covariates = cohort_config["categorical_covariates"]
    treatment_col = cohort_config["treatment"]
    censor_col = cohort_config["outcome_censoring"]
    time_col = cohort_config["outcome_time"]

    generation_train_csv = os.path.join(splits_dir, "real_generation_train.csv")
    downstream_train_csv = os.path.join(splits_dir, "real_downstream_train.csv")
    heldout_test_csv = os.path.join(splits_dir, "real_heldout_test.csv")

    df_generation_train = pd.read_csv(generation_train_csv)
    df_downstream_train = pd.read_csv(downstream_train_csv)
    df_heldout_test = pd.read_csv(heldout_test_csv)

    df_generation_train, df_downstream_train, df_heldout_test = (
        impute_continuous_with_train_median(
            df_generation_train=df_generation_train,
            df_downstream_train=df_downstream_train,
            df_heldout_test=df_heldout_test,
            continuous_cols=list(continuous_covariates),
        )
    )

    df_generation_train.to_csv(generation_train_csv, index=False)
    df_downstream_train.to_csv(downstream_train_csv, index=False)
    df_heldout_test.to_csv(heldout_test_csv, index=False)

    generation_view_train, generation_view_test = train_test_split(
        df_generation_train,
        test_size=0.2,
        random_state=seed,
        stratify=(
            df_generation_train[treatment_col].astype(str)
            + "_"
            + df_generation_train[censor_col].astype(str)
        ),
        shuffle=True,
    )
    generation_view_train = generation_view_train.reset_index(drop=True)
    generation_view_test = generation_view_test.reset_index(drop=True)

    views_dir = os.path.join(PREPARATION_DIR, f"{cohort_name}_{seed}/views")
    os.makedirs(views_dir, exist_ok=True)

    views = [
        {
            "name": "x",
            "columns": [
                c
                for c in df_generation_train.columns
                if c in list(continuous_covariates) + list(categorical_covariates)
            ],
            "continuous_cols": [
                c
                for c in df_generation_train.columns
                if c in list(continuous_covariates)
            ],
            "categorical_cols": [
                c
                for c in df_generation_train.columns
                if c in list(categorical_covariates)
            ],
        },
        {
            "name": "xwct",
            "columns": [
                c
                for c in df_generation_train.columns
                if c
                in list(continuous_covariates)
                + list(categorical_covariates)
                + [treatment_col, censor_col, time_col]
            ],
            "continuous_cols": [
                c
                for c in df_generation_train.columns
                if c in list(continuous_covariates) + [time_col]
            ],
            "categorical_cols": [
                c
                for c in df_generation_train.columns
                if c in list(categorical_covariates) + [treatment_col, censor_col]
            ],
        },
    ]

    for view in views:
        print(f"\n--- VIEW {view['name']} ---")
        out_dir = os.path.join(views_dir, view["name"])
        os.makedirs(out_dir, exist_ok=True)

        df_generation_view_train = generation_view_train[view["columns"]].copy()
        df_generation_view_test = generation_view_test[view["columns"]].copy()
        train_view_path = os.path.join(out_dir, "train.csv")
        test_view_path = os.path.join(out_dir, "test.csv")
        raw_view_path = os.path.join(out_dir, f"raw_{view['name']}.csv")

        df_generation_view_train.to_csv(train_view_path, index=False)
        df_generation_view_test.to_csv(test_view_path, index=False)
        pd.concat([df_generation_view_train, df_generation_view_test], axis=0).to_csv(
            raw_view_path, index=False
        )

        print(f"Wrote train split for '{view['name']}' to {train_view_path}")
        print(f"Wrote test split for '{view['name']}' to {test_view_path}")
        print(f"Wrote raw view '{view['name']}' to {raw_view_path}")
        print(f"Wrote view '{view['name']}': {out_dir}")

        cat_col_idx = [
            df_generation_view_train.columns.get_loc(c)
            for c in view["categorical_cols"]
        ]
        cont_col_idx = [
            df_generation_view_train.columns.get_loc(c) for c in view["continuous_cols"]
        ]

        info = {
            "name": f"{cohort_name}_{view['name']}",
            "task_type": "binclass",
            "header": "infer",
            "column_names": None,
            "num_col_idx": cont_col_idx,
            "cat_col_idx": cat_col_idx,
            "target_col_idx": [],
            "file_type": "csv",
            "data_path": train_view_path,
            "test_path": test_view_path,
            "val_path": None,
        }
        basic_info_path = os.path.join(out_dir, "basic_info.json")
        with open(basic_info_path, "w") as f:
            json.dump(info, f, indent=2)

        print(
            f"Running TabDiff processing for view '{view['name']}' with info file {basic_info_path}"
        )
        subprocess.run(
            [
                sys.executable,
                os.path.join(
                    BASE_DIR, "generation", "third_party", "TabDiff", "process_dataset.py"
                ),
                "--info_file",
                basic_info_path,
                "--save_dir",
                out_dir,
            ],
            check=True,
            cwd=BASE_DIR,
        )

    print("\n--- MAIN SPLIT PREVALENCES ---")
    _print_binary_prevalence_summary(
        "generation_train",
        df_generation_train,
        treatment_col,
        censor_col,
    )
    _print_binary_prevalence_summary(
        "downstream_train",
        df_downstream_train,
        treatment_col,
        censor_col,
    )
    _print_binary_prevalence_summary(
        "heldout_test",
        df_heldout_test,
        treatment_col,
        censor_col,
    )

    print("\n--- VIEW SPLIT PREVALENCES ---")
    _print_binary_prevalence_summary(
        "xwct train",
        generation_view_train,
        treatment_col,
        censor_col,
    )
    _print_binary_prevalence_summary(
        "xwct test",
        generation_view_test,
        treatment_col,
        censor_col,
    )


if __name__ == "__main__":
    main()

# %%
