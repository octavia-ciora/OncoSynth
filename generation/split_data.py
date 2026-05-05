# %%
import argparse
import json
import pandas as pd
import numpy as np
import os
import yaml
from sklearn.model_selection import train_test_split


def compute_split_sizes(N, test_size):
    n_heldout_test = int(round(N * test_size))
    if (N - n_heldout_test) % 2 != 0:
        if n_heldout_test < N:
            n_heldout_test += 1
        else:
            n_heldout_test -= 1
    n_generation_train = (N - n_heldout_test) // 2
    n_downstream_train = n_generation_train
    return n_generation_train, n_downstream_train, n_heldout_test


def validate_dataset(
    df: pd.DataFrame,
    categorical_covariates: list[str],
    continuous_covariates: list[str],
    treatment_col: str,
    censoring_col: str,
    time_col: str,
):
    if df.isna().any().any():
        bad = df.isna().sum()
        bad = bad[bad > 0]
        print(f"Missing values found:\n{bad}")

    for col in [treatment_col, censoring_col]:
        if col not in df.columns:
            raise ValueError(f"Missing column: {col}")
        vals = sorted(df[col].unique())
        if vals != [0, 1]:
            raise ValueError(f"Column '{col}' must be binary [0, 1], found {vals}")

    if time_col not in df.columns:
        raise ValueError(f"Missing column: {time_col}")
    if not pd.api.types.is_numeric_dtype(df[time_col]):
        raise ValueError(f"Column '{time_col}' must be numeric")
    if df[time_col].nunique() <= 1:
        raise ValueError(f"Column '{time_col}' is constant")

    overlap = set(categorical_covariates) & set(continuous_covariates)
    if overlap:
        raise ValueError(
            f"Overlap between categorical and continuous covariates: {overlap}"
        )

    allowed = (
        set(categorical_covariates)
        | set(continuous_covariates)
        | {treatment_col, censoring_col, time_col}
    )
    extra = set(df.columns) - allowed
    if extra:
        raise ValueError(f"Columns not listed in config: {extra}")


parser = argparse.ArgumentParser(
    description="Split dataset into generation_train, downstream_train, and heldout_test sets"
)
parser.add_argument("--cohort", type=str, help="Cohort(s) to process")
parser.add_argument(
    "--cohort_seed",
    type=int,
    required=True,
    help="Cohort seed used for deterministic splitting and output folder naming",
)
args = parser.parse_args()

cohort = args.cohort

base_dir = os.path.dirname(os.path.abspath(__file__))
config_file = os.path.join(base_dir, "config.yaml")
with open(config_file, "r") as f:
    config = yaml.safe_load(f)

RANDOM_SEED = args.cohort_seed
HELDOUT_TEST_SIZE = config["dataset_preparation"]["heldout_test_size"]
BASE_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(config_file)), config["paths"]["base_dir"])
)


cohort_config = config["cohort_configs"][cohort]
categorical_covariates = cohort_config["categorical_covariates"]
continuous_covariates = cohort_config["continuous_covariates"]

input_file = os.path.join(BASE_DIR, cohort_config["cleaned_path"])
df_input = pd.read_csv(input_file)

cont_covs = cohort_config["continuous_covariates"]
cat_covs = cohort_config["categorical_covariates"]
treatment_col = cohort_config["treatment"]
censor_col = cohort_config["outcome_censoring"]
time_col = cohort_config["outcome_time"]

validate_dataset(
    df_input,
    categorical_covariates=cat_covs,
    continuous_covariates=cont_covs,
    treatment_col=treatment_col,
    censoring_col=censor_col,
    time_col=time_col,
)


n_generation_train, n_downstream_train, n_heldout_test = compute_split_sizes(
    N=len(df_input), test_size=HELDOUT_TEST_SIZE
)

df_remaining, df_heldout_test = train_test_split(
    df_input,
    test_size=n_heldout_test,
    random_state=RANDOM_SEED,
    stratify=(
        df_input[treatment_col].astype(str) + "_" + df_input[censor_col].astype(str)
    ),
    shuffle=True,
)
df_generation_train, df_downstream_train = train_test_split(
    df_remaining,
    train_size=n_generation_train,
    test_size=n_downstream_train,
    random_state=RANDOM_SEED,
    stratify=(
        df_remaining[treatment_col].astype(str)
        + "_"
        + df_remaining[censor_col].astype(str)
    ),
    shuffle=True,
)

df_generation_train = df_generation_train.reset_index(drop=True)
df_downstream_train = df_downstream_train.reset_index(drop=True)
df_heldout_test = df_heldout_test.reset_index(drop=True)

print("Split sizes:")
print(f"  generation_train: {len(df_generation_train)}")
print(f"  downstream_train: {len(df_downstream_train)}")
print(f"  heldout_test: {len(df_heldout_test)}")

for col in cat_covs:
    train_cats = set(df_generation_train[col].astype(str).unique())
    for split_name, dsplit in [
        ("downstream_train", df_downstream_train),
        ("heldout_test", df_heldout_test),
    ]:
        split_cats = set(dsplit[col].astype(str).unique())
        unseen = split_cats - train_cats
        if unseen:
            raise ValueError(
                f"Unseen categories in {split_name} for '{col}': {sorted(list(unseen))[:10]}"
            )

for c in cont_covs + [time_col]:
    df_input[c] = df_input[c].astype(np.float32)

for c in cat_covs:
    df_input[c] = df_input[c].astype(str)

df_input[treatment_col] = df_input[treatment_col].astype(np.int64)
df_input[censor_col] = df_input[censor_col].astype(np.int64)

split_dir = os.path.join(
    BASE_DIR,
    cohort_config["real_dir"],
    f"{cohort}_{RANDOM_SEED}/splits",
)
os.makedirs(split_dir, exist_ok=True)

df_generation_train.to_csv(
    os.path.join(split_dir, "real_generation_train.csv"), index=False
)
df_downstream_train.to_csv(
    os.path.join(split_dir, "real_downstream_train.csv"), index=False
)
df_heldout_test.to_csv(os.path.join(split_dir, "real_heldout_test.csv"), index=False)

meta = {
    "cohort": cohort,
    "input_path": str(input_file),
    "config_path": str(config_file),
    "seed": RANDOM_SEED,
    "generation_train_size": n_generation_train / len(df_input),
    "downstream_train_size": n_downstream_train / len(df_input),
    "heldout_test_size": n_heldout_test / len(df_input),
    "stratify": [treatment_col, censor_col],
    "rows": {
        "total": int(len(df_input)),
        "generation_train": int(len(df_generation_train)),
        "downstream_train": int(len(df_downstream_train)),
        "heldout_test": int(len(df_heldout_test)),
    },
}
print(f"Total data size: {len(df_input)} rows, {len(df_input.columns)} columns")
print(f"Split data stored in {split_dir}")
print(
    f"  generation_train: {len(df_generation_train)}, downstream_train: {len(df_downstream_train)}, heldout_test: {len(df_heldout_test)}"
)

with open(os.path.join(split_dir, "meta.json"), "w") as f:
    json.dump(meta, f, indent=2)
print(f"  meta: {meta}")

# %%
