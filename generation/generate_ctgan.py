#!/usr/bin/env python3
import argparse
import os
import sys
import time
from datetime import datetime

import json
import pandas as pd
import yaml
import numpy as np
import torch
from ctgan import CTGAN


def _load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Train CTGAN and generate synthetic data."
    )
    parser.add_argument(
        "--config",
        default=os.path.join("generation", "config.yaml"),
        help="Path to generation config.yaml",
    )
    parser.add_argument("--cohort", default="breast", help="Cohort name")
    parser.add_argument(
        "--epochs",
        type=int,
        default=300,
        help="Number of CTGAN training epochs",
    )
    parser.add_argument(
        "--cohort_seed",
        type=int,
        required=True,
        help="Cohort seed used to select the prepared cohort split",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print planned actions without running them",
    )
    args = parser.parse_args()
    np.random.seed(args.cohort_seed)
    torch.manual_seed(args.cohort_seed)

    config = _load_config(args.config)
    base_root = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(args.config)),
            config["paths"]["base_dir"],
        )
    )
    cohort_cfg = config["cohort_configs"][args.cohort]
    real_dir = cohort_cfg["real_dir"]
    synth_dir = cohort_cfg["synth_dir"]
    outcome_time_col = cohort_cfg["outcome_time"]
    seed = args.cohort_seed
    synth_dir_override = config.get("paths", {}).get("synth_dir_override")

    cohort_split = f"{args.cohort}_{seed}"
    data_dir = os.path.join(base_root, real_dir, cohort_split, "views", "xwct")
    train_csv = os.path.join(data_dir, "train.csv")
    info_json = os.path.join(data_dir, "info.json")
    if synth_dir_override:
        out_dir = (
            synth_dir_override
            if os.path.isabs(synth_dir_override)
            else os.path.join(base_root, synth_dir_override)
        )
    else:
        out_dir = os.path.join(base_root, synth_dir, cohort_split)

    if args.dry_run:
        print(f"Would train CTGAN on {train_csv}")
        print(f"Would write output to {os.path.join(out_dir, 'ctgan.csv')}")
        print(
            f"Completed at {datetime.now().isoformat()} (elapsed {time.time() - start_time:.2f}s)"
        )
        return 0

    df = pd.read_csv(train_csv)

    with open(info_json, "r") as f:
        info = json.load(f)
        column_names = info.get("column_names") or list(df.columns)
        cat_idx = [int(i) for i in info.get("cat_col_idx", [])]
        discrete_columns = [column_names[i] for i in cat_idx if i < len(column_names)]
        n_samples = (
            info["train_num"] + info.get("test_num", 0)
            if "train_num" in info
            else len(df)
        )

    model = CTGAN(
        epochs=args.epochs,
        verbose=True,
    )
    model.fit(df, discrete_columns=discrete_columns)
    syn = model.sample(n_samples)

    if outcome_time_col in syn.columns:
        syn[outcome_time_col] = (
            pd.to_numeric(syn[outcome_time_col], errors="coerce")
            .round()
            .astype("Int64")
        )

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "ctgan.csv")
    syn.to_csv(out_path, index=False)
    print(f"Wrote {out_path}")
    print(
        f"Completed at {datetime.now().isoformat()} (elapsed {time.time() - start_time:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
