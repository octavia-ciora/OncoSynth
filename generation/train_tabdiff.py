#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import yaml
import time
from datetime import datetime


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(description="Train TabDiff for xwct view.")
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
    parser.add_argument("--gpu", type=int, default=1, help="GPU index")
    args = parser.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    config = None
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    base_root = os.path.normpath(
        os.path.join(
            os.path.dirname(os.path.abspath(args.config)), config["paths"]["base_dir"]
        )
    )
    cohort_cfg = config["cohort_configs"][args.cohort]
    real_dir = cohort_cfg["real_dir"]
    model_dir = cohort_cfg["model_dir"]
    seed = args.cohort_seed

    view_root = os.path.join(base_root, real_dir, f"{args.cohort}_{seed}", "views")
    model_root = os.path.join(base_root, model_dir, f"{args.cohort}_{seed}", "tabdiff")

    script = os.path.join(base_dir, "train_models.py")
    cmd = [
        sys.executable,
        script,
        "--view_root",
        view_root,
        "--model_root",
        model_root,
        "--views",
        "xwct",
        "--gpu",
        str(args.gpu),
    ]
    print("Running:", " ".join(cmd))
    code = subprocess.call(cmd)
    if code != 0:
        return code
    print(
        f"Completed at {datetime.now().isoformat()} (elapsed {time.time() - start_time:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
