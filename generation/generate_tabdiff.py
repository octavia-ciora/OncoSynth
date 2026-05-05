#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys
import shutil
import time
from datetime import datetime
import pandas as pd
import yaml


def _pick_best_ema(ckpt_dir: str) -> str | None:
    try:
        candidates = [f for f in os.listdir(ckpt_dir) if f.startswith("best_ema_model")]
    except FileNotFoundError:
        return None
    if not candidates:
        return None

    def _epoch(name: str) -> int:
        try:
            return int(name.split("_")[-1].split(".")[0])
        except Exception:
            return -1

    candidates.sort(key=_epoch, reverse=True)
    return os.path.join(ckpt_dir, candidates[0])


def main() -> int:
    start_time = time.time()
    parser = argparse.ArgumentParser(
        description="Generate synthetic data with a trained TabDiff model."
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
        "--num_samples_to_generate",
        type=int,
        default=None,
        help="Optional number of samples to generate",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )
    args = parser.parse_args()

    config = None
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
    outcome_time_col = cohort_cfg["outcome_time"]
    seed = args.cohort_seed
    synth_dir_override = config.get("paths", {}).get("synth_dir_override")

    data_dir = os.path.join(
        base_root, real_dir, f"{args.cohort}_{seed}", "views", "xwct"
    )
    model_root = os.path.join(
        base_root, model_dir, f"{args.cohort}_{seed}", "tabdiff", "xwct"
    )
    ckpt_dir = os.path.join(model_root, "ckpt")
    ckpt_path = _pick_best_ema(ckpt_dir)
    if ckpt_path is None:
        print(f"No best_ema_model* found in {ckpt_dir}")
        return 1

    if synth_dir_override:
        base_out = (
            synth_dir_override
            if os.path.isabs(synth_dir_override)
            else os.path.join(base_root, synth_dir_override)
        )
    else:
        base_out = os.path.join(base_root, synth_dir, f"{args.cohort}_{seed}")
    out_synth_dir = os.path.join(base_out, "tabdiff")

    num_samples = args.num_samples_to_generate
    if num_samples is None:
        info_path = os.path.join(data_dir, "info.json")
        with open(info_path, "r") as f:
            info = yaml.safe_load(f)
        if isinstance(info, dict) and "train_num" in info and "test_num" in info:
            num_samples = int(info["train_num"]) + int(info["test_num"])

    cmd = [
        sys.executable,
        os.path.join(base_root, "generation", "third_party", "TabDiff", "main.py"),
        "--mode",
        "test",
        "--data_dir",
        data_dir,
        "--synth_dir",
        out_synth_dir,
        "--ckpt_path",
        ckpt_path,
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
            f"Completed at {datetime.now().isoformat()} (elapsed {time.time() - start_time:.2f}s)"
        )
        return 0
    code = subprocess.call(cmd, cwd=base_root)
    if code != 0:
        return code

    epoch = os.path.basename(ckpt_path).split("_")[-1].split(".")[0]
    samples_path = os.path.join(out_synth_dir, "results", str(epoch), "samples.csv")
    if not os.path.exists(samples_path):
        print(f"Expected samples not found at {samples_path}")
        return 1
    xwct_out = os.path.join(out_synth_dir, "xwct_synthetic.csv")
    shutil.copyfile(samples_path, xwct_out)
    print(f"Wrote {xwct_out}")

    final_out = os.path.join(base_out, "tabdiff.csv")
    df_final = pd.read_csv(samples_path)
    if outcome_time_col in df_final.columns:
        df_final[outcome_time_col] = (
            pd.to_numeric(df_final[outcome_time_col], errors="coerce")
            .round()
            .astype("Int64")
        )
    df_final.to_csv(final_out, index=False)
    print(f"Wrote {final_out}")
    metadata_path = os.path.join(out_synth_dir, "run_metadata.json")
    metadata = {
        "cohort": args.cohort,
        "seed": seed,
        "gpu": args.gpu,
        "num_samples_to_generate": num_samples,
        "ckpt_path": ckpt_path,
        "data_dir": data_dir,
        "out_synth_dir": out_synth_dir,
        "xwct_synthetic": xwct_out,
        "final_csv": final_out,
        "generated_at": datetime.now().isoformat(),
    }
    with open(metadata_path, "w") as f:
        yaml.safe_dump(metadata, f)
    print(f"Wrote {metadata_path}")
    print(
        f"Completed at {datetime.now().isoformat()} (elapsed {time.time() - start_time:.2f}s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
