#!/usr/bin/env python3
import argparse
import os
import subprocess
import sys


VIEWS = ["x", "xw", "xwc", "xwct"]


def _build_view_dir(view_root: str, view: str) -> str:
    return os.path.join(view_root, view)


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
    parser = argparse.ArgumentParser(
        description="Train diffusion model for multiple views."
    )
    parser.add_argument(
        "--view_root",
        required=True,
        help="Root folder that contains view subfolders (x, xw, xwc, xwct)",
    )
    parser.add_argument(
        "--model_root",
        required=True,
        help="Root folder where trained models will be stored",
    )
    parser.add_argument("--gpu", type=int, default=1, help="GPU index")
    parser.add_argument(
        "--views",
        default=",".join(VIEWS),
        help="Comma-separated views to train (default: x,xw,xwc,xwct)",
    )
    parser.add_argument(
        "--y_only",
        action="store_true",
        help="Train y_only models for xw/xwc/xwct views",
    )
    parser.add_argument(
        "--dry_run",
        action="store_true",
        help="Print commands without running them",
    )
    args = parser.parse_args()

    view_root = args.view_root
    model_root = args.model_root
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

    views = [v.strip() for v in args.views.split(",") if v.strip()]

    commands = {}
    for view in views:
        if args.y_only and view == "x":
            continue
        data_dir = _build_view_dir(view_root, view)
        run_view = f"{view}_y_only" if args.y_only and view != "x" else view
        synth_dir = os.path.join(model_root, run_view)
        cmd = [
            sys.executable,
            os.path.join(
                repo_root, "generation", "third_party", "TabDiff", "main.py"
            ),
            "--mode",
            "train",
            "--data_dir",
            data_dir,
            "--synth_dir",
            synth_dir,
            "--gpu",
            str(args.gpu),
            "--seed",
            "0",
            "--deterministic",
            "--no_wandb",
        ]
        if args.y_only and view in {"xw", "xwc", "xwct"}:
            cmd.append("--y_only")
            base_synth_dir = os.path.join(model_root, view)
            ckpt_dir = os.path.join(base_synth_dir, "ckpt")
            ckpt_path = _pick_best_ema(ckpt_dir)
            if ckpt_path:
                cmd.extend(["--ckpt_path", ckpt_path])
        commands[view] = cmd

    if args.dry_run:
        for view, cmd in commands.items():
            print(f"[{view}] {' '.join(cmd)}")
        return 0

    for view in views:
        data_dir = _build_view_dir(view_root, view)
        if not os.path.isdir(data_dir):
            print(f"Missing view directory: {data_dir}")
            return 1

    for view in views:
        if view not in commands:
            continue
        cmd = commands[view]
        print(f"[{view}] {' '.join(cmd)}")
        result = subprocess.run(cmd, cwd=repo_root)
        if result.returncode != 0:
            print(f"[{view}] exited with code {result.returncode}")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
