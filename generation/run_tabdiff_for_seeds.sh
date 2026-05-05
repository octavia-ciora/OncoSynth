#!/usr/bin/env bash

set -euo pipefail

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <cohort> <seed1> [seed2 ...]"
    echo "Example: $0 breast 0 1 2 3 4"
    exit 1
fi

cohort="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

for seed in "$@"; do
    echo "Training TabDiff for cohort=${cohort}, cohort_seed=${seed}"
    python "${repo_root}/generation/train_tabdiff.py" \
        --cohort "${cohort}" \
        --gpu 1 \
        --cohort_seed "${seed}"
    echo "Trained TabDiff for cohort=${cohort}, cohort_seed=${seed}"

    echo "Generating TabDiff data for cohort=${cohort}, cohort_seed=${seed}"
    python "${repo_root}/generation/generate_tabdiff.py" \
        --cohort "${cohort}" \
        --gpu 1 \
        --cohort_seed "${seed}"
    echo "Generated TabDiff data for cohort=${cohort}, cohort_seed=${seed}"
done
