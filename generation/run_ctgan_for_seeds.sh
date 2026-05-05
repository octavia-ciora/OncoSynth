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
    echo "Running CTGAN for cohort=${cohort}, cohort_seed=${seed}"
    python "${repo_root}/generation/generate_ctgan.py" \
        --cohort "${cohort}" \
        --epochs 300 \
        --cohort_seed "${seed}"
done
