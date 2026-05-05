#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="${script_dir}"

source "$(conda info --base)/etc/profile.d/conda.sh"
conda create -n eval_env python=3.13.11 -y
conda run -n eval_env python -m pip install --upgrade pip
conda run -n eval_env python -m pip install -r "${repo_root}/requirements_evaluation.txt"


conda run -n eval_env Rscript install_R_packages.R