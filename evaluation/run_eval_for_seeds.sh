#!/usr/bin/env bash

set -euo pipefail

usage() {
    cat <<'EOF'
Usage: evaluation/run_eval_for_seeds.sh <cohort> --horizons <h1> [h2 ...] --seeds <s1> [s2 ...]
Example: evaluation/run_eval_for_seeds.sh breast --horizons 12 24 36 --seeds 0 1 2 3 4
EOF
}

if [ "$#" -lt 5 ]; then
    usage
    exit 1
fi

cohort="$1"
shift
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

horizons=()
seeds=()
current_section=""

while [ "$#" -gt 0 ]; do
    case "$1" in
        --horizons)
            current_section="horizons"
            ;;
        --seeds)
            current_section="seeds"
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        --*)
            echo "Unknown option: $1" >&2
            usage
            exit 1
            ;;
        *)
            if [ "${current_section}" = "horizons" ]; then
                horizons+=("$1")
            elif [ "${current_section}" = "seeds" ]; then
                seeds+=("$1")
            else
                echo "Value '$1' must follow --horizons or --seeds." >&2
                usage
                exit 1
            fi
            ;;
    esac
    shift
done

if [ "${#horizons[@]}" -eq 0 ]; then
    echo "At least one horizon must be provided after --horizons." >&2
    usage
    exit 1
fi

if [ "${#seeds[@]}" -eq 0 ]; then
    echo "At least one seed must be provided after --seeds." >&2
    usage
    exit 1
fi

for seed in "${seeds[@]}"; do
    cohort_folder="${cohort}_${seed}"

    echo "Running fidelity evaluation for cohort_folder=${cohort_folder}"
    python "${repo_root}/evaluation/evaluate_fidelity.py" \
        --cohort_folder "${cohort_folder}"

    for horizon in "${horizons[@]}"; do
        echo "Running utility evaluation for cohort_folder=${cohort_folder}, horizon=${horizon}"
        python "${repo_root}/evaluation/evaluate_utility.py" \
            --cohort_folder "${cohort_folder}" \
            --horizon "${horizon}"
    done
done

echo "Finished running fidelity and utility evaluation for cohort=${cohort}"
