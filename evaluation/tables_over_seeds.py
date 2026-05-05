# %%
import csv
import math
from pathlib import Path

TRAIN_SET = "downstream_train"
PRED_SET = "heldout_test"
PROP_CLASSIFIER = "logistic"
COHORT = "lung"
SEEDS = [0, 1, 2, 3, 4]

METRIC_ORDER = [
    "propensity JSD",
    "propensity MAE",
    "propensity ECE",
    "propensity AUROC",
    "delta ATE",
    "ITE_JSD",
    "U_pehe",
    # "ITE_MAE",
    "ITE_ECE",
    "AUQC",
    "feature importance pearson",
    # "feature importance MAE",
]

FIDELITY_METRIC_ORDER = [
    "univariate_fidelity_all",
    "bivariate_MAE_all",
    "treatment prevalence diff",
    "event prevalence diff",
    "average survival time JSD",
    "RMST diff 36",
    "RMST diff 60",
    "RMST diff 84",
    "RMST diff 120",
]


def fail(message: str) -> None:
    raise SystemExit(f"Error: {message}")


def get_config_path() -> Path:
    return Path(__file__).with_name("config.yaml")


def read_config_lines() -> list[str]:
    config_path = get_config_path()
    if not config_path.exists():
        fail(f"config file not found: {config_path}")
    return config_path.read_text().splitlines()


def get_section_lines(lines: list[str], section_name: str) -> list[str]:
    section_prefix = f"{section_name}:"
    start_idx = None
    section_indent = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_prefix:
            start_idx = idx
            section_indent = len(line) - len(line.lstrip(" "))
            break

    if start_idx is None:
        fail(f"missing section '{section_name}' in {get_config_path()}")

    section_lines = []
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            section_lines.append(line)
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent <= section_indent and not line.startswith(
            " " * (section_indent + 1)
        ):
            break
        section_lines.append(line)
    return section_lines


def get_mapping_value(lines: list[str], key: str) -> str:
    key_prefix = f"{key}:"
    for line in lines:
        stripped = line.strip()
        if stripped.startswith(key_prefix):
            return stripped[len(key_prefix) :].strip().strip('"').strip("'")
    fail(f"missing key '{key}' in config")


def get_nested_section(lines: list[str], section_name: str) -> list[str]:
    section_prefix = f"{section_name}:"
    start_idx = None
    section_indent = None

    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == section_prefix:
            start_idx = idx
            section_indent = len(line) - len(line.lstrip(" "))
            break

    if start_idx is None:
        fail(f"missing nested section '{section_name}' in config")

    nested_lines = []
    for line in lines[start_idx + 1 :]:
        stripped = line.strip()
        if not stripped:
            nested_lines.append(line)
            continue
        current_indent = len(line) - len(line.lstrip(" "))
        if current_indent <= section_indent:
            break
        nested_lines.append(line)
    return nested_lines


def load_config_values(cohort: str) -> dict:
    lines = read_config_lines()

    paths_lines = get_section_lines(lines, "paths")
    base_dir = Path(get_mapping_value(paths_lines, "base_dir")).resolve()

    cohort_configs_lines = get_section_lines(lines, "cohort_configs")
    cohort_lines = get_nested_section(cohort_configs_lines, cohort)
    evaluation_dir = get_mapping_value(cohort_lines, "evaluation_dir")

    evaluation_lines = get_section_lines(lines, "evaluation")
    cohort_eval_lines = get_nested_section(evaluation_lines, cohort)
    rmst_horizon = int(get_mapping_value(cohort_eval_lines, "rmst_horizon"))

    model_order_lines = get_section_lines(lines, "model_order")
    model_order = [
        line.strip()[2:].strip()
        for line in model_order_lines
        if line.strip().startswith("- ")
    ]
    if not model_order:
        fail("model_order is empty in config")

    display_lines = get_section_lines(lines, "model_display_names")
    model_display_names = {}
    for line in display_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or ":" not in stripped:
            continue
        key, value = stripped.split(":", 1)
        model_display_names[key.strip()] = value.strip().strip('"').strip("'")

    return {
        "base_dir": base_dir,
        "evaluation_dir": evaluation_dir,
        "rmst_horizon": rmst_horizon,
        "model_order": model_order,
        "model_display_names": model_display_names,
    }


def require_file(path: Path) -> Path:
    if not path.exists():
        fail(f"required file not found: {path}")
    return path


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    require_file(path)
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_csv_matrix(path: Path) -> tuple[list[str], list[list[str]]]:
    require_file(path)
    with path.open(newline="") as handle:
        rows = list(csv.reader(handle))
    if not rows:
        fail(f"empty CSV: {path}")
    header = rows[0]
    if not header:
        fail(f"missing header in CSV: {path}")
    return header, rows[1:]


def get_row(
    rows: list[dict[str, str]], key_name: str, key_value: str, path: Path
) -> dict[str, str]:
    for row in rows:
        if row.get(key_name) == key_value:
            return row
    fail(f"missing row '{key_value}' keyed by '{key_name}' in {path}")


def get_float_from_row(row: dict[str, str], key: str, path: Path) -> float:
    if key not in row:
        fail(f"missing column '{key}' in {path}")
    return float(row[key])


def get_float_from_indexed_csv(path: Path, row_name: str, col_name: str) -> float:
    header, rows = read_csv_matrix(path)
    if not rows:
        fail(f"no data rows in {path}")
    col_idx = header.index(col_name)

    for row in rows:
        if row and row[0] == row_name:
            if col_idx >= len(row):
                fail(f"row '{row_name}' is missing column '{col_name}' in {path}")
            return float(row[col_idx])

    fail(f"missing row '{row_name}' in {path}")


def get_model_names(cfg: dict) -> list[str]:
    return [
        cfg["model_display_names"].get(model, model) for model in cfg["model_order"]
    ]


def get_cohort_group_evaluation_dir(cfg: dict) -> Path:
    path = cfg["base_dir"] / cfg["evaluation_dir"]
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_utility_evaluation_dir(cfg: dict, cohort_folder: str) -> Path:
    path = get_cohort_group_evaluation_dir(cfg) / cohort_folder / "utility"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_fidelity_evaluation_dir(cfg: dict, cohort_folder: str) -> Path:
    path = get_cohort_group_evaluation_dir(cfg) / cohort_folder / "fidelity"
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_utility_paths(utility_dir: Path, horizon: int) -> dict[str, Path]:
    return {
        "propensity_metrics": utility_dir
        / f"propensity_metrics_{PROP_CLASSIFIER}_train[{TRAIN_SET}]_pred[{PRED_SET}].csv",
        "propensities": utility_dir
        / f"propensities_{PROP_CLASSIFIER}_train[{TRAIN_SET}]_pred[{PRED_SET}].csv",
        "effect_metrics": utility_dir
        / f"effect_metrics_train[{TRAIN_SET}]_pred[{PRED_SET}]_horizon[{horizon}].csv",
        "ite": utility_dir
        / f"ITE_csf_train[{TRAIN_SET}]_pred[{PRED_SET}]_horizon[{horizon}].csv",
        "ite_decile_metrics": utility_dir
        / f"ITE_decile_metrics_train[{TRAIN_SET}]_pred[{PRED_SET}]_horizon[{horizon}].csv",
        "qini_metrics": utility_dir
        / f"qini_metrics_train[{TRAIN_SET}]_pred[{PRED_SET}]_horizon[{horizon}].csv",
        # "feature_importance_correlation": utility_dir
        # / f"feature_importance_correlation_train[{TRAIN_SET}]_horizon[{horizon}].csv",
    }


def get_float_from_row_candidates(
    row: dict[str, str], candidate_keys: list[str], path: Path, metric_name: str
) -> float:
    for key in candidate_keys:
        if key in row:
            try:
                return float(row[key])
            except ValueError as exc:
                fail(
                    f"non-numeric value for column '{key}' while extracting '{metric_name}' in {path}: {row[key]!r}"
                )
                raise exc
    fail(
        f"could not extract '{metric_name}' from {path}; tried columns {candidate_keys}"
    )


def get_float_from_indexed_csv_candidates(
    path: Path, row_candidates: list[str], col_name: str, metric_name: str
) -> float:
    header, rows = read_csv_matrix(path)
    if not rows:
        fail(f"no data rows in {path}")
    try:
        col_idx = header.index(col_name)
    except ValueError:
        fail(f"missing column '{col_name}' in {path}")

    for row_name in row_candidates:
        for row in rows:
            if row and row[0] == row_name:
                if col_idx >= len(row):
                    fail(f"row '{row_name}' is missing column '{col_name}' in {path}")
                try:
                    return float(row[col_idx])
                except ValueError as exc:
                    fail(
                        f"non-numeric value at row '{row_name}', column '{col_name}' while extracting '{metric_name}' in {path}: {row[col_idx]!r}"
                    )
                    raise exc

    fail(
        f"could not extract '{metric_name}' from {path}; tried rows {row_candidates} and column '{col_name}'"
    )


def get_seed_metrics(
    utility_dir: Path, model_name: str, horizon: int
) -> dict[str, float]:
    paths = get_utility_paths(utility_dir, horizon)

    propensity_rows = read_csv_rows(paths["propensity_metrics"])
    propensity_row = get_row(
        propensity_rows, "model", model_name, paths["propensity_metrics"]
    )

    qini_rows = read_csv_rows(paths["qini_metrics"])
    qini_row = get_row(qini_rows, "model", model_name, paths["qini_metrics"])

    return {
        "propensity JSD": get_float_from_row(
            propensity_row, "distance", paths["propensity_metrics"]
        ),
        "propensity MAE": get_float_from_row(
            propensity_row, "MAE", paths["propensity_metrics"]
        ),
        "propensity ECE": get_float_from_row(
            propensity_row, "ece_synth", paths["propensity_metrics"]
        ),
        "propensity AUROC": get_float_from_row_candidates(
            propensity_row,
            ["auroc", "AUROC", "auroc_synth", "AUROC_synth"],
            paths["propensity_metrics"],
            "propensity AUROC",
        ),
        "delta ATE": get_float_from_indexed_csv(
            paths["effect_metrics"], "ATE_dist", model_name
        ),
        "U_pehe": get_float_from_indexed_csv(
            paths["effect_metrics"], "U-PEHE", model_name
        ),
        "ITE_JSD": get_float_from_indexed_csv_candidates(
            paths["effect_metrics"],
            ["ITE_JSD", "JSD_ITE", "ITE_distribution_JSD"],
            model_name,
            "ITE_JSD",
        ),
        # "ITE_MAE": get_float_from_indexed_csv(
        #     paths["effect_metrics"], "ITE_MAE", model_name
        # ),
        "ITE_ECE": get_float_from_indexed_csv(
            paths["ite_decile_metrics"], "ITE_decile_ece", model_name
        ),
        "AUQC": get_float_from_row(qini_row, "auqini", paths["qini_metrics"]),
        # "feature importance pearson": get_float_from_indexed_csv(
        #     paths["feature_importance_correlation"],
        #     "FI_correlation_pearson",
        #     model_name,
        # ),
        # "feature importance MAE": get_float_from_indexed_csv(
        #     paths["feature_importance_correlation"], "FI_MAE", model_name
        # ),
    }


def get_fidelity_summary_path(fidelity_dir: Path) -> Path:
    return fidelity_dir / "fidelity_summary_ref[generation_train].csv"


def get_fidelity_seed_metrics(fidelity_dir: Path, model_name: str) -> dict[str, float]:
    path = get_fidelity_summary_path(fidelity_dir)
    return {
        "univariate_fidelity_all": get_float_from_indexed_csv(
            path, "Average univariate fidelity (all)", model_name
        ),
        "bivariate_MAE_all": get_float_from_indexed_csv(
            path, "Bivariate fidelity MAE (all)", model_name
        ),
        "treatment prevalence diff": get_float_from_indexed_csv(
            path, "Treatment prevalence diff from ref [abs prob]", model_name
        ),
        "event prevalence diff": get_float_from_indexed_csv(
            path, "Event prevalence diff from ref [abs prob]", model_name
        ),
        "average survival time JSD": get_float_from_indexed_csv(
            path, "Average survival time JSD", model_name
        ),
        "RMST diff 36": get_float_from_indexed_csv(path, "RMST diff to 36", model_name),
        "RMST diff 60": get_float_from_indexed_csv(path, "RMST diff to 60", model_name),
        "RMST diff 84": get_float_from_indexed_csv(path, "RMST diff to 84", model_name),
        "RMST diff 120": get_float_from_indexed_csv(
            path, "RMST diff to 120", model_name
        ),
    }


def compute_mean(values: list[float]) -> float:
    return sum(values) / len(values)


def compute_std(values: list[float]) -> float:
    if len(values) <= 1:
        return 0.0
    mean_value = compute_mean(values)
    variance = sum((value - mean_value) ** 2 for value in values) / (len(values) - 1)
    return math.sqrt(variance)


def write_summary(
    output_path: Path,
    model_names: list[str],
    values_by_model: dict[str, dict[str, list[float]]],
    metric_order: list[str],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    header = ["metric"]
    for model_name in model_names:
        header.extend([f"{model_name}_mean", f"{model_name}_std"])

    with output_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for metric_name in metric_order:
            row = [metric_name]
            for model_name in model_names:
                values = values_by_model[model_name][metric_name]
                row.extend([compute_mean(values), compute_std(values)])
            writer.writerow(row)


def main() -> None:
    cfg = load_config_values(COHORT)
    horizon = cfg["rmst_horizon"]
    model_names = get_model_names(cfg)

    values_by_model = {
        model_name: {metric_name: [] for metric_name in METRIC_ORDER}
        for model_name in model_names
    }
    fidelity_values_by_model = {
        model_name: {metric_name: [] for metric_name in FIDELITY_METRIC_ORDER}
        for model_name in model_names
    }

    for seed in SEEDS:
        cohort_folder = f"{COHORT}_{seed}"
        utility_dir = get_utility_evaluation_dir(cfg=cfg, cohort_folder=cohort_folder)
        fidelity_dir = get_fidelity_evaluation_dir(cfg=cfg, cohort_folder=cohort_folder)
        for model_name in model_names:
            seed_metrics = get_seed_metrics(
                utility_dir=utility_dir,
                model_name=model_name,
                horizon=horizon,
            )
            for metric_name in METRIC_ORDER:
                values_by_model[model_name][metric_name].append(
                    seed_metrics[metric_name]
                )
            fidelity_seed_metrics = get_fidelity_seed_metrics(
                fidelity_dir=fidelity_dir,
                model_name=model_name,
            )
            for metric_name in FIDELITY_METRIC_ORDER:
                fidelity_values_by_model[model_name][metric_name].append(
                    fidelity_seed_metrics[metric_name]
                )

    output_dir = get_cohort_group_evaluation_dir(cfg)
    output_path = output_dir / f"tables_over_seeds_horizon[{horizon}].csv"
    write_summary(output_path, model_names, values_by_model, METRIC_ORDER)
    fidelity_output_path = output_dir / "fidelity_tables_over_seeds.csv"
    write_summary(
        fidelity_output_path,
        model_names,
        fidelity_values_by_model,
        FIDELITY_METRIC_ORDER,
    )
    print("Saved utility summary to", output_path)
    print("Saved fidelity summary to", fidelity_output_path)


if __name__ == "__main__":
    main()
