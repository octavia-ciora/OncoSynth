# %%

from __future__ import annotations
import csv
from pathlib import Path
from typing import Any

###############
cohort = "breast"
# cohort = "lung"
###############

COHORT_TABLE_VARIABLES = {
    "breast": {
        "Demographics": [
            "Age_at_Diagnosis",
            "Race_Origin",
            "Marital_Status",
        ],
        "Tumor characteristics": [
            "Tumor_Size",
            "Grade_Recode",
            "Nodes_Positive",
            "ER_Status",
            "PR_Status",
        ],
        "Outcome": [
            "Event_Observed",
            "Survival_Months",
        ],
    },
    "lung": {
        "Demographics": [
            "Age",
            "Sex",
            "Race_Origin",
            "Marital_Status",
        ],
        "Tumor characteristics": [
            "Primary_Site",
            "Laterality",
            # "AJCC_T",
            # "AJCC_N",
            # "AJCC_M",
        ],
        "Other therapies": [
            "Surgery",
            "Chemotherapy",
        ],
        "Outcome": [
            "Event_Observed",
            "Survival_Months",
        ],
    },
}

CATEGORICAL_VARIABLES_CLASSES = {
    "breast": {
        "ER_Status": [0, 1, 2, 3],
        "PR_Status": [0, 1, 2, 3],
        "Grade_Recode": [1, 2, 3, 4, 0],
        "Race_Origin": [0, 1, 2, 3],
        "Marital_Status": [0, 1, 2, 3, 4],
        "Event_Observed": [0, 1],
    },
    "lung": {
        "Sex": [0, 1],
        "Race_Origin": [0, 1, 2, 3],
        "Marital_Status": [0, 1, 2, 3, 4],
        "Primary_Site": [0, 1, 2, 3, 4, 5],
        "Laterality": [0, 1, 2],
        "AJCC_T": [0, 1, 2, 3, 4],
        "AJCC_N": [0, 1, 2, 3],
        "AJCC_M": [0, 1],
        "Event_Observed": [0, 1],
        "Chemotherapy": [1],
        "Surgery": [1],
    },
}

TABLE_COLUMN_COUNT = 4
CONTINUED_TABLE_LABEL = "Table S2"
TOTAL_COLUMN_LABEL = "Total"


def _strip_inline_comment(line: str) -> str:
    in_single = False
    in_double = False
    result: list[str] = []

    for char in line:
        if char == "'" and not in_double:
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
        elif char == "#" and not in_single and not in_double:
            break
        result.append(char)

    return "".join(result).rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if not value:
        return ""

    if (value.startswith('"') and value.endswith('"')) or (
        value.startswith("'") and value.endswith("'")
    ):
        return value[1:-1]

    lower_value = value.lower()
    if lower_value == "true":
        return True
    if lower_value == "false":
        return False
    if lower_value in {"null", "none"}:
        return None

    try:
        return int(value)
    except ValueError:
        pass

    try:
        return float(value)
    except ValueError:
        return value


def _parse_yaml_block(lines: list[str], start: int, indent: int) -> tuple[Any, int]:
    if start >= len(lines):
        return {}, start

    current_indent = len(lines[start]) - len(lines[start].lstrip(" "))
    if current_indent < indent:
        return {}, start

    stripped = lines[start].lstrip(" ")
    if stripped.startswith("- "):
        items: list[Any] = []
        index = start
        while index < len(lines):
            line = lines[index]
            line_indent = len(line) - len(line.lstrip(" "))
            if line_indent != indent or not line.lstrip(" ").startswith("- "):
                break

            item_content = line.lstrip(" ")[2:].strip()
            index += 1
            if item_content:
                items.append(_parse_scalar(item_content))
                continue

            item, index = _parse_yaml_block(lines, index, indent + 2)
            items.append(item)

        return items, index

    mapping: dict[str, Any] = {}
    index = start
    while index < len(lines):
        line = lines[index]
        line_indent = len(line) - len(line.lstrip(" "))
        if line_indent < indent:
            break
        if line_indent > indent:
            raise ValueError(f"Invalid YAML indentation near: {line}")

        stripped_line = line[indent:]
        if stripped_line.startswith("- "):
            break

        key, _, remainder = stripped_line.partition(":")
        key = key.strip()
        remainder = remainder.strip()
        index += 1

        if remainder:
            mapping[key] = _parse_scalar(remainder)
            continue

        if index >= len(lines):
            mapping[key] = {}
            continue

        next_indent = len(lines[index]) - len(lines[index].lstrip(" "))
        if next_indent <= indent:
            mapping[key] = {}
            continue

        value, index = _parse_yaml_block(lines, index, next_indent)
        mapping[key] = value

    return mapping, index


def load_yaml_subset(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
    except ModuleNotFoundError:
        text_lines = []
        for raw_line in path.read_text().splitlines():
            clean_line = _strip_inline_comment(raw_line)
            if clean_line.strip():
                text_lines.append(clean_line)
        parsed, _ = _parse_yaml_block(text_lines, 0, 0)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected YAML mapping in {path}")
        return parsed

    with path.open() as handle:
        loaded = yaml.safe_load(handle)
    if not isinstance(loaded, dict):
        raise ValueError(f"Expected YAML mapping in {path}")
    return loaded


def resolve_cleaned_breast_path(raw_config_path: Path) -> Path:
    raw_config = load_yaml_subset(raw_config_path)
    output_dir = raw_config["output_dir"]
    output_file = raw_config["output_file"]
    return (raw_config_path.parent / output_dir / output_file).resolve()


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def parse_numeric(value: str) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if text == "":
        return None
    try:
        return float(text)
    except ValueError:
        return None


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("Cannot compute quantile of empty list.")
    if len(sorted_values) == 1:
        return sorted_values[0]

    position = (len(sorted_values) - 1) * q
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(sorted_values) - 1)
    weight = position - lower_index
    lower = sorted_values[lower_index]
    upper = sorted_values[upper_index]
    return lower + (upper - lower) * weight


def format_n_pct(count: int, denominator: int) -> str:
    percentage = (100.0 * count / denominator) if denominator else 0.0
    return f"{count:,} ({percentage:.1f})"


def format_median_iqr(values: list[float]) -> str:
    if not values:
        return "--"
    ordered = sorted(values)
    median_value = quantile(ordered, 0.5)
    q1 = quantile(ordered, 0.25)
    q3 = quantile(ordered, 0.75)
    return f"{median_value:.0f} [{q1:.0f}, {q3:.0f}]"


def format_group_n(count: int) -> str:
    return f"$N$={count:,}"


def latex_row(label: str, *values: str) -> str:
    return " & ".join([label, *values]) + " \\\\"


def section_row(title: str) -> str:
    return (
        f"\\multicolumn{{{TABLE_COLUMN_COUNT}}}{{@{{}}l}}{{\\textbf{{{title}}}}} \\\\"
    )


def build_label_map(raw_map: dict[Any, Any]) -> dict[str, str]:
    label_map: dict[str, str] = {}
    for key, value in raw_map.items():
        label_map[str(key)] = str(value)
    return label_map


def normalize_code(value: str) -> str:
    text = str(value).strip()
    numeric_value = parse_numeric(text)
    if numeric_value is None:
        return text
    if numeric_value.is_integer():
        return str(int(numeric_value))
    return str(numeric_value)


def get_group_rows(
    rows: list[dict[str, str]], treatment_column: str
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    treatment0_rows = [
        row for row in rows if normalize_code(row[treatment_column]) == "0"
    ]
    treatment1_rows = [
        row for row in rows if normalize_code(row[treatment_column]) == "1"
    ]
    return treatment0_rows, treatment1_rows


def continuous_summary(rows: list[dict[str, str]], column: str) -> str:
    values = [
        numeric_value
        for numeric_value in (parse_numeric(row.get(column, "")) for row in rows)
        if numeric_value is not None
    ]
    return format_median_iqr(values)


def categorical_summary(
    rows: list[dict[str, str]], column: str, accepted_values: set[str]
) -> str:
    count = sum(
        1 for row in rows if normalize_code(row.get(column, "")) in accepted_values
    )
    return format_n_pct(count, len(rows))


def numeric_binary_summary(rows: list[dict[str, str]], column: str, predicate) -> str:
    count = 0
    for row in rows:
        value = parse_numeric(row.get(column, ""))
        if value is not None and predicate(value):
            count += 1
    return format_n_pct(count, len(rows))


def build_table_lines(
    rows: list[dict[str, str]], cohort_name: str, cohort_config: dict[str, Any]
) -> list[str]:
    treatment_column = str(cohort_config["treatment"])
    continuous_covariates = set(cohort_config["continuous_covariates"])
    categorical_covariates = set(cohort_config["categorical_covariates"])
    outcome_columns = {
        str(cohort_config["outcome_censoring"]),
        str(cohort_config["outcome_time"]),
    }
    display_names = {
        str(key): str(value)
        for key, value in cohort_config["column_display_names"].items()
    }
    label_maps = {
        str(column): build_label_map(raw_map)
        for column, raw_map in cohort_config["label_maps"].items()
    }
    table_sections = COHORT_TABLE_VARIABLES[cohort_name]
    cohort_categorical_classes = CATEGORICAL_VARIABLES_CLASSES.get(cohort_name, {})
    table_variables = {
        variable_name
        for section_variables in table_sections.values()
        for variable_name in section_variables
    }
    configured_columns = (
        continuous_covariates | categorical_covariates | {treatment_column}
    )
    configured_columns |= outcome_columns
    missing_in_config = table_variables - configured_columns
    if missing_in_config:
        raise KeyError(
            f"Required table columns missing from evaluation/config.yaml for cohort '{cohort_name}': {sorted(missing_in_config)}"
        )

    missing_categorical_classes = sorted(
        variable_name
        for variable_name in table_variables
        if variable_name in categorical_covariates
        or variable_name == str(cohort_config["outcome_censoring"])
        if variable_name not in cohort_categorical_classes
    )
    if missing_categorical_classes:
        raise KeyError(
            f"Missing categorical class order in CATEGORICAL_VARIABLES_CLASSES for cohort '{cohort_name}': {missing_categorical_classes}"
        )

    treatment0_rows, treatment1_rows = get_group_rows(rows, treatment_column)

    treatment_labels = label_maps[treatment_column]
    treatment0_label = treatment_labels.get("0", "Treatment 0")
    treatment1_label = treatment_labels.get("1", "Treatment 1")
    treatment0_n = format_group_n(len(treatment0_rows))
    treatment1_n = format_group_n(len(treatment1_rows))
    total_n = format_group_n(len(rows))

    lines = [
        "\\toprule",
        latex_row(
            "\\multirow{2}{*}{\\textbf{Characteristics}}",
            f"\\multicolumn{{1}}{{c}}{{\\textbf{{{treatment0_label}}}}}",
            f"\\multicolumn{{1}}{{c}}{{\\textbf{{{treatment1_label}}}}}",
            f"\\multicolumn{{1}}{{c}}{{\\textbf{{{TOTAL_COLUMN_LABEL}}}}}",
        ),
        f"& \\multicolumn{{1}}{{c}}{{{treatment0_n}}} & \\multicolumn{{1}}{{c}}{{{treatment1_n}}} & \\multicolumn{{1}}{{c}}{{{total_n}}} \\\\",
        "\\midrule",
        "\\endfirsthead",
        "",
        "",
        f"\\multicolumn{{{TABLE_COLUMN_COUNT}}}{{l}}{{\\textit{{{CONTINUED_TABLE_LABEL} continued from previous page}}}}\\\\",
        "\\toprule",
        latex_row(
            "\\textbf{Characteristic}",
            f"\\textbf{{{treatment0_label}}}",
            f"\\textbf{{{treatment1_label}}}",
            f"\\textbf{{{TOTAL_COLUMN_LABEL}}}",
        ),
        f"& {treatment0_n} & {treatment1_n} & {total_n} \\\\",
        "\\midrule",
        "\\endhead",
        "",
        "\\midrule",
        f"\\multicolumn{{{TABLE_COLUMN_COUNT}}}{{r}}{{\\textit{{Continued on next page}}}}\\\\",
        "\\endfoot",
        "",
        "\\bottomrule",
        "\\endlastfoot",
        "",
    ]

    for section_index, (section_name, variables) in enumerate(table_sections.items()):
        if section_index > 0:
            lines.append("")
        lines.append(section_row(section_name))

        for variable_name in variables:
            display_label = display_names.get(
                variable_name, variable_name.replace("_", " ")
            )
            if variable_name == "Survival_Months":
                display_label = display_label.replace(" (months)", "")

            if variable_name in continuous_covariates or variable_name == str(
                cohort_config["outcome_time"]
            ):
                lines.append(
                    latex_row(
                        display_label,
                        continuous_summary(treatment0_rows, variable_name),
                        continuous_summary(treatment1_rows, variable_name),
                        continuous_summary(rows, variable_name),
                    )
                )
                continue

            lines.append(latex_row(display_label, "", "", ""))
            value_map = label_maps.get(variable_name, {})
            sorted_codes = [
                normalize_code(str(code))
                for code in cohort_categorical_classes[variable_name]
            ]

            if len(sorted_codes) == 1:
                code = sorted_codes[0]
                lines.pop()
                lines.append(
                    latex_row(
                        display_label,
                        categorical_summary(treatment0_rows, variable_name, {code}),
                        categorical_summary(treatment1_rows, variable_name, {code}),
                        categorical_summary(rows, variable_name, {code}),
                    )
                )
                continue

            for code in sorted_codes:
                sub_label = value_map.get(code, code)
                lines.append(
                    latex_row(
                        f"\\quad {sub_label}",
                        categorical_summary(treatment0_rows, variable_name, {code}),
                        categorical_summary(treatment1_rows, variable_name, {code}),
                        categorical_summary(rows, variable_name, {code}),
                    )
                )

        lines.append("\\\\")

    return lines


eval_config = "evaluation/config.yaml"
if cohort == "breast":
    raw_config = "data_cleaning/config_raw_breast.yaml"
elif cohort == "lung":
    raw_config = "data_cleaning/config_raw_lung.yaml"
else:
    raise ValueError(f"Unsupported cohort: {cohort}")

repo_root = Path(__file__).resolve().parent.parent
eval_config_path = (repo_root / eval_config).resolve()
raw_config_path = (repo_root / raw_config).resolve()

eval_config = load_yaml_subset(eval_config_path)
cohort_config = eval_config["cohort_configs"][cohort]
cleaned_path = resolve_cleaned_breast_path(raw_config_path)

rows = read_csv_rows(cleaned_path)
latex_lines = build_table_lines(rows, cohort, cohort_config)
for line in latex_lines:
    print(line)
