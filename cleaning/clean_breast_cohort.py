# %%
import pandas as pd
import numpy as np
import os
import yaml

COLUMNS_TO_DROP = []

COLUMN_MAPPING = {
    "RX Summ--Surg Prim Site (1998+)": "Surgery",
    "RX Summ--Systemic/Sur Seq (2007+)": "Treatment_Sequence",
    "Vital status recode (study cutoff used)": "Vital_Status",
    "Survival months": "Survival_Months",
    "Year of diagnosis": "Year_of_Diagnosis",
    "CS tumor size (2004-2015)": "Tumor_Size",
    "Regional nodes positive (1988+)": "Nodes_Positive",
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": "Race_Origin",
    "Grade Recode (thru 2017)": "Grade_Recode",
    "CS site-specific factor 1 (2004-2017 varying by schema)": "ER_Status",
    "CS site-specific factor 2 (2004-2017 varying by schema)": "PR_Status",
    "Marital status at diagnosis": "Marital_Status",
    "Age recode with single ages and 85+": "Age_at_Diagnosis",
    "Chemotherapy recode (yes, no/unk)": "Chemotherapy",
    "Sex": "Sex",
    "Breast - Adjusted AJCC 6th Stage (1988-2015)": "AJCC_Stage",
    "Median household income inflation adj to 2023": "Household_Income",
}

with open("../cleaning/config_raw_breast.yaml", "r") as f:
    config = yaml.safe_load(f)
df_raw = pd.read_csv(config["raw_path"])

print("\n--- Processing Data ---")
df_data = df_raw.copy()

# drop columns
for col in COLUMNS_TO_DROP:
    if col in df_data.columns:
        print(f"Dropping column: {col}")
        df_data = df_data.drop(columns=[col])

expected_cols = set(COLUMN_MAPPING.keys())
actual_cols = set(df_data.columns)
missing_cols = expected_cols - actual_cols
extra_cols = actual_cols - expected_cols

if missing_cols:
    raise ValueError(
        f"ERROR: The following expected columns are missing from the input data: {missing_cols}"
    )
if extra_cols:
    raise ValueError(
        f"ERROR: The input data contains unexpected extra columns: {extra_cols}"
    )

df_data = df_data.rename(columns=COLUMN_MAPPING)

# %% treatment
treatment_counts = (
    df_data["Treatment_Sequence"]
    .value_counts(dropna=False)
    .sort_values(ascending=False)
)
total = treatment_counts.sum()
cum_sum = treatment_counts.cumsum()
print("\nDistinct treatment values sorted by count (with cumulative percentage):")
for val, count in treatment_counts.items():
    cum_pct = (cum_sum[val] / total) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")
treatment_mapping = {
    "Systemic therapy after surgery": 0,
    "Systemic therapy before surgery": 1,
}

df_data["Treatment_Group"] = df_data["Treatment_Sequence"].map(treatment_mapping)
df_data.loc[
    ~df_data["Treatment_Sequence"].isin(treatment_mapping.keys()), "Treatment_Group"
] = np.nan
print("\nValue counts for mapped 'Treatment_Group':")
print(df_data["Treatment_Group"].value_counts(dropna=False))
num_nan = df_data["Treatment_Group"].isna().sum()
if num_nan > 0:
    print(
        f"There are {num_nan} rows with NaN in 'Treatment_Group' (unmapped treatment values)."
    )

df_data = df_data.drop(columns=["Treatment_Sequence"])

# %% outcome (Vital_Status)
if "Vital_Status" not in df_data.columns:
    raise ValueError("ERROR: 'Vital_Status' column is missing from the data.")

vital_counts = (
    df_data["Vital_Status"].value_counts(dropna=False).sort_values(ascending=False)
)
total_vital = vital_counts.sum()
cum_sum_vital = vital_counts.cumsum()
print("\nDistinct Vital_Status values sorted by count (with cumulative percentage):")
for val, count in vital_counts.items():
    cum_pct = (cum_sum_vital[val] / total_vital) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")

df_data["Event_Observed"] = np.where(df_data["Vital_Status"] == "Dead", 1, 0)
print("\nValue counts for 'Event_Observed' (outcome):")
print(df_data["Event_Observed"].value_counts(dropna=False))
df_data = df_data.drop(columns=["Vital_Status"])

# %% Survival Months
if "Survival_Months" not in df_data.columns:
    raise ValueError("ERROR: 'Survival_Months' column is missing from the data.")
df_data["Survival_Months"] = pd.to_numeric(df_data["Survival_Months"], errors="coerce")
print("\nStatistics for 'Survival_Months':")
print(df_data["Survival_Months"].describe())
valid_survival = df_data["Survival_Months"].dropna()
if not valid_survival.empty:
    print("\nDistribution of 'Survival_Months' (Histogram):")
    counts, bin_edges = np.histogram(valid_survival, bins=20)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        print(f"{bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar} {count}")
else:
    print("No valid numeric data found for 'Survival_Months'.")

# %% PR Status
# PR Status according to https://staging.seer.cancer.gov/cs/input/02.05.50/breast/ssf2/?version=/tnm/home/1.1/
if "PR_Status" not in df_data.columns:
    raise ValueError("ERROR: 'PR_Status' column is missing from the data.")

print("\nInitial numeric value counts for 'PR_Status':")
pr_counts = df_data["PR_Status"].value_counts(dropna=False).sort_values(ascending=False)
total_pr = pr_counts.sum()
cum_sum_pr = pr_counts.cumsum()
print("\nDistinct PR_Status values sorted by count (with cumulative percentage):")
for val, count in pr_counts.items():
    cum_pct = (cum_sum_pr[val] / total_pr) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_receptor_status(val):
    if val == "010":  # positive
        return 1
    elif val == "020":  # negative
        return 0
    elif val == "030":  # borderline
        return 2
    elif val == "999":  # unknown
        return 3
    elif val == "998":  # test not done
        return 3
    elif val == "997":  # test ordered, result not in chart
        return 2
    elif val == "996":  # test ordered, result not interpretable
        return 2
    elif val in ["Blank(s)"]:
        return 3
    else:
        raise ValueError(f"Unexpected value in PR_Status: {val}")


df_data["PR_Status"] = df_data["PR_Status"].apply(map_receptor_status)
print("Final mapped value counts for 'PR_Status':")
print(df_data["PR_Status"].value_counts(dropna=False).sort_values(ascending=False))


# %% ER Status
# ER Status https://staging.seer.cancer.gov/cs/input/02.05.50/breast/ssf1/?version=/tnm/home/1.1/
if "ER_Status" not in df_data.columns:
    raise ValueError("ERROR: 'ER_Status' column is missing from the data.")
print("\nInitial numeric value counts for 'ER_Status':")
er_counts = df_data["ER_Status"].value_counts(dropna=False).sort_values(ascending=False)
total_er = er_counts.sum()
cum_sum_er = er_counts.cumsum()
print("\nDistinct ER_Status values sorted by count (with cumulative percentage):")
for val, count in er_counts.items():
    cum_pct = (cum_sum_er[val] / total_er) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")
df_data["ER_Status"] = df_data["ER_Status"].apply(map_receptor_status)
print("Final mapped value counts for 'ER_Status':")
print(df_data["ER_Status"].value_counts(dropna=False).sort_values(ascending=False))


# %%
# Race_Origin
if "Race_Origin" not in df_data.columns:
    raise ValueError("ERROR: 'Race_Origin' column is missing from the data.")
print("\nValue counts for 'Race_Origin':")
race_counts = (
    df_data["Race_Origin"].value_counts(dropna=False).sort_values(ascending=False)
)
total_race = race_counts.sum()
cum_sum_race = race_counts.cumsum()
print("\nDistinct Race_Origin values sorted by count (with cumulative percentage):")
for val, count in race_counts.items():
    cum_pct = (cum_sum_race[val] / total_race) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_race(val):
    s_val = str(val).strip()
    if s_val == "Non-Hispanic White":
        return 0
    elif s_val == "Non-Hispanic Black":
        return 1
    elif s_val == "Hispanic (All Races)":
        return 2
    elif s_val in [
        "Non-Hispanic Asian or Pacific Islander",
        "Non-Hispanic American Indian/Alaska Native",
        "Non-Hispanic Unknown Race",
    ]:
        return 3
    else:
        raise ValueError(f"Unexpected value in Race_Origin: {val}")


df_data["Race_Origin"] = df_data["Race_Origin"].apply(map_race)
print("\nValue counts for mapped 'Race_Origin':")
print(df_data["Race_Origin"].value_counts(dropna=False).sort_values())

# %%
# Tumor Size (Numeric)
if "Tumor_Size" not in df_data.columns:
    raise ValueError("ERROR: 'Tumor_Size' column is missing from the data.")
df_data["Tumor_Size"] = pd.to_numeric(df_data["Tumor_Size"], errors="coerce")
print("\nStatistics for 'Tumor_Size':")
print(df_data["Tumor_Size"].describe())
valid_tumor = df_data["Tumor_Size"].dropna()
if not valid_tumor.empty:
    print("\nDistribution of 'Tumor_Size' (Histogram):")
    counts, bin_edges = np.histogram(valid_tumor, bins=20)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        print(f"{bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar} {count}")
else:
    print("No valid numeric data found for 'Tumor_Size'.")
mask_valid_range = (df_data["Tumor_Size"] >= 0) & (df_data["Tumor_Size"] <= 400)
before_nans = df_data["Tumor_Size"].isna().sum()
df_data.loc[~mask_valid_range, "Tumor_Size"] = np.nan
after_nans = df_data["Tumor_Size"].isna().sum()
print(
    f"\nFiltered 'Tumor_Size' to range [0, 400]. Set {after_nans - before_nans} values to NaN. Total NaN: {after_nans}"
)
print("New statistics for 'Tumor_Size':")
print(df_data["Tumor_Size"].describe())
valid_tumor_clean = df_data["Tumor_Size"].dropna()
if not valid_tumor_clean.empty:
    print("\nDistribution of cleaned 'Tumor_Size' (Histogram):")
    counts, bin_edges = np.histogram(valid_tumor_clean, bins=20)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        print(f"{bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar} {count}")


# %%
# Grade Recode
if "Grade_Recode" not in df_data.columns:
    raise ValueError("ERROR: 'Grade_Recode' column is missing from the data.")
print("\nValue counts for 'Grade_Recode':")
grade_counts = (
    df_data["Grade_Recode"].value_counts(dropna=False).sort_values(ascending=False)
)
total_grade = grade_counts.sum()
cum_sum_grade = grade_counts.cumsum()
print("\nDistinct Grade_Recode values sorted by count (with cumulative percentage):")
for val, count in grade_counts.items():
    cum_pct = (cum_sum_grade[val] / total_grade) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


# Map Grade_Recode: 1, 2, 3, 4. Unknown -> 0
def map_grade(val):
    s_val = str(val).strip()

    if s_val == "Well differentiated; Grade I":
        return 1
    elif s_val == "Moderately differentiated; Grade II":
        return 2
    elif s_val == "Poorly differentiated; Grade III":
        return 3
    elif s_val == "Undifferentiated; anaplastic; Grade IV":
        return 4
    elif s_val in ["Unknown", "Blank(s)"]:
        return 0
    else:
        raise ValueError(f"Unexpected value in Grade_Recode: {val}")


df_data["Grade_Recode"] = df_data["Grade_Recode"].apply(map_grade)
print("\nValue counts for mapped 'Grade_Recode':")
print(df_data["Grade_Recode"].value_counts(dropna=False).sort_values())

# %% Nodes_Positive
if "Nodes_Positive" not in df_data.columns:
    raise ValueError("ERROR: 'Nodes_Positive' column is missing from the data.")
print("\nStatistics for 'Nodes_Positive':")
print(df_data["Nodes_Positive"].describe(percentiles=[0.25, 0.5, 0.75, 0.98, 0.99]))
valid_nodes = df_data["Nodes_Positive"].dropna()
if not valid_nodes.empty:
    print("\nDistribution of 'Nodes_Positive' (Histogram):")
    counts, bin_edges = np.histogram(valid_nodes, bins=20)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        print(f"{bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar} {count}")
else:
    print("No valid numeric data found for 'Nodes_Positive'.")

# Filter Nodes_Positive: <= 90
mask_nodes_valid = df_data["Nodes_Positive"] <= 90
before_nans_nodes = df_data["Nodes_Positive"].isna().sum()
df_data.loc[~mask_nodes_valid, "Nodes_Positive"] = np.nan
after_nans_nodes = df_data["Nodes_Positive"].isna().sum()
print(
    f"\nFiltered 'Nodes_Positive' to <= 90. Set {after_nans_nodes - before_nans_nodes} values to NaN. Total NaN: {after_nans_nodes}"
)
print("New statistics for 'Nodes_Positive':")
print(df_data["Nodes_Positive"].describe(percentiles=[0.25, 0.5, 0.75, 0.98, 0.99]))
valid_nodes_clean = df_data["Nodes_Positive"].dropna()
if not valid_nodes_clean.empty:
    print("\nDistribution of cleaned 'Nodes_Positive' (Histogram):")
    counts, bin_edges = np.histogram(valid_nodes_clean, bins=20)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        print(f"{bin_edges[i]:6.1f} - {bin_edges[i+1]:6.1f} | {bar} {count}")


# %% Marital Status
if "Marital_Status" not in df_data.columns:
    raise ValueError("ERROR: 'Marital_Status' column is missing from the data.")
print("\nValue counts for 'Marital_Status':")
marital_counts = (
    df_data["Marital_Status"].value_counts(dropna=False).sort_values(ascending=False)
)
total_marital = marital_counts.sum()
cum_sum_marital = marital_counts.cumsum()
print("\nDistinct Marital_Status values sorted by count (with cumulative percentage):")
for val, count in marital_counts.items():
    cum_pct = (cum_sum_marital[val] / total_marital) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_marital(val):
    s_val = str(val).strip()
    if s_val in ["Married (including common law)"]:
        return 0
    elif s_val in ["Single (never married)", "Unmarried or Domestic Partner"]:
        return 1
    elif s_val in ["Divorced", "Separated"]:
        return 2
    elif s_val == "Widowed":
        return 3
    elif s_val in ["Unknown"]:
        return 4
    else:
        raise ValueError(f"Unexpected value in Marital_Status: {val}")


df_data["Marital_Status"] = df_data["Marital_Status"].apply(map_marital)
print("\nValue counts for mapped 'Marital_Status':")
print(df_data["Marital_Status"].value_counts(dropna=False).sort_values())

# %% Age_at_Diagnosis
if "Age_at_Diagnosis" not in df_data.columns:
    raise ValueError("ERROR: 'Age_at_Diagnosis' column is missing from the data.")
print("\nValue counts for 'Age_at_Diagnosis':")
age_counts = (
    df_data["Age_at_Diagnosis"].value_counts(dropna=False).sort_values(ascending=False)
)
total_age = age_counts.sum()
cum_sum_age = age_counts.cumsum()
print(
    "\nDistinct Age_at_Diagnosis values sorted by count (with cumulative percentage):"
)
for val, count in age_counts.items():
    cum_pct = (cum_sum_age[val] / total_age) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def parse_age(val):
    if pd.isna(val):
        return np.nan
    s_val = str(val).strip()
    if "85+" in s_val:
        return 85
    clean_val = s_val.replace(" years", "").strip()
    return int(clean_val)


df_data["Age_at_Diagnosis"] = df_data["Age_at_Diagnosis"].apply(parse_age)
print("\nStatistics for 'Age_at_Diagnosis' (Numeric):")
print(df_data["Age_at_Diagnosis"].describe())
valid_age = df_data["Age_at_Diagnosis"].dropna()
if not valid_age.empty:
    print("\nDistribution of 'Age_at_Diagnosis' (Histogram):")
    min_age = int(valid_age.min())
    max_age = int(valid_age.max())
    bins = np.arange(min_age, max_age + 6, 5)
    counts, bin_edges = np.histogram(valid_age, bins=bins)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        low = int(bin_edges[i])
        high = int(bin_edges[i + 1])
        print(f"{low:3d} - {high:3d} | {bar} {count}")


# %% Year_of_Diagnosis
if "Year_of_Diagnosis" not in df_data.columns:
    raise ValueError("ERROR: 'Year_of_Diagnosis' column is missing from the data.")
df_data["Year_of_Diagnosis"] = pd.to_numeric(
    df_data["Year_of_Diagnosis"], errors="coerce"
)
print("\nStatistics for 'Year_of_Diagnosis':")
print(df_data["Year_of_Diagnosis"].describe())
valid_year = df_data["Year_of_Diagnosis"].dropna()
if not valid_year.empty:
    print("\nDistribution of 'Year_of_Diagnosis' (Histogram):")
    min_year = int(valid_year.min())
    max_year = int(valid_year.max())
    bins = np.arange(min_year, max_year + 2, 1)
    counts, bin_edges = np.histogram(valid_year, bins=bins)
    max_count = counts.max()
    width = 50
    for i in range(len(counts)):
        count = counts[i]
        bar_length = int(count / max_count * width) if max_count > 0 else 0
        bar = "#" * bar_length
        val = int(bin_edges[i])
        print(f"{val} | {bar} {count}")

# %% Household income
if "Household_Income" not in df_data.columns:
    raise ValueError("ERROR: 'Household_Income' column is missing from the data.")
print("\nValue counts for 'Household_Income':")
income_counts = (
    df_data["Household_Income"].value_counts(dropna=False).sort_values(ascending=False)
)
total_income = income_counts.sum()
cum_sum_income = income_counts.cumsum()
print(
    "\nDistinct Household_Income values sorted by count (with cumulative percentage):"
)
for val, count in income_counts.items():
    cum_pct = (cum_sum_income[val] / total_income) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_income(val):
    s_val = str(val).strip()
    if s_val == "< $40,000":
        return 0
    elif s_val == "$40,000 - $44,999":
        return 1
    elif s_val == "$45,000 - $49,999":
        return 2
    elif s_val == "$50,000 - $54,999":
        return 3
    elif s_val == "$55,000 - $59,999":
        return 4
    elif s_val == "$60,000 - $64,999":
        return 5
    elif s_val == "$65,000 - $69,999":
        return 6
    elif s_val == "$70,000 - $74,999":
        return 7
    elif s_val == "$75,000 - $79,999":
        return 8
    elif s_val == "$80,000 - $84,999":
        return 9
    elif s_val == "$85,000 - $89,999":
        return 10
    elif s_val == "$90,000 - $94,999":
        return 11
    elif s_val == "$95,000 - $99,999":
        return 12
    elif s_val == "$100,000 - $109,999":
        return 13
    elif s_val == "$110,000 - $119,999":
        return 14
    elif s_val == "$120,000+":
        return 15
    elif s_val in ["Unknown/missing/no match/Not 1990-2023"]:
        return 0
    else:
        raise ValueError(f"Unexpected value in Household_Income: {val}")


df_data["Household_Income"] = df_data["Household_Income"].apply(map_income)
print("\nValue counts for mapped 'Household_Income':")
print(df_data["Household_Income"].value_counts(dropna=False).sort_index())

# %% Surgery
if "Surgery" not in df_data.columns:
    raise ValueError("ERROR: 'Surgery' column is missing from the data.")
print("\nValue counts for 'Surgery':")
surgery_counts = (
    df_data["Surgery"].value_counts(dropna=False).sort_values(ascending=False)
)
total_surgery = surgery_counts.sum()
cum_sum_surgery = surgery_counts.cumsum()
print("\nDistinct Surgery values sorted by count (with cumulative percentage):")
for val, count in surgery_counts.items():
    cum_pct = (cum_sum_surgery[val] / total_surgery) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")

# no surgery
# 10-80 site specific codes
# 90 surgery NOS
# 98 special codes, surgery
# 99 unknown, death certificate only
# https://seer.cancer.gov/data-software/documentation/seerstat/nov2024/TextData.FileDescription-nov2024.pdf


def map_surgery(val):
    s_val = str(val).strip()
    if s_val in ["0"]:
        return 0
    elif s_val in [
        "19",
        "15",
        "12",
        "13",
        "20",
        "21",
        "22",
        "23",
        "24",
        "25",
        "30",
        "33",
        "40",
        "41",
        "42",
        "43",
        "44",
        "45",
        "46",
        "47",
        "48",
        "49",
        "50",
        "51",
        "52",
        "53",
        "54",
        "55",
        "56",
        "58",
        "57",
        "59",
        "60",
        "61",
        "62",
        "63",
        "64",
        "65",
        "66",
        "67",
        "68",
        "69",
        "70",
        "71",
        "72",
        "73",
        "74",
        "75",
        "76",
        "80",
        "90",
    ]:
        return 1
    elif s_val in ["99"]:
        return np.nan
    else:
        raise ValueError(f"Unexpected value in Surgery: {val}")


df_data["Surgery"] = df_data["Surgery"].apply(map_surgery)
print("\nValue counts for mapped 'Surgery':")
print(df_data["Surgery"].value_counts(dropna=False).sort_index())


# %%
print("\n--- Inclusion/exclusion ---")
print(f"Initial number of rows: {len(df_data)}")

# Drop sex other than female
initial_len = len(df_data)
df_data = df_data[df_data["Sex"] == "Female"]
dropped_len_sex = initial_len - len(df_data)
print(
    f"Dropped {dropped_len_sex} rows with 'Sex' not equal to 'Female'. Remaining: {len(df_data)}"
)
df_data = df_data.drop(columns=["Sex"])

# Drop rows with age < 18
initial_len = len(df_data)
df_data = df_data[df_data["Age_at_Diagnosis"] >= 18]


# Diagnosed between 2007 and 2015
df_data = df_data[
    (df_data["Year_of_Diagnosis"] >= 2007) & (df_data["Year_of_Diagnosis"] <= 2015)
]
dropped_len_year_age = initial_len - len(df_data)
print(
    f"Dropped {dropped_len_year_age} rows with 'Year_of_Diagnosis' outside 2007-2015 and age < 18. Remaining: {len(df_data)}"
)

# AJCC IIIA or IIIB
initial_len_stage = len(df_data)
df_data = df_data[df_data["AJCC_Stage"].isin(["IIIA", "IIIB"])]
dropped_len_stage = initial_len_stage - len(df_data)
print(
    f"Dropped {dropped_len_stage} rows with 'AJCC_Stage' not in ['IIIA', 'IIIB']. Remaining: {len(df_data)}"
)
df_data = df_data.drop(columns=["AJCC_Stage"])

# Keep surgery = yes
initial_len_surgery = len(df_data)
df_data = df_data[df_data["Surgery"] == 1]
dropped_len_surgery = initial_len_surgery - len(df_data)
print(
    f"Dropped {dropped_len_surgery} rows with 'Surgery' not performed/unknown. Remaining: {len(df_data)}"
)
df_data = df_data.drop(columns=["Surgery"])

# Keep chemotherapy = yes
initial_len_chemo = len(df_data)
df_data = df_data[df_data["Chemotherapy"] == "Yes"]
dropped_len_chemo = initial_len_chemo - len(df_data)
print(
    f"Dropped {dropped_len_chemo} rows with 'Chemotherapy' not equal to 'Yes'. Remaining: {len(df_data)}"
)
df_data = df_data.drop(columns=["Chemotherapy"])


# Drop rows with NaN in Tumor_Size
initial_len_tumor_nodes = len(df_data)
df_data = df_data.dropna(subset=["Tumor_Size"])

# Drop rows with NaN in Nodes_Positive
df_data = df_data.dropna(subset=["Nodes_Positive"])
dropped_len_tumor_nodes = initial_len_tumor_nodes - len(df_data)
print(
    f"\nDropped {dropped_len_tumor_nodes} rows with invalid 'Nodes_Positive' and 'Tumor_Size'. Remaining: {len(df_data)}"
)

# Drop rows with NaN in Treatment_Group
initial_len_treat = len(df_data)
df_data = df_data.dropna(subset=["Treatment_Group"])
dropped_len = initial_len_treat - len(df_data)
print(
    f"Dropped {dropped_len} rows with NaN in 'Treatment_Group'. Remaining: {len(df_data)}"
)
df_data["Treatment_Group"] = df_data["Treatment_Group"].astype(int)


# Drop rows with NaN in Event_Observed
initial_len_event = len(df_data)
df_data = df_data.dropna(subset=["Event_Observed"])
dropped_len_event = initial_len_event - len(df_data)
print(
    f"Dropped {dropped_len_event} rows with NaN in 'Event_Observed'. Remaining: {len(df_data)}"
)

# Drop rows with NaN in Survival_Months or Survival_Months <= 0
initial_len_survival = len(df_data)
df_data = df_data.dropna(subset=["Survival_Months"])
df_data = df_data[df_data["Survival_Months"] > 0]
dropped_len_survival = initial_len_survival - len(df_data)
print(
    f"Dropped {dropped_len_survival} rows with NaN or non-positive values in 'Survival_Months'. Remaining: {len(df_data)}"
)


# %% per column print value counts and statistics
print("\n--- Final Column Value Counts and Statistics ---")
for col in df_data.columns:
    print(f"\nColumn: {col}")
    if df_data[col].unique().shape[0] > 20 and pd.api.types.is_numeric_dtype(
        df_data[col]
    ):
        print(df_data[col].describe())
    else:
        counts = df_data[col].value_counts(dropna=False)
        total = counts.sum()
        cum_sum = counts.cumsum()
        print("Value counts (with cumulative percentage):")
        for val, count in counts.items():
            cum_pct = (cum_sum[val] / total) * 100
            print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


# %%
nan_counts = df_data.isna().sum()
if nan_counts.sum() > 0:
    for col in df_data.columns:
        num_nans = df_data[col].isna().sum()
        if num_nans > 0:
            print(f"  Column '{col}' has {num_nans} NaN values.")
            # add missingness indicator
            indicator_col = f"{col}_isna"
            df_data[indicator_col] = df_data[col].isna().astype(int)
            print(
                f"  Added missingness indicator column: {indicator_col}, {df_data[indicator_col].value_counts(dropna=False).to_dict()}"
            )

else:
    print("\n[SUCCESS] No NaNs remaining in the dataset.")

# Ensure treatment, event, time are last columns in this order
FINAL_COLS_ORDER = ["Treatment_Group", "Event_Observed", "Survival_Months"]

# Sanity check
for c in FINAL_COLS_ORDER:
    if c not in df_data.columns:
        raise RuntimeError(f"Missing required final column: {c}")

# Reorder columns
other_cols = [c for c in df_data.columns if c not in FINAL_COLS_ORDER]
df_data = df_data[other_cols + FINAL_COLS_ORDER]


print("\nFinal Dataset Column Types:")
print(df_data.dtypes)

print(df_data["Treatment_Group"].value_counts(dropna=False))
print(df_data["Event_Observed"].value_counts(dropna=False))


# Save cleaned data
output_dir = config["output_dir"]
output_file = config["output_file"]
os.makedirs(output_dir, exist_ok=True)
output_path = os.path.join(output_dir, output_file)
df_data.to_csv(output_path, index=False)
print(f"\nCleaned data saved to: {output_path}")
