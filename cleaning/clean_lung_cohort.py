# %%
import pandas as pd
import numpy as np
import os
import yaml

COLUMNS_TO_DROP = [
]


COLUMN_MAPPING = {
    "Age recode with single ages and 85+": "Age",
    "Sex": "Sex",
    "Year of diagnosis": "Year_of_Diagnosis",
    "Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)": "Race_Origin",
    "Primary Site - labeled": "Primary_Site",
    "Laterality": "Laterality",
    "Derived AJCC T, 6th ed (2004-2015)": "AJCC_T",
    "Derived AJCC N, 6th ed (2004-2015)": "AJCC_N",
    "Derived AJCC M, 6th ed (2004-2015)": "AJCC_M",
    "Marital status at diagnosis": "Marital_Status",
    "Median household income inflation adj to 2023": "Household_Income",
    "RX Summ--Surg Prim Site (1998+)": "Surgery",
    "Radiation recode": "Radiation",
    "Chemotherapy recode (yes, no/unk)": "Chemotherapy",
    "First malignant primary indicator": "First_Malignant_Primary_Indicator",
    "Vital status recode (study cutoff used)": "Vital_Status",
    "Survival months": "Survival_Months",
}

with open("../cleaning/config_raw_lung.yaml", "r") as f:
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

# %% Year of Diagnosis
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


# %% Age
if "Age" not in df_data.columns:
    raise ValueError("ERROR: 'Age' column is missing from the data.")
print("\nValue counts for 'Age':")
age_counts = df_data["Age"].value_counts(dropna=False).sort_values(ascending=False)
total_age = age_counts.sum()
cum_sum_age = age_counts.cumsum()
print("\nDistinct Age values sorted by count (with cumulative percentage):")
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


df_data["Age"] = df_data["Age"].apply(parse_age)
print("\nStatistics for 'Age' (Numeric):")
print(df_data["Age"].describe())
age_all = df_data["Age"]
n_missing = age_all.isna().sum()
valid_age = age_all[age_all.notna()]

if not valid_age.empty:
    print("\nDistribution of 'Age' (Histogram):")
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
if n_missing > 0:
    print(f"\nNumber of missing (NaN) 'Age' values: {n_missing}")


# %% -------- INCLUSION CRITERIA (BASE COHORT) ---------
print("Initial number of rows:", len(df_data))

# Drop rows with age < 18
initial_len = len(df_data)
df_data = df_data[df_data["Age"] >= 18]
dropped_len_age = initial_len - len(df_data)
print(f"Dropped {dropped_len_age} rows with 'Age' < 18. Remaining: {len(df_data)}")

# Keep rows with year of diagnosis between 2004 and 2015 (inclusive)
initial_len = len(df_data)
df_data = df_data[
    (df_data["Year_of_Diagnosis"] >= 2004) & (df_data["Year_of_Diagnosis"] <= 2015)
]
dropped_len_year = initial_len - len(df_data)
print(
    f"Dropped {dropped_len_year} rows with 'Year_of_Diagnosis' outside 2004-2015. Remaining: {len(df_data)}"
)


# %% Treatment
if "Radiation" not in df_data.columns:
    raise ValueError("ERROR: 'Radiation' column is missing from the data.")
print("\nValue counts for 'Radiation':")
radiation_counts = (
    df_data["Radiation"].value_counts(dropna=False).sort_values(ascending=False)
)
total_radiation = radiation_counts.sum()
cum_sum_radiation = radiation_counts.cumsum()
print("\nDistinct Radiation values sorted by count (with cumulative percentage):")
for val, count in radiation_counts.items():
    cum_pct = (cum_sum_radiation[val] / total_radiation) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


# Map Radiation
def map_radiation(val):
    s_val = str(val).strip()
    if s_val in [
        "Beam radiation",
        "Radiation, NOS  method or source not specified",
        "Radioactive implants (includes brachytherapy) (1988+)",
        "Radioisotopes (1988+)",
        "Combination of beam with implants or isotopes",
        "Radioisotopes (1988+)",
    ]:
        return 1
    elif s_val in ["None/Unknown"]:
        return 0
    elif s_val in ["Refused (1988+)", "Recommended, unknown if administered"]:
        return np.nan
    else:
        raise ValueError(f"Unexpected value in Radiation: {val}")


df_data["Treatment_Group"] = df_data["Radiation"].apply(map_radiation)
print("\nValue counts for mapped 'Radiation':")
print(df_data["Treatment_Group"].value_counts(dropna=False).sort_index())
df_data = df_data.drop(columns=["Radiation"])

# %% Outcome vital status -> Event_Observed
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
print("\nValue counts for 'Event_Observed' (1=Dead, 0=Alive):")
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

print("\nValue counts for 'Event_Observed' after capping:")
print(df_data["Event_Observed"].value_counts(dropna=False))

# %% Race_Origin
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
print(df_data["Race_Origin"].value_counts(dropna=False).sort_index())

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
print(df_data["Marital_Status"].value_counts(dropna=False).sort_index())


# %% Sex
if "Sex" not in df_data.columns:
    raise ValueError("ERROR: 'Sex' column is missing from the data.")
print("\nValue counts for 'Sex':")
sex_counts = df_data["Sex"].value_counts(dropna=False).sort_values(ascending=False)
total_sex = sex_counts.sum()
cum_sum_sex = sex_counts.cumsum()
print("\nDistinct Sex values sorted by count (with cumulative percentage):")
for val, count in sex_counts.items():
    cum_pct = (cum_sum_sex[val] / total_sex) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_sex(val):
    s_val = str(val).strip()
    if s_val == "Male":
        return 0
    elif s_val == "Female":
        return 1
    else:
        raise ValueError(f"Unexpected value in Sex: {val}")


df_data["Sex"] = df_data["Sex"].apply(map_sex)
print("\nValue counts for mapped 'Sex':")
print(df_data["Sex"].value_counts(dropna=False).sort_index())

# %% Laterality
if "Laterality" not in df_data.columns:
    raise ValueError("ERROR: 'Laterality' column is missing from the data.")
print("\nValue counts for 'Laterality':")
laterality_counts = (
    df_data["Laterality"].value_counts(dropna=False).sort_values(ascending=False)
)
total_laterality = laterality_counts.sum()
cum_sum_laterality = laterality_counts.cumsum()
print("\nDistinct Laterality values sorted by count (with cumulative percentage):")
for val, count in laterality_counts.items():
    cum_pct = (cum_sum_laterality[val] / total_laterality) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_laterality(val):
    s_val = str(val).strip()
    if s_val == "Left - origin of primary":
        return 0
    elif s_val == "Right - origin of primary":
        return 1
    elif s_val == "Not a paired  site":
        return 2
    elif s_val in ["Bilateral, single primary", "Paired site: midline tumor"]:
        return 2
    elif s_val in [
        "Only one side - side unspecified",
        "Paired site, but no information concerning laterality",
    ]:
        return 2
    else:
        raise ValueError(f"Unexpected value in Laterality: {val}")


df_data["Laterality"] = df_data["Laterality"].apply(map_laterality)
print("\nValue counts for mapped 'Laterality':")
print(df_data["Laterality"].value_counts(dropna=False).sort_index())


# %% Primary site
if "Primary_Site" not in df_data.columns:
    raise ValueError("ERROR: 'Primary_Site' column is missing from the data.")
print("\nValue counts for 'Primary_Site':")
primary_site_counts = (
    df_data["Primary_Site"].value_counts(dropna=False).sort_values(ascending=False)
)
total_primary_site = primary_site_counts.sum()
cum_sum_primary_site = primary_site_counts.cumsum()
print("\nDistinct Primary_Site values sorted by count (with cumulative percentage):")
for val, count in primary_site_counts.items():
    cum_pct = (cum_sum_primary_site[val] / total_primary_site) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_primary_site(val):
    s_val = str(val).strip()
    if s_val == "C34.0-Main bronchus":
        return 0
    elif s_val == "C34.1-Upper lobe, lung":
        return 1
    elif s_val == "C34.2-Middle lobe, lung":
        return 2
    elif s_val == "C34.3-Lower lobe, lung":
        return 3
    elif s_val == "C34.8-Overlapping lesion of lung":
        return 4
    elif s_val == "C34.9-Lung, NOS":
        return 5
    else:
        raise ValueError(f"Unexpected value in Primary_Site: {val}")


df_data["Primary_Site"] = df_data["Primary_Site"].apply(map_primary_site)
print("\nValue counts for mapped 'Primary_Site':")
print(df_data["Primary_Site"].value_counts(dropna=False).sort_index())


# %% AJCC staging T
if "AJCC_T" not in df_data.columns:
    raise ValueError("ERROR: 'AJCC_T' column is missing from the data.")
print("\nValue counts for 'AJCC_T':")
ajcc_t_counts = (
    df_data["AJCC_T"].value_counts(dropna=False).sort_values(ascending=False)
)
total_ajcc_t = ajcc_t_counts.sum()
cum_sum_ajcc_t = ajcc_t_counts.cumsum()
print("\nDistinct AJCC_T values sorted by count (with cumulative percentage):")
for val, count in ajcc_t_counts.items():
    cum_pct = (cum_sum_ajcc_t[val] / total_ajcc_t) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_ajcc_t(val):
    s_val = str(val).strip()
    if s_val == "T0":
        return 0
    elif s_val == "T1":
        return 1
    elif s_val == "T2":
        return 2
    elif s_val == "T3":
        return 3
    elif s_val == "T4":
        return 4
    elif s_val == "TX":
        return np.nan
    else:
        raise ValueError(f"Unexpected value in AJCC_T: {val}")


df_data["AJCC_T"] = df_data["AJCC_T"].apply(map_ajcc_t)
print("\nValue counts for mapped 'AJCC_T':")
print(df_data["AJCC_T"].value_counts(dropna=False).sort_index())

# %% AJCC staging N
if "AJCC_N" not in df_data.columns:
    raise ValueError("ERROR: 'AJCC_N' column is missing from the data.")
print("\nValue counts for 'AJCC_N':")
ajcc_n_counts = (
    df_data["AJCC_N"].value_counts(dropna=False).sort_values(ascending=False)
)
total_ajcc_n = ajcc_n_counts.sum()
cum_sum_ajcc_n = ajcc_n_counts.cumsum()
print("\nDistinct AJCC_N values sorted by count (with cumulative percentage):")
for val, count in ajcc_n_counts.items():
    cum_pct = (cum_sum_ajcc_n[val] / total_ajcc_n) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_ajcc_n(val):
    s_val = str(val).strip()
    if s_val == "N0":
        return 0
    elif s_val == "N1":
        return 1
    elif s_val == "N2":
        return 2
    elif s_val == "N3":
        return 3
    elif s_val == "NX":
        return np.nan
    else:
        raise ValueError(f"Unexpected value in AJCC_N: {val}")


df_data["AJCC_N"] = df_data["AJCC_N"].apply(map_ajcc_n)
print("\nValue counts for mapped 'AJCC_N':")
print(df_data["AJCC_N"].value_counts(dropna=False).sort_index())

# %% AJCC staging M
if "AJCC_M" not in df_data.columns:
    raise ValueError("ERROR: 'AJCC_M' column is missing from the data.")
print("\nValue counts for 'AJCC_M':")
ajcc_m_counts = (
    df_data["AJCC_M"].value_counts(dropna=False).sort_values(ascending=False)
)
total_ajcc_m = ajcc_m_counts.sum()
cum_sum_ajcc_m = ajcc_m_counts.cumsum()
print("\nDistinct AJCC_M values sorted by count (with cumulative percentage):")
for val, count in ajcc_m_counts.items():
    cum_pct = (cum_sum_ajcc_m[val] / total_ajcc_m) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_ajcc_m(val):
    s_val = str(val).strip()
    if s_val == "M0":
        return 0
    elif s_val == "M1":
        return 1
    elif s_val == "MX":
        return np.nan
    else:
        raise ValueError(f"Unexpected value in AJCC_M: {val}")


df_data["AJCC_M"] = df_data["AJCC_M"].apply(map_ajcc_m)
print("\nValue counts for mapped 'AJCC_M':")
print(df_data["AJCC_M"].value_counts(dropna=False).sort_index())


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
        "45",
        "46",
        "47",
        "48",
        "55",
        "56",
        "65",
        "66",
        "70",
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

# %% Chemotherapy
if "Chemotherapy" not in df_data.columns:
    raise ValueError("ERROR: 'Chemotherapy' column is missing from the data.")
print("\nValue counts for 'Chemotherapy':")
chemo_counts = (
    df_data["Chemotherapy"].value_counts(dropna=False).sort_values(ascending=False)
)
total_chemo = chemo_counts.sum()
cum_sum_chemo = chemo_counts.cumsum()
print("\nDistinct Chemotherapy values sorted by count (with cumulative percentage):")
for val, count in chemo_counts.items():
    cum_pct = (cum_sum_chemo[val] / total_chemo) * 100
    print(f"  '{val}': {count} rows ({cum_pct:.1f}% cumulative)")


def map_chemotherapy(val):
    s_val = str(val).strip()
    if s_val in ["Yes"]:
        return 1
    elif s_val in ["No/Unknown"]:
        return 0
    else:
        raise ValueError(f"Unexpected value in Chemotherapy: {val}")


df_data["Chemotherapy"] = df_data["Chemotherapy"].apply(map_chemotherapy)
print("\nValue counts for mapped 'Chemotherapy':")
print(df_data["Chemotherapy"].value_counts(dropna=False).sort_index())

# %% -------- Exclusion ---------
print("\n--- Inclusion/exclusion ---")
print("Rows after inclusion:", len(df_data))

# Drop rows with first malignant primary indicator = no
initial_len = len(df_data)
df_data = df_data[df_data["First_Malignant_Primary_Indicator"] == "Yes"]
dropped_len_first = initial_len - len(df_data)
print(
    f"Dropped {dropped_len_first} rows with first malignant primary indicator = no. Remaining: {len(df_data)}"
)
df_data = df_data.drop(columns=["First_Malignant_Primary_Indicator"])

# Drop rows with NaN in AJCC staging, T, N, M
initial_len_ajcc = len(df_data)
tmp_len = initial_len_ajcc
for col in ["AJCC_T", "AJCC_N", "AJCC_M"]:
    df_data = df_data.dropna(subset=[col])
    dropped_len_ajcc = tmp_len - len(df_data)
    tmp_len = len(df_data)
    print(
        f"Dropped {dropped_len_ajcc} rows with NaN in {col}. Remaining: {len(df_data)}"
    )

# Drop unknown primary site surgery
initial_len_surgery = len(df_data)
df_data = df_data.dropna(subset=["Surgery"])
dropped_len_surgery = initial_len_surgery - len(df_data)
print(
    f"Dropped {dropped_len_surgery} rows with NaN in primary site surgery. Remaining: {len(df_data)}"
)

# Drop patients with unknown treatment_group
initial_len = len(df_data)
df_data = df_data[df_data["Treatment_Group"].isin([0, 1])]
dropped_len_treatment = initial_len - len(df_data)
print(
    f"Dropped {dropped_len_treatment} rows with unknown/other treatment. Remaining: {len(df_data)}"
)
df_data["Treatment_Group"] = df_data["Treatment_Group"].astype(int)
print(df_data["Treatment_Group"].value_counts(dropna=False).sort_index())

# Drop rows with NaN in Event_Observed
initial_len_event = len(df_data)
df_data = df_data.dropna(subset=["Event_Observed"])
dropped_len_event = initial_len_event - len(df_data)
print(
    f"Dropped {dropped_len_event} rows with NaN in 'Event_Observed'. Remaining: {len(df_data)}"
)
print(df_data["Event_Observed"].value_counts(dropna=False).sort_index())


# Drop rows with NaN in Survival_Months or Survival_Months <= 0
initial_len_survival = len(df_data)
df_data = df_data.dropna(subset=["Survival_Months"])
df_data = df_data[df_data["Survival_Months"] > 0]
dropped_len_survival = initial_len_survival - len(df_data)
print(
    f"Dropped {dropped_len_survival} rows with NaN or non-positive values in 'Survival_Months'. Remaining: {len(df_data)}"
)


# %%
# per column print value counts and statistics
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

# %%
