import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import warnings
from pandas.api.types import CategoricalDtype

warnings.filterwarnings("ignore")

df = pd.read_excel("medical-dataset.xlsx")

# Display the types of each column in the DataFrame
print("--- Initial Data Types ---")
print(df.dtypes)


# helper function to convert date strings to datetime objects
def convert_date(date_str):
    if pd.isna(date_str):
        return pd.NaT
    try:
        # Try different date formats
        for fmt in ["%Y-%m-%d %H:%M:%S", "%d/%m/%Y", "%Y-%m-%d"]:
            try:
                return pd.to_datetime(date_str, format=fmt)
            except:
                continue
        return pd.to_datetime(date_str, infer_datetime_format=True)
    except:
        return pd.NaT


# Convert date column to datetime objects
df["survey_date"] = df["survey_date"].apply(convert_date)

# Replace '100_110' with '90_plus' and convert to ordered category
df["age"] = df["age"].replace("100_110", "90_plus")
age_order = [
    "0_10",
    "10_20",
    "20_30",
    "30_40",
    "40_50",
    "50_60",
    "60_70",
    "70_80",
    "80_90",
    "90_plus",
]
age_dtype = pd.api.types.CategoricalDtype(categories=age_order, ordered=True)
df["age"] = df["age"].astype(age_dtype)

# print categories of age and thier values
print("\n--- Age Categories ---")
print(df["age"].cat.categories)
print("\n--- Age Value Counts ---")
print(df["age"].value_counts(sort=False))

# Convert categorical variables
categorical_cols = [
    "gender",
    "blood_type",
    "insurance",
    "income",
    "smoking",
    "working",
    "region",
    "country",
]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Convert boolean health conditions
health_conditions = [
    "covid19_positive",
    "covid19_symptoms",
    "covid19_contact",
    "asthma",
    "kidney_disease",
    "liver_disease",
    "compromised_immune",
    "heart_disease",
    "lung_disease",
    "diabetes",
    "hiv_positive",
    "other_chronic",
    "nursing_home",
    "health_worker",
]

for col in health_conditions:
    df[col] = df[col].astype("bool")

print("\n--- Data Types After Conversion ---")
print(df.dtypes)

# Identify numerical columns in the dataset
numerical_cols = df.select_dtypes(include=[np.number]).columns
# dictionary to store skewness results
skewness_results = {}

for col in numerical_cols:
    if col in df.columns:
        skewness = df[col].skew()
        skewness_results[col] = skewness
        print(f"{col}: {skewness:.3f}")

# Visualize distributions
plt.figure(figsize=(15, 10))

# Compute and print mean, median, and mode for selected variables
print("\n--- Mean, Median, Mode for Skew Detection ---")
for col in ["contacts_count", "bmi", "height", "weight"]:
    mode = df[col].mode().iloc[0]
    mean = df[col].mean()
    median = df[col].median()
    print(f"{col:<15} → Mode: {mode:.2f}, Median: {median:.2f}, Mean: {mean:.2f}")

# Plot histograms for selected numerical variables
# for i, col in enumerate(["height", "weight", "bmi", "contacts_count"], 1):
#     plt.subplot(2, 2, i)
#     df[col].hist(bins=30)
#     plt.title(f"Distribution of {col} (Skewness: {df[col].skew():.3f})")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()

# Identify missing values in each column
missing_values = df.isnull().sum()
missing_values = missing_values[
    missing_values > 0
]  # Only show columns with missing values

# Display columns with missing values
print("Columns with missing values:\n")
print(missing_values)

# rules for expected ranges in suspected outliers
rules = {"height": (100, 220), "weight": (30, 200), "bmi": (10, 60)}
# Check for outliers based on the defined rules
for col, (min_val, max_val) in rules.items():
    outliers = df[(df[col] < min_val) | (df[col] > max_val)]
    print(f"{col}: {len(outliers)} values outside expected range")

# for i, col in enumerate(["height", "weight", "bmi", "contacts_count"], 1):
#     plt.subplot(2, 2, i)
#     df[col].hist(bins=30)
#     plt.title(f"Distribution of {col} (Skewness: {df[col].skew():.3f})")
#     plt.xlabel(col)
#     plt.ylabel("Frequency")
# plt.tight_layout()
# plt.show()


# begin data clean up
# Drop region and cocaine due to over 70% missing
df.drop(columns=["region", "cocaine"], inplace=True)


# 🔁 Fill missing categorical values with 'unknown', show before/after
for col in ["insurance", "income"]:
    # Output: show how amount missing before
    missing_before = df[col].isnull().sum()
    print(f"\n--- {col.capitalize()} Missing Values Before: {missing_before} ---")

    if "unknown" not in df[col].cat.categories:
        df[col] = df[col].cat.add_categories(["unknown"])

    df[col] = df[col].fillna("unknown")

    # Output: show 'unknown' value count after
    print(f"\n--- {col.capitalize()} Value Counts After Imputation ---")
    print(df[col].value_counts())

# Fill 'smoking' and 'working' with mode
for col in ["smoking", "working"]:
    missing_before = df[col].isnull().sum()
    df[col] = df[col].fillna(df[col].mode()[0])
    missing_after = df[col].isnull().sum()
    print(f"\n--- {col.capitalize()} ---")
    print(f"Missing before: {missing_before}, after: {missing_after}")
    print(df[col].value_counts())

# Fill 'worried' and 'alcohol' with median
for col in ["worried", "alcohol"]:
    missing_before = df[col].isnull().sum()
    df[col] = df[col].fillna(df[col].median())
    missing_after = df[col].isnull().sum()
    print(f"\n--- {col.capitalize()} ---")
    print(f"Missing before: {missing_before}, after: {missing_after}")
    print(f"Filled with median: {df[col].median()}")

# Fill 'contacts_count' with median and 'public_transport_count' with 0
fill_methods = {
    "contacts_count": df["contacts_count"].median(),
    "public_transport_count": 0,
}

for col, fill_value in fill_methods.items():
    missing_before = df[col].isnull().sum()
    df[col] = df[col].fillna(fill_value)
    missing_after = df[col].isnull().sum()
    print(f"\n--- {col.replace('_', ' ').title()} ---")
    print(f"Missing before: {missing_before}, after: {missing_after}")
    print(f"Filled with: {fill_value}")

rules = {
    "height": (100, 220),  # in cm
    "weight": (30, 200),  # in kg
    "bmi": (10, 60),  # typical healthy/realistic human range
}

# Only apply outlier removal to rows with age >= '20_30'
adult_ages = ["20_30", "30_40", "40_50", "50_60", "60_70", "70_80", "80_90", "90_plus"]
adult_mask = df["age"].isin(adult_ages)

for col, (min_val, max_val) in rules.items():
    before = df.shape[0]
    # Only filter adult rows based on rules
    df = df[~((adult_mask) & ((df[col] < min_val) | (df[col] > max_val)))]
    after = df.shape[0]
    print(
        f"{col}: Removed {before - after} adult rows outside range {min_val}-{max_val}"
    )


# Final structure and null check
print("\nFinal dataset info:")
print(df.info())

print("\nRemaining missing values:")
print(df.isnull().sum()[df.isnull().sum() > 0])
