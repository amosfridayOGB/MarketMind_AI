import os
import pandas as pd

# =====================================================
# LOAD DATA
# =====================================================

DATA_PATH = "data/marketing_campaign.csv"

print("Loading Dataset...")

df = pd.read_csv(
    DATA_PATH,
    sep="\t"
)

print("\nOriginal Shape:")
print(df.shape)

# =====================================================
# DROP ID
# =====================================================

df = df.drop(
    columns=["ID"]
)

# =====================================================
# HANDLE MISSING INCOME
# =====================================================

df["Income"] = df["Income"].fillna(
    df["Income"].median()
)

# =====================================================
# TARGET
# =====================================================

df["target"] = df["Response"]

df = df.drop(
    columns=["Response"]
)

# =====================================================
# SAVE
# =====================================================

os.makedirs(
    "processed",
    exist_ok=True
)

OUTPUT_PATH = (
    "processed/clean_marketing.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nClean Shape:")
print(df.shape)

print("\nTarget Distribution:")
print(
    df["target"]
    .value_counts()
)

print("\nSaved:")
print(OUTPUT_PATH)
