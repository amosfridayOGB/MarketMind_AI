import os
import pandas as pd

# =====================================================
# LOAD DATA
# =====================================================

print("Loading Dataset...")

df = pd.read_csv(
    "processed/clean_marketing.csv"
)

# =====================================================
# AGE
# =====================================================

CURRENT_YEAR = 2026

df["Age"] = (
    CURRENT_YEAR -
    df["Year_Birth"]
)

# =====================================================
# TOTAL CHILDREN
# =====================================================

df["TotalChildren"] = (
    df["Kidhome"] +
    df["Teenhome"]
)

# =====================================================
# TOTAL SPENDING
# =====================================================

df["TotalSpending"] = (

    df["MntWines"] +

    df["MntFruits"] +

    df["MntMeatProducts"] +

    df["MntFishProducts"] +

    df["MntSweetProducts"] +

    df["MntGoldProds"]
)

# =====================================================
# TOTAL PURCHASES
# =====================================================

df["TotalPurchases"] = (

    df["NumWebPurchases"] +

    df["NumCatalogPurchases"] +

    df["NumStorePurchases"]
)

# =====================================================
# CAMPAIGN SUCCESS COUNT
# =====================================================

df["CampaignSuccesses"] = (

    df["AcceptedCmp1"] +

    df["AcceptedCmp2"] +

    df["AcceptedCmp3"] +

    df["AcceptedCmp4"] +

    df["AcceptedCmp5"]
)

# =====================================================
# CUSTOMER TENURE
# =====================================================

df["Dt_Customer"] = pd.to_datetime(
    df["Dt_Customer"],
    format="%d-%m-%Y"
)

latest_date = (
    df["Dt_Customer"]
    .max()
)

df["CustomerTenureDays"] = (
    latest_date -
    df["Dt_Customer"]
).dt.days

# =====================================================
# DROP DATE COLUMN
# =====================================================

df = df.drop(
    columns=["Dt_Customer"]
)

# =====================================================
# SAVE
# =====================================================

os.makedirs(
    "processed",
    exist_ok=True
)

OUTPUT_PATH = (
    "processed/engineered_marketing.csv"
)

df.to_csv(
    OUTPUT_PATH,
    index=False
)

print("\nSaved:")
print(OUTPUT_PATH)

print("\nShape:")
print(df.shape)
