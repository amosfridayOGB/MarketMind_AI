import pandas as pd

df = pd.read_csv(
    "data/marketing_campaign.csv",
    sep="\t"
)

print("\nSHAPE")
print(df.shape)

print("\nCOLUMNS")
print(df.columns.tolist())

print("\nMISSING VALUES")
print(df.isnull().sum())

print("\nHEAD")
print(df.head())