import pandas as pd

df = pd.read_csv(
    "processed/engineered_marketing.csv"
)

print(df.columns.tolist())