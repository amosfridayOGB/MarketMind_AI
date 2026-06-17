import joblib
import pandas as pd

metrics = joblib.load(
    "models/model_metrics.pkl"
)

print(metrics)

feature_df = pd.read_csv(
    "models/feature_importance.csv"
)

print("\nTop Features")
print(
    feature_df.head(20)
)
