import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from xgboost import XGBClassifier

# =====================================================
# LOAD DATA
# =====================================================

df = pd.read_csv(
    "processed/engineered_marketing.csv"
)

print("\nDataset Loaded")
print(df.shape)

# =====================================================
# TARGET
# =====================================================

X = df.drop(columns=["target"])
y = df["target"]

# =====================================================
# FEATURE TYPES
# =====================================================

categorical_features = [
    "Education",
    "Marital_Status"
]

numeric_features = [
    col for col in X.columns
    if col not in categorical_features
]

# =====================================================
# PREPROCESSING
# =====================================================

categorical_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

numeric_transformer = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            categorical_transformer,
            categorical_features
        ),
        (
            "num",
            numeric_transformer,
            numeric_features
        )
    ]
)

# =====================================================
# TRAIN TEST SPLIT
# =====================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# =====================================================
# CLASS WEIGHT
# =====================================================

negative = (y_train == 0).sum()
positive = (y_train == 1).sum()

scale_weight = negative / positive

print("\nScale Weight:")
print(scale_weight)

# =====================================================
# MODEL
# =====================================================

model = XGBClassifier(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    objective="binary:logistic",
    eval_metric="auc",
    scale_pos_weight=scale_weight,
    random_state=42
)

pipeline = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            model
        )
    ]
)

# =====================================================
# TRAIN
# =====================================================

print("\nTraining XGBoost...")

pipeline.fit(
    X_train,
    y_train
)

# =====================================================
# PREDICT
# =====================================================

probs = pipeline.predict_proba(
    X_test
)[:, 1]

preds = (
    probs >= 0.50
).astype(int)

# =====================================================
# METRICS
# =====================================================

accuracy = accuracy_score(
    y_test,
    preds
)

precision = precision_score(
    y_test,
    preds,
    zero_division=0
)

recall = recall_score(
    y_test,
    preds,
    zero_division=0
)

f1 = f1_score(
    y_test,
    preds,
    zero_division=0
)

auc = roc_auc_score(
    y_test,
    probs
)

print("\nMODEL PERFORMANCE")
print("-" * 50)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"ROC AUC   : {auc:.4f}")

print("\nCONFUSION MATRIX")
print(
    confusion_matrix(
        y_test,
        preds
    )
)

print("\nCLASSIFICATION REPORT")
print(
    classification_report(
        y_test,
        preds,
        zero_division=0
    )
)

# =====================================================
# SAVE MODEL
# =====================================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    pipeline,
    "models/marketmind_xgb.pkl"
)

# =====================================================
# SAVE METRICS
# =====================================================

metrics = {
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "roc_auc": auc
}

joblib.dump(
    metrics,
    "models/model_metrics.pkl"
)

# =====================================================
# SAVE ROC
# =====================================================

fpr, tpr, _ = roc_curve(
    y_test,
    probs
)

roc_df = pd.DataFrame({
    "fpr": fpr,
    "tpr": tpr
})

roc_df.to_csv(
    "models/roc_curve.csv",
    index=False
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

feature_names = (
    pipeline.named_steps[
        "preprocessor"
    ].get_feature_names_out()
)

importance = (
    pipeline.named_steps[
        "model"
    ].feature_importances_
)

feature_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance
})

feature_df = feature_df.sort_values(
    by="importance",
    ascending=False
)

feature_df.to_csv(
    "models/feature_importance.csv",
    index=False
)

print("\nModel Saved:")
print("models/marketmind_xgb.pkl")

print("\nTraining Complete.")
