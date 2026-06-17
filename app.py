import os
import joblib
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# =====================================================
# PAGE CONFIG
# =====================================================

st.set_page_config(
    page_title="MarketMind AI",
    page_icon="📈",
    layout="wide"
)

# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    return pd.read_csv(
        "processed/engineered_marketing.csv"
    )

df = load_data()

# =====================================================
# LOAD MODEL
# =====================================================

@st.cache_resource
def load_model():
    return joblib.load(
        "models/marketmind_xgb.pkl"
    )

model = load_model()

# =====================================================
# LOAD METRICS
# =====================================================

@st.cache_data
def load_metrics():

    path = "models/model_metrics.pkl"

    if os.path.exists(path):
        return joblib.load(path)

    return {}

metrics = load_metrics()

if "f1_score" not in metrics and "f1" in metrics:
    metrics["f1_score"] = metrics["f1"]

# =====================================================
# LOAD FEATURE IMPORTANCE
# =====================================================

@st.cache_data
def load_feature_importance():

    path = "models/feature_importance.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()

feature_df = load_feature_importance()

# =====================================================
# LOAD ROC CURVE
# =====================================================

@st.cache_data
def load_roc():

    path = "models/roc_curve.csv"

    if os.path.exists(path):
        return pd.read_csv(path)

    return pd.DataFrame()

roc_df = load_roc()

# =====================================================
# HEADER
# =====================================================

st.title("📈 MarketMind AI")

st.markdown("""
### Customer Conversion Prediction Platform

MarketMind AI leverages machine learning to predict customer campaign responses and uncover actionable marketing intelligence.

**Built With:**

- Python
- XGBoost
- Scikit-Learn
- Pandas
- Plotly
- Streamlit
""")

st.divider()

# =====================================================
# KPI SECTION
# =====================================================

total_customers = len(df)

responders = int(
    df["target"].sum()
)

response_rate = (
    responders /
    total_customers
) * 100

accuracy = metrics.get(
    "accuracy",
    0
)

roc_auc = metrics.get(
    "roc_auc",
    0
)

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Customers",
    f"{total_customers:,}"
)

col2.metric(
    "Responders",
    f"{responders:,}"
)

col3.metric(
    "Response Rate",
    f"{response_rate:.2f}%"
)

col4.metric(
    "ROC-AUC",
    f"{roc_auc:.4f}"
)

st.divider()

# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.header("Filters")

education_filter = st.sidebar.selectbox(
    "Education",
    ["All"] +
    sorted(
        df["Education"]
        .astype(str)
        .unique()
        .tolist()
    )
)

marital_filter = st.sidebar.selectbox(
    "Marital Status",
    ["All"] +
    sorted(
        df["Marital_Status"]
        .astype(str)
        .unique()
        .tolist()
    )
)

filtered_df = df.copy()

if education_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Education"]
        == education_filter
    ]

if marital_filter != "All":
    filtered_df = filtered_df[
        filtered_df["Marital_Status"]
        == marital_filter
    ]

# =====================================================
# DATASET EXPLORER
# =====================================================

st.header("📋 Customer Dataset Explorer")

st.dataframe(
    filtered_df.head(100),
    width="stretch"
)

# =====================================================
# CUSTOMER ANALYTICS
# =====================================================

st.header("👥 Customer Analytics")

col1, col2 = st.columns(2)

with col1:

    education_counts = (
        filtered_df["Education"]
        .value_counts()
    )

    fig_edu = px.bar(
        education_counts,
        title="Education Distribution",
        labels={
            "index": "Education",
            "value": "Count"
        }
    )

    st.plotly_chart(
        fig_edu,
        width="stretch"
    )

with col2:

    marital_counts = (
        filtered_df["Marital_Status"]
        .value_counts()
    )

    fig_marital = px.pie(
        values=marital_counts.values,
        names=marital_counts.index,
        title="Marital Status Distribution"
    )

    st.plotly_chart(
        fig_marital,
        width="stretch"
    )

# =====================================================
# INCOME ANALYTICS
# =====================================================

st.header("💰 Income Analytics")

fig_income = px.histogram(
    filtered_df,
    x="Income",
    nbins=30,
    title="Customer Income Distribution"
)

st.plotly_chart(
    fig_income,
    width="stretch"
)

# =====================================================
# SPENDING ANALYTICS
# =====================================================

st.header("🛒 Spending Intelligence")

spending_df = pd.DataFrame({
    "Category": [
        "Wine",
        "Fruits",
        "Meat",
        "Fish",
        "Sweets",
        "Gold"
    ],
    "Spending": [
        filtered_df["MntWines"].sum(),
        filtered_df["MntFruits"].sum(),
        filtered_df["MntMeatProducts"].sum(),
        filtered_df["MntFishProducts"].sum(),
        filtered_df["MntSweetProducts"].sum(),
        filtered_df["MntGoldProds"].sum()
    ]
})

fig_spend = px.bar(
    spending_df,
    x="Category",
    y="Spending",
    title="Spending by Product Category"
)

st.plotly_chart(
    fig_spend,
    width="stretch"
)

# =====================================================
# PURCHASE CHANNELS
# =====================================================

st.header("🏪 Purchase Channels")

purchase_df = pd.DataFrame({
    "Channel": [
        "Web",
        "Catalog",
        "Store"
    ],
    "Purchases": [
        filtered_df["NumWebPurchases"].sum(),
        filtered_df["NumCatalogPurchases"].sum(),
        filtered_df["NumStorePurchases"].sum()
    ]
})

fig_purchase = px.pie(
    purchase_df,
    values="Purchases",
    names="Channel",
    title="Purchase Channel Distribution"
)

st.plotly_chart(
    fig_purchase,
    width="stretch"
)

# =====================================================
# FEATURE IMPORTANCE
# =====================================================

st.header("🔥 Feature Importance")

if not feature_df.empty:

    fig_feat = px.bar(
        feature_df.head(15),
        x="importance",
        y="feature",
        orientation="h",
        title="Top Predictive Features"
    )

    st.plotly_chart(
        fig_feat,
        width="stretch"
    )

else:

    st.warning(
        "Feature importance file not found."
    )

# =====================================================
# ROC CURVE
# =====================================================

st.header("📈 ROC Curve")

if not roc_df.empty:

    fig_roc = go.Figure()

    fig_roc.add_trace(
        go.Scatter(
            x=roc_df["fpr"],
            y=roc_df["tpr"],
            mode="lines",
            name="ROC Curve"
        )
    )

    fig_roc.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="Random Classifier"
        )
    )

    fig_roc.update_layout(
        title=f"ROC Curve (AUC = {roc_auc:.4f})",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate"
    )

    st.plotly_chart(
        fig_roc,
        width="stretch"
    )

# =====================================================
# MODEL PERFORMANCE
# =====================================================

st.header("🎯 Model Performance")

c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy",
    f"{metrics.get('accuracy', 0):.4f}"
)

c2.metric(
    "Precision",
    f"{metrics.get('precision', 0):.4f}"
)

c3.metric(
    "Recall",
    f"{metrics.get('recall', 0):.4f}"
)

c4.metric(
    "F1 Score",
    f"{metrics.get('f1_score', 0):.4f}"
)

# =====================================================
# PREDICTION TOOL
# =====================================================

st.header("🤖 Customer Conversion Predictor")

education = st.selectbox(
    "Education",
    sorted(
        df["Education"]
        .unique()
    )
)

marital_status = st.selectbox(
    "Marital Status",
    sorted(
        df["Marital_Status"]
        .unique()
    )
)

income = st.number_input(
    "Income",
    value=50000
)

age = st.number_input(
    "Age",
    value=40
)

children = st.number_input(
    "Total Children",
    value=1
)

total_spending = st.number_input(
    "Total Spending",
    value=500
)

campaign_successes = st.number_input(
    "Previous Campaign Successes",
    value=0
)

if st.button(
    "Predict Conversion Probability"
):

    prediction_df = pd.DataFrame({

        "Year_Birth": [2026 - age],
        "Education": [education],
        "Marital_Status": [marital_status],
        "Income": [income],
        "Kidhome": [children],
        "Teenhome": [0],
        "Recency": [30],
        "MntWines": [100],
        "MntFruits": [50],
        "MntMeatProducts": [100],
        "MntFishProducts": [50],
        "MntSweetProducts": [50],
        "MntGoldProds": [50],
        "NumDealsPurchases": [2],
        "NumWebPurchases": [5],
        "NumCatalogPurchases": [2],
        "NumStorePurchases": [4],
        "NumWebVisitsMonth": [5],
        "AcceptedCmp3": [0],
        "AcceptedCmp4": [0],
        "AcceptedCmp5": [0],
        "AcceptedCmp1": [0],
        "AcceptedCmp2": [0],
        "Complain": [0],
        "Z_CostContact": [3],
        "Z_Revenue": [11],
        "Age": [age],
        "TotalChildren": [children],
        "TotalSpending": [total_spending],
        "TotalPurchases": [11],
        "CampaignSuccesses": [campaign_successes],
        "CustomerTenureDays": [365]
    })

    probability = (
        model.predict_proba(
            prediction_df
        )[0][1]
    )

    st.subheader(
        f"Conversion Probability: {probability:.2%}"
    )

    if probability >= 0.70:

        st.success(
            "HIGH CONVERSION LIKELIHOOD"
        )

    elif probability >= 0.40:

        st.warning(
            "MEDIUM CONVERSION LIKELIHOOD"
        )

    else:

        st.error(
            "LOW CONVERSION LIKELIHOOD"
        )

# =====================================================
# FOOTER
# =====================================================

st.divider()

st.markdown("""
### About MarketMind AI

MarketMind AI is a machine learning-powered marketing intelligence platform designed to predict customer campaign responses and provide actionable business insights.

**Applications**

- Customer Segmentation
- Campaign Optimization
- Lead Scoring
- Marketing Analytics
- Customer Intelligence
- Predictive Modeling

Built by **Amos Friday Ogbonna**
""")
