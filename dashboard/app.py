
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import sqlite3
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import add_engineered_features, NUMERIC_FEATURES, CATEGORICAL_FEATURES  # noqa: E402

DB_PATH = ROOT / "data" / "processed" / "bank_churn.db"
MODEL_PATH = ROOT / "models" / "best_model.joblib"

st.set_page_config(page_title="Bank Customer Churn", layout="wide")


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_customer_view():
    conn = sqlite3.connect(DB_PATH)
    query = (ROOT / "sql" / "02_full_customer_view.sql").read_text()
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


pipe = load_model()
df = load_customer_view()

st.title("Bank Customer Churn — Risk Dashboard")

tab_overview, tab_predict = st.tabs(["Overview", "Predict Churn Risk"])

# ---------------- Overview ----------------
with tab_overview:
    st.metric("Overall churn rate", f"{df['exited'].mean():.1%}")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Churn rate by geography")
        st.bar_chart(df.groupby("geography")["exited"].mean().sort_values(ascending=False))
    with col2:
        st.subheader("Churn rate by number of products")
        st.bar_chart(df.groupby("num_of_products")["exited"].mean().sort_values(ascending=False))

    st.subheader("High-value / high-risk segments")
    med_balance = df["balance"].median()
    med_salary = df["estimated_salary"].median()
    df["value_segment"] = np.where(
        (df["balance"] > med_balance) & (df["estimated_salary"] > med_salary),
        "high_value", "standard_value",
    )
    df["risk_segment"] = np.where(
        (df["is_active_member"] == 0) | (df["num_of_products"] == 1),
        "high_risk", "lower_risk",
    )
    seg = df.groupby(["value_segment", "risk_segment"])["exited"].agg(n_customers="count", churn_rate="mean")
    st.dataframe(seg.style.format({"churn_rate": "{:.1%}"}))

# ---------------- Predict ----------------
with tab_predict:
    st.subheader("Enter a customer's details")
    c1, c2, c3 = st.columns(3)
    with c1:
        credit_score = st.number_input("Credit score", 300, 900, 650)
        age = st.number_input("Age", 18, 100, 40)
        tenure = st.number_input("Tenure (years)", 0, 15, 5)
    with c2:
        balance = st.number_input("Balance", 0.0, 300000.0, 50000.0, step=1000.0)
        estimated_salary = st.number_input("Estimated salary", 0.0, 300000.0, 100000.0, step=1000.0)
        num_of_products = st.selectbox("Number of products", [1, 2, 3, 4])
    with c3:
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        card_type = st.selectbox("Card type", ["SILVER", "GOLD", "PLATINUM", "DIAMOND"])

    c4, c5, c6 = st.columns(3)
    with c4:
        has_cr_card = st.checkbox("Has credit card", value=True)
    with c5:
        is_active_member = st.checkbox("Active member", value=True)
    with c6:
        point_earned = st.number_input("Loyalty points", 0, 1000, 500)
    satisfaction_score = st.slider("Satisfaction score", 1, 5, 3)

    if st.button("Predict churn risk"):
        row = pd.DataFrame([{
            "credit_score": credit_score, "age": age, "tenure": tenure,
            "balance": balance, "num_of_products": num_of_products,
            "has_cr_card": int(has_cr_card), "is_active_member": int(is_active_member),
            "estimated_salary": estimated_salary, "point_earned": point_earned,
            "satisfaction_score": satisfaction_score,
            "geography": geography, "gender": gender, "card_type": card_type,
        }])
        row_fe = add_engineered_features(row)
        X_input = row_fe[NUMERIC_FEATURES + CATEGORICAL_FEATURES]

        proba = pipe.predict_proba(X_input)[0, 1]
        st.metric("Predicted churn probability", f"{proba:.1%}")

        prep = pipe.named_steps["prep"]
        clf = pipe.named_steps["clf"]
        X_transformed = prep.transform(X_input)
        feature_names = prep.get_feature_names_out()

        explainer = shap.TreeExplainer(clf)
        shap_values = explainer.shap_values(X_transformed)
        if isinstance(shap_values, list):
            sv = shap_values[1][0]
        elif np.ndim(shap_values) == 3:
            sv = shap_values[0, :, 1]
        else:
            sv = shap_values[0]

        expected_value = explainer.expected_value
        if isinstance(expected_value, (list, np.ndarray)):
            expected_value = expected_value[1]

        explanation = shap.Explanation(
            values=sv, base_values=expected_value,
            data=X_transformed[0], feature_names=list(feature_names),
        )
        st.subheader("Why this prediction?")
        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation, show=False)
        st.pyplot(fig)