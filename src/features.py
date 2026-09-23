"""
Feature engineering shared by modeling.py and the dashboard.
"""
import pandas as pd

LEAKAGE_COLS = ["complain"]
ID_COLS = ["customer_id", "surname"]
TARGET = "exited"

NUMERIC_FEATURES = [
    "credit_score", "age", "tenure", "balance", "num_of_products",
    "has_cr_card", "is_active_member", "estimated_salary",
    "point_earned", "satisfaction_score",
    "balance_salary_ratio", "products_activity_score",
]
CATEGORICAL_FEATURES = ["geography", "gender", "card_type"]


def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Floor the denominator: a few rows have an unrealistically tiny
    # estimated_salary, which would blow the ratio up to absurd values.
    safe_salary = df["estimated_salary"].clip(lower=1000.0)
    df["balance_salary_ratio"] = (df["balance"] / safe_salary).clip(upper=10.0)
    df["products_activity_score"] = df["num_of_products"] * df["is_active_member"]
    return df


def make_model_frame(df: pd.DataFrame):
    df = add_engineered_features(df)
    drop_cols = [c for c in ID_COLS + LEAKAGE_COLS if c in df.columns]
    y = df[TARGET]
    X = df.drop(columns=[TARGET] + drop_cols)
    return X, y