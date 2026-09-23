import sqlite3
import sys
import warnings
from pathlib import Path

import joblib
import lightgbm as lgb
import pandas as pd
import xgboost as xgb
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, make_model_frame

DB_PATH = ROOT.parent / "data" / "processed" / "bank_churn.db"
MODEL_DIR = ROOT.parent / "models"
MODEL_DIR.mkdir(exist_ok=True)

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(open(ROOT.parent / "sql" / "02_full_customer_view.sql").read(), conn)
conn.close()

X, y = make_model_frame(df)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Tree models don't need scaling (they split on raw thresholds, invariant to
# monotonic rescaling) -- skipping it here means our later SHAP explanations
# show real values (age=46) instead of z-scores (age=0.67). Logistic
# Regression still needs scaling since it's distance/scale-sensitive.
preprocessor_scaled = ColumnTransformer([
    ("num", StandardScaler(), NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])
preprocessor_raw = ColumnTransformer([
    ("num", "passthrough", NUMERIC_FEATURES),
    ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
])

spw = (y_train == 0).sum() / (y_train == 1).sum()  # for XGBoost's scale_pos_weight

models = {
    "Logistic Regression": (preprocessor_scaled,
        LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)),
    "Random Forest": (preprocessor_raw,
        RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                random_state=42, n_jobs=-1)),
    "XGBoost": (preprocessor_raw,
        xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                           scale_pos_weight=spw, eval_metric="logloss", random_state=42)),
    "LightGBM": (preprocessor_raw,
        lgb.LGBMClassifier(n_estimators=300, max_depth=6, learning_rate=0.05,
                            class_weight="balanced", random_state=42, verbosity=-1)),
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scoring = {"roc_auc": "roc_auc", "f1": "f1", "precision": "precision", "recall": "recall"}

results = []
for name, (prep, clf) in models.items():
    pipe = Pipeline([("prep", prep), ("clf", clf)])
    cv_res = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)

    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]

    results.append({
        "model": name,
        "cv_roc_auc": cv_res["test_roc_auc"].mean(),
        "test_precision": precision_score(y_test, y_pred),
        "test_recall": recall_score(y_test, y_pred),
        "test_f1": f1_score(y_test, y_pred),
        "test_roc_auc": roc_auc_score(y_test, y_proba),
    })
    print(f"\n=== {name} ===")
    print(classification_report(y_test, y_pred, target_names=["Stayed", "Churned"]))

# --- Compare SMOTE (oversampling) against class_weight balancing, on XGBoost ---
smote_pipe = ImbPipeline([
    ("prep", preprocessor_raw),
    ("smote", SMOTE(random_state=42)),
    ("clf", xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.08,
                               eval_metric="logloss", random_state=42)),
])
smote_pipe.fit(X_train, y_train)
y_pred_s = smote_pipe.predict(X_test)
y_proba_s = smote_pipe.predict_proba(X_test)[:, 1]
results.append({
    "model": "XGBoost + SMOTE",
    "cv_roc_auc": None,
    "test_precision": precision_score(y_test, y_pred_s),
    "test_recall": recall_score(y_test, y_pred_s),
    "test_f1": f1_score(y_test, y_pred_s),
    "test_roc_auc": roc_auc_score(y_test, y_proba_s),
})
print("\n=== XGBoost + SMOTE ===")
print(classification_report(y_test, y_pred_s, target_names=["Stayed", "Churned"]))

results_df = pd.DataFrame(results).sort_values("test_roc_auc", ascending=False)
print("\n", results_df.round(4).to_string(index=False))
results_df.to_csv(ROOT.parent / "reports" / "model_comparison.csv", index=False)

# save the best model (by test ROC-AUC) for later use in SHAP + dashboard
best_name = results_df.iloc[0]["model"]
best_pipe = smote_pipe if best_name == "XGBoost + SMOTE" else Pipeline(
    [("prep", models[best_name][0]), ("clf", models[best_name][1])]
).fit(X_train, y_train)
joblib.dump(best_pipe, MODEL_DIR / "best_model.joblib")
print(f"\nBest model: {best_name} -> saved to models/best_model.joblib")

# save train/test split for reuse in the SHAP step
X_train.to_pickle(ROOT.parent / "data" / "processed" / "X_train.pkl")
X_test.to_pickle(ROOT.parent / "data" / "processed" / "X_test.pkl")
y_train.to_pickle(ROOT.parent / "data" / "processed" / "y_train.pkl")
y_test.to_pickle(ROOT.parent / "data" / "processed" / "y_test.pkl")