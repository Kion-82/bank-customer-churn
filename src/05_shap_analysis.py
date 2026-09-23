"""
SHAP interpretability for the churn model.

Random Forest is a tree ensemble -> shap.TreeExplainer (exact, fast).
KernelExplainer (slow, model-agnostic approximation) would only be needed
for a model type SHAP has no dedicated algorithm for, e.g. an SVM.

best_model.joblib is a full sklearn Pipeline (prep -> clf), but
TreeExplainer needs the raw trees and data in the exact transformed shape
they were trained on. So we split the pipeline apart, transform X_test
through `prep` ourselves, and explain `clf` on that transformed matrix.
"""
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = ROOT / "models"
FIG_DIR = ROOT / "reports" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

pipe = joblib.load(MODEL_DIR / "best_model.joblib")
X_test = pd.read_pickle(ROOT / "data" / "processed" / "X_test.pkl")

prep = pipe.named_steps["prep"]
clf = pipe.named_steps["clf"]

X_test_transformed = prep.transform(X_test)
feature_names = prep.get_feature_names_out()
X_test_df = pd.DataFrame(X_test_transformed, columns=feature_names, index=X_test.index)

explainer = shap.TreeExplainer(clf)
shap_values = explainer.shap_values(X_test_df)

# Depending on the shap/sklearn version, binary-classifier output shape varies.
# Normalize to "SHAP values for the churn (class=1) prediction".
if isinstance(shap_values, list):
    shap_values_churn = shap_values[1]
elif shap_values.ndim == 3:
    shap_values_churn = shap_values[:, :, 1]
else:
    shap_values_churn = shap_values

expected_value = explainer.expected_value
if isinstance(expected_value, (list, np.ndarray)):
    expected_value = expected_value[1]

# --- 1. Global importance (bar) ---
plt.figure()
shap.summary_plot(shap_values_churn, X_test_df, plot_type="bar", show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_bar.png", dpi=150)
plt.close()

# --- 2. Global importance + direction (beeswarm) ---
plt.figure()
shap.summary_plot(shap_values_churn, X_test_df, show=False)
plt.tight_layout()
plt.savefig(FIG_DIR / "shap_summary.png", dpi=150)
plt.close()

# --- 3. Individual customer waterfalls ---
proba = pipe.predict_proba(X_test)[:, 1]
highest_idx = int(np.argmax(proba))
lowest_idx = int(np.argmin(proba))

print(f"Highest risk: row {X_test.index[highest_idx]}, prob={proba[highest_idx]:.2%}")
print(f"Lowest risk:  row {X_test.index[lowest_idx]}, prob={proba[lowest_idx]:.2%}")

explanation = shap.Explanation(
    values=shap_values_churn,
    base_values=np.full(len(X_test_df), expected_value),
    data=X_test_df.values,
    feature_names=list(feature_names),
)

for label, idx in [("highest_risk", highest_idx), ("lowest_risk", lowest_idx)]:
    plt.figure()
    shap.plots.waterfall(explanation[idx], show=False)
    plt.tight_layout()
    plt.savefig(FIG_DIR / f"shap_waterfall_{label}.png", dpi=150, bbox_inches="tight")
    plt.close()

print("Saved SHAP figures to reports/figs/")