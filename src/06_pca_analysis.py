"""
PCA analysis of the numeric features.

  Part A - redundancy / collinearity diagnostics
           (correlation matrix, VIF, PCA scree plot + loadings)
  Part B - ablation: the same models with vs. without PCA on the numeric block
           (+ a sweep over the number of components)

Everything is fit on the training split only (X_train.pkl), consistent with the
rest of the pipeline. X_test is used only for the final holdout metrics.

Outputs: reports/figs/pca_*.png, reports/pca_comparison.csv, reports/pca_sweep.csv
"""
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from features import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402

FIG_DIR = ROOT / "reports" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)
warnings.filterwarnings("ignore")

RANDOM_STATE = 42
PALETTE = sns.color_palette("colorblind")

# PCA is meant for continuous/count variables. 0/1 flags are kept out of the PCA
# block and passed straight through to the model.
BINARY_FLAGS = ["has_cr_card", "is_active_member"]
PCA_FEATURES = [c for c in NUMERIC_FEATURES if c not in BINARY_FLAGS]


# ------------------------------------------------------------------ Part A
def compute_vif(df: pd.DataFrame) -> pd.Series:
    """VIF_i = 1 / (1 - R^2_i), where R^2_i comes from regressing feature i on all the others."""
    out = {}
    for col in df.columns:
        others = df.drop(columns=col)
        r2 = LinearRegression().fit(others, df[col]).score(others, df[col])
        out[col] = 1.0 / (1.0 - r2) if r2 < 1 else np.inf
    return pd.Series(out).sort_values(ascending=False)


def part_a_diagnostics(X_train: pd.DataFrame) -> None:
    num = X_train[PCA_FEATURES]

    # A1. correlation matrix
    corr = num.corr()
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, vmin=-1, vmax=1,
                square=True, cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Correlation of numeric features (training set)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_corr_heatmap.png", dpi=150)
    plt.close(fig)

    pairs = (corr.where(np.triu(np.ones(corr.shape, dtype=bool), k=1))
                 .stack().rename("corr").to_frame())
    pairs["abs_corr"] = pairs["corr"].abs()
    print("\n=== Top 5 correlated feature pairs ===")
    print(pairs.sort_values("abs_corr", ascending=False).head(5)["corr"].round(3).to_string())

    # A2. VIF
    print("\n=== VIF (rule of thumb: >5 moderate, >10 severe collinearity) ===")
    print(compute_vif(num).round(2).to_string())

    # A3. PCA on standardized features
    Z = StandardScaler().fit_transform(num)
    pca = PCA(random_state=RANDOM_STATE).fit(Z)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n95 = int(np.searchsorted(cum, 0.95) + 1)
    print(f"\n=== PCA: {len(PCA_FEATURES)} numeric features -> {n95} components reach 95% variance ===")
    print(pd.DataFrame({"explained": evr, "cumulative": cum},
                       index=[f"PC{i + 1}" for i in range(len(evr))]).round(3).to_string())

    x = np.arange(1, len(evr) + 1)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.bar(x, evr, color=PALETTE[0], label="Per component")
    ax.plot(x, cum, marker="o", color=PALETTE[1], label="Cumulative")
    ax.axhline(0.95, linestyle="--", color="grey", linewidth=1)
    ax.set_xticks(x)
    ax.set_xlabel("Principal component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("PCA scree plot (standardized numeric features)")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_scree.png", dpi=150)
    plt.close(fig)

    # A4. loadings of the first 4 components: which original features form each PC
    k = min(4, len(evr))
    loadings = pd.DataFrame(pca.components_[:k].T, index=PCA_FEATURES,
                            columns=[f"PC{i + 1}" for i in range(k)])
    fig, ax = plt.subplots(figsize=(6.5, 5))
    sns.heatmap(loadings, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("PCA loadings (first components)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_loadings.png", dpi=150)
    plt.close(fig)
    print("\n=== Loadings ===")
    print(loadings.round(2).to_string())


# ------------------------------------------------------------------ Part B
def make_preprocessor(use_pca: bool, scale_numeric: bool, n_components=0.95):
    if use_pca:
        num_branch = Pipeline([
            ("scale", StandardScaler()),
            ("pca", PCA(n_components=n_components, random_state=RANDOM_STATE)),
        ])
    else:
        num_branch = StandardScaler() if scale_numeric else "passthrough"
    return ColumnTransformer([
        ("num", num_branch, PCA_FEATURES),
        ("flags", "passthrough", BINARY_FLAGS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def get_models(spw: float):
    # (estimator, scale numeric features when PCA is NOT used) -- same setup as 04_modeling.py
    return {
        "Logistic Regression": (LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE), True),
        "Random Forest": (RandomForestClassifier(
            n_estimators=300, max_depth=8, class_weight="balanced",
            random_state=RANDOM_STATE, n_jobs=-1), False),
        "XGBoost": (xgb.XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.08, eval_metric="logloss",
            scale_pos_weight=spw, random_state=RANDOM_STATE), False),
    }


def part_b_ablation(X_train, X_test, y_train, y_test, cv):
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    rows = []
    for name, (clf, scale_when_raw) in get_models(spw).items():
        for use_pca in (False, True):
            pipe = Pipeline([("prep", make_preprocessor(use_pca, scale_when_raw)), ("clf", clf)])
            cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")
            pipe.fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            n_comp = (pipe.named_steps["prep"].named_transformers_["num"]
                      .named_steps["pca"].n_components_ if use_pca else len(PCA_FEATURES))
            rows.append({
                "model": name,
                "numeric_block": f"PCA (95% var, {n_comp} comps)" if use_pca else "raw features",
                "cv_roc_auc": cv_auc.mean(),
                "cv_auc_std": cv_auc.std(),
                "test_roc_auc": roc_auc_score(y_test, proba),
                "test_precision": precision_score(y_test, pred),
                "test_recall": recall_score(y_test, pred),
                "test_f1": f1_score(y_test, pred),
            })
            print(f"done: {name} | {'PCA' if use_pca else 'raw'}")
    res = pd.DataFrame(rows)
    res.to_csv(ROOT / "reports" / "pca_comparison.csv", index=False)
    print("\n=== With vs. without PCA ===")
    print(res.round(4).to_string(index=False))

    # grouped bar chart of test ROC-AUC
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    models = res["model"].unique()
    xpos = np.arange(len(models))
    w = 0.36
    raw = res[res["numeric_block"] == "raw features"].set_index("model").loc[models]
    pc = res[res["numeric_block"] != "raw features"].set_index("model").loc[models]
    ax.bar(xpos - w / 2, raw["test_roc_auc"], w, color=PALETTE[0], label="Raw numeric features")
    ax.bar(xpos + w / 2, pc["test_roc_auc"], w, color=PALETTE[1], label="PCA (95% variance)")
    ax.set_xticks(xpos)
    ax.set_xticklabels(models)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Test ROC-AUC")
    ax.set_title("Effect of PCA on the numeric block")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_model_comparison.png", dpi=150)
    plt.close(fig)
    return res


def part_b_sweep(X_train, y_train, cv):
    """CV ROC-AUC as a function of the number of retained components."""
    spw = (y_train == 0).sum() / (y_train == 1).sum()
    models = get_models(spw)
    sweep_models = ["Logistic Regression", "Random Forest"]
    rows = []
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for i, name in enumerate(sweep_models):
        clf, scale_when_raw = models[name]
        base = cross_val_score(
            Pipeline([("prep", make_preprocessor(False, scale_when_raw)), ("clf", clf)]),
            X_train, y_train, cv=cv, scoring="roc_auc").mean()
        ks = list(range(1, len(PCA_FEATURES) + 1))
        scores = []
        for k in ks:
            pipe = Pipeline([("prep", make_preprocessor(True, scale_when_raw, n_components=k)),
                             ("clf", clf)])
            scores.append(cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc").mean())
            rows.append({"model": name, "n_components": k, "cv_roc_auc": scores[-1],
                         "baseline_raw_cv_roc_auc": base})
        ax.plot(ks, scores, marker="o", color=PALETTE[i], label=f"{name} + PCA(k)")
        ax.axhline(base, linestyle="--", color=PALETTE[i], alpha=0.7,
                   label=f"{name}, raw features")
        print(f"sweep done: {name}")
    ax.set_xlabel("Number of principal components kept")
    ax.set_ylabel("5-fold CV ROC-AUC")
    ax.set_title("Performance vs. number of PCA components")
    ax.set_xticks(range(1, len(PCA_FEATURES) + 1))
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "pca_sweep.png", dpi=150)
    plt.close(fig)
    pd.DataFrame(rows).to_csv(ROOT / "reports" / "pca_sweep.csv", index=False)


def main():
    proc = ROOT / "data" / "processed"
    X_train = pd.read_pickle(proc / "X_train.pkl")
    X_test = pd.read_pickle(proc / "X_test.pkl")
    y_train = pd.read_pickle(proc / "y_train.pkl")
    y_test = pd.read_pickle(proc / "y_test.pkl")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    part_a_diagnostics(X_train)
    part_b_ablation(X_train, X_test, y_train, y_test, cv)
    part_b_sweep(X_train, y_train, cv)
    print("\nSaved figures to reports/figs/ and tables to reports/")


if __name__ == "__main__":
    main()