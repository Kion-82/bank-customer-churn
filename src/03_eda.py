import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "bank_churn.db"
FIG_DIR = ROOT / "reports" / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# use a colorblind-safe palette instead of default matplotlib colors
palette = sns.color_palette("colorblind")
sns.set_theme(style="whitegrid")

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql_query(open(ROOT / "sql" / "02_full_customer_view.sql").read(), conn)
conn.close()

print("Overall churn rate: {:.2%}".format(df["exited"].mean()))

# --- Chart 1: churn class distribution (shows the class imbalance) ---
fig, ax = plt.subplots(figsize=(4.5, 4))
counts = df["exited"].value_counts().sort_index()
ax.bar(["Stayed", "Churned"], counts.values, color=[palette[0], palette[3]])
for i, v in enumerate(counts.values):
    ax.text(i, v + 50, f"{v}\n({v/len(df):.1%})", ha="center", fontsize=10)
ax.set_title("Churn Class Distribution")
fig.tight_layout()
fig.savefig(FIG_DIR / "01_churn_distribution.png", dpi=150)
plt.close(fig)

# --- Chart 2: churn rate by geography and number of products ---
fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))

geo = df.groupby("geography")["exited"].mean().sort_values(ascending=False) * 100
axes[0].bar(geo.index, geo.values, color=palette[4])
axes[0].set_title("Churn Rate by Country")
axes[0].set_ylabel("Churn rate (%)")

prod = df.groupby("num_of_products")["exited"].mean().sort_index() * 100
axes[1].bar(prod.index.astype(str), prod.values, color=palette[2])
axes[1].set_title("Churn Rate by Number of Products")
axes[1].set_xlabel("Number of products")

fig.tight_layout()
fig.savefig(FIG_DIR / "02_churn_by_geo_products.png", dpi=150)
plt.close(fig)

# --- Chart 3: correlation heatmap (numeric features vs. target) ---
num_cols = ["credit_score", "age", "tenure", "balance", "num_of_products",
            "has_cr_card", "is_active_member", "estimated_salary",
            "point_earned", "satisfaction_score", "complain", "exited"]
corr = df[num_cols].corr()

fig, ax = plt.subplots(figsize=(8, 6.5))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax)
ax.set_title("Correlation Heatmap")
fig.tight_layout()
fig.savefig(FIG_DIR / "03_correlation_heatmap.png", dpi=150)
plt.close(fig)

print(f"Figures saved to {FIG_DIR}")