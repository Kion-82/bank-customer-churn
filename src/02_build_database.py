import sqlite3
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = ROOT / "data" / "raw" / "Customer-Churn-Records.csv"
DB_PATH = ROOT / "data" / "processed" / "bank_churn.db"

df = pd.read_csv(RAW_CSV)

# Rename to SQL-friendly names
df = df.rename(columns={
    "CustomerId": "customer_id",
    "CreditScore": "credit_score",
    "Geography": "geography",
    "Gender": "gender",
    "Age": "age",
    "Tenure": "tenure",
    "HasCrCard": "has_cr_card",
    "IsActiveMember": "is_active_member",
    "EstimatedSalary": "estimated_salary",
    "Balance": "balance",
    "NumOfProducts": "num_of_products",
    "Card Type": "card_type",
    "Point Earned": "point_earned",
    "Satisfaction Score": "satisfaction_score",
    "Exited": "exited",
    "Complain": "complain",
})

customers = df[["customer_id", "credit_score", "geography", "gender", "age",
                "tenure", "has_cr_card", "is_active_member", "estimated_salary"]]

accounts = df[["customer_id", "balance", "num_of_products", "card_type",
               "point_earned", "satisfaction_score"]]

churn_status = df[["customer_id", "exited", "complain"]]

DB_PATH.parent.mkdir(parents=True, exist_ok=True)
conn = sqlite3.connect(DB_PATH)

customers.to_sql("customers", conn, index=False, if_exists="replace")
accounts.to_sql("accounts", conn, index=False, if_exists="replace")
churn_status.to_sql("churn_status", conn, index=False, if_exists="replace")

# Index the foreign keys so later JOINs are fast
conn.execute("CREATE INDEX IF NOT EXISTS idx_accounts_customer ON accounts(customer_id)")
conn.execute("CREATE INDEX IF NOT EXISTS idx_churn_customer ON churn_status(customer_id)")
conn.commit()

for tbl in ("customers", "accounts", "churn_status"):
    n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
    print(f"{tbl}: {n} rows")

conn.close()
print(f"\nSQLite DB written to {DB_PATH}")