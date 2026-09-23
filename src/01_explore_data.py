import pandas as pd
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
df = pd.read_csv(ROOT / "data" / "raw" / "Customer-Churn-Records.csv")

print("Basic info")
print("Shape", df.shape)
print("\nColumn type:\n", df.dtypes)

print("\nMissing values")
print(df.isnull().sum())

print("\nTarget distribution (Exited)")
print(df["Exited"].value_counts())
print(df["Exited"].value_counts(normalize=True))

print("\nNumeric summary")
print(df.describe())

print("\nFirst 5 rows")
print(df.head())

print("\nCorrelation with target")
print(df.corr(numeric_only=True)["Exited"].sort_values(ascending=False))