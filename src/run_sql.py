"""
Usage: python src/run_sql.py sql/01_churn_rate_by_segment.sql
Runs every statement in a .sql file against the project database and
prints each result as a table.
"""
import sqlite3
import sys
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB_PATH = ROOT / "data" / "processed" / "bank_churn.db"

def run_sql_file(sql_path: Path):
    conn = sqlite3.connect(DB_PATH)
    raw = sql_path.read_text()

    # strip comment-only lines, then split into individual statements
    lines = [l for l in raw.splitlines() if not l.strip().startswith("--")]
    statements = [s.strip() for s in "\n".join(lines).split(";") if s.strip()]

    for i, stmt in enumerate(statements):
        print(f"\n--- Query {i+1} ---")
        df = pd.read_sql_query(stmt, conn)
        print(df.to_string(index=False))

    conn.close()

if __name__ == "__main__":
    run_sql_file(Path(sys.argv[1]))