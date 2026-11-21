"""DuckDB helper: exposes the raw CSVs as SQL views and runs the files in sql/."""

from pathlib import Path

import duckdb

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
SQL = ROOT / "sql"

VIEWS = {
    "customers": "customers.csv",
    "behavior_raw": "daily_user_behavior.csv",
    "orders": "orders.csv",
    "campaigns": "campaigns.csv",
    "ad_performance": "ad_performance_daily.csv",
    "coupon_program": "coupon_program_assignment.csv",
}


def connect() -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    for view, file in VIEWS.items():
        con.execute(f"CREATE VIEW {view} AS SELECT * FROM read_csv_auto('{(RAW / file).as_posix()}')")
    # cleaned behaviour table: drop exact duplicate rows produced by logging retries
    con.execute("CREATE VIEW behavior AS SELECT DISTINCT * FROM behavior_raw")
    return con


def run(con, name: str, **params):
    """Run sql/<name>.sql (the last statement's result is returned as a DataFrame)."""
    text = (SQL / f"{name}.sql").read_text()
    for k, v in params.items():
        text = text.replace("{{" + k + "}}", str(v))
    statements = [s for s in text.split(";") if s.strip() and not all(
        line.strip().startswith("--") or not line.strip() for line in s.splitlines())]
    for s in statements[:-1]:
        con.execute(s)
    return con.execute(statements[-1]).df()
