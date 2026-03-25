"""
ingest.py
---------
Reads raw CSVs, cleans the data, and loads it into a normalized
SQLite database with the following relational schema:

    regions (region_id PK, province_code, province_name, region_type)
    sectors (sector_id PK, sector_code, sector_name, sector_category)
    employment (employment_id PK, region_id FK, sector_id FK, year,
                employed_thousands, avg_annual_salary_cad,
                unemployment_rate_pct, part_time_pct)
    demographics (demo_id PK, region_id FK, year, pct_female,
                  pct_indigenous, pct_visible_minority, median_age,
                  pct_union_coverage)
"""

import os
import sqlite3
import pandas as pd

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "employment.db")

PROVINCE_REGIONS = {
    "AB": "Prairie",
    "BC": "Pacific",
    "MB": "Prairie",
    "NB": "Atlantic",
    "NL": "Atlantic",
    "NS": "Atlantic",
    "NT": "North",
    "NU": "North",
    "ON": "Central",
    "PE": "Atlantic",
    "QC": "Central",
    "SK": "Prairie",
    "YT": "North",
}

CREATE_REGIONS = """
CREATE TABLE IF NOT EXISTS regions (
    region_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    province_code TEXT    NOT NULL UNIQUE,
    province_name TEXT    NOT NULL,
    region_type   TEXT    NOT NULL   -- Atlantic / Central / Prairie / Pacific / North
);
"""

CREATE_SECTORS = """
CREATE TABLE IF NOT EXISTS sectors (
    sector_id       INTEGER PRIMARY KEY AUTOINCREMENT,
    sector_code     TEXT    NOT NULL UNIQUE,
    sector_name     TEXT    NOT NULL,
    sector_category TEXT    NOT NULL
);
"""

CREATE_EMPLOYMENT = """
CREATE TABLE IF NOT EXISTS employment (
    employment_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id             INTEGER NOT NULL REFERENCES regions(region_id),
    sector_id             INTEGER NOT NULL REFERENCES sectors(sector_id),
    year                  INTEGER NOT NULL,
    employed_thousands    REAL    NOT NULL,
    avg_annual_salary_cad REAL    NOT NULL,
    unemployment_rate_pct REAL    NOT NULL,
    part_time_pct         REAL    NOT NULL,
    UNIQUE(region_id, sector_id, year)
);
"""

CREATE_DEMOGRAPHICS = """
CREATE TABLE IF NOT EXISTS demographics (
    demo_id              INTEGER PRIMARY KEY AUTOINCREMENT,
    region_id            INTEGER NOT NULL REFERENCES regions(region_id),
    year                 INTEGER NOT NULL,
    pct_female           REAL    NOT NULL,
    pct_indigenous       REAL    NOT NULL,
    pct_visible_minority REAL    NOT NULL,
    median_age           REAL    NOT NULL,
    pct_union_coverage   REAL    NOT NULL,
    UNIQUE(region_id, year)
);
"""


def clean_employment(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna(subset=["province_code", "sector_code", "year", "employed_thousands"])
    df["employed_thousands"] = pd.to_numeric(df["employed_thousands"], errors="coerce")
    df["avg_annual_salary_cad"] = pd.to_numeric(df["avg_annual_salary_cad"], errors="coerce")
    df["unemployment_rate_pct"] = pd.to_numeric(df["unemployment_rate_pct"], errors="coerce").clip(0, 25)
    df["part_time_pct"] = pd.to_numeric(df["part_time_pct"], errors="coerce").clip(0, 100)
    df["year"] = df["year"].astype(int)
    df = df[df["employed_thousands"] > 0]
    return df.reset_index(drop=True)


def clean_demographics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    df = df.dropna()
    for col in ["pct_female", "pct_indigenous", "pct_visible_minority", "pct_union_coverage"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").clip(0, 100)
    df["median_age"] = pd.to_numeric(df["median_age"], errors="coerce").clip(18, 70)
    df["year"] = df["year"].astype(int)
    return df.reset_index(drop=True)


def build_schema(conn: sqlite3.Connection):
    cur = conn.cursor()
    cur.executescript(CREATE_REGIONS + CREATE_SECTORS + CREATE_EMPLOYMENT + CREATE_DEMOGRAPHICS)
    conn.commit()
    print("  Schema created.")


def load_regions(conn: sqlite3.Connection, emp_df: pd.DataFrame):
    cur = conn.cursor()
    provinces = emp_df[["province_code", "province_name"]].drop_duplicates()
    for _, row in provinces.iterrows():
        region_type = PROVINCE_REGIONS.get(row["province_code"], "Other")
        cur.execute(
            "INSERT OR IGNORE INTO regions (province_code, province_name, region_type) VALUES (?, ?, ?)",
            (row["province_code"], row["province_name"], region_type),
        )
    conn.commit()
    print(f"  Loaded {cur.execute('SELECT COUNT(*) FROM regions').fetchone()[0]} regions.")


def load_sectors(conn: sqlite3.Connection, emp_df: pd.DataFrame):
    cur = conn.cursor()
    sectors = emp_df[["sector_code", "sector_name", "sector_category"]].drop_duplicates()
    for _, row in sectors.iterrows():
        cur.execute(
            "INSERT OR IGNORE INTO sectors (sector_code, sector_name, sector_category) VALUES (?, ?, ?)",
            (row["sector_code"], row["sector_name"], row["sector_category"]),
        )
    conn.commit()
    print(f"  Loaded {cur.execute('SELECT COUNT(*) FROM sectors').fetchone()[0]} sectors.")


def load_employment(conn: sqlite3.Connection, emp_df: pd.DataFrame):
    cur = conn.cursor()
    region_map = {r[0]: r[1] for r in cur.execute("SELECT province_code, region_id FROM regions")}
    sector_map = {r[0]: r[1] for r in cur.execute("SELECT sector_code, sector_id FROM sectors")}

    rows = []
    for _, row in emp_df.iterrows():
        rid = region_map.get(row["province_code"])
        sid = sector_map.get(row["sector_code"])
        if rid and sid:
            rows.append((
                rid, sid,
                int(row["year"]),
                float(row["employed_thousands"]),
                float(row["avg_annual_salary_cad"]),
                float(row["unemployment_rate_pct"]),
                float(row["part_time_pct"]),
            ))

    cur.executemany(
        """INSERT OR IGNORE INTO employment
           (region_id, sector_id, year, employed_thousands,
            avg_annual_salary_cad, unemployment_rate_pct, part_time_pct)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM employment").fetchone()[0]
    print(f"  Loaded {count:,} employment records.")


def load_demographics(conn: sqlite3.Connection, demo_df: pd.DataFrame):
    cur = conn.cursor()
    region_map = {r[0]: r[1] for r in cur.execute("SELECT province_code, region_id FROM regions")}

    rows = []
    for _, row in demo_df.iterrows():
        rid = region_map.get(row["province_code"])
        if rid:
            rows.append((
                rid,
                int(row["year"]),
                float(row["pct_female"]),
                float(row["pct_indigenous"]),
                float(row["pct_visible_minority"]),
                float(row["median_age"]),
                float(row["pct_union_coverage"]),
            ))

    cur.executemany(
        """INSERT OR IGNORE INTO demographics
           (region_id, year, pct_female, pct_indigenous,
            pct_visible_minority, median_age, pct_union_coverage)
           VALUES (?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    count = cur.execute("SELECT COUNT(*) FROM demographics").fetchone()[0]
    print(f"  Loaded {count:,} demographic records.")


def main():
    emp_csv = os.path.join(RAW_DIR, "statcan_employment.csv")
    demo_csv = os.path.join(RAW_DIR, "statcan_demographics.csv")

    print("Reading raw CSVs...")
    emp_raw = pd.read_csv(emp_csv)
    demo_raw = pd.read_csv(demo_csv)
    print(f"  Employment: {len(emp_raw):,} rows | Demographics: {len(demo_raw):,} rows")

    print("Cleaning data...")
    emp_df = clean_employment(emp_raw)
    demo_df = clean_demographics(demo_raw)
    print(f"  After cleaning — Employment: {len(emp_df):,} | Demographics: {len(demo_df):,}")

    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    if os.path.exists(DB_PATH):
        os.remove(DB_PATH)

    print(f"Building SQLite database at {DB_PATH} ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")

    build_schema(conn)
    load_regions(conn, emp_df)
    load_sectors(conn, emp_df)
    load_employment(conn, emp_df)
    load_demographics(conn, demo_df)
    conn.close()
    print("Ingestion complete.")


if __name__ == "__main__":
    main()
