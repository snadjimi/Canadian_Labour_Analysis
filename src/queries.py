"""
queries.py
----------
Runs analytical SQL queries against the employment.db database.
Uses JOINs, aggregations, and window functions to surface employment
trends by sector, region, and year.

Returns a dict of named DataFrames consumed by visualize.py.
"""

import os
import sqlite3
import pandas as pd

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "db", "employment.db")


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


# ---------------------------------------------------------------------------
# Q1 — Total public sector employment by year (national)
# ---------------------------------------------------------------------------
Q_NATIONAL_TREND = """
SELECT
    e.year,
    ROUND(SUM(e.employed_thousands), 1)          AS total_employed_thousands,
    ROUND(AVG(e.avg_annual_salary_cad), 0)        AS avg_salary_cad,
    ROUND(AVG(e.unemployment_rate_pct), 2)         AS avg_unemployment_rate,
    ROUND(AVG(e.part_time_pct), 2)                 AS avg_part_time_pct
FROM employment e
GROUP BY e.year
ORDER BY e.year;
"""

# ---------------------------------------------------------------------------
# Q2 — Employment by sector and year + year-over-year growth (window function)
# ---------------------------------------------------------------------------
Q_SECTOR_TREND = """
WITH sector_totals AS (
    SELECT
        s.sector_code,
        s.sector_name,
        s.sector_category,
        e.year,
        ROUND(SUM(e.employed_thousands), 2)         AS employed_thousands,
        ROUND(AVG(e.avg_annual_salary_cad), 0)       AS avg_salary_cad
    FROM employment e
    JOIN sectors s ON e.sector_id = s.sector_id
    GROUP BY s.sector_code, e.year
)
SELECT
    sector_code,
    sector_name,
    sector_category,
    year,
    employed_thousands,
    avg_salary_cad,
    LAG(employed_thousands) OVER (
        PARTITION BY sector_code ORDER BY year
    )                                               AS prev_year_thousands,
    ROUND(
        100.0 * (employed_thousands - LAG(employed_thousands) OVER (
            PARTITION BY sector_code ORDER BY year
        )) / NULLIF(LAG(employed_thousands) OVER (
            PARTITION BY sector_code ORDER BY year
        ), 0),
        2
    )                                               AS yoy_growth_pct
FROM sector_totals
ORDER BY sector_code, year;
"""

# ---------------------------------------------------------------------------
# Q3 — Employment by region and year (aggregated across sectors)
# ---------------------------------------------------------------------------
Q_REGION_TREND = """
SELECT
    r.region_type,
    r.province_code,
    r.province_name,
    e.year,
    ROUND(SUM(e.employed_thousands), 2)             AS total_employed_thousands,
    ROUND(AVG(e.avg_annual_salary_cad), 0)           AS avg_salary_cad,
    ROUND(AVG(e.unemployment_rate_pct), 2)            AS avg_unemployment_rate
FROM employment e
JOIN regions r ON e.region_id = r.region_id
GROUP BY r.province_code, e.year
ORDER BY r.province_code, e.year;
"""

# ---------------------------------------------------------------------------
# Q4 — Cumulative employment growth since 2010 per province (window function)
# ---------------------------------------------------------------------------
Q_CUMULATIVE_GROWTH = """
WITH province_year AS (
    SELECT
        r.province_code,
        r.province_name,
        r.region_type,
        e.year,
        SUM(e.employed_thousands) AS employed_thousands
    FROM employment e
    JOIN regions r ON e.region_id = r.region_id
    GROUP BY r.province_code, e.year
),
base AS (
    SELECT province_code, employed_thousands AS base_thousands
    FROM province_year
    WHERE year = 2010
)
SELECT
    py.province_code,
    py.province_name,
    py.region_type,
    py.year,
    ROUND(py.employed_thousands, 2)                  AS employed_thousands,
    ROUND(
        100.0 * (py.employed_thousands - b.base_thousands) / b.base_thousands,
        2
    )                                                AS pct_growth_since_2010
FROM province_year py
JOIN base b USING (province_code)
ORDER BY py.province_code, py.year;
"""

# ---------------------------------------------------------------------------
# Q5 — Sector share of total public employment per year
# ---------------------------------------------------------------------------
Q_SECTOR_SHARE = """
WITH year_totals AS (
    SELECT year, SUM(employed_thousands) AS national_total
    FROM employment
    GROUP BY year
),
sector_year AS (
    SELECT
        s.sector_code,
        s.sector_name,
        e.year,
        SUM(e.employed_thousands) AS sector_total
    FROM employment e
    JOIN sectors s ON e.sector_id = s.sector_id
    GROUP BY s.sector_code, e.year
)
SELECT
    sy.sector_code,
    sy.sector_name,
    sy.year,
    ROUND(sy.sector_total, 2)                         AS employed_thousands,
    ROUND(100.0 * sy.sector_total / yt.national_total, 2) AS share_pct
FROM sector_year sy
JOIN year_totals yt USING (year)
ORDER BY sy.year, sy.sector_code;
"""

# ---------------------------------------------------------------------------
# Q6 — Top 5 provinces by public sector growth (2010→2023) with demographics
# ---------------------------------------------------------------------------
Q_TOP_PROVINCES = """
WITH emp_2010 AS (
    SELECT r.province_code, SUM(e.employed_thousands) AS emp_2010
    FROM employment e JOIN regions r ON e.region_id = r.region_id
    WHERE e.year = 2010 GROUP BY r.province_code
),
emp_2023 AS (
    SELECT r.province_code, r.province_name, r.region_type,
           SUM(e.employed_thousands) AS emp_2023,
           AVG(e.avg_annual_salary_cad) AS avg_salary_2023
    FROM employment e JOIN regions r ON e.region_id = r.region_id
    WHERE e.year = 2023 GROUP BY r.province_code
),
demo_2023 AS (
    SELECT r.province_code,
           d.pct_female, d.pct_union_coverage, d.median_age
    FROM demographics d JOIN regions r ON d.region_id = r.region_id
    WHERE d.year = 2023
)
SELECT
    e23.province_code,
    e23.province_name,
    e23.region_type,
    ROUND(e10.emp_2010, 1)                                AS employed_2010,
    ROUND(e23.emp_2023, 1)                                AS employed_2023,
    ROUND(100.0*(e23.emp_2023-e10.emp_2010)/e10.emp_2010, 1) AS growth_pct,
    ROUND(e23.avg_salary_2023, 0)                          AS avg_salary_2023,
    d.pct_female,
    d.pct_union_coverage,
    d.median_age
FROM emp_2023 e23
JOIN emp_2010 e10 USING (province_code)
JOIN demo_2023 d USING (province_code)
ORDER BY growth_pct DESC
LIMIT 5;
"""

# ---------------------------------------------------------------------------
# Q7 — Salary rank per sector per year (dense_rank window function)
# ---------------------------------------------------------------------------
Q_SALARY_RANK = """
WITH sector_avg AS (
    SELECT
        s.sector_name,
        e.year,
        ROUND(AVG(e.avg_annual_salary_cad), 0) AS avg_salary
    FROM employment e
    JOIN sectors s ON e.sector_id = s.sector_id
    GROUP BY s.sector_name, e.year
)
SELECT
    sector_name,
    year,
    avg_salary,
    DENSE_RANK() OVER (PARTITION BY year ORDER BY avg_salary DESC) AS salary_rank
FROM sector_avg
ORDER BY year, salary_rank;
"""

# ---------------------------------------------------------------------------
# Q8 — Regional summary statistics (latest year)
# ---------------------------------------------------------------------------
Q_REGIONAL_SUMMARY = """
SELECT
    r.region_type,
    e.year,
    COUNT(DISTINCT r.province_code)              AS num_provinces,
    ROUND(SUM(e.employed_thousands), 1)           AS total_employed_thousands,
    ROUND(AVG(e.avg_annual_salary_cad), 0)         AS avg_salary,
    ROUND(AVG(e.unemployment_rate_pct), 2)          AS avg_unemployment_rate,
    ROUND(AVG(e.part_time_pct), 2)                  AS avg_part_time_pct
FROM employment e
JOIN regions r ON e.region_id = r.region_id
GROUP BY r.region_type, e.year
ORDER BY r.region_type, e.year;
"""

QUERIES = {
    "national_trend":    Q_NATIONAL_TREND,
    "sector_trend":      Q_SECTOR_TREND,
    "region_trend":      Q_REGION_TREND,
    "cumulative_growth": Q_CUMULATIVE_GROWTH,
    "sector_share":      Q_SECTOR_SHARE,
    "top_provinces":     Q_TOP_PROVINCES,
    "salary_rank":       Q_SALARY_RANK,
    "regional_summary":  Q_REGIONAL_SUMMARY,
}


def run_all() -> dict[str, pd.DataFrame]:
    conn = get_connection()
    results = {}
    for name, sql in QUERIES.items():
        df = pd.read_sql_query(sql, conn)
        results[name] = df
        print(f"  [{name}] → {len(df)} rows")
    conn.close()
    return results


if __name__ == "__main__":
    print("Running SQL queries...")
    dfs = run_all()
    for name, df in dfs.items():
        print(f"\n=== {name} ===")
        print(df.head(5).to_string(index=False))
