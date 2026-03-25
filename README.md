# Canadian Public Sector Labour Market Analysis

A data pipeline that ingests and cleans Statistics Canada public employment datasets, loads them into a normalized SQLite database, runs SQL analyses using joins, aggregations, and window functions, and exports findings to a structured HTML report with visualizations.

**Tech stack:** Python 3.11+, pandas, SQLite, matplotlib, seaborn

---

## Project Structure

```
canadian-labour-analysis/
├── data/
│   └── raw/                   # Generated CSV datasets
│       ├── statcan_employment.csv
│       └── statcan_demographics.csv
├── db/
│   └── employment.db          # Normalized SQLite database
├── output/
│   └── report.html            # Self-contained HTML report
├── src/
│   ├── generate_data.py       # Synthetic StatCan-style data generator
│   ├── ingest.py              # Data cleaning + SQLite ingestion
│   ├── queries.py             # SQL analytical queries
│   ├── visualize.py           # Charts + HTML report builder
│   └── main.py                # Pipeline orchestrator
└── requirements.txt
```

---

## Database Schema

Four normalized relational tables:

```sql
regions     (region_id, province_code, province_name, region_type)
sectors     (sector_id, sector_code, sector_name, sector_category)
employment  (employment_id, region_id→, sector_id→, year,
             employed_thousands, avg_annual_salary_cad,
             unemployment_rate_pct, part_time_pct)
demographics(demo_id, region_id→, year, pct_female, pct_indigenous,
             pct_visible_minority, median_age, pct_union_coverage)
```

---

## SQL Analyses

| Query | Technique |
|---|---|
| National employment trend | Aggregation (`SUM`, `AVG`) |
| Sector growth over time | `LAG()` window function (YoY %) |
| Provincial employment trend | Multi-table `JOIN` + `GROUP BY` |
| Cumulative growth since 2010 | Self-join + `WITH` CTE |
| Sector share of total employment | Subquery + division |
| Top 5 provinces by growth | Multi-CTE with demographics join |
| Salary ranking by sector | `DENSE_RANK()` window function |
| Regional summary statistics | `GROUP BY` with `COUNT(DISTINCT)` |

---

## Quickstart

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the full pipeline
python src/main.py

# 3. Open the report
open output/report.html     # macOS
xdg-open output/report.html # Linux
```

---

## Data Notes

The datasets are synthetically generated to mirror the structure and magnitudes of Statistics Canada public-use employment tables:

- **14-10-0066-01** — Employment by industry (NAICS), Canada, provinces and territories
- **14-10-0027-01** — Employment by province and economic region

Sector-level growth rates, COVID-19 shocks (2020–2021), regional distributions, and salary progression are calibrated against publicly available StatCan summary data.

---

## Key Findings (2010–2023)

- Total public sector employment grew across all provinces, driven largely by **Health Care and Social Assistance**.
- The **2020 COVID shock** produced the sharpest single-year contraction in Educational Services (−8%) and Local Government (−5%), while Health Care grew counter-cyclically.
- **Utilities** consistently carry the highest average salaries; **Local Government** the lowest.
- Northern territories (NT, NU, YT) showed the highest relative growth rates off a small base.
- Union coverage remains above 55% across all regions, with the Atlantic provinces highest.
