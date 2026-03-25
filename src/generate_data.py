"""
generate_data.py
----------------
Generates realistic synthetic Statistics Canada-style public sector
employment datasets and saves them as CSVs in data/raw/.

The data mirrors the structure of StatCan table 14-10-0066-01
(Employment by industry) and 14-10-0027-01 (Employment by province),
covering federal, provincial/territorial, and local government sectors
from 2010 to 2023.
"""

import os
import numpy as np
import pandas as pd

SEED = 42
rng = np.random.default_rng(SEED)

PROVINCES = [
    ("AB", "Alberta"),
    ("BC", "British Columbia"),
    ("MB", "Manitoba"),
    ("NB", "New Brunswick"),
    ("NL", "Newfoundland and Labrador"),
    ("NS", "Nova Scotia"),
    ("NT", "Northwest Territories"),
    ("NU", "Nunavut"),
    ("ON", "Ontario"),
    ("PE", "Prince Edward Island"),
    ("QC", "Quebec"),
    ("SK", "Saskatchewan"),
    ("YT", "Yukon"),
]

SECTORS = [
    ("PS-FED", "Federal government", "Public Administration"),
    ("PS-PROV", "Provincial and territorial government", "Public Administration"),
    ("PS-LOCAL", "Local government", "Public Administration"),
    ("PS-HLTH", "Health care and social assistance (public)", "Health & Social"),
    ("PS-EDU", "Educational services (public)", "Education"),
    ("PS-UTIL", "Utilities (public)", "Utilities"),
    ("PS-TRANS", "Transportation and warehousing (public)", "Transportation"),
    ("PS-DEF", "Defence and security services", "Public Administration"),
]

YEARS = list(range(2010, 2024))

# Base employment (thousands) per province for each sector
# Roughly calibrated to real StatCan magnitudes
BASE_EMPLOYMENT = {
    "AB":    {"PS-FED": 28, "PS-PROV": 95, "PS-LOCAL": 55, "PS-HLTH": 115, "PS-EDU": 85, "PS-UTIL": 18, "PS-TRANS": 12, "PS-DEF": 10},
    "BC":    {"PS-FED": 42, "PS-PROV": 130, "PS-LOCAL": 80, "PS-HLTH": 160, "PS-EDU": 115, "PS-UTIL": 22, "PS-TRANS": 18, "PS-DEF": 15},
    "MB":    {"PS-FED": 14, "PS-PROV": 55, "PS-LOCAL": 30, "PS-HLTH": 60, "PS-EDU": 42, "PS-UTIL": 8,  "PS-TRANS": 7,  "PS-DEF": 5},
    "NB":    {"PS-FED": 10, "PS-PROV": 35, "PS-LOCAL": 18, "PS-HLTH": 38, "PS-EDU": 28, "PS-UTIL": 5,  "PS-TRANS": 4,  "PS-DEF": 4},
    "NL":    {"PS-FED": 8,  "PS-PROV": 30, "PS-LOCAL": 15, "PS-HLTH": 32, "PS-EDU": 23, "PS-UTIL": 4,  "PS-TRANS": 3,  "PS-DEF": 3},
    "NS":    {"PS-FED": 12, "PS-PROV": 38, "PS-LOCAL": 20, "PS-HLTH": 42, "PS-EDU": 30, "PS-UTIL": 6,  "PS-TRANS": 5,  "PS-DEF": 8},
    "NT":    {"PS-FED": 3,  "PS-PROV": 8,  "PS-LOCAL": 4,  "PS-HLTH": 6,  "PS-EDU": 4,  "PS-UTIL": 1,  "PS-TRANS": 1,  "PS-DEF": 1},
    "NU":    {"PS-FED": 2,  "PS-PROV": 5,  "PS-LOCAL": 2,  "PS-HLTH": 4,  "PS-EDU": 3,  "PS-UTIL": 1,  "PS-TRANS": 1,  "PS-DEF": 0},
    "ON":    {"PS-FED": 110, "PS-PROV": 320, "PS-LOCAL": 210, "PS-HLTH": 420, "PS-EDU": 300, "PS-UTIL": 55, "PS-TRANS": 50, "PS-DEF": 30},
    "PE":    {"PS-FED": 4,  "PS-PROV": 12, "PS-LOCAL": 6,  "PS-HLTH": 12, "PS-EDU": 8,  "PS-UTIL": 1,  "PS-TRANS": 1,  "PS-DEF": 1},
    "QC":    {"PS-FED": 65, "PS-PROV": 250, "PS-LOCAL": 130, "PS-HLTH": 300, "PS-EDU": 210, "PS-UTIL": 40, "PS-TRANS": 35, "PS-DEF": 14},
    "SK":    {"PS-FED": 12, "PS-PROV": 48, "PS-LOCAL": 28, "PS-HLTH": 52, "PS-EDU": 38, "PS-UTIL": 7,  "PS-TRANS": 6,  "PS-DEF": 4},
    "YT":    {"PS-FED": 3,  "PS-PROV": 7,  "PS-LOCAL": 3,  "PS-HLTH": 5,  "PS-EDU": 4,  "PS-UTIL": 1,  "PS-TRANS": 1,  "PS-DEF": 1},
}

# Annual growth rates (%) per sector — reflects real trends
SECTOR_GROWTH = {
    "PS-FED":   0.5,
    "PS-PROV":  1.0,
    "PS-LOCAL": 1.2,
    "PS-HLTH":  2.5,   # rapid growth due to aging population
    "PS-EDU":   1.5,
    "PS-UTIL":  0.3,
    "PS-TRANS": 0.8,
    "PS-DEF":   0.6,
}

# COVID shock years (2020 and 2021) — sectoral impacts
COVID_SHOCK = {
    2020: {"PS-FED": 0.02, "PS-PROV": -0.03, "PS-LOCAL": -0.05, "PS-HLTH": 0.04, "PS-EDU": -0.08, "PS-UTIL": -0.01, "PS-TRANS": -0.06, "PS-DEF": 0.01},
    2021: {"PS-FED": 0.01, "PS-PROV":  0.01, "PS-LOCAL":  0.01, "PS-HLTH": 0.06, "PS-EDU":  0.02, "PS-UTIL":  0.01, "PS-TRANS":  0.02, "PS-DEF": 0.01},
}

AVG_SALARY_BASE = {
    "PS-FED":   92000,
    "PS-PROV":  78000,
    "PS-LOCAL": 68000,
    "PS-HLTH":  72000,
    "PS-EDU":   75000,
    "PS-UTIL":  88000,
    "PS-TRANS": 70000,
    "PS-DEF":   84000,
}

SALARY_GROWTH = 0.025  # ~2.5% annual wage growth


def generate_employment_records():
    records = []
    for prov_code, prov_name in PROVINCES:
        for sector_code, sector_name, sector_category in SECTORS:
            base = BASE_EMPLOYMENT[prov_code][sector_code]
            for i, year in enumerate(YEARS):
                growth = (1 + SECTOR_GROWTH[sector_code] / 100) ** i
                shock = COVID_SHOCK.get(year, {}).get(sector_code, 0)
                noise = rng.normal(0, 0.015)  # ±1.5% random variation
                employed_thousands = round(base * growth * (1 + shock + noise), 2)

                salary = AVG_SALARY_BASE[sector_code] * (1 + SALARY_GROWTH) ** i
                salary += rng.normal(0, salary * 0.03)
                salary = round(salary, 0)

                unemployment_rate = round(rng.uniform(3.5, 8.5), 1)
                part_time_pct = round(rng.uniform(12, 28), 1)

                records.append({
                    "province_code": prov_code,
                    "province_name": prov_name,
                    "sector_code": sector_code,
                    "sector_name": sector_name,
                    "sector_category": sector_category,
                    "year": year,
                    "employed_thousands": employed_thousands,
                    "avg_annual_salary_cad": salary,
                    "unemployment_rate_pct": unemployment_rate,
                    "part_time_pct": part_time_pct,
                })
    return pd.DataFrame(records)


def generate_workforce_demographics():
    """Generates a supplementary demographics table per region/year."""
    records = []
    for prov_code, prov_name in PROVINCES:
        for year in YEARS:
            i = year - YEARS[0]
            records.append({
                "province_code": prov_code,
                "year": year,
                "pct_female": round(rng.uniform(48, 62), 1),
                "pct_indigenous": round(rng.uniform(2, 18), 1),
                "pct_visible_minority": round(rng.uniform(5, 32), 1),
                "median_age": round(rng.uniform(40, 47) + i * 0.1, 1),
                "pct_union_coverage": round(rng.uniform(55, 85), 1),
            })
    return pd.DataFrame(records)


def main():
    out_dir = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
    os.makedirs(out_dir, exist_ok=True)

    print("Generating employment dataset...")
    emp_df = generate_employment_records()
    emp_path = os.path.join(out_dir, "statcan_employment.csv")
    emp_df.to_csv(emp_path, index=False)
    print(f"  Saved {len(emp_df):,} rows → {emp_path}")

    print("Generating workforce demographics dataset...")
    demo_df = generate_workforce_demographics()
    demo_path = os.path.join(out_dir, "statcan_demographics.csv")
    demo_df.to_csv(demo_path, index=False)
    print(f"  Saved {len(demo_df):,} rows → {demo_path}")

    print("Data generation complete.")


if __name__ == "__main__":
    main()
