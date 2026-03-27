"""
forecast.py
-----------
Uses scikit-learn to fit polynomial trend models per sector and
forecasts Canadian public sector employment from 2024 to 2028.

Returns DataFrames consumed by visualize.py for chart rendering.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

FORECAST_YEARS = list(range(2024, 2029))


def _poly_model(degree: int = 2):
    return make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())


def forecast_sector_employment(sector_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fits a degree-2 polynomial regression per sector on historical
    employed_thousands vs year, then forecasts through 2028.

    Returns a DataFrame with columns:
        sector_name, year, employed_thousands, is_forecast
    """
    records = []

    for sector_name, group in sector_df.groupby("sector_name"):
        group = group.sort_values("year")
        X = group["year"].values.reshape(-1, 1)
        y = group["employed_thousands"].values

        model = _poly_model(degree=2)
        model.fit(X, y)

        # Historical actuals
        for _, row in group.iterrows():
            records.append({
                "sector_name": sector_name,
                "year": int(row["year"]),
                "employed_thousands": round(float(row["employed_thousands"]), 2),
                "is_forecast": False,
            })

        # Forecasted values
        X_future = np.array(FORECAST_YEARS).reshape(-1, 1)
        y_pred = model.predict(X_future)
        for yr, pred in zip(FORECAST_YEARS, y_pred):
            records.append({
                "sector_name": sector_name,
                "year": yr,
                "employed_thousands": round(max(float(pred), 0.0), 2),
                "is_forecast": True,
            })

    return pd.DataFrame(records)


def forecast_national_employment(national_df: pd.DataFrame) -> pd.DataFrame:
    """
    Forecasts total national public sector employment and average salary
    through 2028 using polynomial (employment) and linear (salary) regression.

    Returns a DataFrame with columns:
        year, total_employed_thousands, avg_salary_cad, is_forecast
    """
    df = national_df.sort_values("year")
    X = df["year"].values.reshape(-1, 1)

    emp_model = _poly_model(degree=2)
    emp_model.fit(X, df["total_employed_thousands"].values)

    sal_model = LinearRegression()
    sal_model.fit(X, df["avg_salary_cad"].values)

    records = []

    # Historical actuals
    for _, row in df.iterrows():
        records.append({
            "year": int(row["year"]),
            "total_employed_thousands": round(float(row["total_employed_thousands"]), 1),
            "avg_salary_cad": round(float(row["avg_salary_cad"]), 0),
            "is_forecast": False,
        })

    # Forecasted values
    X_future = np.array(FORECAST_YEARS).reshape(-1, 1)
    emp_pred = emp_model.predict(X_future)
    sal_pred = sal_model.predict(X_future)

    for yr, ep, sp in zip(FORECAST_YEARS, emp_pred, sal_pred):
        records.append({
            "year": yr,
            "total_employed_thousands": round(max(float(ep), 0.0), 1),
            "avg_salary_cad": round(max(float(sp), 0.0), 0),
            "is_forecast": True,
        })

    return pd.DataFrame(records)


def run_forecasts(dfs: dict) -> dict:
    """
    Entry point: runs all forecast models and returns a dict of DataFrames.
    Keys: 'sector_forecast', 'national_forecast'
    """
    print("  Forecasting sector employment (2024–2028)...")
    sector_forecast = forecast_sector_employment(dfs["sector_trend"])

    print("  Forecasting national employment (2024–2028)...")
    national_forecast = forecast_national_employment(dfs["national_trend"])

    return {
        "sector_forecast": sector_forecast,
        "national_forecast": national_forecast,
    }
