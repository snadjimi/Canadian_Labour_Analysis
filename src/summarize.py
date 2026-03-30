"""
summarize.py
------------
Calls the Claude API to generate a natural language summary of key
findings from the Canadian Public Sector Labour Market Analysis pipeline.

Requires the ANTHROPIC_API_KEY environment variable to be set.
Falls back to a placeholder message if the key is missing.
"""

import json
import os

import anthropic

MODEL = "claude-3-5-haiku-20241022"


def _build_prompt(dfs: dict, stats: dict, forecasts: dict) -> str:
    """Constructs the data context string sent to Claude."""

    nat = dfs["national_trend"]
    first_year = int(nat["year"].min())
    last_year  = int(nat["year"].max())

    # Average YoY growth per sector
    sector_growth = (
        dfs["sector_trend"]
        .groupby("sector_name")["yoy_growth_pct"]
        .mean()
        .round(2)
        .to_dict()
    )

    # Top provinces by employment growth
    top_prov = (
        dfs["top_provinces"][["province_name", "growth_pct"]]
        .to_dict(orient="records")
    )

    # 2028 national forecast
    nat_fc = forecasts["national_forecast"]
    row_2028 = nat_fc[nat_fc["year"] == 2028].iloc[0]
    forecast_summary = {
        "year": 2028,
        "projected_employed_thousands": float(row_2028["total_employed_thousands"]),
        "projected_avg_salary_cad": float(row_2028["avg_salary_cad"]),
    }

    data_context = {
        "analysis_period": f"{first_year}–{last_year}",
        "key_statistics": stats,
        "sector_avg_yoy_growth_pct": sector_growth,
        "top_provinces_by_employment_growth": top_prov,
        "national_forecast_2028": forecast_summary,
    }

    return (
        "You are a labour market economist advising a Canadian government policy team.\n"
        "Analyze the following Statistics Canada public sector employment data and write\n"
        "a concise, insightful 3-paragraph executive summary (roughly 200 words total).\n\n"
        "Structure your response as follows:\n"
        "  Paragraph 1 — Overall employment and salary trends across the analysis period.\n"
        "  Paragraph 2 — Notable sector and regional differences.\n"
        "  Paragraph 3 — What the 2024–2028 forecast suggests for workforce planning.\n\n"
        "Write in clear, professional prose. Do not use bullet points or headings.\n\n"
        f"Data:\n{json.dumps(data_context, indent=2)}"
    )


def generate_summary(dfs: dict, stats: dict, forecasts: dict) -> str:
    """
    Generates a natural language summary via the Claude API.

    Parameters
    ----------
    dfs       : dict of DataFrames from queries.run_all()
    stats     : dict of summary statistics from visualize.summary_stats()
    forecasts : dict of forecast DataFrames from forecast.run_forecasts()

    Returns
    -------
    A plain-text summary string (or a fallback message if the API key is absent).
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return (
            "LLM summary unavailable — set the ANTHROPIC_API_KEY environment variable "
            "to enable AI-generated insights powered by Claude."
        )

    client = anthropic.Anthropic(api_key=api_key)
    prompt = _build_prompt(dfs, stats, forecasts)

    message = client.messages.create(
        model=MODEL,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )
    return message.content[0].text
