"""
visualize.py
------------
Produces charts from query results and renders a self-contained HTML
report with embedded figures (base64) and summary statistics tables.
"""

import base64
import io
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import pandas as pd

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "output")

PALETTE = sns.color_palette("tab10")
sns.set_theme(style="whitegrid", palette="tab10", font_scale=1.05)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def fig_to_b64(fig: plt.Figure) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return encoded


def thousands_fmt(x, _):
    return f"{x:,.0f}k"


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def chart_national_trend(df: pd.DataFrame) -> str:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1 = axes[0]
    ax1.plot(df["year"], df["total_employed_thousands"], marker="o", linewidth=2.2,
             color=PALETTE[0])
    ax1.set_title("Total Public Sector Employment (Canada)", fontweight="bold")
    ax1.set_xlabel("Year")
    ax1.set_ylabel("Employed (thousands)")
    ax1.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_fmt))
    ax1.set_xticks(df["year"])
    ax1.tick_params(axis="x", rotation=45)

    ax2 = axes[1]
    ax2.plot(df["year"], df["avg_salary_cad"] / 1000, marker="s", linewidth=2.2,
             color=PALETTE[1])
    ax2.set_title("Average Annual Salary (Public Sector)", fontweight="bold")
    ax2.set_xlabel("Year")
    ax2.set_ylabel("Avg. Salary (CAD thousands)")
    ax2.set_xticks(df["year"])
    ax2.tick_params(axis="x", rotation=45)

    fig.suptitle("National Employment Trends — 2010–2023", fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_sector_employment(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="year", columns="sector_name",
                           values="employed_thousands", aggfunc="sum")
    fig, ax = plt.subplots(figsize=(13, 5.5))
    pivot.plot(ax=ax, linewidth=2, marker=".")
    ax.set_title("Public Sector Employment by Sector (2010–2023)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Employed (thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_fmt))
    ax.set_xticks(df["year"].unique())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_yoy_growth(df: pd.DataFrame) -> str:
    df_clean = df.dropna(subset=["yoy_growth_pct"])
    pivot = df_clean.pivot_table(index="year", columns="sector_name",
                                 values="yoy_growth_pct", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(13, 5))
    pivot.plot(kind="bar", ax=ax, width=0.75)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title("Year-over-Year Employment Growth by Sector (%)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("YoY Growth (%)")
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_regional_heatmap(df: pd.DataFrame) -> str:
    pivot = df.pivot_table(index="province_code", columns="year",
                           values="total_employed_thousands", aggfunc="sum")
    # Normalize each row to 2010=100 for relative comparison
    pivot_norm = pivot.div(pivot[2010], axis=0) * 100

    fig, ax = plt.subplots(figsize=(14, 6))
    sns.heatmap(pivot_norm, ax=ax, cmap="YlOrRd", annot=True, fmt=".0f",
                linewidths=0.4, linecolor="#cccccc",
                cbar_kws={"label": "Index (2010 = 100)"})
    ax.set_title("Public Sector Employment Growth Index by Province (2010 = 100)",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Province")
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_sector_share(df: pd.DataFrame) -> str:
    years = [2010, 2015, 2020, 2023]
    df_sub = df[df["year"].isin(years)].copy()
    pivot = df_sub.pivot_table(index="year", columns="sector_name",
                               values="share_pct", aggfunc="sum")

    fig, axes = plt.subplots(1, len(years), figsize=(16, 5))
    colors = sns.color_palette("tab10", n_colors=len(pivot.columns))

    for i, year in enumerate(years):
        ax = axes[i]
        row = pivot.loc[year]
        wedges, texts, autotexts = ax.pie(
            row.values, labels=None, autopct="%1.0f%%",
            colors=colors, startangle=90,
            textprops={"fontsize": 7.5},
        )
        ax.set_title(str(year), fontweight="bold")

    fig.legend(pivot.columns, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.05))
    fig.suptitle("Sector Share of Total Public Employment", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_salary_rank(df: pd.DataFrame) -> str:
    latest = df[df["year"] == df["year"].max()].sort_values("avg_salary", ascending=True)
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(latest["sector_name"], latest["avg_salary"] / 1000,
                   color=PALETTE[:len(latest)])
    ax.set_xlabel("Avg. Annual Salary (CAD thousands)")
    ax.set_title(f"Sector Salary Ranking ({latest['year'].iloc[0]})", fontweight="bold")
    for bar, val in zip(bars, latest["avg_salary"]):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"${val:,.0f}", va="center", fontsize=8.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_national_forecast(df: pd.DataFrame) -> str:
    historical = df[~df["is_forecast"]]
    forecast   = df[df["is_forecast"]]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    for ax, col, label, color_idx, ylabel in [
        (axes[0], "total_employed_thousands", "Employment (thousands)", 0, "Employed (thousands)"),
        (axes[1], "avg_salary_cad",           "Avg. Salary (CAD)",      1, "Avg. Salary (CAD thousands)"),
    ]:
        ax.plot(historical["year"], historical[col] if col == "total_employed_thousands"
                else historical[col] / 1000,
                marker="o", linewidth=2.2, color=PALETTE[color_idx], label="Historical")
        ax.plot(forecast["year"], forecast[col] if col == "total_employed_thousands"
                else forecast[col] / 1000,
                marker="o", linewidth=2.2, linestyle="--", color=PALETTE[2], label="Forecast")
        ax.axvline(x=2023.5, color="#aaa", linestyle=":", linewidth=1.5)
        ax.set_xlabel("Year")
        ax.set_ylabel(ylabel)
        ax.set_xticks(sorted(df["year"].unique()))
        ax.tick_params(axis="x", rotation=45)
        ax.legend(fontsize=8.5)

    axes[0].set_title("National Employment — Historical & Forecast", fontweight="bold")
    axes[0].yaxis.set_major_formatter(mticker.FuncFormatter(thousands_fmt))
    axes[1].set_title("Avg. Salary — Historical & Forecast", fontweight="bold")

    fig.suptitle("National Forecast 2024–2028 (Polynomial Trend Model)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_sector_forecast(df: pd.DataFrame) -> str:
    # Select top 4 sectors by 2023 employment
    top_sectors = (
        df[(df["year"] == 2023) & (~df["is_forecast"])]
        .nlargest(4, "employed_thousands")["sector_name"]
        .tolist()
    )
    df_top = df[df["sector_name"].isin(top_sectors)]
    colors = sns.color_palette("tab10", n_colors=len(top_sectors))

    fig, ax = plt.subplots(figsize=(13, 5.5))
    for i, (sector, group) in enumerate(df_top.groupby("sector_name")):
        hist = group[~group["is_forecast"]]
        fc   = group[group["is_forecast"]]
        ax.plot(hist["year"], hist["employed_thousands"],
                marker=".", linewidth=2, color=colors[i], label=sector)
        ax.plot(fc["year"], fc["employed_thousands"],
                marker=".", linewidth=2, linestyle="--", color=colors[i])

    ax.axvline(x=2023.5, color="#aaa", linestyle=":", linewidth=1.5, label="Forecast boundary")
    ax.set_title("Sector Employment — Historical & Forecast (Top 4 Sectors)", fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Employed (thousands)")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(thousands_fmt))
    ax.set_xticks(sorted(df["year"].unique()))
    ax.tick_params(axis="x", rotation=45)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8.5)
    fig.tight_layout()
    return fig_to_b64(fig)


def chart_cumulative_growth(df: pd.DataFrame) -> str:
    # Show top 6 provinces by 2023 growth
    latest = df[df["year"] == 2023].nlargest(6, "pct_growth_since_2010")
    top_provs = latest["province_code"].tolist()
    df_top = df[df["province_code"].isin(top_provs)]

    fig, ax = plt.subplots(figsize=(12, 5))
    for prov, group in df_top.groupby("province_code"):
        ax.plot(group["year"], group["pct_growth_since_2010"],
                marker=".", linewidth=2, label=prov)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_title("Cumulative Employment Growth Since 2010 — Top 6 Provinces",
                 fontweight="bold")
    ax.set_xlabel("Year")
    ax.set_ylabel("Growth since 2010 (%)")
    ax.set_xticks(df["year"].unique())
    ax.tick_params(axis="x", rotation=45)
    ax.legend(title="Province", fontsize=9)
    fig.tight_layout()
    return fig_to_b64(fig)


# ---------------------------------------------------------------------------
# Summary stats helper
# ---------------------------------------------------------------------------

def summary_stats(dfs: dict) -> dict:
    nat = dfs["national_trend"]
    first = nat[nat["year"] == nat["year"].min()].iloc[0]
    last  = nat[nat["year"] == nat["year"].max()].iloc[0]

    total_growth = round(
        100 * (last["total_employed_thousands"] - first["total_employed_thousands"])
        / first["total_employed_thousands"], 1
    )
    salary_growth = round(
        100 * (last["avg_salary_cad"] - first["avg_salary_cad"])
        / first["avg_salary_cad"], 1
    )
    fastest_sector = (
        dfs["sector_trend"]
        .groupby("sector_name")["yoy_growth_pct"]
        .mean()
        .idxmax()
    )
    return {
        "year_range": f"{int(first['year'])}–{int(last['year'])}",
        "total_employed_2023": f"{last['total_employed_thousands']:,.1f}k",
        "total_growth_pct": f"{total_growth}%",
        "avg_salary_2023": f"${last['avg_salary_cad']:,.0f}",
        "salary_growth_pct": f"{salary_growth}%",
        "fastest_growing_sector": fastest_sector,
        "avg_unemployment_latest": f"{last['avg_unemployment_rate']:.1f}%",
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>Canadian Public Sector Labour Market Analysis</title>
  <style>
    body {{ font-family: 'Segoe UI', Arial, sans-serif; margin: 0; background: #f5f7fa; color: #222; }}
    header {{ background: #cc0000; color: #fff; padding: 28px 40px; }}
    header h1 {{ margin: 0; font-size: 1.7rem; }}
    header p {{ margin: 6px 0 0; opacity: .85; font-size: 0.95rem; }}
    main {{ max-width: 1100px; margin: 0 auto; padding: 32px 24px; }}
    h2 {{ border-left: 5px solid #cc0000; padding-left: 12px; color: #1a1a1a; }}
    h3 {{ color: #444; }}
    .cards {{ display: flex; flex-wrap: wrap; gap: 16px; margin-bottom: 32px; }}
    .card {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.08);
             padding: 20px 24px; flex: 1 1 160px; min-width: 150px; }}
    .card .label {{ font-size: .78rem; color: #888; text-transform: uppercase; letter-spacing: .05em; }}
    .card .value {{ font-size: 1.5rem; font-weight: 700; color: #cc0000; margin-top: 4px; }}
    .card .sub   {{ font-size: .82rem; color: #555; margin-top: 2px; }}
    .chart-block {{ background: #fff; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,.07);
                    padding: 20px; margin-bottom: 28px; }}
    .chart-block img {{ width: 100%; height: auto; display: block; }}
    table {{ border-collapse: collapse; width: 100%; font-size: .88rem; }}
    th {{ background: #cc0000; color: #fff; padding: 8px 12px; text-align: left; }}
    td {{ padding: 7px 12px; border-bottom: 1px solid #e8e8e8; }}
    tr:hover td {{ background: #fafafa; }}
    footer {{ text-align: center; padding: 20px; color: #aaa; font-size: .8rem; }}
  </style>
</head>
<body>
<header>
  <h1>Canadian Public Sector Labour Market Analysis</h1>
  <p>Data Source: Statistics Canada (synthetic dataset mirroring tables 14-10-0066-01 &amp;
     14-10-0027-01) &nbsp;|&nbsp; Period: {year_range}</p>
</header>
<main>

  <h2>Key Findings</h2>
  <div class="cards">
    <div class="card">
      <div class="label">Total Employed (2023)</div>
      <div class="value">{total_employed_2023}</div>
      <div class="sub">public sector workers</div>
    </div>
    <div class="card">
      <div class="label">Employment Growth</div>
      <div class="value">{total_growth_pct}</div>
      <div class="sub">since {year_start}</div>
    </div>
    <div class="card">
      <div class="label">Avg. Salary (2023)</div>
      <div class="value">{avg_salary_2023}</div>
      <div class="sub">CAD per year</div>
    </div>
    <div class="card">
      <div class="label">Salary Growth</div>
      <div class="value">{salary_growth_pct}</div>
      <div class="sub">since {year_start}</div>
    </div>
    <div class="card">
      <div class="label">Fastest Growing Sector</div>
      <div class="value" style="font-size:1rem;">{fastest_growing_sector}</div>
      <div class="sub">by avg. YoY growth</div>
    </div>
    <div class="card">
      <div class="label">Avg. Unemployment Rate</div>
      <div class="value">{avg_unemployment_latest}</div>
      <div class="sub">latest year (public sector)</div>
    </div>
  </div>

  <h2>National Employment Trends</h2>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_national}" alt="National Employment Trends"/>
  </div>

  <h2>Employment by Sector</h2>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_sector_emp}" alt="Employment by Sector"/>
  </div>

  <h2>Year-over-Year Growth by Sector</h2>
  <p>Negative values during 2020–2021 reflect the COVID-19 pandemic shock on public employment,
     particularly in education and transportation. Health care shows counter-cyclical growth.</p>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_yoy}" alt="YoY Growth by Sector"/>
  </div>

  <h2>Sector Composition Over Time</h2>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_share}" alt="Sector Share of Employment"/>
  </div>

  <h2>Provincial Employment Heat Map</h2>
  <p>Index where 2010 = 100. Values above 100 indicate employment growth relative to 2010 levels.</p>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_heatmap}" alt="Provincial Employment Heatmap"/>
  </div>

  <h2>Cumulative Growth — Top 6 Provinces</h2>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_cum}" alt="Cumulative Growth"/>
  </div>

  <h2>Salary Ranking by Sector (2023)</h2>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_salary}" alt="Salary Rank"/>
  </div>

  <h2>Top 5 Provinces by Employment Growth (2010–2023)</h2>
  <div class="chart-block">
    {table_top_provinces}
  </div>

  <h2>Regional Summary (2023)</h2>
  <div class="chart-block">
    {table_regional_summary}
  </div>

  <h2>AI-Generated Executive Summary</h2>
  <div class="chart-block">
    <div style="line-height:1.75; font-size:.95rem; color:#333;">
      {llm_summary}
    </div>
    <p style="margin-top:12px; font-size:.78rem; color:#999;">
      Generated by Claude (claude-3-5-haiku) via the Anthropic API based on query results.
    </p>
  </div>

  <h2>Employment Forecast — 2024–2028</h2>
  <p>Forecasts produced using polynomial degree-2 regression fitted on 2010–2023 actuals
     per sector. Dashed lines indicate projected values; dotted vertical line marks the
     historical/forecast boundary.</p>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_national_forecast}" alt="National Forecast"/>
  </div>
  <div class="chart-block">
    <img src="data:image/png;base64,{chart_sector_forecast}" alt="Sector Forecast"/>
  </div>

  <h2>Methodology</h2>
  <ul>
    <li>Data generated to match the structure and magnitude of Statistics Canada public-use
        employment tables (CANSIM 14-10-0066-01 and 14-10-0027-01).</li>
    <li>Data cleaned with <strong>pandas</strong>: deduplication, type coercion, and range
        validation before database load.</li>
    <li>Stored in a <strong>normalized SQLite</strong> database across four relational tables:
        <code>regions</code>, <code>sectors</code>, <code>employment</code>,
        <code>demographics</code>.</li>
    <li>Analytical queries use SQL <strong>JOINs</strong>, <strong>GROUP BY aggregations</strong>,
        and <strong>window functions</strong> (<code>LAG</code>, <code>DENSE_RANK</code>) to
        derive YoY growth, cumulative change, and salary rankings.</li>
    <li>Visualizations produced with <strong>matplotlib</strong> and <strong>seaborn</strong>;
        report rendered as a self-contained HTML file.</li>
  </ul>

</main>
<footer>Generated by the Canadian Public Sector Labour Market Analysis pipeline &nbsp;|&nbsp;
Data: Statistics Canada (synthetic) &nbsp;|&nbsp; 2025</footer>
</body>
</html>
"""


def df_to_html_table(df: pd.DataFrame) -> str:
    return df.to_html(index=False, border=0, classes="", na_rep="—")


def build_report(dfs: dict, forecasts: dict = None, llm_summary: str = ""):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    stats = summary_stats(dfs)
    year_start = stats["year_range"].split("–")[0]

    print("  Rendering charts...")
    chart_national   = chart_national_trend(dfs["national_trend"])
    chart_sector_emp = chart_sector_employment(dfs["sector_trend"])
    chart_yoy        = chart_yoy_growth(dfs["sector_trend"])
    chart_heatmap    = chart_regional_heatmap(dfs["region_trend"])
    chart_share      = chart_sector_share(dfs["sector_share"])
    chart_salary     = chart_salary_rank(dfs["salary_rank"])
    chart_cum        = chart_cumulative_growth(dfs["cumulative_growth"])

    print("  Rendering forecast charts...")
    forecasts = forecasts or {}
    chart_nat_fc  = chart_national_forecast(forecasts["national_forecast"]) if forecasts else ""
    chart_sect_fc = chart_sector_forecast(forecasts["sector_forecast"])   if forecasts else ""

    top_prov_table = df_to_html_table(dfs["top_provinces"])

    reg_2023  = dfs["regional_summary"][dfs["regional_summary"]["year"] == 2023]
    reg_table = df_to_html_table(reg_2023.drop(columns="year"))

    # Wrap LLM summary paragraphs in <p> tags for HTML
    formatted_summary = "".join(
        f"<p>{p.strip()}</p>" for p in llm_summary.split("\n\n") if p.strip()
    ) if llm_summary else "<p><em>LLM summary not available.</em></p>"

    html = HTML_TEMPLATE.format(
        year_range=stats["year_range"],
        year_start=year_start,
        total_employed_2023=stats["total_employed_2023"],
        total_growth_pct=stats["total_growth_pct"],
        avg_salary_2023=stats["avg_salary_2023"],
        salary_growth_pct=stats["salary_growth_pct"],
        fastest_growing_sector=stats["fastest_growing_sector"],
        avg_unemployment_latest=stats["avg_unemployment_latest"],
        chart_national=chart_national,
        chart_sector_emp=chart_sector_emp,
        chart_yoy=chart_yoy,
        chart_heatmap=chart_heatmap,
        chart_share=chart_share,
        chart_salary=chart_salary,
        chart_cum=chart_cum,
        table_top_provinces=top_prov_table,
        table_regional_summary=reg_table,
        llm_summary=formatted_summary,
        chart_national_forecast=chart_nat_fc,
        chart_sector_forecast=chart_sect_fc,
    )

    report_path = os.path.join(OUTPUT_DIR, "report.html")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    print(f"  Report saved → {report_path}")
    return report_path
