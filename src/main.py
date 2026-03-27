"""
main.py
-------
Orchestrates the full pipeline:
  1. Generate synthetic Statistics Canada-style data
  2. Ingest and clean into normalized SQLite database
  3. Run SQL analytical queries
  4. Produce visualizations and HTML report
"""

import sys
import os
import time

# Allow sibling imports when run directly
sys.path.insert(0, os.path.dirname(__file__))

import generate_data
import ingest
import queries
import visualize
import forecast
import summarize


def step(label: str):
    print(f"\n{'='*60}")
    print(f"  {label}")
    print(f"{'='*60}")


def main():
    t0 = time.time()
    print("Canadian Public Sector Labour Market Analysis Pipeline")
    print("=" * 60)

    step("Step 1/6 — Generating raw datasets")
    generate_data.main()

    step("Step 2/6 — Ingesting data into SQLite")
    ingest.main()

    step("Step 3/6 — Running SQL analyses")
    print("Running SQL queries...")
    dfs = queries.run_all()

    step("Step 4/6 — Forecasting employment (2024–2028)")
    forecasts = forecast.run_forecasts(dfs)

    step("Step 5/6 — Generating AI executive summary (Claude API)")
    stats = visualize.summary_stats(dfs)
    llm_summary = summarize.generate_summary(dfs, stats, forecasts)
    print(f"  Summary generated ({len(llm_summary)} chars)")

    step("Step 6/6 — Building report")
    report_path = visualize.build_report(dfs, forecasts=forecasts, llm_summary=llm_summary)

    elapsed = round(time.time() - t0, 1)
    print(f"\nDone in {elapsed}s.")
    print(f"Open the report: {os.path.abspath(report_path)}")


if __name__ == "__main__":
    main()
