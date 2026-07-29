"""WS2 — build the longitudinal carbon firm-metric-year panel from RAW data.

The processed `long_clean.parquet` collapses to a latest-value snapshot and
drops `metric_year`. The raw Clarity AI files (pipe-delimited) DO carry the full
time dimension, so this reads them directly and preserves it.

For each carbon metric it keeps: perm_id, company_name, metric_name, group,
metric_year (date), provenance (REPORTED/ESTIMATED/CALCULATED/...), value, unit,
reported_date, industry, country. Output: outputs/carbon_panel.parquet plus a
small carbon_panel_by_year.parquet aggregate for the app.

Reads each unique source once (the *.csv.gz copies) to avoid double-counting the
files that exist as both .csv and .csv.gz.

Run:  python -m src.build_carbon_panel   (or `make carbon-panel`)
"""
from __future__ import annotations

import glob
import os

import pandas as pd

from src.analysis import CARBON_METRICS

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")
RAW = os.path.join(HERE, "..", "..", "Raw data ESG data 2025Feb1")

USE_COLS = [
    "company_name", "perm_id", "data_type", "disclosure", "metric_name",
    "metric_unit", "metric_value", "metric_year", "metric_period",
    "reported_date", "pillar", "headquarter_country",
]


def _unique_sources() -> list:
    """One path per source file; prefer .csv.gz so duplicated files aren't
    counted twice (several exist as both .csv and .csv.gz)."""
    gz = {os.path.basename(p)[:-3]: p for p in glob.glob(os.path.join(RAW, "*.csv.gz"))}
    plain = {os.path.basename(p): p for p in glob.glob(os.path.join(RAW, "*.csv"))}
    names = set(gz) | set(plain)
    return [gz.get(n, plain.get(n)) for n in sorted(names)]


def build() -> None:
    carbon = set(CARBON_METRICS)
    frames = []
    for path in _unique_sources():
        comp = "gzip" if path.endswith(".gz") else None
        for chunk in pd.read_csv(path, sep="|", dtype=str, compression=comp,
                                 usecols=USE_COLS, chunksize=400_000):
            hit = chunk[chunk["metric_name"].isin(carbon)]
            if len(hit):
                frames.append(hit)
    df = pd.concat(frames, ignore_index=True)

    # types + derived fields
    df["year"] = pd.to_datetime(df["metric_year"], errors="coerce").dt.year
    df["value_num"] = pd.to_numeric(df["metric_value"], errors="coerce")
    df["group"] = df["metric_name"].map(CARBON_METRICS)
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)

    # a firm reports one value per metric-year; drop exact dup rows, then keep the
    # last reported_date if a firm-metric-year appears more than once.
    df = df.sort_values("reported_date")
    df = df.drop_duplicates(["perm_id", "metric_name", "year"], keep="last")

    # attach industry from company_meta
    meta = pd.read_parquet(os.path.join(OUT, "company_meta.parquet"))
    df["perm_id"] = df["perm_id"].astype(str)
    meta["perm_id"] = meta["perm_id"].astype(str)
    df = df.merge(meta[["perm_id", "industry"]].drop_duplicates(),
                  on="perm_id", how="left")

    keep = ["perm_id", "company_name", "industry", "headquarter_country",
            "metric_name", "group", "year", "disclosure", "value_num",
            "metric_unit", "reported_date"]
    panel = df[keep].reset_index(drop=True)
    panel.to_parquet(os.path.join(OUT, "carbon_panel.parquet"))

    # app-friendly aggregate: coverage + provenance mix per metric-year
    panel["is_reported"] = (panel["disclosure"] == "REPORTED").astype(int)
    panel["is_estimated"] = (panel["disclosure"] == "ESTIMATED").astype(int)
    by_year = (panel.groupby(["metric_name", "group", "year"])
               .agg(observations=("perm_id", "count"),
                    companies=("perm_id", "nunique"),
                    reported_share=("is_reported", "mean"),
                    estimated_share=("is_estimated", "mean"))
               .reset_index())
    by_year.to_parquet(os.path.join(OUT, "carbon_panel_by_year.parquet"))

    yrs = sorted(panel["year"].unique())
    print(f"carbon_panel: {len(panel):,} obs | {panel.perm_id.nunique():,} firms "
          f"| {panel.metric_name.nunique()} metrics | years {yrs[0]}-{yrs[-1]}")
    print(f"carbon_panel_by_year: {len(by_year):,} rows")


if __name__ == "__main__":
    build()
