"""
Data preparation pipeline for the ESG showcase project.

Turns ~6.6M rows of long-format, mixed-type ESG metric data (across six
category files) into analysis-ready artifacts:

  outputs/long_clean.parquet      tidy long table (one row per company-metric)
  outputs/wide_matrix.parquet     companies x metrics matrix (numeric, for PCA)
  outputs/company_meta.parquet    perm_id -> name, country, industry
  outputs/metric_catalog.parquet  metric_name -> pillar, category, coverage

Design decisions (documented because they ARE the data-science work):
  * Source = the pre-categorized files (already E/S/G x Risk/Opportunity tagged,
    comma-delimited with headers) rather than the raw '|' files.
  * Keep only the latest observation per (company, metric) to collapse the
    year dimension -- the wide matrix is a current-state snapshot.
  * Yes/No metrics -> 1/0 ; numeric metrics kept as-is, standardized later.
  * Pivot to wide, then drop ultra-sparse metrics/companies so PCA is meaningful.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
CATEGORIZED = os.path.join(ROOT, "categorized_esg_data(25Feb)")
INDUSTRY = os.path.join(
    ROOT, "Industry Matching List(Clarity AI dataset)1", "industry.csv"
)
OUT = os.path.join(HERE, "..", "outputs")
os.makedirs(OUT, exist_ok=True)

# Coverage thresholds for the wide matrix (tuneable).
MIN_COMPANIES_PER_METRIC = 2000   # drop metrics observed for too few companies
MIN_METRICS_PER_COMPANY = 5       # drop companies with too few metrics

USECOLS = [
    "company_name", "perm_id", "data_type", "disclosure", "metric_name",
    "metric_unit", "metric_value", "metric_year", "pillar",
    "headquarter_country", "category",
]


def load_long() -> pd.DataFrame:
    """Read and concatenate all six categorized files into one long table."""
    files = sorted(glob.glob(os.path.join(CATEGORIZED, "*", "*.csv")))
    if not files:
        raise FileNotFoundError(f"No categorized CSVs under {CATEGORIZED}")
    frames = []
    for f in files:
        print(f"  reading {os.path.basename(f)} ...", flush=True)
        df = pd.read_csv(
            f, usecols=USECOLS, dtype=str, on_bad_lines="skip",
            low_memory=False,
        )
        frames.append(df)
    long = pd.concat(frames, ignore_index=True)
    print(f"  raw long rows: {len(long):,}")
    return long


def to_numeric_value(df: pd.DataFrame) -> pd.DataFrame:
    """Coerce metric_value to numeric; map Yes/No units to 1/0."""
    df = df.copy()
    val = pd.to_numeric(df["metric_value"], errors="coerce")
    # Yes/No metrics already arrive as 0/1 numeric strings, so coercion handles
    # them. Drop rows that are non-numeric (free text, blanks).
    df["value_num"] = val
    df = df.dropna(subset=["value_num", "perm_id", "metric_name"])
    return df


def latest_per_company_metric(df: pd.DataFrame) -> pd.DataFrame:
    """Keep the most recent observation per (perm_id, metric_name)."""
    df = df.copy()
    df["year_sort"] = pd.to_numeric(
        df["metric_year"].str[:4], errors="coerce"
    ).fillna(0)
    df = df.sort_values("year_sort")
    df = df.drop_duplicates(subset=["perm_id", "metric_name"], keep="last")
    return df


def build_company_meta(df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        df[["perm_id", "company_name", "headquarter_country"]]
        .drop_duplicates(subset="perm_id")
        .reset_index(drop=True)
    )
    ind = pd.read_csv(INDUSTRY, dtype=str)
    ind = ind.rename(columns={"Industry": "industry", "perm_id": "perm_id"})
    ind = ind[["perm_id", "industry"]].drop_duplicates(subset="perm_id")
    meta = meta.merge(ind, on="perm_id", how="left")
    meta["industry"] = meta["industry"].fillna("Unknown")
    return meta


def build_metric_catalog(df: pd.DataFrame) -> pd.DataFrame:
    cat = (
        df.groupby("metric_name")
        .agg(
            pillar=("pillar", "first"),
            category=("category", "first"),
            unit=("metric_unit", "first"),
            n_companies=("perm_id", "nunique"),
        )
        .reset_index()
        .sort_values("n_companies", ascending=False)
    )
    return cat


def build_wide(df: pd.DataFrame, catalog: pd.DataFrame) -> pd.DataFrame:
    keep_metrics = catalog.loc[
        catalog["n_companies"] >= MIN_COMPANIES_PER_METRIC, "metric_name"
    ]
    sub = df[df["metric_name"].isin(keep_metrics)]
    wide = sub.pivot_table(
        index="perm_id", columns="metric_name",
        values="value_num", aggfunc="mean",
    )
    # Drop companies with too few observed metrics.
    enough = wide.notna().sum(axis=1) >= MIN_METRICS_PER_COMPANY
    wide = wide[enough]
    print(
        f"  wide matrix: {wide.shape[0]:,} companies x {wide.shape[1]} metrics"
        f"  ({wide.notna().mean().mean():.1%} filled)"
    )
    return wide


def main() -> None:
    print("1/5 loading long data ...")
    long = load_long()

    print("2/5 numeric coercion ...")
    long = to_numeric_value(long)
    print(f"  numeric rows: {len(long):,}")

    # Save the disclosure breakdown from the FULL (pre-collapse) data so the
    # reported-vs-estimated split counts every observation, not just latest.
    disc = long.loc[
        long["disclosure"].isin(["REPORTED", "ESTIMATED"]),
        ["metric_name", "pillar", "disclosure"],
    ]
    disc.to_parquet(os.path.join(OUT, "disclosure_long.parquet"), index=False)
    print(f"  disclosure rows: {len(disc):,}")

    print("3/5 collapsing to latest per company-metric ...")
    long = latest_per_company_metric(long)
    print(f"  collapsed rows: {len(long):,}")

    long.to_parquet(os.path.join(OUT, "long_clean.parquet"), index=False)

    print("4/5 building metadata + catalog ...")
    meta = build_company_meta(long)
    catalog = build_metric_catalog(long)
    meta.to_parquet(os.path.join(OUT, "company_meta.parquet"), index=False)
    catalog.to_parquet(os.path.join(OUT, "metric_catalog.parquet"), index=False)
    print(f"  companies: {len(meta):,} | metrics: {len(catalog):,}")

    print("5/5 building wide matrix ...")
    wide = build_wide(long, catalog)
    wide.to_parquet(os.path.join(OUT, "wide_matrix.parquet"))

    print("\nDONE. Artifacts written to outputs/")


if __name__ == "__main__":
    main()
