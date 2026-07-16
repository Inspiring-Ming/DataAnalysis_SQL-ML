"""Build a SQLite database from the parquet outputs.

Each outputs/<name>.parquet becomes a table <name>. Index columns (perm_id,
metric_name, industry) are written as real columns and indexed so joins and
lookups are fast.

Two builds:
  * full (default) -> outputs/esg_full.db  -- ALL tables incl. the 5.7M-row
        raw observation tables. ~960 MB; for local SQL exploration only.
  * --slim         -> outputs/esg.db       -- analytical tables only; small
        (~50 MB), committed to the repo and used by the hosted app.

Run:  python -m src.build_sqlite          # full, local
      python -m src.build_sqlite --slim   # slim, hosted (committed)
"""
from __future__ import annotations

import argparse
import glob
import os
import sqlite3

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "..", "outputs")

# join/lookup keys to index per table (only created if the column exists)
KEYS = {
    "perm_id", "metric_name", "industry", "cluster", "pillar", "disclosure",
    "headquarter_country",
}

# raw observation tables excluded from the slim (hosted) build -- too large.
HEAVY = {"disclosure_long", "long_clean"}


def build(db_path: str, slim: bool) -> None:
    if os.path.exists(db_path):
        os.remove(db_path)
    con = sqlite3.connect(db_path)
    try:
        for f in sorted(glob.glob(os.path.join(OUT, "*.parquet"))):
            name = os.path.basename(f).replace(".parquet", "")
            if slim and name in HEAVY:
                print(f"  (skip heavy table `{name}` in slim build)")
                continue
            df = pd.read_parquet(f)
            # promote a named index (perm_id, metric_name, industry) to a column
            if df.index.name is not None:
                df = df.reset_index()
            df.to_sql(name, con, if_exists="replace", index=False)
            for col in df.columns:
                if col in KEYS:
                    safe = col.replace('"', '""')
                    con.execute(
                        f'CREATE INDEX IF NOT EXISTS "ix_{name}_{col}" '
                        f'ON "{name}" ("{safe}")'
                    )
            print(f"  {name:22s} {len(df):>9,} rows -> table `{name}`")
        con.commit()
        con.execute("VACUUM")
    finally:
        con.close()
    size_mb = os.path.getsize(db_path) / 1e6
    print(f"\nDONE. Wrote {db_path}  ({size_mb:.1f} MB)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--slim", action="store_true",
                    help="analytical tables only (hosted build -> esg.db)")
    args = ap.parse_args()
    db = os.path.join(OUT, "esg.db" if args.slim else "esg_full.db")
    build(db, slim=args.slim)


if __name__ == "__main__":
    main()
