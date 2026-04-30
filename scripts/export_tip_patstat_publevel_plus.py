#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Run the TIP/PATSTAT export query for the publication-level corpus and write Parquet outputs. "
            "Requires access to EPO TIP PATSTAT BigQuery via epo.tipdata.patstat.PatstatClient."
        )
    )
    p.add_argument(
        "--env",
        default="PROD",
        help="TIP environment identifier passed to PatstatClient (default: PROD).",
    )
    p.add_argument(
        "--sql-file",
        default="scripts/tip_patstat_export_global_corpus_publevel_plus.sql",
        help="Path to the SQL file to execute.",
    )
    p.add_argument(
        "--out-parquet",
        default="data/raw/global_corpus_publevel_plus.parquet",
        help="Publication-level output parquet path.",
    )
    p.add_argument(
        "--out-families-parquet",
        default="data/raw/global_corpus_families.parquet",
        help="Family-level unique ID output parquet path.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from epo.tipdata.patstat import PatstatClient

    sql_path = Path(args.sql_file)
    sql = sql_path.read_text(encoding="utf-8")

    patstat = PatstatClient(env=args.env)
    res = patstat.sql_query(sql, use_legacy_sql=False)
    df = pd.DataFrame(res)

    out_parquet = Path(args.out_parquet)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_parquet, index=False)

    families = df[["docdb_family_id"]].drop_duplicates()
    out_families = Path(args.out_families_parquet)
    out_families.parent.mkdir(parents=True, exist_ok=True)
    families.to_parquet(out_families, index=False)

    print("Rows, Cols:", df.shape)
    print("Unique families:", len(families))


if __name__ == "__main__":
    main()
