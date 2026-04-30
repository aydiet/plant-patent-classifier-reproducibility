#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Export a 1-row-per-DOCDB-family modeling corpus from an enriched publication-level parquet. "
            "Keeps only families that have at least one row with both final-English title and final-English abstract. "
            "Selects a deterministic representative family member and writes a family-level parquet."
        )
    )
    ap.add_argument(
        "--parquet",
        type=Path,
        default=Path("data/derived/global_corpus_publevel_plus_text_enriched_apple_nllb.parquet"),
        help="Input enriched publication-level parquet (local, gitignored).",
    )
    ap.add_argument(
        "--out-parquet",
        type=Path,
        default=Path("data/derived/family_modeling_corpus_en_complete.parquet"),
        help="Output family-level parquet (local, gitignored).",
    )
    ap.add_argument(
      "--exclude-families-csv",
      type=Path,
      default=None,
      help=(
        "Optional CSV containing docdb_family_id values to exclude (e.g. the labeled 600 families). "
        "Useful to prevent leakage when exporting an inference pool."
      ),
    )
    args = ap.parse_args()

    if not args.parquet.exists():
        raise SystemExit(f"Not found: {args.parquet}")

    con = duckdb.connect(database=":memory:")
    args.out_parquet.parent.mkdir(parents=True, exist_ok=True)

    parquet_posix = args.parquet.as_posix()
    out_posix = args.out_parquet.as_posix()

    with_prefix = "WITH"
    exclude_where = ""
    if args.exclude_families_csv is not None:
      if not args.exclude_families_csv.exists():
        raise SystemExit(f"Not found: {args.exclude_families_csv}")
      exclude_posix = args.exclude_families_csv.as_posix().replace("'", "''")
      with_prefix = (
        "WITH excluded_families AS ("
        "SELECT CAST(docdb_family_id AS BIGINT) AS docdb_family_id "
        f"FROM read_csv_auto('{exclude_posix}')"
        "),"
      )
      exclude_where = "AND docdb_family_id NOT IN (SELECT docdb_family_id FROM excluded_families)"

    # Representative selection rule (deterministic, per family):
    # 1) require both title and abstract present (non-empty)
    # 2) prefer larger combined length(title)+length(abstract)
    # 3) tie-breaker: longer abstract
    # 4) tie-breaker: larger appln_id
    export_sql = f"""
    COPY (
      {with_prefix}
      base AS (
        SELECT
          docdb_family_id,
          appln_id,

          appln_title_en_final,
          title_en_final_source,
          title_mt_engine,
          title_mt_rejected,
          title_mt_reject_reason,

          appln_abstract_en_final,
          abstract_en_final_source,
          abstract_mt_engine,
          abstract_mt_rejected,
          abstract_mt_reject_reason
        FROM read_parquet('{parquet_posix}')
        WHERE
          appln_title_en_final IS NOT NULL
          AND length(trim(appln_title_en_final)) > 0
          AND appln_abstract_en_final IS NOT NULL
          AND length(trim(appln_abstract_en_final)) > 0
          {exclude_where}
      ),
      ranked AS (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY docdb_family_id
            ORDER BY
              (length(appln_title_en_final) + length(appln_abstract_en_final)) DESC,
              length(appln_abstract_en_final) DESC,
              appln_id DESC
          ) AS rn
        FROM base
      )
      SELECT
        docdb_family_id,
        appln_id AS rep_appln_id,

        appln_title_en_final AS title_en,
        title_en_final_source AS title_source,
        title_mt_engine,
        title_mt_rejected,
        title_mt_reject_reason,

        appln_abstract_en_final AS abstract_en,
        abstract_en_final_source AS abstract_source,
        abstract_mt_engine,
        abstract_mt_rejected,
        abstract_mt_reject_reason,

        (title_en || '\n\n' || abstract_en) AS text_en
      FROM ranked
      WHERE rn = 1
    ) TO '{out_posix}' (FORMAT PARQUET, COMPRESSION ZSTD);
    """

    con.execute(export_sql)

    families_total = con.execute(
        f"SELECT COUNT(DISTINCT docdb_family_id) FROM read_parquet('{parquet_posix}')"
    ).fetchone()[0]
    families_complete = con.execute(
        f"""
        SELECT COUNT(DISTINCT docdb_family_id)
        FROM read_parquet('{parquet_posix}')
        WHERE
          appln_title_en_final IS NOT NULL
          AND length(trim(appln_title_en_final)) > 0
          AND appln_abstract_en_final IS NOT NULL
          AND length(trim(appln_abstract_en_final)) > 0
        """
    ).fetchone()[0]

    out_rows = con.execute(
        f"SELECT COUNT(*) FROM read_parquet('{out_posix}')"
    ).fetchone()[0]

    print(f"Wrote: {args.out_parquet}")
    print(f"Families total (input): {families_total:,}")
    print(f"Families with complete English title+abstract (input filter): {families_complete:,}")
    print(f"Rows in output (1 per family): {out_rows:,}")


if __name__ == "__main__":
    main()
