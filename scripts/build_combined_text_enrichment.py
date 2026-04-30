from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pandas as pd


_NLLB_ENGINE = "nllb200_coreml_256"


def _sql_nllb_boilerplate_predicate(col_name: str) -> str:
    # Conservative heuristic: reject known hallucinated EU Commission/regulation boilerplate.
    needles = [
        "the commission",
        "implementing acts",
        "delegated acts",
        "this regulation",
    ]
    return "(" + " OR ".join([f"lower({col_name}) LIKE '%{n}%'" for n in needles]) + ")"


@dataclass(frozen=True)
class EnrichmentReport:
    rows_total: int
    families_total: int
    families_with_family_en_title: int
    families_with_family_en_abstract: int
    families_with_mt_title: int
    families_with_mt_abstract: int
    families_with_mt_title_rejected: int
    families_with_mt_abstract_rejected: int
    rows_with_final_title_en: int
    rows_with_final_abstract_en: int


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fetchone_int(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None:
        raise RuntimeError(f"No rows for: {sql}")
    return int(row[0])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a single derived parquet that combines: (1) family-level English backfill candidates (from existing English within family) "
            "and (2) MT outputs (e.g., Apple Translation + NLLB fallback), without overwriting original title/abstract fields."
        )
    )
    p.add_argument(
        "--input",
        default="data/raw/global_corpus_publevel_plus.parquet",
        help="Input publication-level parquet (raw).",
    )
    p.add_argument(
        "--translations-jsonl",
        default="metadata/family_translation_results_combined.jsonl",
        help="Merged translation results JSONL (Apple primary + NLLB fallback).",
    )
    p.add_argument(
        "--output",
        default="data/derived/global_corpus_publevel_plus_text_enriched.parquet",
        help="Output derived parquet.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute stats only; do not write parquet.",
    )
    return p.parse_args()


def _load_translations_df(jsonl_path: Path) -> pd.DataFrame:
    rows: list[dict] = []
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    if not rows:
        return pd.DataFrame(columns=[
            "docdb_family_id",
            "field",
            "translated_text",
            "ok",
            "source_lg",
            "source_appln_id",
            "source_text_sha256",
            "engine",
            "os_version",
            "created_at_utc",
        ])
    return pd.DataFrame(rows)


def compute_and_optionally_write(
    input_path: Path,
    translations_jsonl: Path,
    output_path: Path,
    dry_run: bool,
) -> EnrichmentReport:
    con = duckdb.connect(database=":memory:")
    con.execute(f"CREATE VIEW corpus AS SELECT * FROM read_parquet({_sql_string_literal(str(input_path))})")

    df = _load_translations_df(translations_jsonl)
    con.register("translation_rows", df)
    con.execute(
        """
        CREATE TEMP TABLE translations AS
        SELECT
          docdb_family_id::BIGINT AS docdb_family_id,
          field::VARCHAR AS field,
          source_appln_id::BIGINT AS source_appln_id,
          source_lg::VARCHAR AS source_lg,
          target_lg::VARCHAR AS target_lg,
          source_text_sha256::VARCHAR AS source_text_sha256,
          translated_text::VARCHAR AS translated_text,
          ok::BOOLEAN AS ok,
          error::VARCHAR AS error,
          engine::VARCHAR AS engine,
          os_version::VARCHAR AS os_version,
          created_at_utc::VARCHAR AS created_at_utc
        FROM translation_rows
        """
    )

    # Compute best English title/abstract per family (deterministic: longest text, tie-break by larger appln_id)
    con.execute(
        """
        CREATE TEMP TABLE family_en_title AS
        WITH base AS (
          SELECT
            docdb_family_id,
            appln_id,
            NULLIF(trim(appln_title), '') AS appln_title,
            NULLIF(lower(trim(title_lg)), '') AS title_lg
          FROM corpus
        ),
        ranked AS (
          SELECT
            docdb_family_id,
            appln_id AS en_title_source_appln_id,
            appln_title AS appln_title_en_family,
            ROW_NUMBER() OVER (
              PARTITION BY docdb_family_id
              ORDER BY length(appln_title) DESC, appln_id DESC
            ) AS rn
          FROM base
          WHERE appln_title IS NOT NULL AND title_lg = 'en'
        )
        SELECT
          docdb_family_id,
          en_title_source_appln_id,
          appln_title_en_family
        FROM ranked
        WHERE rn = 1
        """
    )

    con.execute(
        """
        CREATE TEMP TABLE family_en_abstract AS
        WITH base AS (
          SELECT
            docdb_family_id,
            appln_id,
            NULLIF(trim(appln_abstract), '') AS appln_abstract,
            NULLIF(lower(trim(abstract_lg)), '') AS abstract_lg
          FROM corpus
        ),
        ranked AS (
          SELECT
            docdb_family_id,
            appln_id AS en_abstract_source_appln_id,
            appln_abstract AS appln_abstract_en_family,
            ROW_NUMBER() OVER (
              PARTITION BY docdb_family_id
              ORDER BY length(appln_abstract) DESC, appln_id DESC
            ) AS rn
          FROM base
          WHERE appln_abstract IS NOT NULL AND abstract_lg = 'en'
        )
        SELECT
          docdb_family_id,
          en_abstract_source_appln_id,
          appln_abstract_en_family
        FROM ranked
        WHERE rn = 1
        """
    )

    # MT lookups
    nllb_reject_pred = f"(engine = '{_NLLB_ENGINE}' AND translated_text IS NOT NULL AND {_sql_nllb_boilerplate_predicate('translated_text')})"
    con.execute(
        f"""
        CREATE TEMP TABLE title_mt AS
        SELECT
          docdb_family_id,
          CASE WHEN {nllb_reject_pred} THEN NULL ELSE translated_text END AS appln_title_en_mt,
          source_lg AS title_mt_source_lg,
          source_appln_id AS title_mt_source_appln_id,
          source_text_sha256 AS title_mt_source_sha256,
          engine AS title_mt_engine,
          os_version AS title_mt_os_version,
          created_at_utc AS title_mt_created_at_utc,
          CASE WHEN {nllb_reject_pred} THEN TRUE ELSE FALSE END AS title_mt_rejected,
          CASE WHEN {nllb_reject_pred} THEN 'nllb_boilerplate' ELSE NULL END AS title_mt_reject_reason
        FROM translations
        WHERE field = 'title' AND ok = true AND translated_text IS NOT NULL
        """
    )
    con.execute(
        f"""
        CREATE TEMP TABLE abstract_mt AS
        SELECT
          docdb_family_id,
          CASE WHEN {nllb_reject_pred} THEN NULL ELSE translated_text END AS appln_abstract_en_mt,
          source_lg AS abstract_mt_source_lg,
          source_appln_id AS abstract_mt_source_appln_id,
          source_text_sha256 AS abstract_mt_source_sha256,
          engine AS abstract_mt_engine,
          os_version AS abstract_mt_os_version,
          created_at_utc AS abstract_mt_created_at_utc,
          CASE WHEN {nllb_reject_pred} THEN TRUE ELSE FALSE END AS abstract_mt_rejected,
          CASE WHEN {nllb_reject_pred} THEN 'nllb_boilerplate' ELSE NULL END AS abstract_mt_reject_reason
        FROM translations
        WHERE field = 'abstract' AND ok = true AND translated_text IS NOT NULL
        """
    )

    rows_total = _fetchone_int(con, "SELECT COUNT(*) FROM corpus")
    families_total = _fetchone_int(con, "SELECT COUNT(DISTINCT docdb_family_id) FROM corpus")
    families_with_family_en_title = _fetchone_int(con, "SELECT COUNT(*) FROM family_en_title")
    families_with_family_en_abstract = _fetchone_int(con, "SELECT COUNT(*) FROM family_en_abstract")
    families_with_mt_title = _fetchone_int(
      con, "SELECT COUNT(DISTINCT docdb_family_id) FROM title_mt WHERE appln_title_en_mt IS NOT NULL"
    )
    families_with_mt_abstract = _fetchone_int(
      con, "SELECT COUNT(DISTINCT docdb_family_id) FROM abstract_mt WHERE appln_abstract_en_mt IS NOT NULL"
    )
    families_with_mt_title_rejected = _fetchone_int(
      con, "SELECT COUNT(DISTINCT docdb_family_id) FROM title_mt WHERE title_mt_rejected = TRUE"
    )
    families_with_mt_abstract_rejected = _fetchone_int(
      con, "SELECT COUNT(DISTINCT docdb_family_id) FROM abstract_mt WHERE abstract_mt_rejected = TRUE"
    )

    # Final per-row English text selection:
    # - Prefer row's own English text
    # - Else prefer family English text (from within-family English)
    # - Else prefer MT English text
    con.execute(
        """
        CREATE TEMP VIEW enriched AS
        SELECT
          c.*,

          -- Family-English backfill columns
          ft.appln_title_en_family,
          ft.en_title_source_appln_id,
          fa.appln_abstract_en_family,
          fa.en_abstract_source_appln_id,

          -- MT columns
          tm.appln_title_en_mt,
          tm.title_mt_source_lg,
          tm.title_mt_source_appln_id,
          tm.title_mt_source_sha256,
          tm.title_mt_engine,
          tm.title_mt_os_version,
          tm.title_mt_created_at_utc,
          tm.title_mt_rejected,
          tm.title_mt_reject_reason,

          am.appln_abstract_en_mt,
          am.abstract_mt_source_lg,
          am.abstract_mt_source_appln_id,
          am.abstract_mt_source_sha256,
          am.abstract_mt_engine,
          am.abstract_mt_os_version,
          am.abstract_mt_created_at_utc,
          am.abstract_mt_rejected,
          am.abstract_mt_reject_reason,

          -- Final English text fields (do not overwrite originals)
          CASE
            WHEN c.appln_title IS NOT NULL AND lower(c.title_lg) = 'en' THEN c.appln_title
            WHEN ft.appln_title_en_family IS NOT NULL THEN ft.appln_title_en_family
            WHEN tm.appln_title_en_mt IS NOT NULL THEN tm.appln_title_en_mt
            ELSE NULL
          END AS appln_title_en_final,

          CASE
            WHEN c.appln_title IS NOT NULL AND lower(c.title_lg) = 'en' THEN 'original_en'
            WHEN ft.appln_title_en_family IS NOT NULL THEN 'family_en'
            WHEN tm.appln_title_en_mt IS NOT NULL THEN 'mt_en'
            ELSE NULL
          END AS title_en_final_source,

          CASE
            WHEN c.appln_abstract IS NOT NULL AND lower(c.abstract_lg) = 'en' THEN c.appln_abstract
            WHEN fa.appln_abstract_en_family IS NOT NULL THEN fa.appln_abstract_en_family
            WHEN am.appln_abstract_en_mt IS NOT NULL THEN am.appln_abstract_en_mt
            ELSE NULL
          END AS appln_abstract_en_final,

          CASE
            WHEN c.appln_abstract IS NOT NULL AND lower(c.abstract_lg) = 'en' THEN 'original_en'
            WHEN fa.appln_abstract_en_family IS NOT NULL THEN 'family_en'
            WHEN am.appln_abstract_en_mt IS NOT NULL THEN 'mt_en'
            ELSE NULL
          END AS abstract_en_final_source

        FROM corpus c
        LEFT JOIN family_en_title ft USING (docdb_family_id)
        LEFT JOIN family_en_abstract fa USING (docdb_family_id)
        LEFT JOIN title_mt tm USING (docdb_family_id)
        LEFT JOIN abstract_mt am USING (docdb_family_id)
        """
    )

    rows_with_final_title_en = _fetchone_int(con, "SELECT COUNT(*) FROM enriched WHERE appln_title_en_final IS NOT NULL")
    rows_with_final_abstract_en = _fetchone_int(
        con, "SELECT COUNT(*) FROM enriched WHERE appln_abstract_en_final IS NOT NULL"
    )

    report = EnrichmentReport(
        rows_total=rows_total,
        families_total=families_total,
        families_with_family_en_title=families_with_family_en_title,
        families_with_family_en_abstract=families_with_family_en_abstract,
        families_with_mt_title=families_with_mt_title,
        families_with_mt_abstract=families_with_mt_abstract,
        families_with_mt_title_rejected=families_with_mt_title_rejected,
        families_with_mt_abstract_rejected=families_with_mt_abstract_rejected,
        rows_with_final_title_en=rows_with_final_title_en,
        rows_with_final_abstract_en=rows_with_final_abstract_en,
    )

    if dry_run:
        return report

    output_path.parent.mkdir(parents=True, exist_ok=True)
    con.execute(
        f"COPY (SELECT * FROM enriched) TO {_sql_string_literal(str(output_path))} (FORMAT PARQUET, COMPRESSION ZSTD)"
    )
    return report


def main() -> None:
    args = parse_args()
    report = compute_and_optionally_write(
        input_path=Path(args.input),
        translations_jsonl=Path(args.translations_jsonl),
        output_path=Path(args.output),
        dry_run=bool(args.dry_run),
    )

    print("Combined text enrichment complete.")
    print(f"Rows total: {report.rows_total:,}")
    print(f"Families total: {report.families_total:,}")
    print(f"Families with family English title: {report.families_with_family_en_title:,}")
    print(f"Families with family English abstract: {report.families_with_family_en_abstract:,}")
    print(f"Families with MT title: {report.families_with_mt_title:,}")
    print(f"Families with MT abstract: {report.families_with_mt_abstract:,}")
    print(f"Families with MT title rejected (QC): {report.families_with_mt_title_rejected:,}")
    print(f"Families with MT abstract rejected (QC): {report.families_with_mt_abstract_rejected:,}")
    print(f"Rows with final English title: {report.rows_with_final_title_en:,}")
    print(f"Rows with final English abstract: {report.rows_with_final_abstract_en:,}")
    if args.dry_run:
        print("(dry-run) Did not write derived parquet.")
    else:
        print(f"Wrote derived parquet: {args.output}")


if __name__ == "__main__":
    main()
