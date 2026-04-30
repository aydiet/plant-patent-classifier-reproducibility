from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import duckdb


@dataclass(frozen=True)
class CandidateCounts:
    n_families_total: int
    n_title_candidates: int
    n_abstract_candidates: int
    n_total_candidates: int


def _markdown_table(rows: list[tuple[str, int]], col1_name: str, col2_name: str) -> str:
  lines = [f"| {col1_name} | {col2_name} |", "|---|---:|"]
  for k, v in rows:
    key = k if k else "(missing)"
    lines.append(f"| {key} | {v:,} |")
  return "\n".join(lines)


def _sql_string_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _fetchone_scalar_int(con: duckdb.DuckDBPyConnection, sql: str) -> int:
    row = con.execute(sql).fetchone()
    if row is None or len(row) == 0:
        raise RuntimeError(f"Expected 1-row result for query, got none: {sql}")
    return int(row[0])


def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Build a family-level translation candidate list for Apple Translation. "
            "Selects one title and/or one abstract per family ONLY when no English exists anywhere in the family. "
            "Outputs JSONL candidates suitable for a Swift CLI translator."
        )
    )
    p.add_argument(
        "--input",
        default="data/raw/global_corpus_publevel_plus.parquet",
        help="Input publication-level parquet.",
    )
    p.add_argument(
        "--output-jsonl",
        default="metadata/family_translation_candidates.jsonl",
        help="Output JSONL candidate file.",
    )
    p.add_argument(
        "--report-md",
        default="metadata/family_translation_candidates_report.md",
        help="Markdown report output path.",
    )
    p.add_argument(
        "--target-language",
        default="en",
        help="Target language identifier (default: en).",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute counts/report only; do not write JSONL output.",
    )
    return p.parse_args()


def compute_and_optionally_write(
    input_path: Path,
    output_jsonl: Path,
    report_md: Path,
    target_language: str,
    dry_run: bool,
) -> CandidateCounts:
    con = duckdb.connect(database=":memory:")
    input_sql = _sql_string_literal(str(input_path))
    con.execute(f"CREATE VIEW corpus AS SELECT * FROM read_parquet({input_sql})")

    # Candidate rule:
    # - Only families with at least one non-empty text in that field
    # - AND with zero English texts for that field anywhere in the family
    # Deterministic source selection within family:
    #   1) prefer longer text
    #   2) tie-breaker by larger appln_id

    # Title candidates
    con.execute(
        """
        CREATE TEMP TABLE title_candidates AS
        WITH base AS (
          SELECT
            docdb_family_id,
            appln_id,
            NULLIF(trim(appln_title), '') AS appln_title,
            NULLIF(lower(trim(title_lg)), '') AS title_lg
          FROM corpus
        ),
        fam_flags AS (
          SELECT
            docdb_family_id,
            MAX(CASE WHEN appln_title IS NOT NULL THEN 1 ELSE 0 END) AS fam_has_any_title,
            MAX(CASE WHEN appln_title IS NOT NULL AND title_lg = 'en' THEN 1 ELSE 0 END) AS fam_has_title_en
          FROM base
          GROUP BY docdb_family_id
        ),
        eligible AS (
          SELECT f.docdb_family_id
          FROM fam_flags f
          WHERE f.fam_has_any_title = 1 AND f.fam_has_title_en = 0
        ),
        ranked AS (
          SELECT
            b.docdb_family_id,
            b.appln_id AS source_appln_id,
            b.title_lg AS source_lg,
            b.appln_title AS source_text,
            length(b.appln_title) AS source_text_len,
            ROW_NUMBER() OVER (
              PARTITION BY b.docdb_family_id
              ORDER BY length(b.appln_title) DESC, b.appln_id DESC
            ) AS rn
          FROM base b
          JOIN eligible e USING (docdb_family_id)
          WHERE b.appln_title IS NOT NULL
        )
        SELECT
          docdb_family_id,
          'title' AS field,
          source_appln_id,
          source_lg,
          source_text,
          source_text_len
        FROM ranked
        WHERE rn = 1
        """
    )

    # Abstract candidates
    con.execute(
        """
        CREATE TEMP TABLE abstract_candidates AS
        WITH base AS (
          SELECT
            docdb_family_id,
            appln_id,
            NULLIF(trim(appln_abstract), '') AS appln_abstract,
            NULLIF(lower(trim(abstract_lg)), '') AS abstract_lg
          FROM corpus
        ),
        fam_flags AS (
          SELECT
            docdb_family_id,
            MAX(CASE WHEN appln_abstract IS NOT NULL THEN 1 ELSE 0 END) AS fam_has_any_abstract,
            MAX(CASE WHEN appln_abstract IS NOT NULL AND abstract_lg = 'en' THEN 1 ELSE 0 END) AS fam_has_abstract_en
          FROM base
          GROUP BY docdb_family_id
        ),
        eligible AS (
          SELECT f.docdb_family_id
          FROM fam_flags f
          WHERE f.fam_has_any_abstract = 1 AND f.fam_has_abstract_en = 0
        ),
        ranked AS (
          SELECT
            b.docdb_family_id,
            b.appln_id AS source_appln_id,
            b.abstract_lg AS source_lg,
            b.appln_abstract AS source_text,
            length(b.appln_abstract) AS source_text_len,
            ROW_NUMBER() OVER (
              PARTITION BY b.docdb_family_id
              ORDER BY length(b.appln_abstract) DESC, b.appln_id DESC
            ) AS rn
          FROM base b
          JOIN eligible e USING (docdb_family_id)
          WHERE b.appln_abstract IS NOT NULL
        )
        SELECT
          docdb_family_id,
          'abstract' AS field,
          source_appln_id,
          source_lg,
          source_text,
          source_text_len
        FROM ranked
        WHERE rn = 1
        """
    )

    n_families_total = _fetchone_scalar_int(con, "SELECT COUNT(DISTINCT docdb_family_id) FROM corpus")
    n_title_candidates = _fetchone_scalar_int(con, "SELECT COUNT(*) FROM title_candidates")
    n_abstract_candidates = _fetchone_scalar_int(con, "SELECT COUNT(*) FROM abstract_candidates")
    n_total_candidates = n_title_candidates + n_abstract_candidates

    title_lang_counts = con.execute(
      """
      SELECT COALESCE(source_lg, '') AS source_lg, COUNT(*)::INTEGER AS n
      FROM title_candidates
      GROUP BY 1
      ORDER BY n DESC, source_lg ASC
      """
    ).fetchall()
    abstract_lang_counts = con.execute(
      """
      SELECT COALESCE(source_lg, '') AS source_lg, COUNT(*)::INTEGER AS n
      FROM abstract_candidates
      GROUP BY 1
      ORDER BY n DESC, source_lg ASC
      """
    ).fetchall()

    output_jsonl.parent.mkdir(parents=True, exist_ok=True)
    report_md.parent.mkdir(parents=True, exist_ok=True)

    # Write report (always)
    report_md.write_text(
        "\n".join(
            [
                "# Family translation candidate report",
                "",
                f"**Input:** `{input_path.as_posix()}`  ",
                f"**Output (JSONL):** `{output_jsonl.as_posix()}`  ",
                f"**Target language:** `{target_language}`",
                "",
                "This selects one **title** and/or one **abstract** per `docdb_family_id` for translation **only** when the family contains no English for that field.",
                "",
                "Selection rule (deterministic): choose the longest available text in the family; tie-breaker by larger `appln_id`.",
                "",
                "## Counts",
                "",
                f"- Families total (distinct): {n_families_total:,}",
                f"- Title candidates (families with no English title): {n_title_candidates:,}",
                f"- Abstract candidates (families with no English abstract): {n_abstract_candidates:,}",
                f"- Total candidate rows: {n_total_candidates:,}",
                "",
          "## Source language breakdown",
          "",
          "These are the source languages present in the candidate set (by field). Apple Translation requires the corresponding language pairs to be installed in System Settings.",
          "",
          "### Title candidates",
          "",
          _markdown_table([(str(lg), int(n)) for lg, n in title_lang_counts], "source_lg", "candidates"),
          "",
          "### Abstract candidates",
          "",
          _markdown_table([(str(lg), int(n)) for lg, n in abstract_lang_counts], "source_lg", "candidates"),
          "",
                "## Notes",
                "",
                "- Candidate rows are meant to be consumed by a Swift CLI that uses Apple’s Translation framework.",
                "- The CLI should group rows by `source_lg` because Apple batch translation APIs require a consistent source language per batch.",
                "",
            ]
        ),
        encoding="utf-8",
    )

    if dry_run:
        return CandidateCounts(
            n_families_total=n_families_total,
            n_title_candidates=n_title_candidates,
            n_abstract_candidates=n_abstract_candidates,
            n_total_candidates=n_total_candidates,
        )

    # Stream JSONL (keeps memory small; candidate tables are small anyway)
    target_language_norm = target_language.strip().lower()
    with output_jsonl.open("w", encoding="utf-8") as f:
        for field_table in ("title_candidates", "abstract_candidates"):
            rows = con.execute(
                f"SELECT docdb_family_id, field, source_appln_id, source_lg, source_text, source_text_len FROM {field_table}"
            ).fetchall()
            for docdb_family_id, field, source_appln_id, source_lg, source_text, source_text_len in rows:
                source_lg_norm = None if source_lg is None else str(source_lg).strip().lower()
                text = str(source_text)
                # Hash includes language pairing so caching can be safe across target changes.
                sha_basis = f"{source_lg_norm}\n{target_language_norm}\n{text}"
                record = {
                    "docdb_family_id": int(docdb_family_id),
                    "field": field,
                    "source_appln_id": int(source_appln_id),
                    "source_lg": source_lg_norm,
                    "target_lg": target_language_norm,
                    "source_text": text,
                    "source_text_len": int(source_text_len) if source_text_len is not None else None,
                    "source_text_sha256": _sha256_text(sha_basis),
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

    return CandidateCounts(
        n_families_total=n_families_total,
        n_title_candidates=n_title_candidates,
        n_abstract_candidates=n_abstract_candidates,
        n_total_candidates=n_total_candidates,
    )


def main() -> None:
    args = parse_args()
    counts = compute_and_optionally_write(
        input_path=Path(args.input),
        output_jsonl=Path(args.output_jsonl),
        report_md=Path(args.report_md),
        target_language=str(args.target_language),
        dry_run=bool(args.dry_run),
    )

    print("Family translation candidate build complete.")
    print(f"Families total: {counts.n_families_total:,}")
    print(f"Title candidates: {counts.n_title_candidates:,}")
    print(f"Abstract candidates: {counts.n_abstract_candidates:,}")
    print(f"Total candidate rows: {counts.n_total_candidates:,}")
    if args.dry_run:
        print("(dry-run) Did not write JSONL candidates.")
    else:
        print(f"Wrote candidates: {args.output_jsonl}")
    print(f"Wrote report: {args.report_md}")


if __name__ == "__main__":
    main()
