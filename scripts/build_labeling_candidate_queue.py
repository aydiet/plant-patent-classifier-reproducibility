#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import duckdb


def _parse_csv_prefixes(s: str) -> list[str]:
    parts = [p.strip().upper() for p in s.split(",")]
    return [p for p in parts if p]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Build a candidate queue for manual labeling (binary plant vs non-plant) at DOCDB-family level. "
            "Uses the family-level English text corpus for stable 1-row-per-family text, and joins to the "
            "enriched publication-level parquet to bring in CPC/IPC/date metadata for stratification. "
            "Outputs a CSV with candidates split across plant-likely, non-plant-likely, and borderline buckets."
        )
    )
    ap.add_argument(
        "--family-parquet",
        type=Path,
        default=Path("data/derived/family_modeling_corpus_en_complete.parquet"),
        help="Family-level corpus parquet (1 row per docdb_family_id).",
    )
    ap.add_argument(
        "--publevel-parquet",
        type=Path,
        default=Path("data/derived/global_corpus_publevel_plus_text_enriched_apple_nllb.parquet"),
        help=(
            "Enriched publication-level parquet used only to join metadata like cpc_list/ipc_list/dates. "
            "Must contain appln_id and the metadata columns."
        ),
    )
    ap.add_argument(
        "--out-csv",
        type=Path,
        default=Path("metadata/labeling_candidates_600.csv"),
        help="Output CSV to label (committable).",
    )
    ap.add_argument(
        "--n-total",
        type=int,
        default=600,
        help="Total number of candidates to output (approximately 50/50 across buckets).",
    )
    ap.add_argument(
        "--n-borderline",
        type=int,
        default=0,
        help=(
            "How many candidates to reserve for the borderline bucket (0 = auto = 20% of --n-total). "
            "The remainder is split across plant_likely and non_plant_likely."
        ),
    )
    ap.add_argument(
        "--seed",
        type=str,
        default="2026-01-13",
        help="Deterministic seed string used only for stable ordering/sampling.",
    )
    ap.add_argument(
        "--plant-cpc-prefixes",
        type=str,
        default="A01H,A01G,A01C,A01D,A01F,A01M,A01N",
        help=(
            "Comma-separated CPC/IPC prefixes used to define a plant-likely bucket. "
            "Example: 'A01H,A01G'."
        ),
    )
    ap.add_argument(
        "--negative-cpc-prefixes",
        type=str,
        default="G06,H04,H01,F16,F02,B60,B65",
        help=(
            "Comma-separated CPC/IPC prefixes used to define a non-plant-likely bucket. "
            "This keeps negatives mostly in clearly non-plant domains."
        ),
    )

    args = ap.parse_args()

    if not args.family_parquet.exists():
        raise SystemExit(f"Not found: {args.family_parquet}")
    if not args.publevel_parquet.exists():
        raise SystemExit(f"Not found: {args.publevel_parquet}")
    if args.n_total <= 0:
        raise SystemExit("--n-total must be > 0")

    plant_prefixes = _parse_csv_prefixes(args.plant_cpc_prefixes)
    neg_prefixes = _parse_csv_prefixes(args.negative_cpc_prefixes)
    if not plant_prefixes:
        raise SystemExit("--plant-cpc-prefixes must not be empty")
    if not neg_prefixes:
        raise SystemExit("--negative-cpc-prefixes must not be empty")

    out_csv = args.out_csv
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(database=":memory:")

    family_p = args.family_parquet.as_posix()
    pub_p = args.publevel_parquet.as_posix()

    con.execute(f"CREATE VIEW fam AS SELECT * FROM read_parquet('{family_p}')")
    con.execute(f"CREATE VIEW pub AS SELECT * FROM read_parquet('{pub_p}')")

    # `pub` is publication-level and may contain multiple rows per `appln_id`.
    # Collapse to a deterministic one-row-per-application view before joining,
    # otherwise we will duplicate families in the labeling queue.
    con.execute(
        """
        CREATE TEMP TABLE pub_by_appln AS
        SELECT
          appln_id,
          arg_max(cpc_list, length(COALESCE(cpc_list, ''))) AS cpc_list,
          arg_max(ipc_list, length(COALESCE(ipc_list, ''))) AS ipc_list,
          min(appln_filing_date) AS appln_filing_date,
          max(publn_date) AS publn_date,
          arg_max(appln_auth, publn_date) AS appln_auth,
          arg_max(publn_auth, publn_date) AS publn_auth,
          min(earliest_priority_or_filing_date) AS earliest_priority_or_filing_date,
          max(COALESCE(CAST(has_ep AS INTEGER), 0))::INTEGER AS has_ep,
          max(COALESCE(CAST(has_wo AS INTEGER), 0))::INTEGER AS has_wo
        FROM pub
        GROUP BY appln_id
        """
    )

    # Join in stratification metadata from the representative appln_id.
    # Note: This assumes rep_appln_id exists in the family parquet (as created by export_family_modeling_corpus_en.py).
    con.execute(
        """
        CREATE TEMP TABLE fam_plus AS
        SELECT
          f.docdb_family_id,
          f.rep_appln_id,
          f.title_en,
          f.abstract_en,
          f.text_en,

          p.cpc_list,
          p.ipc_list,

          p.appln_filing_date,
          p.publn_date,
          p.appln_auth,
          p.publn_auth,

          p.earliest_priority_or_filing_date,
          p.has_ep,
          p.has_wo
        FROM fam f
                LEFT JOIN pub_by_appln p
          ON p.appln_id = f.rep_appln_id
        """
    )

    # --- Heuristics ---
    # CPC/IPC lists are stored as strings with tokens separated by '; '.
    # We check for prefix matches in either the first token or subsequent tokens.
    def _prefix_or_clause(column: str, prefixes: list[str]) -> str:
        parts: list[str] = []
        for pref in prefixes:
            # token start or after '; '
            parts.append(f"{column} LIKE '{pref}%' OR {column} LIKE '%; {pref}%'")
        return "(" + " OR ".join(parts) + ")"

    plant_cpc_match = (
        _prefix_or_clause("COALESCE(cpc_list, '')", plant_prefixes)
        + " OR "
        + _prefix_or_clause("COALESCE(ipc_list, '')", plant_prefixes)
    )

    neg_cpc_match = (
        _prefix_or_clause("COALESCE(cpc_list, '')", neg_prefixes)
        + " OR "
        + _prefix_or_clause("COALESCE(ipc_list, '')", neg_prefixes)
    )

    # Text keywords: conservative, but broad enough to yield a usable plant-likely pool.
    # Option A (your definition): variety-level AND technology-level inventions are plant-related.
    variety_kw = [
        "cultivar",
        "plant variety",
        "new and distinct",
        "line designated",
        "inbred line",
        "hybrid plant",
        "rootstock",
        "germplasm",
    ]
    technology_kw = [
        "plant breeding",
        "marker-assisted",
        "genomic selection",
        "trait introgression",
        "plant transformation",
        "plant regeneration",
        "micropropagation",
        "doubled haploid",
        "agrobacterium",
        "plant promoter",
        "plant expression",
        "transgenic plant",
        "genome editing",
        "crispr",
        "crop",
        "seed coating",
    ]
    kw = list(dict.fromkeys(variety_kw + technology_kw))
    kw_clause = " OR ".join([f"lower(COALESCE(text_en, '')) LIKE '%{k}%'" for k in kw])
    kw_clause = f"({kw_clause})" if kw_clause else "(FALSE)"

    # Core signals (used to build mutually exclusive buckets).
    plant_signal = f"(({plant_cpc_match}) OR ({kw_clause}))"
    neg_signal = f"({neg_cpc_match})"
    strong_plant_signal = f"(({plant_cpc_match}) AND ({kw_clause}))"

    # Simple strata for deterministic diversity during sampling.
    # - time_bin from earliest_priority_or_filing_date (fallback to appln_filing_date)
    # - cpc_section from the first character of the first CPC token (fallback to IPC)
    time_bin_sql = """
            CASE
                WHEN COALESCE(earliest_priority_or_filing_date, appln_filing_date) IS NULL THEN 'unknown'
                WHEN EXTRACT(year FROM COALESCE(earliest_priority_or_filing_date, appln_filing_date)) < 1990 THEN '<1990'
                WHEN EXTRACT(year FROM COALESCE(earliest_priority_or_filing_date, appln_filing_date)) < 2000 THEN '1990s'
                WHEN EXTRACT(year FROM COALESCE(earliest_priority_or_filing_date, appln_filing_date)) < 2010 THEN '2000s'
                WHEN EXTRACT(year FROM COALESCE(earliest_priority_or_filing_date, appln_filing_date)) < 2020 THEN '2010s'
                ELSE '2020s+'
            END
    """.strip()

    cpc_section_sql = """
            CASE
                WHEN COALESCE(cpc_list, '') <> '' THEN upper(substr(cpc_list, 1, 1))
                WHEN COALESCE(ipc_list, '') <> '' THEN upper(substr(ipc_list, 1, 1))
                ELSE 'unknown'
            END
    """.strip()

    con.execute(
                f"""
                CREATE TEMP TABLE fam_scored AS
                SELECT
                    *,
                    ({plant_cpc_match}) AS has_plant_cpc,
                    ({kw_clause}) AS has_plant_kw,
                    ({plant_signal}) AS has_plant_signal,
                    ({neg_signal}) AS has_neg_signal,
                    ({strong_plant_signal}) AS has_strong_plant_signal,
                    EXTRACT(year FROM COALESCE(earliest_priority_or_filing_date, appln_filing_date))::INTEGER AS priority_or_filing_year,
                    {time_bin_sql} AS time_bin,
                    {cpc_section_sql} AS cpc_section,
                    ({time_bin_sql} || '|' || {cpc_section_sql}) AS stratum
                FROM fam_plus
                """
    )

    # Define mutually exclusive buckets.
    # plant_likely: strong plant evidence (plant CPC/IPC AND plant keywords) and not explicitly negative.
    # non_plant_likely: explicitly negative CPC/IPC and no plant evidence.
    # borderline: remaining cases with at least some plant evidence but weaker/mixed signals.
    con.execute(
                """
                CREATE TEMP TABLE plant_likely AS
                SELECT *,
                    'plant_likely' AS bucket
                FROM fam_scored
                WHERE has_strong_plant_signal AND NOT has_neg_signal
                """
    )

    con.execute(
                """
                CREATE TEMP TABLE non_plant_likely AS
                SELECT *,
                    'non_plant_likely' AS bucket
                FROM fam_scored
                WHERE
                    has_neg_signal
                    AND NOT has_plant_signal
                """
    )

    con.execute(
                """
                CREATE TEMP TABLE borderline AS
                SELECT *,
                    'borderline' AS bucket
                FROM fam_scored
                WHERE
                    has_plant_signal
                    AND NOT (has_strong_plant_signal AND NOT has_neg_signal)
                """
    )

    # Deterministic sampling:
    # - Allocate bucket sizes (defaults: 40/40/20 for plant/non-plant/borderline)
    # - Within each bucket, do deterministic round-robin across strata (time_bin x CPC section)
    # - All ordering is driven by md5(docdb_family_id || seed) so reruns are stable
    if args.n_borderline < 0:
        raise SystemExit("--n-borderline must be >= 0")

    n_total = args.n_total
    n_borderline = args.n_borderline if args.n_borderline > 0 else max(1, int(round(n_total * 0.20)))
    if n_borderline >= n_total:
        raise SystemExit("--n-borderline must be < --n-total")
    remaining = n_total - n_borderline
    n_plant = remaining // 2
    n_nonplant = remaining - n_plant

    seed_sql = args.seed.replace("'", "''")

    con.execute("CREATE TEMP TABLE bucket_limits(bucket VARCHAR, n_target BIGINT)")
    con.execute(
        f"""
        INSERT INTO bucket_limits VALUES
          ('plant_likely', {n_plant}),
          ('non_plant_likely', {n_nonplant}),
          ('borderline', {n_borderline})
        """
    )

    # Output schema for labeling.
    # label: empty string initially; you will fill with 1/0 or yes/no.
    output_sql = f"""
        COPY (
            WITH all_candidates AS (
                SELECT * FROM plant_likely
                UNION ALL
                SELECT * FROM non_plant_likely
                UNION ALL
                SELECT * FROM borderline
            ),
            ranked AS (
                SELECT
                    docdb_family_id,
                    rep_appln_id,
                    bucket,
                    cpc_list,
                    ipc_list,
                    cpc_section,
                    time_bin,
                    priority_or_filing_year,
                    stratum,
                    earliest_priority_or_filing_date,
                    appln_filing_date,
                    publn_date,
                    appln_auth,
                    publn_auth,
                    has_ep,
                    has_wo,
                    title_en,
                    abstract_en,
                    text_en,
                    row_number() OVER (
                        PARTITION BY bucket, stratum
                        ORDER BY md5(CAST(docdb_family_id AS VARCHAR) || '{seed_sql}')
                    ) AS rn_in_stratum,
                    md5(stratum || bucket || '{seed_sql}') AS stratum_hash
                FROM all_candidates
            ),
            picked AS (
                SELECT
                    r.*,
                    bl.n_target,
                    row_number() OVER (
                        PARTITION BY r.bucket
                        ORDER BY r.rn_in_stratum ASC, r.stratum_hash ASC
                    ) AS rn_in_bucket
                FROM ranked r
                JOIN bucket_limits bl USING (bucket)
            )
            SELECT
                docdb_family_id,
                rep_appln_id,
                bucket,
                cpc_section,
                time_bin,
                priority_or_filing_year,
                cpc_list,
                ipc_list,
                earliest_priority_or_filing_date,
                appln_filing_date,
                publn_date,
                appln_auth,
                publn_auth,
                has_ep,
                has_wo,
                title_en,
                abstract_en,
                text_en,
                '' AS label,
                '' AS subtype,
                '' AS confidence,
                '' AS notes
            FROM picked
            WHERE rn_in_bucket <= n_target
        ) TO '{out_csv.as_posix()}' (HEADER, DELIMITER ',', QUOTE '"', ESCAPE '"');
        """

    con.execute(output_sql)

    plant_row = con.execute("SELECT COUNT(*) FROM plant_likely").fetchone()
    nonplant_row = con.execute("SELECT COUNT(*) FROM non_plant_likely").fetchone()
    borderline_row = con.execute("SELECT COUNT(*) FROM borderline").fetchone()
    assert plant_row is not None and nonplant_row is not None and borderline_row is not None
    plant_n = plant_row[0]
    nonplant_n = nonplant_row[0]
    borderline_n = borderline_row[0]

    print(f"Wrote: {out_csv}")
    print(f"Plant-likely pool size: {plant_n:,}")
    print(f"Non-plant-likely pool size: {nonplant_n:,}")
    print(f"Borderline pool size: {borderline_n:,}")
    print(
        f"Requested total: {n_total:,} (plant={n_plant:,}, nonplant={n_nonplant:,}, borderline={n_borderline:,}, seed='{args.seed}')"
    )

    if plant_n < n_plant:
        print("WARNING: plant-likely pool smaller than requested; relax strong-plant criteria or add prefixes/keywords.")
    if nonplant_n < n_nonplant:
        print("WARNING: non-plant-likely pool smaller than requested; expand negative prefixes.")
    if borderline_n < n_borderline:
        print("WARNING: borderline pool smaller than requested; relax borderline criteria.")


if __name__ == "__main__":
    main()
