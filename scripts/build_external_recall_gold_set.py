#!/usr/bin/env python
"""
Build the positive-only external recall set from two independent source registries.

The script maps source-registry patent records to DOCDB families, checks overlap
with the 600-family labeled set, and writes one representative English
title/abstract row per matched family. Registry inputs are not redistributed in
the public Appendix D companion repository.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import duckdb
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--enriched",
        default="data/derived/global_corpus_publevel_plus_text_enriched_apple_nllb.parquet",
        help="Text-enriched publication-level corpus.",
    )
    parser.add_argument(
        "--registry-a-parquet",
        default="metadata/external_registry_a_positives_2026-03-03.parquet",
        help="Positive families from the first external registry.",
    )
    parser.add_argument(
        "--registry-b-xlsx",
        default="data/raw/industry_patent_pool_register_oct2025.xlsx",
        help="Raw workbook from the second external source registry.",
    )
    parser.add_argument(
        "--registry-b-sheets",
        nargs="+",
        default=["ILP Patent Register", "Expired Patents"],
        help="Workbook sheets containing source-registry patent records.",
    )
    parser.add_argument("--labeled-csv", default="metadata/labeling_candidates_600.csv")
    parser.add_argument("--splits-csv", default="metadata/labeling_splits_420_90_90.csv")
    parser.add_argument(
        "--out-parquet",
        default="metadata/gold_test_external_combined_positives.parquet",
    )
    parser.add_argument(
        "--out-report",
        default="metadata/gold_test_external_combined_report.md",
    )
    parser.add_argument(
        "--out-registry-b-mapping",
        default="metadata/external_recall_registry_b_family_mapping.csv",
    )
    return parser.parse_args()


def parse_registry_b_workbook(path: str, sheets: list[str]) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for sheet in sheets:
        df = pd.read_excel(path, sheet_name=sheet)
        named_cols = [c for c in df.columns if not str(c).startswith("Unnamed")]
        df = df[named_cols].dropna(how="all").copy()
        print(f'  Sheet "{sheet}": {len(df)} rows')

        for _, row in df.iterrows():
            member = str(row.get("ILP Member", "")).strip()
            title = str(row.get("PCT Patent Title", "")).strip()
            trait = str(row.get("Trait / Characteristic", "")).strip()
            species = str(row.get("Related Vegetable (primary use)", "")).strip()

            raw_numbers: list[str] = []
            for col in [
                "PCT Publication Number",
                "EP Publication Number (incl. Hyperlink to EspaceNet) ",
                "US or other national publication number",
            ]:
                val = str(row.get(col, "")).strip()
                if val and val not in {"nan", "n.a.", "NaN", "None"}:
                    raw_numbers.append(val)

            for raw in raw_numbers:
                raw = re.sub(r"https?://\S+", "", raw)
                for token in re.findall(r"[A-Z]{2}\s*[\d,]+[A-Z]*\d*", raw):
                    token_clean = token.strip().replace(",", "")
                    match = re.match(r"^([A-Z]{2})\s*(\d+)", token_clean)
                    if not match:
                        continue
                    records.append(
                        {
                            "publn_auth": match.group(1),
                            "publn_nr": match.group(2),
                            "raw_number": token.strip(),
                            "source_sheet": sheet,
                            "source_member": member,
                            "source_title": title,
                            "source_trait": trait,
                            "source_species": species,
                        }
                    )

    out = pd.DataFrame(records).drop_duplicates(subset=["publn_auth", "publn_nr"])
    print(f"  Unique registry-B publication numbers: {len(out)}")
    return out


def main() -> None:
    args = parse_args()
    base = Path(__file__).resolve().parents[1]
    out_report = base / args.out_report
    out_report.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("1. Parsing second external registry")
    print("=" * 60)
    registry_b_df = parse_registry_b_workbook(
        str(base / args.registry_b_xlsx),
        args.registry_b_sheets,
    )

    print("\n" + "=" * 60)
    print("2. Mapping registry-B records to DOCDB families")
    print("=" * 60)
    con = duckdb.connect()
    con.execute(
        "CREATE VIEW enriched AS SELECT * FROM read_parquet(?)",
        [str(base / args.enriched)],
    )
    con.execute("CREATE TABLE registry_b_pubs AS SELECT * FROM registry_b_df")

    registry_b_matches = con.execute(
        """
        SELECT DISTINCT
            b.publn_auth AS source_auth,
            b.publn_nr AS source_nr,
            b.raw_number,
            b.source_sheet,
            b.source_member,
            b.source_title,
            e.docdb_family_id
        FROM registry_b_pubs b
        JOIN enriched e
          ON CAST(e.publn_nr AS VARCHAR) = b.publn_nr
         AND e.publn_auth = b.publn_auth
        """
    ).fetchdf()

    matched_keys = set(zip(registry_b_matches["source_auth"], registry_b_matches["source_nr"]))
    registry_b_families = set(registry_b_matches["docdb_family_id"].astype(int))
    unmatched_b = registry_b_df[
        ~registry_b_df.apply(lambda r: (r["publn_auth"], r["publn_nr"]) in matched_keys, axis=1)
    ]
    print(
        f"  Matched {len(matched_keys)}/{len(registry_b_df)} publication numbers "
        f"to {len(registry_b_families)} DOCDB families"
    )
    registry_b_matches.to_csv(base / args.out_registry_b_mapping, index=False)

    print("\n" + "=" * 60)
    print("3. Combining external source families")
    print("=" * 60)
    registry_a_df = pd.read_parquet(base / args.registry_a_parquet)
    registry_a_families = set(registry_a_df["docdb_family_id"].astype(int).unique())
    overlap = registry_a_families & registry_b_families
    combined_families = registry_a_families | registry_b_families

    print(f"  Registry A families: {len(registry_a_families)}")
    print(f"  Registry B families: {len(registry_b_families)}")
    print(f"  Overlap: {len(overlap)}")
    print(f"  Combined total: {len(combined_families)}")

    labeled = pd.read_csv(base / args.labeled_csv)
    labeled_families = set(labeled["docdb_family_id"].astype(int))
    overlap_labeled = combined_families & labeled_families
    print(f"  Overlap with labeled 600-family set: {len(overlap_labeled)}")

    splits = pd.read_csv(base / args.splits_csv)
    for split_name in ["train", "val", "test"]:
        split_families = set(splits[splits["split"] == split_name]["docdb_family_id"].astype(int))
        print(f"  Overlap with {split_name:>5s} split: {len(combined_families & split_families)}")

    print("\n" + "=" * 60)
    print("4. Selecting representative family text")
    print("=" * 60)
    combined_list = ",".join(str(f) for f in sorted(combined_families))
    combined_df = con.execute(
        f"""
        WITH ranked AS (
            SELECT *,
                LENGTH(COALESCE(appln_title_en_final, ''))
                + LENGTH(COALESCE(appln_abstract_en_final, '')) AS text_len,
                ROW_NUMBER() OVER (
                    PARTITION BY docdb_family_id
                    ORDER BY
                        LENGTH(COALESCE(appln_title_en_final, ''))
                        + LENGTH(COALESCE(appln_abstract_en_final, '')) DESC,
                        appln_id DESC
                ) AS rn
            FROM enriched
            WHERE docdb_family_id IN ({combined_list})
        )
        SELECT * FROM ranked WHERE rn = 1
        """
    ).fetchdf()

    def source_for_family(family_id: int) -> str:
        in_a = family_id in registry_a_families
        in_b = family_id in registry_b_families
        if in_a and in_b:
            return "registry_a+registry_b"
        if in_a:
            return "registry_a"
        return "registry_b"

    combined_df["gold_label"] = 1
    combined_df["gold_source"] = combined_df["docdb_family_id"].astype(int).map(source_for_family)
    combined_df["gold_confidence"] = "high"
    combined_df = combined_df[[c for c in combined_df.columns if c not in {"rn", "text_len"}]]

    has_title = combined_df["appln_title_en_final"].notna() & (
        combined_df["appln_title_en_final"].str.len() > 0
    )
    has_abstract = combined_df["appln_abstract_en_final"].notna() & (
        combined_df["appln_abstract_en_final"].str.len() > 0
    )
    has_both = has_title & has_abstract
    combined_df.to_parquet(base / args.out_parquet, compression="zstd")
    print(f"  Total families: {len(combined_df)}")
    print(f"  With title and abstract: {int(has_both.sum())}")
    print(f"  Missing title or abstract: {int((~has_both).sum())}")

    source_counts = combined_df["gold_source"].value_counts()
    report = f"""# External Recall Gold Set - Positives Only

**Generated by:** `scripts/build_external_recall_gold_set.py`

## Source Summary

| Source | Unique families | Also in other source |
|---|---:|---:|
| Registry A | {len(registry_a_families)} | {len(overlap)} |
| Registry B | {len(registry_b_families)} | {len(overlap)} |

## Combined Set

| Metric | Value |
|---|---:|
| Total unique DOCDB families | {len(combined_df)} |
| Registry A only | {int(source_counts.get('registry_a', 0))} |
| Registry B only | {int(source_counts.get('registry_b', 0))} |
| In both sources | {int(source_counts.get('registry_a+registry_b', 0))} |
| With title and abstract | {int(has_both.sum())} |
| Missing title or abstract | {int((~has_both).sum())} |
| Overlap with 600-family labeled set | {len(overlap_labeled)} |

## Registry-B Mapping Summary

- Total publication numbers parsed: {len(registry_b_df)}
- Matched to enriched corpus: {len(matched_keys)}
- Unmatched: {len(unmatched_b)}
- Unique DOCDB families from registry B: {len(registry_b_families)}
"""
    out_report.write_text(report, encoding="utf-8")
    print(f"  Saved: {args.out_parquet}")
    print(f"  Saved report: {args.out_report}")


if __name__ == "__main__":
    main()
