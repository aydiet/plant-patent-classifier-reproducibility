#!/usr/bin/env python3
"""
Generate all statistics and intermediate CSVs for Section 4: Corpus Analysis.

Outputs go to metadata/corpus_analysis_*.csv for tables and
paper/figures/ for plotting data.

Usage:
    .venv/bin/python scripts/analyze_plant_corpus_landscape.py
"""

import pathlib
import duckdb
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "global_corpus_publevel_plus.parquet"
SCORED = ROOT / "data" / "derived" / "family_modeling_corpus_en_complete__paecter_precision_seed2026-01-29__FULL__with_subtype.parquet"
META = ROOT / "metadata"
FIGDATA = ROOT / "paper" / "figures"
FIGDATA.mkdir(parents=True, exist_ok=True)

con = duckdb.connect()

# ── Helper: create family-level analysis table ────────────────────────────────
# Join scored predictions to publication-level data, derive fields, collapse to family level

print("Building family-level analysis table...")
con.sql(f"""
CREATE OR REPLACE TABLE analysis_base AS
WITH pub AS (
    SELECT *,
        YEAR(earliest_priority_or_filing_date::DATE) AS filing_year
    FROM read_parquet('{RAW}')
),
scored AS (
    SELECT docdb_family_id, label_pred, p_yes, subtype_pred, p_technology
    FROM read_parquet('{SCORED}')
),
fam AS (
    SELECT
        p.docdb_family_id,
        MIN(p.filing_year) AS filing_year,
        -- Plant status from scored
        MAX(s.label_pred) AS plant_pred,
        MAX(s.subtype_pred) AS subtype_pred,
        -- Family scope: distinct publication authorities
        COUNT(DISTINCT p.publn_auth) AS family_scope,
        MAX(p.has_ep) AS has_ep,
        MAX(p.has_wo) AS has_wo,
        -- First applicant name (from any pub in family — take first non-null)
        FIRST(p.applicant_names ORDER BY p.appln_id) FILTER (WHERE p.applicant_names IS NOT NULL AND p.applicant_names != '') AS applicant_names_raw,
        -- First applicant country
        FIRST(p.applicant_country_codes ORDER BY p.appln_id) FILTER (WHERE p.applicant_country_codes IS NOT NULL AND p.applicant_country_codes != '') AS applicant_country_raw,
        -- Primary application authority (earliest filing)
        FIRST(p.appln_auth ORDER BY p.appln_filing_date, p.appln_id) AS primary_appln_auth,
        -- CPC and IPC lists (from any pub)
        FIRST(p.cpc_list ORDER BY p.appln_id) FILTER (WHERE p.cpc_list IS NOT NULL AND p.cpc_list != '') AS cpc_list,
        FIRST(p.ipc_list ORDER BY p.appln_id) FILTER (WHERE p.ipc_list IS NOT NULL AND p.ipc_list != '') AS ipc_list,
        -- US Plant Patent flag (35 USC 161): any P-kind US publication
        MAX(CASE WHEN p.publn_auth = 'US' AND TRIM(p.publn_kind) IN ('P','P1','P2','P3','P4','P9') THEN 1 ELSE 0 END) AS has_us_plant_patent,
        -- Utility model flag
        MAX(CASE WHEN TRIM(p.appln_kind) = 'U' THEN 1 ELSE 0 END) AS has_utility_model,
        -- Publication count
        COUNT(DISTINCT p.appln_id) AS n_applications,
        COUNT(*) AS n_publications
    FROM pub p
    LEFT JOIN scored s ON p.docdb_family_id = s.docdb_family_id
    GROUP BY p.docdb_family_id
)
SELECT
    f.*,
    -- Derive first applicant name (before first semicolon)
    CASE 
        WHEN f.applicant_names_raw LIKE '%;%' 
        THEN TRIM(SPLIT_PART(f.applicant_names_raw, ';', 1))
        ELSE TRIM(f.applicant_names_raw)
    END AS first_applicant,
    -- Derive first applicant country
    CASE 
        WHEN f.applicant_country_raw LIKE '%;%' 
        THEN TRIM(SPLIT_PART(f.applicant_country_raw, ';', 1))
        ELSE TRIM(f.applicant_country_raw)
    END AS applicant_country,
    -- Is plant-related
    CASE WHEN f.plant_pred = 'yes' THEN 1 ELSE 0 END AS is_plant,
    -- Subtype flags
    CASE WHEN f.subtype_pred = 'variety' THEN 1 ELSE 0 END AS is_variety,
    CASE WHEN f.subtype_pred = 'technology' THEN 1 ELSE 0 END AS is_technology,
    -- Has A01H code
    CASE WHEN f.cpc_list LIKE '%A01H%' OR f.ipc_list LIKE '%A01H%' THEN 1 ELSE 0 END AS has_a01h,
    -- US Plant Patent and utility model (pass through)
    f.has_us_plant_patent,
    f.has_utility_model,
    -- CPC section (first letter)
    CASE
        WHEN f.cpc_list IS NOT NULL AND f.cpc_list != '' THEN LEFT(TRIM(SPLIT_PART(f.cpc_list, ';', 1)), 1)
        WHEN f.ipc_list IS NOT NULL AND f.ipc_list != '' THEN LEFT(TRIM(SPLIT_PART(f.ipc_list, ';', 1)), 1)
        ELSE NULL
    END AS cpc_section
FROM fam f
WHERE f.filing_year BETWEEN 1985 AND 2023
""")

total = con.sql("SELECT COUNT(*) FROM analysis_base").fetchone()[0]
plant = con.sql("SELECT SUM(is_plant) FROM analysis_base").fetchone()[0]
print(f"Analysis base: {total:,} families (1985-2023), {plant:,} plant-related")

# ═══════════════════════════════════════════════════════════════════════════════
# H1: Growth trends — plant-related vs. baseline
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H1: Growth trends ===")
h1 = con.sql("""
SELECT 
    filing_year AS year,
    COUNT(*) AS all_families,
    SUM(is_plant) AS plant_families,
    SUM(is_variety) AS variety_families,
    SUM(is_technology) AS technology_families
FROM analysis_base
GROUP BY filing_year
ORDER BY filing_year
""").df()
h1.to_csv(META / "corpus_analysis_h1_growth_trends.csv", index=False)
print(h1.tail(10).to_string())

# CAGR calculations
for period_name, y0, y1 in [("1985-2023", 1985, 2023), ("2000-2023", 2000, 2023), ("2010-2023", 2010, 2023)]:
    row0 = h1[h1.year == y0].iloc[0]
    row1 = h1[h1.year == y1].iloc[0]
    n = y1 - y0
    for col in ["all_families", "plant_families"]:
        if row0[col] > 0:
            cagr = (row1[col] / row0[col]) ** (1/n) - 1
            print(f"  {period_name} CAGR {col}: {cagr*100:.2f}%")

# ═══════════════════════════════════════════════════════════════════════════════
# H2: Applicant concentration
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H2: Applicant concentration ===")

# Top 50 raw applicants (plant-related)
h2_top50 = con.sql("""
SELECT 
    UPPER(TRIM(first_applicant)) AS applicant_upper,
    first_applicant AS applicant_example,
    COUNT(*) AS family_count,
    SUM(is_variety) AS variety_count,
    SUM(is_technology) AS technology_count
FROM analysis_base
WHERE is_plant = 1 AND first_applicant IS NOT NULL AND first_applicant != ''
GROUP BY UPPER(TRIM(first_applicant)), first_applicant
ORDER BY family_count DESC
LIMIT 50
""").df()
h2_top50.to_csv(META / "corpus_analysis_h2_top50_applicants_raw.csv", index=False)
print("Top-20 applicants (plant-related, raw):")
print(h2_top50.head(20)[["applicant_example", "family_count", "variety_count", "technology_count"]].to_string())

# Overall HHI (plant-related)
hhi_plant = con.sql("""
WITH counts AS (
    SELECT UPPER(TRIM(first_applicant)) AS app, COUNT(*) AS n
    FROM analysis_base
    WHERE is_plant = 1 AND first_applicant IS NOT NULL AND first_applicant != ''
    GROUP BY UPPER(TRIM(first_applicant))
),
total AS (SELECT SUM(n) AS total_n FROM counts)
SELECT ROUND(SUM((n * 1.0 / total_n) * (n * 1.0 / total_n) * 10000), 2) AS hhi
FROM counts, total
""").fetchone()[0]
print(f"\nPlant-related HHI: {hhi_plant}")

hhi_all = con.sql("""
WITH counts AS (
    SELECT UPPER(TRIM(first_applicant)) AS app, COUNT(*) AS n
    FROM analysis_base
    WHERE first_applicant IS NOT NULL AND first_applicant != ''
    GROUP BY UPPER(TRIM(first_applicant))
),
total AS (SELECT SUM(n) AS total_n FROM counts)
SELECT ROUND(SUM((n * 1.0 / total_n) * (n * 1.0 / total_n) * 10000), 2) AS hhi
FROM counts, total
""").fetchone()[0]
print(f"Overall HHI: {hhi_all}")

# ═══════════════════════════════════════════════════════════════════════════════
# H3: Geographic specialization
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H3: Geographic specialization ===")
h3 = con.sql("""
SELECT 
    primary_appln_auth AS auth,
    COUNT(*) AS all_families,
    SUM(is_plant) AS plant_families,
    ROUND(100.0 * SUM(is_plant) / COUNT(*), 2) AS plant_share_pct,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS auth_share_of_total_pct,
    ROUND(100.0 * SUM(is_plant) / SUM(SUM(is_plant)) OVER (), 2) AS auth_share_of_plant_pct
FROM analysis_base
GROUP BY primary_appln_auth
ORDER BY plant_families DESC
LIMIT 20
""").df()
h3.to_csv(META / "corpus_analysis_h3_geography.csv", index=False)
print(h3.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# H4: Internationalization trend
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H4: Internationalization trends ===")
h4 = con.sql("""
SELECT 
    filing_year AS year,
    -- All families
    ROUND(AVG(family_scope), 3) AS avg_scope_all,
    ROUND(100.0 * AVG(has_ep), 2) AS pct_ep_all,
    ROUND(100.0 * AVG(has_wo), 2) AS pct_wo_all,
    -- Plant families
    ROUND(AVG(CASE WHEN is_plant = 1 THEN family_scope END), 3) AS avg_scope_plant,
    ROUND(100.0 * AVG(CASE WHEN is_plant = 1 THEN has_ep END), 2) AS pct_ep_plant,
    ROUND(100.0 * AVG(CASE WHEN is_plant = 1 THEN has_wo END), 2) AS pct_wo_plant
FROM analysis_base
GROUP BY filing_year
ORDER BY filing_year
""").df()
h4.to_csv(META / "corpus_analysis_h4_internationalization.csv", index=False)
print(h4.tail(10).to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# H5: Patent instrument breakdown (plant families)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H5: Patent instrument breakdown ===")
h5c = con.sql("""
SELECT
    CASE WHEN is_variety = 1 THEN 'Variety'
         WHEN is_technology = 1 THEN 'Technology'
         ELSE 'Other' END AS subtype,
    CASE
        WHEN has_us_plant_patent = 1 THEN 'US Plant Patent (35 USC 161)'
        WHEN has_utility_model = 1 THEN 'Utility Model'
        ELSE 'Invention Patent'
    END AS instrument,
    COUNT(*) AS n_families,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (PARTITION BY
        CASE WHEN is_variety = 1 THEN 'Variety'
             WHEN is_technology = 1 THEN 'Technology'
             ELSE 'Other' END), 1) AS pct_of_subtype
FROM analysis_base
WHERE is_plant = 1
GROUP BY subtype, instrument
ORDER BY subtype, n_families DESC
""").df()
h5c.to_csv(META / "corpus_analysis_h5c_patent_instruments.csv", index=False)
print(h5c.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# H6: Temporal divergence variety vs technology
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H6: Variety vs Technology over time ===")
# (Data already in h1)
print("(See h1 output - variety_families and technology_families columns)")

# ═══════════════════════════════════════════════════════════════════════════════
# H7: Geographic differentiation by subtype
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H7: Geography by subtype ===")
h7 = con.sql("""
SELECT 
    primary_appln_auth AS auth,
    SUM(is_variety) AS variety_families,
    SUM(is_technology) AS technology_families,
    SUM(is_plant) AS plant_total,
    ROUND(100.0 * SUM(is_variety) / NULLIF(SUM(is_plant), 0), 2) AS variety_share_pct,
    ROUND(100.0 * SUM(is_technology) / NULLIF(SUM(is_plant), 0), 2) AS tech_share_pct
FROM analysis_base
WHERE is_plant = 1
GROUP BY primary_appln_auth
ORDER BY plant_total DESC
LIMIT 15
""").df()
h7.to_csv(META / "corpus_analysis_h7_geography_by_subtype.csv", index=False)
print(h7.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# H8: Top applicants by subtype
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H8: Top applicants - Variety ===")
h8_var = con.sql("""
SELECT 
    UPPER(TRIM(first_applicant)) AS applicant,
    COUNT(*) AS family_count
FROM analysis_base
WHERE is_variety = 1 AND first_applicant IS NOT NULL AND first_applicant != ''
GROUP BY UPPER(TRIM(first_applicant))
ORDER BY family_count DESC
LIMIT 20
""").df()
print(h8_var.to_string())

print("\n=== H8: Top applicants - Technology ===")
h8_tech = con.sql("""
SELECT 
    UPPER(TRIM(first_applicant)) AS applicant,
    COUNT(*) AS family_count
FROM analysis_base
WHERE is_technology = 1 AND first_applicant IS NOT NULL AND first_applicant != ''
GROUP BY UPPER(TRIM(first_applicant))
ORDER BY family_count DESC
LIMIT 20
""").df()
print(h8_tech.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# H9: Family scope by subtype
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H9: Family scope by subtype ===")
h9 = con.sql("""
SELECT 
    subtype_pred AS subtype,
    COUNT(*) AS n_families,
    ROUND(AVG(family_scope), 3) AS avg_scope,
    ROUND(MEDIAN(family_scope), 1) AS median_scope,
    ROUND(100.0 * AVG(has_ep), 2) AS pct_ep,
    ROUND(100.0 * AVG(has_wo), 2) AS pct_wo
FROM analysis_base
WHERE is_plant = 1
GROUP BY subtype_pred
""").df()
print(h9.to_string())

# ═══════════════════════════════════════════════════════════════════════════════
# H10: Classifier uplift over A01H
# ═══════════════════════════════════════════════════════════════════════════════
print("\n=== H10: Classifier uplift ===")
h10 = con.sql("""
SELECT 
    SUM(is_plant) AS plant_families,
    SUM(CASE WHEN is_plant = 1 AND has_a01h = 1 THEN 1 ELSE 0 END) AS plant_with_a01h,
    SUM(CASE WHEN is_plant = 1 AND has_a01h = 0 THEN 1 ELSE 0 END) AS plant_without_a01h,
    ROUND(100.0 * SUM(CASE WHEN is_plant = 1 AND has_a01h = 0 THEN 1 ELSE 0 END) / SUM(is_plant), 2) AS uplift_pct
FROM analysis_base
""").df()
print(h10.to_string())

# CPC section distribution for plant-related families
h10_cpc = con.sql("""
SELECT 
    COALESCE(cpc_section, 'Unknown') AS section,
    COUNT(*) AS plant_families,
    ROUND(100.0 * COUNT(*) / SUM(COUNT(*)) OVER (), 2) AS pct
FROM analysis_base
WHERE is_plant = 1
GROUP BY cpc_section
ORDER BY plant_families DESC
""").df()
h10_cpc.to_csv(META / "corpus_analysis_h10_cpc_sections.csv", index=False)
print("\nCPC section distribution (plant-related):")
print(h10_cpc.to_string())

print("\n=== DONE ===")
print(f"Output CSVs saved to {META}/corpus_analysis_*.csv")
